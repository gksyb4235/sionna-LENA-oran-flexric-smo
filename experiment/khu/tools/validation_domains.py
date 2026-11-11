"""Check replay controls against the frozen ns-3 schedule domains.

TxP uses the existing MITIGATION_PLAN_ALLOW_CONTINUOUS_TXP=1 execution mode.
This module checks inputs only: it never rounds controls or changes plans.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PARAMETERS = (
    'txp_5g', 'txp_4g1', 'txp_4g2',
    'ret_5g', 'ret_4g1', 'ret_4g2',
    'cio_5g', 'cio_4g1', 'cio_4g2', 'ttt_ms', 'hys_db',
)
RET_VALUES = (3, 5, 7, 9, 12)
TTT_VALUES = (128, 256, 340, 512, 720)


def violation(parameter, value):
    """Return a domain-error description, or None for an admissible value."""
    if parameter not in PARAMETERS:
        return 'Unknown parameter; expected one of the eleven frozen coordinates'
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 'Control value must be a finite number'
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return 'Nonfinite parameter'
    if not math.isfinite(number):
        return 'Nonfinite parameter'
    if parameter == 'ttt_ms':
        return None if number in TTT_VALUES else 'TTT must be one of 128, 256, 340, 512, 720 ms'
    if parameter == 'hys_db':
        allowed = .5 <= number <= 5 and abs(2 * number - round(2 * number)) < 1e-9
        return None if allowed else 'HYS requires [0.5,5] in 0.5 dB steps'
    if parameter.startswith('txp_'):
        return None if 0 <= number <= 60 else 'Continuous TxP requires [0,60] dBm'
    if parameter.startswith('ret_'):
        allowed = any(abs(number - target) < 1e-7 for target in RET_VALUES)
        return None if allowed else 'RET must match an exact cache angle: 3, 5, 7, 9, 12 degrees'
    low, high = (5, 9) if parameter == 'cio_5g' else (-2, 2)
    allowed = low <= number <= high and abs(number - round(number)) < 1e-9
    return None if allowed else f'CIO requires integer values in [{low},{high}] dB'


def _report_value(value):
    # Keep reports valid JSON even when the rejected input contains NaN/Infinity.
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def audit_jobs(jobs):
    """Return every invalid initial/event control without mutating the inputs."""
    issues = []
    for job in jobs:
        stages = [('initial', 0, job['initial_parameters'])]
        stages.extend(('event', event['time_s'], event['updates']) for event in job['events'])
        for phase, time_s, controls in stages:
            for parameter, value in controls.items():
                reason = violation(parameter, value)
                if reason is not None:
                    issues.append(dict(job_id=job['job_id'], job_index=job.get('index'),
                                       phase=phase, time_s=time_s, parameter=parameter,
                                       value=_report_value(value), reason=reason))
    return issues


def assert_admissible_jobs(jobs):
    """Fail before any simulation if the selected jobs contain invalid controls."""
    issues = audit_jobs(jobs)
    if not issues:
        return
    sample = '; '.join(f"{r['job_id']} {r['phase']} t={r['time_s']}s "
                       f"{r['parameter']}={r['value']!r}: {r['reason']}"
                       for r in issues[:8])
    count = len({r['job_id'] for r in issues})
    raise ValueError(f'Frozen control-domain preflight failed: {len(issues)} invalid values '
                     f'across {count} jobs. {sample}. '
                     'Preserve these plans; regenerate them using admissible candidates. '
                     'No control value was rounded or changed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, help='Explicit optional audit-report path')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    issues = audit_jobs(manifest['jobs'])
    report = dict(manifest=str(args.manifest.resolve()), jobs_checked=len(manifest['jobs']),
                  invalid_jobs=len({r['job_id'] for r in issues}), invalid_values=len(issues),
                  admissible=not issues, txp_mode='continuous_0_to_60', issues=issues)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8', newline='\n')
    print(payload, end='')
    return 1 if issues else 0


if __name__ == '__main__':
    raise SystemExit(main())
