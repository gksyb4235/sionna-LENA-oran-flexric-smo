"""Frozen offline-DRL evaluation: four held-out traces x four matched arms.

The simulator runs in Linux/WSL. This runner never trains, selects, or overwrites
policies, never contacts E2/RIC, and only terminates its own child process groups.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
BACKEND = Path('/home/user/LENA-oran-flexric-smo')
SCHEMA = 'offline-drl-frozen-factorial-v1'
ARMS = {'none': 'none', 'lb_local': 'lb', 'mo_local': 'mo', 'both_local': 'both'}


def now():
    return datetime.now(timezone.utc).isoformat()


def process_start_ticks(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
        return int(raw[raw.rfind(')') + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def terminal_campaign_state(state, records, pending):
    """Derive terminal counts from final worker records, never a pre-stop poll."""
    complete = sorted(key for key, record in records.items() if record.get('status') == 'complete')
    failed = sorted(key for key, record in records.items() if record.get('status') != 'complete')
    return {**state, 'finished_utc': now(), 'updated_utc': now(), 'jobs': records,
            'running_ids': [], 'running_count': 0, 'completed_ids': complete,
            'completed_count': len(complete), 'failed_jobs': failed, 'failed_ids': failed,
            'failed_count': len(failed), 'pending_ids': sorted(pending),
            'pending_count': len(pending), 'started_count': len(records), 'memory_throttled': False}


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_child(root, name):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name):
        raise ValueError(f'Unsafe identifier {name!r}')
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Path escapes experiment root')
    return result


def make_plan():
    jobs = []
    for trace in (1419, 1420, 1421, 1422):
        for arm, mode in ARMS.items():
            jobs.append(dict(id=f'test_{arm}_t{trace}', phase='test', arm=arm,
                             mode=mode, reward='local', policy='eval', trace_seed=trace,
                             learning_seed=909, lb_start_s=180, mo_start_s=540))
    return dict(schema=SCHEMA, created_utc=now(), backend_root=str(BACKEND),
                binary='bin/ns3-drl-offline', controller_module='drl_offline.controller',
                checkpoint_dir='selected', duration_s=900, n_ues=300,
                control_start_s=180, control_interval_s=30, rng_seed=1, rng_run=1,
                test_traces=[1419, 1420, 1421, 1422],
                baseline={'txp_dbm': 43, 'ret_deg': 9, 'cio_db': [6, 0, -1], 'ttt_ms': 512, 'hys_db': 1},
                online_learning=False, policy_selection='completed offline; never use these test outcomes for selection',
                requested_workers=16, launch_stagger_s=8, estimated_job_ram_gib=1.5,
                reserved_ram_gib=3.0, jobs=jobs)


def read_plan(path):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if plan.get('schema') != SCHEMA or plan.get('online_learning') is not False:
        raise ValueError('Only a frozen offline-policy evaluation plan is accepted')
    if 'trace_overrides' in plan or 'trace_override_metadata' in plan:
        raise ValueError('Main evaluation must use original held-out mobility traces without overrides')
    if plan.get('duration_s') != 900 or plan.get('control_interval_s') != 30:
        raise ValueError('The paired evaluation requires 900-second runs and 30-second controls')
    if set(plan['test_traces']) != {1419, 1420, 1421, 1422}:
        raise ValueError('Unexpected held-out traces')
    pairs, identifiers = set(), set()
    for job in plan['jobs']:
        safe_child(ROOT / 'results', job['id'])
        if job['id'] in identifiers:
            raise ValueError('Duplicate job id')
        identifiers.add(job['id'])
        if job.get('phase') != 'test' or job.get('policy') != 'eval':
            raise ValueError('Training or exploratory evaluation is forbidden')
        if job.get('arm') not in ARMS or job.get('mode') != ARMS[job['arm']]:
            raise ValueError('Unexpected evaluation arm')
        if (job.get('lb_start_s'), job.get('mo_start_s')) != (180, 540):
            raise ValueError('Matched arms must have identical app activation times')
        pair = (job['trace_seed'], job['arm'])
        if pair in pairs:
            raise ValueError('Duplicate trace/arm')
        pairs.add(pair)
    if pairs != {(trace, arm) for trace in plan['test_traces'] for arm in ARMS}:
        raise ValueError('Evaluation must contain all 16 matched trace/arm jobs')
    for key in ('binary', 'checkpoint_dir'):
        relative = Path(plan[key])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'{key} escapes the isolated experiment')
        destination = (ROOT / relative).resolve()
        allowed = destination.is_relative_to(ROOT.resolve())
        # Historical plans keep checkpoint_dir="selected". Only the exact
        # relocated checkpoint directory is an allowed external destination.
        if key == 'checkpoint_dir' and relative.parts and relative.parts[0] == 'selected':
            study = ROOT.resolve()
            if study.parent.name == 'studies' and study.parent.parent.name == 'khu':
                expected = study.parent.parent / 'results' / study.name / relative
                allowed = allowed or (destination == expected and expected.resolve() == expected)
        if not allowed:
            raise ValueError(f'{key} escapes the isolated experiment')
    return plan


def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('AGORA_', 'RIC_', 'DRL_'))}
    env.pop('PYTHONHOME', None)
    env['PYTHONPATH'] = str(ROOT / 'python') + os.pathsep + str(BACKEND / '.venv/lib/python3.12/site-packages')
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    return env


def trace_path(plan, trace_seed):
    overrides = plan.get('trace_overrides', {})
    if overrides:
        if plan.get('schema') != SCHEMA + '-smoke':
            raise ValueError('Trace overrides are restricted to smoke tests')
        key = str(trace_seed)
        if key not in overrides:
            raise ValueError(f'Missing smoke trace override: {trace_seed}')
        path = Path(overrides[key]).resolve()
        if not path.is_relative_to((ROOT / 'smoke').resolve()):
            raise ValueError('Smoke trace override must stay inside the isolated smoke output tree')
        return path
    return BACKEND / f'scenarios/khu-real/ue_position/ue_positions_seed{trace_seed}.csv'


def sim_argv(plan, spec, directory):
    cache = BACKEND / 'scenarios/khu-real/ret_caches/tilt_9deg'
    labels = ('gNB_5G', 'gNB_4G_1', 'gNB_4G_2')
    base = plan['baseline']
    def field(key, value):
        values = value if isinstance(value, list) else [value] * 3
        return '--' + key + '=' + ','.join(f'{label}:{v:g}' for label, v in zip(labels, values))
    return [str(ROOT / plan['binary']), '--centralFrequency18=1.8e9', '--bandwidth18=10e6',
            '--centralFrequency35=3.5e9', '--bandwidth35=10e6', '--numerology=0',
            f'--sionnaCacheFile18={cache}/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5',
            f'--sionnaCacheFile35={cache}/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5',
            '--sionnaUpdatePeriod=1s', '--udpBaseIntervalMs=200', '--trafficMultiplierEnabled=true',
            f'--simTime={spec.get("duration_s", plan["duration_s"])}', '--kpiReportInterval=1.0',
            f'--N_Ues={spec.get("n_ues", plan["n_ues"])}',
            f'--sumoTrace={trace_path(plan, spec["trace_seed"])}',
            f'--gnbPositions={BACKEND}/scenarios/khu-real/gnbs-ret.csv',
            f'--resultsRoot={directory}/ns3', '--runTag=simulation',
            f'--cellKpiCsvPath={directory}/cell_kpi.csv', f'--ueKpiCsvPath={directory}/ue_kpi.csv',
            f'--RngSeed={plan["rng_seed"]}', f'--RngRun={plan["rng_run"]}',
            '--ns3::NrHelper::E2ModeNr=false', '--ns3::NrHelper::UseIdealRrc=true',
            field('cellTxPowerDbm', base['txp_dbm']), field('cellRetTiltDeg', base['ret_deg']),
            field('cellCioDb', base['cio_db']), '--cellRetBearingDeg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10',
            f'--handoverTtt={base["ttt_ms"]}', f'--handoverHysteresis={base["hys_db"]}',
            field('cellTttMs', base['ttt_ms']), field('cellHysteresisDb', base['hys_db'])]


def controller_argv(plan, spec, control):
    return [sys.executable, '-m', plan['controller_module'], '--control-dir', str(control),
            '--mode', spec['mode'], '--load-dir', str(ROOT / plan['checkpoint_dir']),
            '--lb-start-s', str(spec['lb_start_s']), '--mo-start-s', str(spec['mo_start_s']),
            '--policy', 'eval', '--seed', str(spec['learning_seed'])]


def last_time(directory):
    try:
        with (directory / 'cell_kpi.csv').open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 16384))
            lines = stream.read().decode('utf-8').splitlines()
        for line in reversed(lines):
            try:
                return float(line.split(',')[0])
            except ValueError:
                continue
    except OSError:
        pass
    return None


def verify(directory, duration, spec=None):
    """Verify complete simulation, actual readback, timing, and policy immutability."""
    directory = Path(directory)
    issues, slots, rows, summary, timing = [], set(), [], None, []
    try:
        with (directory / 'cell_kpi.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                rows.append(row)
                key = float(row['time_s']), int(row['cell_id'])
                if key in slots:
                    issues.append('duplicate_cell_time')
                slots.add(key)
        expected = {(i + 0.5, cell) for i in range(int(duration)) for cell in (1, 2, 3)}
        if slots != expected:
            issues.append(f'incomplete_cell_grid:{len(slots)}/{int(duration)*3}')
        observation = json.loads((directory / 'control/observation.json').read_text())
        if not observation['terminal'] or abs(float(observation['time_s']) - duration) > 1e-6:
            issues.append('missing_terminal_observation')
        summary = json.loads((directory / 'control/summary.json').read_text())
        if summary.get('status') != 'complete' or abs(summary.get('last_time_s', -1) - duration) > 1e-6:
            issues.append('controller_not_complete_at_stop')
        if summary.get('policy') != 'eval' or summary.get('online_learning') is not False:
            issues.append('controller_not_explicitly_frozen')
        if summary.get('model_hashes_before') != summary.get('model_hashes_after'):
            issues.append('policy_weights_changed')
        if not isinstance(summary.get('model_hashes_before'), dict):
            issues.append('missing_policy_hash_attestation')
        if any(summary.get(key) != 0 for key in ('online_gradient_updates', 'online_replay_inserts', 'exploratory_actions')):
            issues.append('online_learning_or_exploration_detected')
        receipts = [json.loads(line) for line in (directory / 'control/receipts.jsonl').read_text().splitlines()]
        actions = [json.loads(line) for line in (directory / 'control/actions.jsonl').read_text().splitlines()]
        if not receipts or receipts[-1]['status'] != 'terminal':
            issues.append('missing_terminal_receipt')
        if any(r['status'] not in ('applied', 'held', 'terminal') for r in receipts):
            issues.append('rejected_control_receipt')
        nonterminal = [a for a in actions if not a['terminal']]
        applied = [r for r in receipts if r['status'] in ('applied', 'held')]
        if len(nonterminal) != len(applied):
            issues.append('action_receipt_count_mismatch')
        start = float((spec or {}).get('control_start_s', 180))
        interval = float((spec or {}).get('control_interval_s', 30))
        expected_times, next_time = [], start
        while next_time < duration - 1e-6:
            expected_times.append(next_time)
            next_time += interval
        if len(nonterminal) != len(expected_times):
            issues.append('planned_control_count_mismatch')
        for index, (action, receipt) in enumerate(zip(nonterminal, applied)):
            planned = expected_times[index] if index < len(expected_times) else None
            if planned is None or abs(action['time_s'] - planned) > 1e-6:
                issues.append('control_not_applied_at_planned_simulation_time')
            if action['seq'] != receipt['seq'] or abs(action['time_s'] - receipt['time_s']) > 1e-6:
                issues.append('action_receipt_sequence_or_time_mismatch')
            if action['updates'] != receipt['updates']:
                issues.append('action_receipt_updates_mismatch')
            for key, value in action['updates'].items():
                if abs(receipt['after'][key] - value) > 1e-6:
                    issues.append('action_receipt_readback_mismatch')
            if spec:
                allowed = set()
                if spec['mode'] in ('lb', 'both') and action['time_s'] >= spec['lb_start_s']:
                    allowed.update(('cio_5g', 'cio_4g1', 'cio_4g2'))
                if spec['mode'] in ('mo', 'both') and action['time_s'] >= spec['mo_start_s']:
                    allowed.update(('ttt_ms', 'hys_db'))
                if set(action['updates']) - allowed:
                    issues.append('unexpected_app_or_preactivation_control')
            timing.append(dict(seq=action['seq'], planned_simulation_s=planned,
                               requested_simulation_s=action['time_s'], applied_simulation_s=receipt['time_s'],
                               controller_ms=action.get('controller_ms'),
                               per_app_decision_ms={name: app.get('decision_ms') for name, app in action.get('apps', {}).items()}))
        if applied:
            labels = {1: '5g', 2: '4g1', 3: '4g2'}
            state, index = dict(applied[0]['before']), 0
            for row in sorted(rows, key=lambda item: float(item['time_s'])):
                while index < len(applied) and applied[index]['time_s'] <= float(row['time_s']):
                    state = dict(applied[index]['after'])
                    index += 1
                label = labels[int(row['cell_id'])]
                for column, parameter in (('cio_bias_db', 'cio_' + label), ('ttt_ms', 'ttt_ms'), ('hysteresis_db', 'hys_db')):
                    if abs(float(row[column]) - state[parameter]) > 1e-6:
                        issues.append(f'cell_parameter_state_mismatch:{row["time_s"]}:{parameter}')
    except (OSError, KeyError, ValueError, TypeError) as error:
        issues.append('verification:' + str(error))
    return dict(passed=not issues, issues=sorted(set(issues)), cell_rows=len(slots), controller_summary=summary,
                control_timing=timing)


def memory_available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024 ** 2
    return 0.0


def preflight(plan):
    import numpy
    import torch
    if sys.platform != 'linux':
        raise RuntimeError('Simulation evaluation requires Linux/WSL')
    paths = [ROOT / plan['binary'], ROOT / 'python' / Path(*plan['controller_module'].split('.')).with_suffix('.py')]
    paths += [ROOT / plan['checkpoint_dir'] / f'{app}.pt' for app in ('lb', 'mo')]
    paths += [trace_path(plan, trace) for trace in plan['test_traces']]
    paths += [Path(metadata['source']) for metadata in plan.get('trace_override_metadata', {}).values()]
    paths += [BACKEND / f'scenarios/khu-real/ret_caches/tilt_9deg/sionna_rt_cache_{band}_2x2_1x1_lzf.h5' for band in ('1p8ghz', '3p5ghz')]
    paths += [BACKEND / 'scenarios/khu-real/gnbs-ret.csv']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not os.access(ROOT / plan['binary'], os.X_OK):
        raise PermissionError('Frozen simulator binary is not executable')
    return dict(checked_utc=now(), available_ram_gib=memory_available_gib(), jobs=len(plan['jobs']),
                online_learning=False, plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                versions={'python': sys.version, 'torch': torch.__version__, 'numpy': numpy.__version__},
                input_hashes=[dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size) for path in paths],
                source_hashes=[dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
                               for folder in ('python', 'evaluation') for path in sorted((ROOT / folder).rglob('*.py'))],
                linked_library_hashes=[dict(path=str(path), sha256=sha(path)) for path in sorted((BACKEND / 'build/lib').glob('*.so'))])


def check_frozen_inputs(plan, reference_path=None):
    reference = json.loads(Path(reference_path or ROOT / 'preflight.json').read_text())
    if reference['plan_sha256'] != hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest():
        raise RuntimeError('Frozen evaluation plan changed')
    for field in ('input_hashes', 'source_hashes', 'linked_library_hashes'):
        for entry in reference[field]:
            path = Path(entry['path'])
            path = path if path.is_absolute() else ROOT / path
            if not path.is_file() or sha(path) != entry['sha256']:
                raise RuntimeError(f'Frozen evaluation input changed: {path}')


def executions():
    records = {}
    for path in (ROOT / 'results').glob('*/execution.json'):
        try:
            record = json.loads(path.read_text())
            records[record['spec']['id']] = record
        except (OSError, ValueError, KeyError):
            pass
    return records


def run_job(plan, spec, timeout=172800, results_root=None, reference_path=None):
    if not any(spec == candidate for candidate in plan['jobs']):
        raise ValueError('Worker spec differs from immutable evaluation plan')
    check_frozen_inputs(plan, reference_path)
    directory = safe_child(Path(results_root or ROOT / 'results'), spec['id'])
    directory.mkdir(parents=True, exist_ok=False)
    control = directory / 'control'
    control.mkdir()
    duration = spec.get('duration_s', plan['duration_s'])
    record = dict(spec=spec, status='launching', started_utc=now(), worker_pid=os.getpid(), worker_start_ticks=process_start_ticks(os.getpid()),
                  directory=str(directory), ns3_returncode=None, controller_returncode=None)
    atomic(directory / 'execution.json', record)
    args, ctrl_args = sim_argv(plan, spec, directory), controller_argv(plan, spec, control)
    atomic(directory / 'command.json', {'ns3': args, 'controller': ctrl_args})
    (directory / 'command.sh').write_text('#!/usr/bin/env bash\n' + shlex.join(args) + '\n', encoding='utf-8')
    env = environment()
    env.update(DRL_CONTROL_DIR=str(control), DRL_START_S=str(spec.get('control_start_s', plan['control_start_s'])),
               DRL_INTERVAL_S=str(spec.get('control_interval_s', plan['control_interval_s'])), DRL_TIMEOUT_S='300')
    processes, previous_handlers = [], {}
    started = time.monotonic()
    def terminate(signum=None, frame=None):
        for process in processes:
            if process.poll() is None:
                process.terminate()
        if signum:
            raise KeyboardInterrupt(f'worker signal {signum}')
    for number in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[number] = signal.signal(number, terminate)
    try:
        with (directory / 'controller.log').open('xb', buffering=0) as clog, (directory / 'ns3.log').open('xb', buffering=0) as slog:
            controller = subprocess.Popen(ctrl_args, cwd=BACKEND, env=env, stdout=clog, stderr=subprocess.STDOUT)
            processes.append(controller)
            simulator = subprocess.Popen(args, cwd=BACKEND, env=env, stdout=slog, stderr=subprocess.STDOUT)
            processes.append(simulator)
            record.update(status='running', controller_pid=controller.pid, ns3_pid=simulator.pid,
                          controller_start_ticks=process_start_ticks(controller.pid), ns3_start_ticks=process_start_ticks(simulator.pid))
            while simulator.poll() is None:
                if controller.poll() is not None and (not (control / 'summary.json').exists() or controller.returncode != 0):
                    raise RuntimeError(f'Controller exited early: {controller.returncode}')
                if time.monotonic() - started > timeout:
                    raise TimeoutError('ns3 job exceeded wall-clock timeout')
                record.update(wall_seconds=time.monotonic() - started, last_cell_time_s=last_time(directory), updated_utc=now())
                atomic(directory / 'execution.json', record)
                time.sleep(2)
            record['ns3_returncode'] = simulator.returncode
            if simulator.returncode:
                raise RuntimeError(f'ns3 exited {simulator.returncode}')
            controller.wait(timeout=60)
            record['controller_returncode'] = controller.returncode
            if controller.returncode:
                raise RuntimeError(f'Controller exited {controller.returncode}')
            record['verification'] = verify(directory, duration, spec)
            record['status'] = 'complete' if record['verification']['passed'] else 'failed'
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for name, process in zip(('controller', 'ns3'), processes):
            record[name + '_returncode'] = process.returncode
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        record.update(finished_utc=now(), wall_seconds=time.monotonic() - started, last_cell_time_s=last_time(directory))
        atomic(directory / 'execution.json', record)
    return record


def make_smoke_plan(n_ues=3):
    """Bridge/serialization test only; its tiny UE population is not an RL score."""
    if not 1 <= n_ues <= 300:
        raise ValueError('Smoke UE count must be within [1, 300]')
    plan = make_plan()
    plan.update(schema=SCHEMA + '-smoke', duration_s=120, n_ues=n_ues,
                control_start_s=90, control_interval_s=15, test_traces=[1419],
                requested_workers=2, purpose='Contract/readback smoke only; never included in performance evaluation')
    plan['jobs'] = [dict(id=f'smoke_{mode}_t1419', phase='smoke', mode=mode,
                        arm=arm, policy='eval', reward='local', trace_seed=1419,
                        learning_seed=909, lb_start_s=90, mo_start_s=90,
                        control_start_s=90, control_interval_s=15)
                    for arm, mode in (('none', 'none'), ('both_local', 'both'))]
    return plan


def filter_smoke_trace(source, destination, n_ues):
    """Keep all rows for the earliest active UE IDs; never alter the source CSV."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError('Smoke trace must be a new file separate from its original input')
    if not 1 <= n_ues <= 300:
        raise ValueError('Smoke UE count must be within [1, 300]')
    source_hash = sha(source)
    with source.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(line for line in stream if line.strip() and not line.lstrip().startswith('#'))
        fieldnames = reader.fieldnames
        if not fieldnames or set(('time_s', 'ue_id', 'x', 'y', 'z', 'active')) - set(fieldnames):
            raise ValueError('Smoke trace requires time_s, ue_id, x, y, z, active columns')
        rows = list(reader)
    first_active, first_seen = {}, {}
    for index, row in enumerate(rows):
        name = row['ue_id']
        timestamp = float(row['time_s'])
        if not name or not math.isfinite(timestamp):
            raise ValueError('Invalid UE ID or time in source mobility trace')
        first_seen.setdefault(name, index)
        if row['active'].strip().lower() in ('true', '1', 'yes'):
            first_active[name] = min(first_active.get(name, math.inf), timestamp)
    ordered = sorted(first_active, key=lambda name: (first_active[name], first_seen[name], name))
    if len(ordered) < n_ues:
        raise ValueError(f'Only {len(ordered)} active UE IDs exist; cannot select {n_ues}')
    selected = ordered[:n_ues]
    selected_set = set(selected)
    filtered = [row for row in rows if row['ue_id'] in selected_set]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', newline='', encoding='utf-8') as stream:
        stream.write(f'# Isolated smoke-only subset: {n_ues} earliest-active UE IDs; source SHA256={source_hash}\n')
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)
    if sha(source) != source_hash:
        raise RuntimeError('Original trace changed while preparing smoke subset')
    return dict(source=str(source), source_sha256=source_hash, output=str(destination),
                output_sha256=sha(destination), selected_ue_ids=selected,
                selected_ue_first_active_s={name: first_active[name] for name in selected},
                source_ue_count=len(first_seen), output_ue_count=len(selected),
                source_rows=len(rows), output_rows=len(filtered),
                selection='Earliest active timestamp; original first appearance breaks ties. Preserve all rows of selected IDs.')


def read_smoke_plan(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / 'smoke').resolve()):
        raise ValueError('Smoke plan must be inside the isolated smoke output tree')
    plan = json.loads(path.read_text(encoding='utf-8'))
    expected = make_smoke_plan(plan.get('n_ues', 0))
    for key in ('schema', 'duration_s', 'control_start_s', 'control_interval_s', 'test_traces', 'jobs', 'online_learning',
                'binary', 'controller_module', 'checkpoint_dir', 'baseline'):
        if plan.get(key) != expected[key]:
            raise ValueError(f'Unexpected smoke plan contract: {key}')
    if set(plan.get('trace_overrides', {})) != {'1419'} or set(plan.get('trace_override_metadata', {})) != {'1419'}:
        raise ValueError('Smoke plan requires its own explicitly filtered trace input')
    trace = trace_path(plan, 1419)
    metadata = plan['trace_override_metadata']['1419']
    if not trace.is_relative_to(path.parent / 'inputs') or Path(metadata['output']).resolve() != trace:
        raise ValueError('Smoke trace must belong to the same smoke session')
    if metadata.get('output_ue_count') != plan['n_ues'] or sha(trace) != metadata.get('output_sha256'):
        raise ValueError('Smoke trace subset count/hash mismatch')
    return plan


def smoke(tag='contract_120s', n_ues=3):
    """Run two independent smoke jobs, separate from plan.json and results/."""
    if sys.platform != 'linux':
        raise RuntimeError('Simulator smoke tests require Linux/WSL')
    session = safe_child(ROOT / 'smoke', tag)
    session.mkdir(parents=True, exist_ok=False)
    plan = make_smoke_plan(n_ues)
    subset = filter_smoke_trace(trace_path(plan, 1419), session / 'inputs' / f'ue_positions_seed1419_n{n_ues}.csv', n_ues)
    plan['trace_overrides'] = {'1419': subset['output']}
    plan['trace_override_metadata'] = {'1419': subset}
    plan_path = session / 'plan.json'
    atomic(plan_path, plan)
    pending = {job['id']: job for job in plan['jobs']}
    running, records = {}, {}
    state = dict(status='running', stage='smoke-only', runner_pid=os.getpid(),
                 started_utc=now(), directory=str(session), online_learning=False)
    atomic(session / 'status.json', state)
    stopping = False
    previous_handlers = {}
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    for number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[number] = signal.signal(number, stop)
    try:
        atomic(session / 'preflight.json', preflight(plan))
        next_launch = 0.0
        while pending or running:
            for job_id, (process, log) in list(running.items()):
                if process.poll() is None:
                    continue
                log.close()
                record_path = session / 'results' / job_id / 'execution.json'
                record = json.loads(record_path.read_text()) if record_path.exists() else {'status': 'failed', 'error': 'Missing worker execution record'}
                if record.get('status') not in ('complete', 'failed'):
                    record.update(status='failed', error=f'Worker exited {process.returncode} without completion')
                records[job_id] = record
                del running[job_id]
            if any(record['status'] == 'failed' for record in records.values()):
                raise RuntimeError('Smoke job failed; evidence is preserved under its unique smoke tag')
            if stopping:
                state['status'] = 'interrupted'
                raise KeyboardInterrupt('Smoke stop requested')
            available = memory_available_gib()
            needed = plan['reserved_ram_gib'] + plan['estimated_job_ram_gib']
            if pending and len(running) < 2 and time.monotonic() >= next_launch and available >= needed:
                job = next(iter(pending.values()))
                specs = session / 'job_specs'
                specs.mkdir(exist_ok=True)
                spec_path = specs / (job['id'] + '.json')
                atomic(spec_path, job)
                log = (specs / (job['id'] + '.worker.log')).open('xb', buffering=0)
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run-smoke-job',
                                            '--plan', str(plan_path), '--spec', str(spec_path)],
                                           cwd=BACKEND, env=environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[job['id']] = process, log
                del pending[job['id']]
                next_launch = time.monotonic() + plan['launch_stagger_s']
            state.update(updated_utc=now(), pending_ids=list(pending), running_ids=list(running),
                         available_ram_gib=available, memory_throttled=bool(pending and available < needed), jobs=records)
            atomic(session / 'status.json', state)
            if pending or running:
                time.sleep(2)
        check_frozen_inputs(plan, session / 'preflight.json')
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status=state['status'] if state['status'] != 'running' else 'failed', error=f'{type(error).__name__}: {error}')
        for process, _ in running.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log in running.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            log.close()
    finally:
        for path in (session / 'results').glob('*/execution.json'):
            record = json.loads(path.read_text())
            records[record['spec']['id']] = record
        state.update(finished_utc=now(), jobs=records, running_ids=[])
        atomic(session / 'status.json', state)
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    return state


def campaign(plan_path, workers=16):
    if workers < 1 or workers > 16:
        raise ValueError('Workers must be in [1, 16]')
    plan = read_plan(plan_path)
    if (ROOT / 'campaign_status.json').exists():
        raise FileExistsError('Evaluation already started; existing evidence is never overwritten')
    lock = ROOT / '.evaluation-launch.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'start_ticks': process_start_ticks(os.getpid()), 'created_utc': now()}))
    pending = {job['id']: job for job in plan['jobs']}
    owned_ids, running, records = set(pending), {}, {}
    state = dict(status='running', started_utc=now(), runner_pid=os.getpid(), runner_start_ticks=process_start_ticks(os.getpid()), workers=workers,
                 total_jobs=len(pending), stage='frozen-test', online_learning=False)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, stop)
    try:
        atomic(ROOT / 'preflight.json', preflight(plan))
        next_launch = 0.0
        while pending or running:
            records = {key: value for key, value in executions().items() if key in owned_ids}
            for job_id, (process, log) in list(running.items()):
                if process.poll() is None:
                    continue
                log.close()
                path = ROOT / 'results' / job_id / 'execution.json'
                if path.is_file():
                    records[job_id] = json.loads(path.read_text())
                if records.get(job_id, {}).get('status') not in ('complete', 'failed'):
                    record = records.get(job_id, {'spec': next(job for job in plan['jobs'] if job['id'] == job_id)})
                    record.update(status='failed', directory=str(path.parent), error=f'Worker exited {process.returncode} without terminal record')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic(path, record)
                    records[job_id] = record
                del running[job_id]
            failed = [key for key, record in records.items() if record['status'] == 'failed']
            if failed:
                state.update(status='failed', failed_jobs=failed, failure_trigger_ids=failed)
                raise RuntimeError('Evaluation halted after a failed job; preserve raw failure before retry')
            if stopping:
                state['status'] = 'interrupted'
                raise KeyboardInterrupt('Evaluation stop requested')
            available = memory_available_gib()
            needed = plan['reserved_ram_gib'] + plan['estimated_job_ram_gib']
            if pending and len(running) < workers and time.monotonic() >= next_launch and available >= needed:
                job = next(iter(pending.values()))
                specs = ROOT / 'job_specs'
                specs.mkdir(exist_ok=True)
                spec_path = specs / (job['id'] + '.json')
                atomic(spec_path, job)
                log = (specs / (job['id'] + '.worker.log')).open('xb', buffering=0)
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run-job', '--plan', str(plan_path), '--spec', str(spec_path)],
                                           cwd=BACKEND, env=environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[job['id']] = process, log
                del pending[job['id']]
                next_launch = time.monotonic() + plan['launch_stagger_s']
            state.update(updated_utc=now(), pending_ids=list(pending), running_ids=list(running),
                         completed_ids=[key for key, value in records.items() if value['status'] == 'complete'],
                         available_ram_gib=available, launch_memory_requirement_gib=needed,
                         memory_throttled=bool(pending and available < needed), jobs=records)
            atomic(ROOT / 'campaign_status.json', state)
            if pending or running:
                time.sleep(3)
        check_frozen_inputs(plan)
        subprocess.run([sys.executable, str(ROOT / 'evaluation/report.py'), '--root', str(ROOT)], check=True, env=environment())
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status=state['status'] if state['status'] != 'running' else 'failed', error=f'{type(error).__name__}: {error}')
        for process, _ in running.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log in running.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            log.close()
    finally:
        records = {key: value for key, value in executions().items() if key in owned_ids}
        for key, record in records.items():
            if record.get('status') not in ('complete', 'failed'):
                record.update(status='failed', finished_utc=now(), error='Owned worker terminated without a terminal execution record')
                atomic(ROOT / 'results' / key / 'execution.json', record)
        state = terminal_campaign_state(state, records, pending)
        atomic(ROOT / 'campaign_status.json', state)
        lock.unlink(missing_ok=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init', 'preflight', 'run-job', 'launch', 'campaign', 'status', 'verify', 'smoke', 'run-smoke-job'))
    parser.add_argument('--plan', type=Path, default=ROOT / 'plan.json')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--smoke-tag', default='contract_120s')
    parser.add_argument('--smoke-n-ues', '--n-ues', dest='smoke_n_ues', type=int, default=3)
    args = parser.parse_args()
    if args.command != 'status' and (ROOT / '.archive-in-progress').exists():
        raise RuntimeError('Failed-attempt archival is unfinished; inspect preserved evidence before starting another operation')
    if args.command == 'init':
        if sys.platform != 'linux':
            raise RuntimeError('Initialize the runtime plan in WSL so recorded paths are POSIX paths')
        if args.plan.exists():
            raise FileExistsError(args.plan)
        atomic(args.plan, make_plan())
        print(args.plan)
        return 0
    if args.command == 'status':
        status_path = ROOT / 'campaign_status.json'
        campaign_state = json.loads(status_path.read_text()) if status_path.exists() else {}
        print(json.dumps({'campaign_status': campaign_state.get('status', 'not_started'),
                          'running_count': len(campaign_state.get('running_ids', [])),
                          'pending_count': len(campaign_state.get('pending_ids', [])),
                          'memory_throttled': campaign_state.get('memory_throttled'),
                          'jobs': [dict(id=key, status=value['status'], time_s=last_time(Path(value['directory'])))
                                   for key, value in executions().items()]}, indent=2))
        return 0
    if args.command == 'smoke':
        result = smoke(args.smoke_tag, args.smoke_n_ues)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'complete' else 1
    if args.command == 'run-smoke-job':
        if args.spec is None:
            parser.error('--spec is required for run-smoke-job')
        plan = read_smoke_plan(args.plan)
        spec_path = args.spec.resolve()
        if not spec_path.is_relative_to(args.plan.resolve().parent):
            parser.error('Smoke spec must be inside the same smoke session')
        result = run_job(plan, json.loads(spec_path.read_text(encoding='utf-8')), timeout=7200,
                         results_root=args.plan.resolve().parent / 'results',
                         reference_path=args.plan.resolve().parent / 'preflight.json')
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'complete' else 1
    plan = read_plan(args.plan)
    if args.command == 'preflight':
        print(json.dumps(preflight(plan), indent=2))
        return 0
    if args.command == 'run-job':
        if args.spec is None:
            parser.error('--spec is required for run-job')
        result = run_job(plan, json.loads(args.spec.read_text(encoding='utf-8')))
    elif args.command == 'verify':
        result = {'jobs': {job['id']: verify(ROOT / 'results' / job['id'], plan['duration_s'], job) for job in plan['jobs']}}
        result['status'] = 'complete' if all(job['passed'] for job in result['jobs'].values()) else 'failed'
        atomic(ROOT / 'completion_verification.json', result)
    else:
        result = campaign(args.plan.resolve(), args.workers)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
