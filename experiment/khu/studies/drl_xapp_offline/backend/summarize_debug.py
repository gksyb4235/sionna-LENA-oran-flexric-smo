"""Preserve evidence from the two isolated SIGILL reproductions, without reruns."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
backend=root/'backend'
failed=root/'results/test_none_t1421'
diagnostic=backend/'debug_1421_sigill_02'
short=backend/'debug_1421_sigill_01'


def prefix(path):
    lines=path.read_bytes().splitlines(keepends=True)
    selected=[lines[0]]
    for line in lines[1:]:
        if float(line.split(b',',1)[0]) <= 19.5:
            selected.append(line)
    return b''.join(selected),len(selected)-1


before,n=prefix(failed/'cell_kpi.csv')
after,m=prefix(diagnostic/'cell_kpi.csv')
other,k=prefix(short/'cell_kpi.csv')
assert n==m==60 and before==after
record={
    'schema':'offline-drl-native-crash-reproduction-v1',
    'recorded_utc':datetime.now(timezone.utc).isoformat(),
    'failed_run':str(failed),'original_signal':'SIGILL',
    'original_returncode':-4,'original_last_kpi_time_s':19.5,
    'short_horizon_diagnostic':json.loads((short/'execution.json').read_text()),
    'short_horizon_prefix_equal':before==other,
    'short_horizon_limitation':'Changing simTime to25 changes the pre-crash traffic/KPI trajectory, so it is not a faithful reproduction.',
    'exact_horizon_diagnostic':json.loads((diagnostic/'execution.json').read_text()),
    'exact_horizon_prefix':{
        'rows':n,'columns':len(before.splitlines()[0].split(b',')),
        'last_time_s':19.5,'byte_identical':before==after,
        'sha256':hashlib.sha256(before).hexdigest(),
    },
    'exact_horizon_stop':'Reached30.5s; diagnostic watchdog deliberately terminated only its child withSIGTERM. This is not another spontaneous crash.',
    'changes_for_diagnostic':'Fresh output paths plus local LD_PRELOAD signal handler; same900s horizon, params, trace, RNG andDRLcontrol initialization.',
    'source_fix_applied':False,
    'root_cause':'Unconfirmed native/runtime SIGILL; not reproduced on identical pre-crash simulation trajectory. No evidence supports a model/controller/TTT source fix.',
    'recommendation':'Preserve failed attempt and dump; fresh identical16-run attempt with explicit monitoring beyond19.5s for everyseed1421 arm. Do not count incomplete runs as performance results.',
    'raw_results_modified':False,
}
(backend/'debug_reproduction_summary.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
