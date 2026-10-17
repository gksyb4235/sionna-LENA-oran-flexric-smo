"""Explicit background launch for only the isolated DRL campaign."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

root = Path(__file__).resolve().parent
if (root / 'launch_record.json').exists() or (root / 'campaign_status.json').exists():
    raise RuntimeError('Campaign was already launched; inspect status, do not duplicate')
log = (root / 'campaign.log').open('xb', buffering=0)
args = [sys.executable, str(root / 'campaign.py'), 'campaign', '--plan', str(root / 'plan.json'), '--workers', '8']
process = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           start_new_session=True, env=dict(os.environ))
log.close()
record = dict(started_utc=datetime.now(timezone.utc).isoformat(), pid=process.pid, argv=args,
              log=str(root / 'campaign.log'), total_planned_runs=64, maximum_parallel_simulators=8)
(root / 'launch_record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
