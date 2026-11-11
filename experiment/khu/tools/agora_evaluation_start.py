#!/usr/bin/env python3
"""Start one canonical native queue after checking existing local retry activity."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from agora_evaluation_allocation import load_allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', required=True, choices=['pc1', 'pc2', 'pc3', 'pc4'])
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / 'experiment/khu/agora_evaluation_v2/manifest.json').read_text())
    if not manifest.get('ready_for_execution'):
        raise SystemExit('Draft manifest; no simulation was started')
    allocation = load_allocation(root, manifest)
    # The old retry dispatcher is separate and must finish before PC1 starts.
    if args.partition == 'pc1':
        old = root / 'experiment/khu/results/rq3_retry_20261111'
        records = list(old.glob('rq3_*/execution.json')) + [old / 'launch.json']
        for path in records:
            if not path.exists():
                continue
            pid = json.loads(path.read_text()).get('pid')
            proc = Path(f'/proc/{pid}/cmdline')
            try:
                command = proc.read_bytes()
            except FileNotFoundError:
                continue
            if b'restart_imported_rq3.py' in command or (b'ns3.48-khu-agora-conflict-sweep' in command and str(old).encode() in command):
                raise SystemExit('Existing PC1 retry processes are still alive; wait for their completion')
    output = Path(os.environ.get('AGORA_OUTPUT', str(root / 'experiment/khu/results/agora_evaluation_v2'))).resolve()
    logs = output / '_launchers'
    logs.mkdir(parents=True, exist_ok=True)
    latest = logs / (args.partition + '_latest.json')
    if latest.exists():
        pid = json.loads(latest.read_text())['pid']
        proc = Path(f'/proc/{pid}/cmdline')
        if proc.exists() and b'agora_evaluation_sweep.py' in proc.read_bytes():
            raise SystemExit('This canonical queue is already running')
    workers = allocation['workers'][args.partition]
    command = [sys.executable, '-u', str(root / 'experiment/khu/tools/agora_evaluation_sweep.py'),
               '--partition', args.partition, '--workers', str(workers), '--output', str(output)]
    if args.retry_failed:
        command.append('--retry-failed')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    log = logs / f'{args.partition}_{stamp}.log'
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=child.pid, partition=args.partition, workers=workers, cores_per_job=1,
                  jobs=allocation['partitions'][args.partition],
                  allocation_sha256=allocation['allocation_sha256'],
                  output=str(output), log=str(log), command=command)
    latest.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
