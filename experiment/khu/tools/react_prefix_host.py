#!/usr/bin/env python3
"""Start one disjoint host partition of the R5 prefix replay experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

TAG='react_prefix_20261124'
REPO=Path(__file__).resolve().parents[3]

def load_partition(host):
    package=REPO/'experiment/khu'/TAG
    manifest_path=package/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    allocation=json.loads((package/'allocation.json').read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=allocation['manifest_sha256']:
        raise ValueError('Manifest changed after host allocation')
    all_ids=[]
    for part in allocation['hosts'].values():
        if len(part['ids'])>part['cpu_capacity']:raise ValueError('More jobs than available host cores')
        all_ids.extend(part['ids'])
    if len(all_ids)!=len(set(all_ids)) or set(all_ids)!=set(manifest['queues']['r5_new']):
        raise ValueError('Host allocation must cover the new queue exactly once')
    return allocation['hosts'][host],manifest

def active(record):
    p=Path(f'/proc/{record["pid"]}/cmdline')
    if not p.exists():return False
    return [x.decode() for x in p.read_bytes().split(b'\0') if x]==record['argv']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',choices=['PC1','PC2','PC3','PC4','Server'])
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--status',action='store_true')
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    part,manifest=load_partition(args.host)
    output=REPO/part['output_relative']
    runner=REPO/'experiment/khu/tools/react_prefix_sweep.py'
    command=[sys.executable,str(runner),'--queue','r5_new','--ids',','.join(map(str,part['ids'])),
             '--workers',str(len(part['ids'])),'--output',str(output)]
    recordpath=output/'host_launch.json'
    if args.status:
        counts={};rows=[]
        for i in part['ids']:
            jid=manifest['jobs'][i]['job_id'];p=output/jid/'execution.json'
            r=json.loads(p.read_text()) if p.exists() else {}
            status=r.get('status','not_started');counts[status]=counts.get(status,0)+1
            rows.append(dict(job_id=jid,status=status))
        record=json.loads(recordpath.read_text()) if recordpath.exists() else None
        print(json.dumps(dict(host=args.host,output=str(output),counts=counts,jobs=rows,
            launcher_running=active(record) if record else None),indent=2));return
    if args.dry_run or args.preflight:
        subprocess.run(command+(['--dry-run'] if args.dry_run else ['--preflight']),check=True,cwd=REPO);return
    if args.host=='PC1':
        raise ValueError('PC1 already owns 16 running jobs. Use status.sh; do not start a second launcher.')
    if os.name!='posix':raise ValueError('Run in Linux/WSL/container')
    import fcntl
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.host_start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if recordpath.exists():
            record=json.loads(recordpath.read_text())
            if active(record):raise ValueError('This host partition is already running')
            raise ValueError('A previous invocation exists. Inspect status before restarting; no automatic retries.')
        if len(os.sched_getaffinity(0))<len(part['ids']):
            raise ValueError(f'This partition needs {len(part["ids"])} allowed CPUs')
        quota=Path('/sys/fs/cgroup/cpu.max')
        if quota.exists():
            q,p=quota.read_text().split()
            if q!='max' and int(q)/int(p)<len(part['ids']):raise ValueError('Container CPU quota is too small')
        logfile=output/'host_launcher.log'
        with logfile.open('ab',buffering=0) as log:
            child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(host=args.host,pid=child.pid,argv=command,workers=len(part['ids']),cores_per_job=1,
            output=str(output),log=str(logfile),started_at_utc=datetime.now(timezone.utc).isoformat())
        recordpath.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2))

if __name__=='__main__':main()
