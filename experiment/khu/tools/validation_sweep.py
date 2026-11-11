#!/usr/bin/env python3
"""Run disjoint frozen all-method replay queues. No LLM or retraining."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path

BUNDLE=Path(os.environ.get('VALIDATION_BUNDLE',str(Path(__file__).resolve().parents[3])))
REPO=Path(os.environ.get('VALIDATION_REPO',str(BUNDLE)))
MANIFEST=BUNDLE/'experiment/khu/validation_20261111/manifest.json'

def parse_ids(text,count):
    values=[]
    for item in text.split(','):
        ends=item.split('-')
        if len(ends)==1:values.append(int(item))
        elif len(ends)==2:
            a,b=map(int,ends)
            if a>b:raise ValueError('Reversed ID range')
            values.extend(range(a,b+1))
        else:raise ValueError('Invalid ID range')
    if len(values)!=len(set(values)) or any(i<0 or i>=count for i in values):raise ValueError('Duplicate or out-of-range job IDs')
    return values

def main():
    p=argparse.ArgumentParser(description=__doc__)
    select=p.add_mutually_exclusive_group(required=True);select.add_argument('--partition',choices=['local_retry','pc1','pc2','pc3','pc4','server_container','new_all','prefix_recheck','reference_repeat','reuse_verified']);select.add_argument('--ids')
    p.add_argument('--workers',type=int,default=1);p.add_argument('--shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--memory-gib',type=float,default=8);p.add_argument('--output',type=Path)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--status',action='store_true');mode.add_argument('--preflight',action='store_true')
    p.add_argument('--retry-failed',action='store_true');args=p.parse_args()
    spec=importlib.util.spec_from_file_location('rq3_frozen',REPO/'experiment/khu/tools/rq3_sweep.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    m=r.read(MANIFEST)
    if m['schema']!='agora-all-variants-replay-v1':raise ValueError('Unexpected manifest')
    if args.shards<1 or not 0<=args.shard_index<args.shards:raise ValueError('Invalid shard')
    ids=parse_ids(args.ids,m['count']) if args.ids else [int(j.split('_')[1]) for j in m['partitions'][args.partition]]
    ids=ids[args.shard_index::args.shards];jobs=[m['jobs'][i] for i in ids]
    if not jobs:raise ValueError('Empty job selection')
    if args.partition=='reuse_verified' and not (args.status or args.dry_run):raise ValueError('Verified reused jobs are not a production queue')
    if args.memory_gib<=0:raise ValueError('Memory limit must be positive')
    output=(args.output or Path(os.environ.get('VALIDATION_OUTPUT',str(REPO/'experiment/khu/results/validation_20261111')))).resolve()
    if args.status:
        for j in jobs:
            f=output/j['job_id']/'execution.json';print(j['job_id'],r.read(f)['status'] if f.exists() else 'not_started')
        return
    workers=min(args.workers,len(jobs));slots=r.cpu_slots(1,workers,sorted(os.sched_getaffinity(0)))
    cpu_max=Path('/sys/fs/cgroup/cpu.max')
    if cpu_max.exists():
        q,period=cpu_max.read_text().split()
        if q!='max' and workers>int(q)/int(period):raise ValueError('Worker count exceeds Docker CPU quota')
    description=dict(jobs=ids,count=len(jobs),workers=workers,cores_per_job=1,cpu_slots=slots,output=str(output),manifest_sha256=r.sha(MANIFEST),memory_limit_per_job_gib=args.memory_gib)
    print(json.dumps(description),flush=True)
    if args.dry_run:return
    cache=Path(os.environ.get('RET_CACHE_ROOT',str(REPO/'experiment/khu/caches/ret_caches')))
    provenance=r.preflight(REPO,BUNDLE,m,jobs,cache,r.clean_env(REPO))
    signature=hashlib.sha256((r.RUNNER_VERSION+r.sha(MANIFEST)+provenance['binary_sha256']+json.dumps(m['backend_source_sha256'],sort_keys=True)).encode()).hexdigest()
    print('Preflight passed',flush=True)
    if args.preflight:return
    os.environ['RQ3_MEMORY_LIMIT_BYTES']=str(int(args.memory_gib*1024**3))
    output.mkdir(parents=True,exist_ok=True)
    r.save(output/f'queue_{args.partition or "ids"}_shard{args.shard_index}.json',description)
    runargs=argparse.Namespace(start=min(ids),end=max(ids),retry_failed=args.retry_failed)
    raise SystemExit(r.run(runargs,REPO,BUNDLE,m,jobs,slots,output,cache,provenance,signature))

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
    except Exception as e:raise SystemExit(f'ERROR: {e}')
