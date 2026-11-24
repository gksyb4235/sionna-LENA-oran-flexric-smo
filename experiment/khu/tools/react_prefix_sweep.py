#!/usr/bin/env python3
"""Run archived R5 prefix plans through the existing frozen ns-3 runner."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

TAG='react_prefix_20261124'
BUNDLE=Path(os.environ.get('AGORA_BUNDLE',str(Path(__file__).resolve().parents[3])))
REPO=Path(os.environ.get('AGORA_REPO',str(BUNDLE)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True,choices=['retry_failures','prefix_probe','r5_new','r5_repair','prefix_all_after_fix'])
    parser.add_argument('--ids',help='Optional comma-separated numeric IDs, restricted to the chosen queue')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--memory-gib',type=float,default=8)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--retry-failed',action='store_true')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--status',action='store_true')
    args=parser.parse_args()
    manifest_path=BUNDLE/f'experiment/khu/{TAG}/manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['schema']=='agora-r5-prefix-replay-v1'
    ids=manifest['queues'][args.queue]
    if args.ids:
        requested=[int(i) for i in args.ids.split(',')]
        if len(requested)!=len(set(requested)) or not set(requested)<=set(ids):
            raise ValueError('IDs must be a unique subset of the selected queue')
        ids=requested
    jobs=[manifest['jobs'][i] for i in ids]
    if any(j['job_id'] in manifest.get('excluded_job_ids',[]) for j in jobs):
        raise ValueError('User-excluded job cannot be executed')
    output=(args.output or REPO/f'experiment/khu/results/{TAG}').resolve()
    if args.status:
        for job in jobs:
            p=output/job['job_id']/'execution.json'
            print(job['job_id'],json.loads(p.read_text())['status'] if p.exists() else 'not_started')
        return
    if args.workers<1 or args.memory_gib<=0:
        raise ValueError('Positive workers and memory limit required')
    if not jobs:
        print('Empty queue');return
    workers=min(args.workers,len(jobs))
    description=dict(queue=args.queue,ids=ids,count=len(ids),workers=workers,cores_per_job=1,
        output=str(output),new_llm_calls=False,gnn_inference=False,
        rng_seed=manifest['rng_seed'],rng_run=manifest['rng_run'],
        estimated_memory_limit_total_gib=workers*args.memory_gib,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    print(json.dumps(description,indent=2),flush=True)
    if args.dry_run:return
    if os.name!='posix':raise ValueError('Execution requires Linux or WSL; dry-run is platform independent')
    if not manifest['ready_for_execution']:raise ValueError('Draft manifest')
    from validation_domains import assert_admissible_jobs
    assert_admissible_jobs(jobs)
    spec=importlib.util.spec_from_file_location('frozen_prefix',BUNDLE/'experiment/khu/tools/agora_frozen_sweep.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    slots=runner.cpu_slots(1,workers,sorted(os.sched_getaffinity(0)))
    quota=Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        q,period=quota.read_text().split()
        if q!='max' and workers>int(q)/int(period):raise ValueError('Workers exceed container CPU quota')
    cache=Path(os.environ.get('RET_CACHE_ROOT',str(REPO/'experiment/khu/caches/ret_caches')))
    provenance=runner.preflight(REPO,BUNDLE,manifest,jobs,cache,runner.clean_env(REPO))
    print('Preflight passed',flush=True)
    if args.preflight:return
    signature=hashlib.sha256((runner.RUNNER_VERSION+runner.sha(manifest_path)+provenance['binary_sha256']+
        json.dumps(manifest['backend_source_sha256'],sort_keys=True)).encode()).hexdigest()
    os.environ['RQ3_MEMORY_LIMIT_BYTES']=str(int(args.memory_gib*1024**3))
    output.mkdir(parents=True,exist_ok=True)
    runner.save(output/f'queue_{args.queue}.json',{**description,'cpu_slots':slots})
    runargs=argparse.Namespace(start=min(ids),end=max(ids),retry_failed=args.retry_failed)
    raise SystemExit(runner.run(runargs,REPO,BUNDLE,manifest,jobs,slots,output,cache,provenance,signature))

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
