#!/usr/bin/env python3
"""Execute only the finalized canonical replay queue; no planning or API calls."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from validation_domains import assert_admissible_jobs
from agora_evaluation_allocation import load_allocation

BUNDLE = Path(os.environ.get('AGORA_BUNDLE', str(Path(__file__).resolve().parents[3])))
REPO = Path(os.environ.get('AGORA_REPO', str(BUNDLE)))
MANIFEST = BUNDLE / 'experiment/khu/agora_evaluation_v2/manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--partition', choices=['pc1', 'pc2', 'pc3', 'pc4', 'server'])
    selector.add_argument('--ids', help='Explicit comma-separated numeric IDs for a reviewed retry')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--memory-gib', type=float, default=8)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--retry-failed', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('frozen', BUNDLE / 'experiment/khu/tools/agora_frozen_sweep.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifest = runner.read(MANIFEST)
    if manifest['schema'] != 'agora-canonical-replay-v2':
        raise ValueError('Incorrect canonical manifest')
    allocation = load_allocation(BUNDLE, manifest)
    ids = ([int(x) for x in args.ids.split(',')] if args.ids else
           [int(j.split('_')[1]) for j in allocation['partitions'][args.partition]])
    if len(ids) != len(set(ids)) or any(i < 0 or i >= len(manifest['jobs']) for i in ids):
        raise ValueError('Duplicate or out-of-range job ID')
    jobs = [manifest['jobs'][i] for i in ids]
    output = (args.output or Path(os.environ.get('AGORA_OUTPUT', str(REPO / 'experiment/khu/results/agora_evaluation_v2')))).resolve()
    if args.status:
        for job in jobs:
            receipt = output / job['job_id'] / 'execution.json'
            print(job['job_id'], runner.read(receipt)['status'] if receipt.exists() else 'not_started')
        return
    if not manifest.get('ready_for_execution'):
        raise ValueError('Draft package: planning inventory is not final. No jobs were launched.')
    if any(not job['required_by_current_methods'] for job in jobs):
        raise ValueError('An archival-only job was selected')
    assert_admissible_jobs(jobs)
    if not jobs:
        print('No queued jobs on this machine.')
        return
    if args.workers < 1 or args.memory_gib <= 0:
        raise ValueError('Positive workers/memory required')
    workers = min(args.workers, len(jobs))
    slots = runner.cpu_slots(1, workers, sorted(os.sched_getaffinity(0)))
    quota = Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        q, period = quota.read_text().split()
        if q != 'max' and workers > int(q) / int(period):
            raise ValueError('Worker count exceeds container CPU quota')
    description = dict(partition=args.partition, ids=ids, count=len(ids), workers=workers,
                       cores_per_job=1, manifest_sha256=runner.sha(MANIFEST), output=str(output),
                       allocation_sha256=allocation['allocation_sha256'],
                       cpu_slots=slots, memory_limit_per_job_gib=args.memory_gib)
    print(json.dumps(description), flush=True)
    if args.dry_run:
        return
    cache = Path(os.environ.get('RET_CACHE_ROOT', str(REPO / 'experiment/khu/caches/ret_caches')))
    provenance = runner.preflight(REPO, BUNDLE, manifest, jobs, cache, runner.clean_env(REPO))
    print('Preflight passed', flush=True)
    if args.preflight:
        return
    signature = hashlib.sha256((runner.RUNNER_VERSION + runner.sha(MANIFEST) +
                                 provenance['binary_sha256'] + json.dumps(manifest['backend_source_sha256'], sort_keys=True)).encode()).hexdigest()
    os.environ['RQ3_MEMORY_LIMIT_BYTES'] = str(int(args.memory_gib * 1024**3))
    output.mkdir(parents=True, exist_ok=True)
    runner.save(output / f'queue_{args.partition or "explicit_ids"}.json', description)
    runargs = argparse.Namespace(start=min(ids), end=max(ids), retry_failed=args.retry_failed)
    raise SystemExit(runner.run(runargs, REPO, BUNDLE, manifest, jobs, slots, output, cache, provenance, signature))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        raise SystemExit(f'ERROR: {error}')
