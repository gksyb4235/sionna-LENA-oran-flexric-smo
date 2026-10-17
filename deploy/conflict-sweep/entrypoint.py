"""CPU-only frozen sweep runtime; accepts FIRST LAST CPUS_PER_JOB PARALLEL_JOBS."""
import argparse, json, os, sys
from pathlib import Path
ROOT=Path('/opt/agora')
BUNDLE=ROOT/'experiments/conflict_scenarios_500/sweep'
sys.path.insert(0,str(BUNDLE))
import sweep
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('start',type=int,nargs='?')
    p.add_argument('end',type=int,nargs='?')
    p.add_argument('cores',type=int,nargs='?')
    p.add_argument('workers',type=int,nargs='?')
    p.add_argument('--mode',choices=['run','plan','preflight','status','collect'],default='run')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--resources',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--report',type=Path)
    a=p.parse_args()
    if a.resources:
        print(json.dumps(dict(allowed_cpu_ids=sorted(os.sched_getaffinity(0)),cpu_quota=sweep.cpu_quota(),
            available_memory_gib=(sweep.memory_available() or 0)/1024**3,
            gpu_devices=list(map(str,Path('/dev').glob('nvidia*')))),indent=2));return
    if a.smoke:
        sys.path.insert(0,str(BUNDLE/'backend'))
        from smoke_backend import run
        result=run(ROOT,ROOT/'results/smoke')
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result['passed'] else 1)
    if None in (a.start,a.end,a.cores,a.workers): p.error('Specify FIRST LAST CPUS_PER_JOB PARALLEL_JOBS')
    mode='plan' if a.dry_run else a.mode
    manifest=sweep.validate_manifest(sweep.read_json(BUNDLE/'data/execution_manifest.json'))
    jobs=sweep.jobs_in_range(manifest,manifest['jobs'],a.start,a.end)
    if a.cores<1 or a.workers<1: p.error('CPU and worker counts must be positive')
    if mode in ('run','preflight','plan'):
        slots=sweep.cpu_slots(a.cores,a.workers)
        print(json.dumps(dict(mode=mode,selected_jobs=len(jobs),cpu_slots=slots)),flush=True)
    if mode=='plan': return
    config=BUNDLE/'config.container.json'
    if mode=='run':
        cfg=sweep.read_json(config)
        memory=sweep.memory_available()
        if memory is None: p.error('Cannot determine available RAM; do not start unattended')
        if memory < (cfg['memory_reserve_gib']+cfg['estimated_job_gib'])*1024**3:
            p.error('Insufficient available RAM even for one worker plus reserve')
    argv=['sweep.py',mode,'--manifest',str(BUNDLE/'data/execution_manifest.json'),'--config',str(config),
          '--start-index',str(a.start),'--end-index',str(a.end),'--cores-per-job',str(a.cores),'--workers',str(a.workers)]
    if mode=='run': argv.append('--resume')
    if a.retry_failed: argv.append('--retry-failed')
    if a.report: argv.extend(['--report',str(a.report)])
    sys.argv=argv
    sweep.main()
if __name__=='__main__':
    try: main()
    except (ValueError,OSError,RuntimeError) as exc: raise SystemExit(str(exc))
