#!/usr/bin/env python3
"""Four-argument distributed sweep: first ID, last ID, CPUs/job, parallel jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import conflict_sweep as entry

sys.path.insert(0, str(entry.BUNDLE))
import sweep


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('start',type=int,help='First conflict ID, 1-based and inclusive')
    parser.add_argument('end',type=int,help='Last conflict ID, inclusive; original 1..500 or reserve 501..550')
    parser.add_argument('cores_per_job',type=int)
    parser.add_argument('parallel_jobs',type=int)
    parser.add_argument('--mode',choices=('run','plan','preflight','status','collect'),default='run')
    parser.add_argument('--dry-run',action='store_true',help='Show the selected IDs and CPU slots without setup or simulation')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--dataset',choices=('auto','original500','reserve50'),default='auto')
    parser.add_argument('--config',type=Path,help='Explicit machine config; preserve its output_root')
    args=parser.parse_args()
    try:
        mode='plan' if args.dry_run else args.mode
        if args.cores_per_job<1 or args.parallel_jobs<1:
            raise ValueError('CPUs/job and parallel jobs must be positive')
        dataset=entry.select_dataset(args.start,args.end,args.dataset)
        manifest_path,base_config=entry.dataset_paths(dataset)
        manifest=sweep.validate_manifest(sweep.read_json(manifest_path))
        selected=sweep.jobs_in_range(manifest,manifest['jobs'],args.start,args.end)
        slots=sweep.cpu_slots(args.cores_per_job,args.parallel_jobs) if mode in ('run','plan','preflight') else None
        config_path=args.config.resolve() if args.config else base_config.with_name('config.range.local.json')
        if args.config and not config_path.is_file():
            raise FileNotFoundError(f'Explicit config does not exist: {config_path}')
        output=entry.default_output(dataset)
        if config_path.exists():
            existing=sweep.read_json(config_path)
            if existing['ns3_root']!=str(entry.ROOT):
                raise ValueError(f'Inspect existing config from a different checkout: {config_path}')
            output=existing['output_root']
        summary=dict(command=mode,dataset=dataset,manifest=str(manifest_path),config=str(config_path),
            range_inclusive=[args.start,args.end],selected_jobs=len(selected),
            first_id=selected[0]['job_id'],last_id=selected[-1]['job_id'],
            cores_per_job=args.cores_per_job,parallel_jobs=args.parallel_jobs,cpu_slots=slots,
            output_root=str(sweep.output_root({'ns3_root':str(entry.ROOT),'output_root':output})),
            simulation_launched=False)
        if mode=='plan':
            summary['job_ids']=[j['job_id'] for j in selected]
            if args.report:
                with args.report.open('x',encoding='utf-8') as handle:
                    json.dump(summary,handle,indent=2);handle.write('\n')
            print(json.dumps(summary,indent=2))
            return
        if mode in ('run','preflight'):
            entry.setup(build=True,dataset=dataset)
        if not config_path.exists():
            config=sweep.read_json(base_config)
            config['output_root']=output
            with config_path.open('x',encoding='utf-8') as handle:
                json.dump(config,handle,indent=2);handle.write('\n')
        config=sweep.read_json(config_path)
        if config['ns3_root']!=str(entry.ROOT):
            raise ValueError(f'Inspect existing config from a different checkout: {config_path}')
        summary['output_root']=str(sweep.output_root(config))
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        argv=[sys.executable,entry.BUNDLE/'sweep.py',mode,'--manifest',manifest_path,
              '--config',config_path,'--start-index',str(args.start),'--end-index',str(args.end),
              '--cores-per-job',str(args.cores_per_job),'--workers',str(args.parallel_jobs)]
        if mode=='run':
            argv.append('--resume')
            if args.retry_failed: argv.append('--retry-failed')
        if args.report: argv.extend(['--report',args.report])
        result=subprocess.run([str(x) for x in argv],cwd=entry.ROOT)
        raise SystemExit(result.returncode)
    except (OSError,ValueError,RuntimeError,KeyError,subprocess.CalledProcessError) as error:
        parser.exit(2,f'Error: {error}\n')


if __name__=='__main__':main()
