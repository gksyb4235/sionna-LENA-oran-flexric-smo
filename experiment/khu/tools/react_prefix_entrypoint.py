"""Run the fixed Server partition inside the existing frozen Docker image."""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys

from react_prefix_host import load_partition

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--status',action='store_true')
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--preflight',action='store_true')
    parser.add_argument('--memory-gib',type=float,default=8)
    args=parser.parse_args()
    if args.memory_gib<=0:raise ValueError('Memory limit must be positive')
    part,_=load_partition('Server')
    if len(part['ids'])!=21 or part['cores_per_job']!=1:
        raise ValueError('Expected 21 one-core Server jobs')
    if not args.status and not args.dry_run:
        path=Path('/opt/agora/deploy/rq3/entrypoint.py')
        spec=importlib.util.spec_from_file_location('frozen_container',path)
        runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
        runtime.verify_runtime_and_caches()
    os.environ.update(AGORA_BUNDLE='/bundle',AGORA_REPO='/opt/agora',AGORA_OUTPUT='/results')
    argv=['/bundle/experiment/khu/tools/react_prefix_sweep.py','--queue','r5_new',
          '--ids',','.join(map(str,part['ids'])),'--workers','21','--output','/results',
          '--memory-gib',str(args.memory_gib)]
    for flag in ('status','dry_run','preflight'):
        if getattr(args,flag):argv.append('--'+flag.replace('_','-'))
    sys.argv=argv
    runpy.run_path(argv[0],run_name='__main__')

if __name__=='__main__':main()
