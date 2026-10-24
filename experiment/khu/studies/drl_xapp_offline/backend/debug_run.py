"""Run one isolated short reproduction with signal diagnostics, never a campaign."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/user/LENA-oran-flexric-smo')
EXP=ROOT/'experiment/khu/studies/drl_xapp_offline'
BACKEND=EXP/'backend'
source=BACKEND/'failed_command.json'
if not source.exists():
    original=EXP/'results/test_none_t1421/command.json'
    source.write_bytes(original.read_bytes())
parser=argparse.ArgumentParser()
parser.add_argument('--name',default='debug_1421_sigill_01')
parser.add_argument('--sim-time',type=int,default=25)
parser.add_argument('--stop-after',type=float,default=0)
parser.add_argument('--bridge',action='store_true')
args=parser.parse_args()
if not args.name.startswith('debug_1421_') or '/' in args.name or '..' in args.name:
    raise ValueError('Invalid debug folder name')
directory=BACKEND/args.name
directory.mkdir(exist_ok=False)
library=BACKEND/'debug_sigill.so'
subprocess.run(['g++','-shared','-fPIC','-g','-O0',str(BACKEND/'debug_sigill.cc'),'-o',str(library)],check=True)
command=json.loads(source.read_text())['ns3']
replacement={'--simTime':str(args.sim_time),'--resultsRoot':str(directory/'ns3'),
             '--cellKpiCsvPath':str(directory/'cell_kpi.csv'),
             '--ueKpiCsvPath':str(directory/'ue_kpi.csv')}
command=[arg.split('=')[0]+'='+replacement[arg.split('=')[0]] if arg.split('=')[0] in replacement else arg for arg in command]
environment={k:v for k,v in os.environ.items() if not k.startswith(('DRL_','AGORA_','RIC_'))}
environment['PYTHONPATH']=str(EXP/'python')+os.pathsep+str(ROOT/'.venv/lib/python3.12/site-packages')
environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',LD_PRELOAD=str(library))
if args.bridge:
    environment.update(DRL_CONTROL_DIR=str(directory/'control'),DRL_START_S='180',DRL_INTERVAL_S='30',DRL_TIMEOUT_S='300')
(directory/'command.json').write_text(json.dumps(command,indent=2)+'\n')
started=time.time()
with (directory/'process.log').open('w') as handle:
    process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    reached=False;last_time=None
    while process.poll() is None:
        csv=directory/'cell_kpi.csv'
        if args.stop_after and csv.exists():
            try:
                lines=csv.read_text().splitlines()
                for line in reversed(lines):
                    if len(line.split(','))>10:
                        last_time=float(line.split(',')[0]);break
                if last_time is not None and last_time>=args.stop_after:
                    reached=True;process.terminate();break
            except (ValueError,OSError): pass
        if time.time()-started>300:
            process.terminate();break
        time.sleep(.25)
    process.wait(timeout=20)
result={'returncode':process.returncode,'wall_seconds':time.time()-started,'directory':str(directory),
        'debug_stop_reached':reached,'last_time':last_time,'sim_time':args.sim_time,'bridge':args.bridge}
(directory/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
