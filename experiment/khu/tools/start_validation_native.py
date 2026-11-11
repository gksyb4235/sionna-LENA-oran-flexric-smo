"""Start one assigned PC queue in the background; preserve logs and prior attempts."""
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

def live_prior_retry_pids(root, manifest, partition, proc_root=Path('/proc')):
 prior=manifest['allocation_model']['already_started_jobs'][partition]
 if not prior:return []
 output=root/'experiment/khu/results/rq3_retry_20261111';pids=set()
 launch=output/'launch.json'
 if launch.exists():
  pid=json.loads(launch.read_text()).get('pid');cmd=proc_root/str(pid)/'cmdline'
  try:
   if b'restart_imported_rq3.py' in cmd.read_bytes():pids.add(pid)
  except FileNotFoundError:pass
 for jid in prior:
  receipt=output/jid/'execution.json'
  if not receipt.exists():continue
  pid=json.loads(receipt.read_text()).get('pid');cmd=proc_root/str(pid)/'cmdline'
  try:
   command=cmd.read_bytes()
   if b'ns3' in command and b'khu' in command:pids.add(pid)
  except FileNotFoundError:pass
 return sorted(pids)

def main():
 p=argparse.ArgumentParser();p.add_argument('--partition',required=True,choices=['pc1','pc2','pc3','pc4']);p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
 root=Path(__file__).resolve().parents[3];m=json.loads((root/'experiment/khu/validation_20261111/manifest.json').read_text());workers=m['allocation_model']['workers'][a.partition]
 prior=live_prior_retry_pids(root,m,a.partition)
 if prior:raise SystemExit(f"PC1 existing retries are still running (PIDs: {prior}). Wait for them to finish; run.sh starts only the additional {len(m['partitions'][a.partition])} jobs.")
 output=Path(os.environ.get('VALIDATION_OUTPUT',str(root/'experiment/khu/results/validation_20261111'))).resolve();logs=output/'_launchers';logs.mkdir(parents=True,exist_ok=True)
 latest=logs/(a.partition+'_latest.json')
 if latest.exists():
  old=json.loads(latest.read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
  if proc.exists() and b'validation_sweep.py' in proc.read_bytes():raise SystemExit('This PC queue is already running; inspect status.sh')
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');log=logs/(a.partition+'_'+stamp+'.log')
 command=[sys.executable,'-u',str(root/'experiment/khu/tools/validation_sweep.py'),'--partition',a.partition,'--workers',str(workers),'--output',str(output)]
 if a.retry_failed:command.append('--retry-failed')
 with log.open('xb') as stream:
  child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 record=dict(pid=child.pid,partition=a.partition,workers=workers,cores_per_job=1,log=str(log),output=str(output),command=command)
 latest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
