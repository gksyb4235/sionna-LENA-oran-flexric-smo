"""Detach only the isolated, already prepared 16-run evaluation campaign."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = Path('/home/user/LENA-oran-flexric-smo/experiments/drl_xapp_offline')


def main():
    if sys.platform != 'linux' or ROOT.resolve() != EXPECTED.resolve():
        raise RuntimeError('Only the isolated WSL experiment can be launched')
    if (ROOT/'campaign_status.json').exists() or (ROOT/'background_launch.json').exists():
        raise RuntimeError('Evaluation already initiated; inspect status instead of relaunching')
    if not (ROOT/'plan.json').exists():
        raise RuntimeError('Run init before background launch; campaign will persist its preflight manifest')
    command=[sys.executable, str(ROOT/'evaluation/campaign.py'), 'launch', '--workers', '16']
    with (ROOT/'campaign_console.log').open('xb') as log, open(os.devnull, 'rb') as stdin:
        process=subprocess.Popen(command,cwd=ROOT,stdin=stdin,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(1)
    record=dict(pid=process.pid,command=command,started_unix_s=time.time(),exit_code=process.poll())
    (ROOT/'background_launch.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    if record['exit_code'] is not None: raise RuntimeError('Campaign exited immediately; see log')


if __name__=='__main__': main()
