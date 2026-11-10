#!/usr/bin/env python3
"""Distributed RQ3 ns-3 replay. No LLM, inference service, or network calls."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

CELLS = [('5g','gNB_5G'),('4g1','gNB_4G_1'),('4g2','gNB_4G_2')]
FIELDS = [('txp','tx_power_dbm'),('ret','ret_tilt_deg'),('cio','cio_bias_db')]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)


def cpu_slots(cores, workers, available):
    if cores < 1 or workers < 1 or cores*workers > len(available):
        raise ValueError(f'Need {cores} x {workers} CPUs; allowed CPU count={len(available)}')
    return [available[i*cores:(i+1)*cores] for i in range(workers)]


def select(jobs, first, last):
    if not 0 <= first <= last < len(jobs):
        raise ValueError(f'Inclusive index range must be within 0..{len(jobs)-1}')
    result=jobs[first:last+1]
    assert [j['index'] for j in result]==list(range(first,last+1))
    return result


def clean_env(repo):
    env={k:v for k,v in os.environ.items() if not k.startswith(('AGORA_','RIC_','MITIGATION_PLAN_'))}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MITIGATION_PLAN_ALLOW_CONTINUOUS_TXP='1')
    env.pop('PYTHONHOME',None)
    sites=sorted((repo/'.venv/lib').glob('python*/site-packages'))
    if sites: env['PYTHONPATH']=str(sites[-1])
    return env


def cache_file(root, tilt, band):
    return root/f'tilt_{int(tilt)}deg'/f'sionna_rt_cache_{band}_2x2_1x1_lzf.h5'


def initial_caches(cache, p):
    a,b=p['ret_4g1'],p['ret_4g2']
    low=cache_file(cache,a,'1p8ghz') if a==b else cache/'merged_percell'/f'merged_1p8ghz_4g1_{int(a)}deg_4g2_{int(b)}deg.h5'
    return low,cache_file(cache,p['ret_5g'],'3p5ghz')


def argv(repo, bundle, manifest, job, cache, output, cpus, memory):
    p=job['initial_parameters']; low,high=initial_caches(cache,p)
    def cell_values(family):
        return ','.join(f'{name}:{p[f"{family}_{label}"]:g}' for label,name in CELLS)
    return ['taskset','-c',','.join(map(str,cpus)),'prlimit',f'--as={memory}',
            str(repo/'build/scratch/agora-conflict-sweep/ns3.48-khu-agora-conflict-sweep-default'),
            '--centralFrequency18=1.8e9','--bandwidth18=10e6','--centralFrequency35=3.5e9',
            '--bandwidth35=10e6','--numerology=0',f'--sionnaCacheFile18={low}',f'--sionnaCacheFile35={high}',
            '--sionnaUpdatePeriod=1s','--udpBaseIntervalMs=200','--trafficMultiplierEnabled=true',
            '--simTime=900','--kpiReportInterval=1','--N_Ues=300',
            f'--sumoTrace={bundle/job["trace"]}',f'--gnbPositions={bundle/manifest["gnb_positions"]}',
            f'--resultsRoot={output}',f'--runTag={job["job_id"]}','--RngSeed=1','--RngRun=1',
            '--ns3::NrHelper::E2ModeNr=false','--ns3::NrHelper::UseIdealRrc=true',
            f'--parameterSchedule={bundle/job["schedule"]}',f'--parameterScheduleRetCacheRoot={cache}',
            '--cellRetBearingDeg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10',
            '--cellTxPowerDbm='+cell_values('txp'),'--cellRetTiltDeg='+cell_values('ret'),
            '--cellCioDb='+cell_values('cio'),f'--handoverTtt={p["ttt_ms"]:g}',
            f'--handoverHysteresis={p["hys_db"]:g}',
            '--cellTttMs='+','.join(f'{name}:{p["ttt_ms"]:g}' for _,name in CELLS),
            '--cellHysteresisDb='+','.join(f'{name}:{p["hys_db"]:g}' for _,name in CELLS)]


def preflight(repo,bundle,manifest,jobs,cache,env):
    binary=repo/'build/scratch/agora-conflict-sweep/ns3.48-khu-agora-conflict-sweep-default'
    if not os.access(binary,os.X_OK):
        raise ValueError(f'Missing executable: {binary}; build khu-agora-conflict-sweep first')
    for tool in ['taskset','prlimit']:
        if not shutil.which(tool): raise ValueError(f'Missing Linux tool: {tool}')
    for rel,expected in manifest['backend_source_sha256'].items():
        path=repo/rel
        if not path.is_file() or sha(path)!=expected:
            raise ValueError(f'Backend source mismatch: {rel}. Compare frozen backend_reference before building.')
    hashes={}
    def verify(path,expected=None):
        key=str(path)
        if key not in hashes:
            hashes[key]=sha(path)
        if expected is not None and hashes[key]!=expected:
            raise ValueError(f'Input hash mismatch: {path}')
    verify(bundle/manifest['gnb_positions'],manifest['gnb_positions_sha256'])
    pairs=set()
    for job in jobs:
        verify(bundle/job['trace'],job['trace_sha256'])
        verify(bundle/job['historical_csv'],job['historical_csv_sha256'])
        verify(bundle/job['schedule'],job['schedule_sha256'])
        p=job['initial_parameters']; pairs.add((p['ret_5g'],p['ret_4g1'],p['ret_4g2']))
        for key,value in list(p.items())+[(k,v) for e in job['events'] for k,v in e['updates'].items()]:
            if key.startswith('ret_'):
                if value not in {3,5,7,9,12}: raise ValueError(f'Unsupported RET {value}')
                verify(cache_file(cache,value,'3p5ghz' if key=='ret_5g' else '1p8ghz'))
    # Serialize merges across independent ranges on the same machine.
    import fcntl
    with (cache/'.rq3_merge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for a,b,c in sorted(pairs):
            command=[sys.executable,str(bundle/'experiment/khu/rq3_final/build_merged_ret_cache.py'),
                     '--tilt-5g',str(int(a)),'--tilt-4g1',str(int(b)),'--tilt-4g2',str(int(c)),
                     '--merged-dir',str(cache/'merged_percell')]
            subprocess.run(command,env={**env,'RET_CACHE_ROOT':str(cache)},check=True,capture_output=True,text=True)
    for job in jobs:
        for path in initial_caches(cache,job['initial_parameters']): verify(path)
    return dict(binary_sha256=sha(binary),file_sha256=hashes,backend_source_sha256=manifest['backend_source_sha256'])


def expected_controls(job,t):
    state=dict(job['initial_parameters'])
    for event in job['events']:
        if event['time_s'] <= t: state.update(event['updates'])
    return state


def close_value(a,b):
    try:
        x,y=float(a),float(b)
        return (math.isnan(x) and math.isnan(y)) or x==y or math.isclose(x,y,rel_tol=1e-7,abs_tol=1e-6)
    except (ValueError,TypeError):
        return a==b


def verify_output(bundle,job,folder,returncode):
    issues=[]; prefix={}; hashes={}
    try:
        for name in ['cell_kpi.csv','ue_kpi.csv','parameter_changes.csv']:
            path=folder/name
            if not path.is_file() or path.stat().st_size==0: raise ValueError('Missing '+name)
            hashes[name]=sha(path)
        if returncode!=0: issues.append(f'exit={returncode}')
        with (folder/'cell_kpi.csv').open(newline='') as f: rows=list(csv.DictReader(f))
        keys=[(float(r['time_s']),int(r['cell_id'])) for r in rows]
        if len(keys)!=2700 or set(keys)!={(s+.5,c) for s in range(900) for c in (1,2,3)}:
            issues.append('incomplete_or_duplicate_cell_grid')
        for r in rows:
            t=float(r['time_s']); c=int(r['cell_id']); p=expected_controls(job,t)
            label=CELLS[c-1][0]
            checks=[(column,p[f'{family}_{label}']) for family,column in FIELDS]+[
                ('ttt_ms',p['ttt_ms']),('hysteresis_db',p['hys_db'])]
            if any(not close_value(r[k],v) for k,v in checks):
                issues.append(f'control_readback_mismatch_at_{t}_cell_{c}');break
        with (folder/'parameter_changes.csv').open(newline='') as f: receipts=list(csv.DictReader(f))
        wanted=[(e['time_s'],p,v) for e in job['events'] for p,v in e['updates'].items()]
        actual=[(float(r['scheduled_time_s']),r['parameter'],float(r['after'])) for r in receipts]
        if sorted(actual)!=sorted(wanted) or any(r['status']!='applied' or not close_value(r['scheduled_time_s'],r['applied_time_s']) for r in receipts):
            issues.append('event_receipt_mismatch')
        with (bundle/job['historical_csv']).open(newline='') as f: original=list(csv.DictReader(f))
        current={(float(r['time_s']),int(r['cell_id'])):r for r in rows}
        mismatch=0; first=None
        for r in original:
            if float(r['time_s'])>=job['t0_s']: continue
            other=current.get((float(r['time_s']),int(r['cell_id'])))
            different=[] if other is not None else ['missing_row']
            if other is not None:
                different=[k for k in r if k not in other or not close_value(r[k],other[k])]
            if different:
                mismatch+=1
                if first is None: first=dict(time_s=r['time_s'],cell_id=r['cell_id'],columns=different)
        prefix=dict(historical_prefix_match=mismatch==0,mismatching_rows=mismatch,first_difference=first,
                    interpretation='A mismatch blocks attribution versus historical CSV; compare matched replay controls after collecting all machines.')
    except Exception as error:
        issues.append(str(error))
    return dict(execution_verified=not issues,issues=issues,output_sha256=hashes,**prefix)


def completed(folder,signature):
    try:
        record=read(folder/'execution.json')
        return (record['status']=='complete' and record['signature']==signature and
                all(sha(folder/name)==value for name,value in record['verification']['output_sha256'].items()))
    except (KeyError,OSError,ValueError):
        return False


def job_signature(base,job,cache,provenance):
    paths=set(initial_caches(cache,job['initial_parameters']))
    pairs=list(job['initial_parameters'].items())+[(k,v) for e in job['events'] for k,v in e['updates'].items()]
    for key,value in pairs:
        if key.startswith('ret_'):
            paths.add(cache_file(cache,value,'3p5ghz' if key=='ret_5g' else '1p8ghz'))
    hashes={p.relative_to(cache).as_posix():provenance['file_sha256'][str(p)] for p in paths}
    return hashlib.sha256((base+json.dumps(hashes,sort_keys=True)).encode()).hexdigest()


def run(args,repo,bundle,manifest,jobs,slots,output,cache,provenance,signature):
    import fcntl
    output.mkdir(parents=True,exist_ok=True)
    locks=[]; running={}; pending=list(jobs); failures=[]; successes=[]
    env=clean_env(repo)
    memory=int(os.environ.get('RQ3_MEMORY_LIMIT_BYTES','3221225472'))
    if memory<=0: raise ValueError('RQ3_MEMORY_LIMIT_BYTES must be positive')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+f'_{os.getpid()}'
    batch=output/'_batches'/f'{args.start}_{args.end}_{stamp}.json'
    def status():
        save(batch,dict(updated=now(),range_inclusive=[args.start,args.end],cpu_slots=slots,
                        running=[x['job']['index'] for x in running.values()],pending=[j['index'] for j in pending],
                        successes=successes,failures=failures,signature=signature,provenance=provenance))
    def stop(sig,frame):
        raise KeyboardInterrupt
    old_int=signal.signal(signal.SIGINT,stop);old_term=signal.signal(signal.SIGTERM,stop)
    try:
        # A process-level CPU reservation prevents concurrent sweeps from silently sharing slots.
        for cpu in sorted({c for s in slots for c in s}):
            handle=open(f'/tmp/agora_rq3_cpu_{os.getuid()}_{cpu}.lock','a')
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        while pending or running:
            for slot,item in list(running.items()):
                rc=item['process'].poll()
                if rc is None: continue
                item['log'].close(); v=verify_output(bundle,item['job'],item['folder'],rc)
                state='complete' if v['execution_verified'] else 'failed'
                record={**item['record'],'status':state,'finished_at':now(),'returncode':rc,'verification':v}
                save(item['folder']/'execution.json',record)
                (successes if state=='complete' else failures).append(item['job']['index'])
                print(f'[{state}] {item["job"]["job_id"]} historical_prefix={v.get("historical_prefix_match")}',flush=True)
                item['lock'].close();del running[slot];status()
            for slot,cpus in enumerate(slots):
                if slot in running or not pending: continue
                job=pending.pop(0);folder=output/job['job_id']
                current_signature=job_signature(signature,job,cache,provenance)
                lockdir=output/'_locks';lockdir.mkdir(exist_ok=True)
                joblock=(lockdir/(job['job_id']+'.lock')).open('a')
                fcntl.flock(joblock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if completed(folder,current_signature):
                    print('[skip]',job['job_id'],flush=True);successes.append(job['index']);joblock.close();continue
                if folder.exists():
                    if not args.retry_failed:
                        joblock.close();raise ValueError(f'{folder} is incomplete or has different provenance. Inspect, then use --retry-failed to archive and retry.')
                    resolved=folder.resolve()
                    if resolved.parent!=output.resolve(): raise ValueError('Unsafe archive path')
                    archive=output/'_attempts';archive.mkdir(exist_ok=True)
                    folder.rename(archive/(job['job_id']+'_'+stamp))
                folder.mkdir();log=(folder/'launcher.log').open('xb')
                command=argv(repo,bundle,manifest,job,cache,output,cpus,memory)
                record=dict(status='launching',job_index=job['index'],job_id=job['job_id'],cpus=cpus,
                            started_at=now(),argv=command,signature=current_signature,provenance=provenance)
                save(folder/'execution.json',record)
                try:
                    process=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except Exception:
                    log.close();joblock.close();raise
                record.update(status='running',pid=process.pid);save(folder/'execution.json',record)
                running[slot]=dict(process=process,record=record,job=job,folder=folder,log=log,lock=joblock)
                print(f'[launch] {job["job_id"]} cpu={cpus}',flush=True);status()
            if running: time.sleep(.5)
        status();return 1 if failures else 0
    finally:
        for item in running.values():
            if item['process'].poll() is None:
                os.killpg(item['process'].pid,signal.SIGTERM)
                try: item['process'].wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(item['process'].pid,signal.SIGKILL);item['process'].wait()
            item['log'].close();item['lock'].close()
            save(item['folder']/'execution.json',{**item['record'],'status':'interrupted','finished_at':now()})
        for lock in locks: lock.close()
        signal.signal(signal.SIGINT,old_int);signal.signal(signal.SIGTERM,old_term)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['start','end','cores_per_job','parallel_jobs']: parser.add_argument(name,type=int)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    bundle=Path(__file__).resolve().parents[3]
    repo=Path(os.environ.get('RQ3_NS3_ROOT',str(bundle))).resolve()
    manifest_path=bundle/'experiment/khu/rq3_final/manifest.json';manifest=read(manifest_path)
    if manifest.get('schema')!='rq3-numbered-replay-v1': raise ValueError('Unexpected manifest schema')
    jobs=select(manifest['jobs'],args.start,args.end)
    output=Path(os.environ.get('RQ3_RESULTS_ROOT',str(repo/'experiment/khu/results/rq3_final'))).resolve()
    cache=Path(os.environ.get('RET_CACHE_ROOT',str(repo/'experiment/khu/caches/ret_caches'))).resolve()
    available=sorted(os.sched_getaffinity(0))
    slots=cpu_slots(args.cores_per_job,args.parallel_jobs,available)
    if args.status:
        for job in jobs:
            path=output/job['job_id']/'execution.json'
            print(job['index'],read(path)['status'] if path.exists() else 'not_started')
        return 0
    if args.dry_run:
        print(json.dumps(dict(range_inclusive=[args.start,args.end],job_count=len(jobs),cpu_slots=slots,
                             output_root=str(output),simulation_started=False),indent=2))
        for j in jobs: print(j['index'],j['job_id'],j['kind'],j['candidate_id'],f'H{j["horizon_minutes"]}')
        return 0
    provenance=preflight(repo,bundle,manifest,jobs,cache,clean_env(repo))
    # A job signature is independent of selected range and CPU allocation.
    # Cache hashes, binary, source and all manifest data bind reuse across machines.
    signature=hashlib.sha256((sha(manifest_path)+provenance['binary_sha256']+
                             json.dumps(manifest['backend_source_sha256'],sort_keys=True)).encode()).hexdigest()
    if args.preflight:
        print(json.dumps(dict(preflight_passed=True,jobs=len(jobs),signature=signature,simulation_started=False),indent=2));return 0
    return run(args,repo,bundle,manifest,jobs,slots,output,cache,provenance,signature)


if __name__=='__main__':
    try: sys.exit(main())
    except KeyboardInterrupt: print('Interrupted; partial outputs retained.',file=sys.stderr);sys.exit(130)
    except Exception as e: print(f'ERROR: {e}',file=sys.stderr);sys.exit(1)
