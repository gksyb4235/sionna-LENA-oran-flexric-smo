"""Frozen offline-trained policies controlling a single isolated ns-3 bridge."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .common import (CELLS,PARAMETERS,Y_NAMES,STATE_DIM,STATE_SCHEMA,CHECKPOINT_SCHEMA,
                     action_catalog,valid_mask,apply_actions,states,metrics)

FIELDS=('time_s','interval_s','cell_id','num_ues','delivered_megabits_interval',
        'generated_load_mbps','prb_used_reg','prb_capacity_reg','ho_attempts_interval',
        'ho_successes_interval','ho_failures_interval','pingpongs_interval','rlfs_interval',
        'delay_sum_ms_interval','delay_samples_interval','tx_power_dbm','ret_tilt_deg')


def atomic_json(path,value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def weights_digest(model):
    result=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        result.update(name.encode());result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def _ratio(n,d,default=0.):
    return float(n/d) if d>0 else float(default)


def read_history(path,now):
    """Read only flushed rows strictly before now; return three exact 30s bins."""
    if now<90:
        raise ValueError('Frozen policy requires90 completed observation seconds')
    rows=[]
    with Path(path).open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle)
        if set(FIELDS)-set(reader.fieldnames or []):
            raise ValueError(f'Missing required columns: {sorted(set(FIELDS)-set(reader.fieldnames or []))}')
        for raw in reader:
            timestamp=float(raw['time_s'])
            if now-90<=timestamp<now:
                row={k:float(raw[k]) for k in FIELDS}
                if not all(np.isfinite(v) for v in row.values()):
                    raise ValueError('Nonfinite completed observation field')
                rows.append(row)
    rows.sort(key=lambda r:(r['time_s'],r['cell_id']))
    if len(rows)!=270:
        raise ValueError(f'Expected270 cell rows for90 completed seconds, got{len(rows)}')
    times=np.asarray([r['time_s'] for r in rows]).reshape(90,3)
    cells=np.asarray([r['cell_id'] for r in rows]).reshape(90,3)
    expected=np.arange(now-90,now)+.5
    if not np.allclose(times,expected[:,None]) or not np.array_equal(cells,np.tile([1,2,3],(90,1))):
        raise ValueError('Missing/duplicate/irregular1-second cell reports')
    if not all(abs(r['interval_s']-1)<1e-8 for r in rows):
        raise ValueError('This policy requires1-second KPI report intervals')
    fixed=np.column_stack([np.asarray([r[k] for r in rows]).reshape(90,3) for k in ['tx_power_dbm','ret_tilt_deg']])
    if not np.allclose(fixed,fixed[0]):
        raise ValueError('TxP/RET must stay constant during this isolated evaluation')
    history=[]
    exact=[]
    for index in range(3):
        group=rows[index*90:(index+1)*90]
        total=lambda k:sum(r[k] for r in group)
        prb=[];ues=[];goodput=[]
        for cid in [1,2,3]:
            selected=[r for r in group if r['cell_id']==cid]
            subtotal=lambda k:sum(r[k] for r in selected)
            prb.append(_ratio(subtotal('prb_used_reg'),subtotal('prb_capacity_reg')))
            ues.append(subtotal('num_ues')/30)
            goodput.append(subtotal('delivered_megabits_interval')/30)
        event=[total(k) for k in ['ho_attempts_interval','ho_successes_interval','ho_failures_interval','pingpongs_interval','rlfs_interval']]
        delay_num=total('delay_sum_ms_interval');delay_den=total('delay_samples_interval')
        history.append(prb+ues+goodput+event+[_ratio(delay_num,delay_den),total('generated_load_mbps')/30,sum(ues),sum(goodput)])
        exact.append(dict(start_s=now-90+index*30,end_s=now-60+index*30,
                          delay_sum_ms=delay_num,delay_samples=delay_den))
    exact_delay=_ratio(sum(v['delay_sum_ms'] for v in exact),sum(v['delay_samples'] for v in exact))
    history=np.asarray(history,dtype=float)
    approximate_delay=float(metrics(history)['long']['delay_ms'][0])
    return history,fixed[-1],dict(causal_last_timestamp=float(times[-1,0]),windows=exact,
        short_delay_definition='exact packet-weighted mean from raw delay sum and samples',
        long_delay_definition='goodput-weighted mean of three30s packet-weighted means; same approximation used in offline state',
        long_delay_is_approximate=True,exact_long_delay_ms=exact_delay,
        state_long_delay_ms=approximate_delay,long_delay_approximation_error_ms=approximate_delay-exact_delay)


def json_metrics(value):
    if isinstance(value,dict):
        return {k:json_metrics(v) for k,v in value.items()}
    if isinstance(value,np.ndarray):
        result=value[0]
        return result.tolist() if result.ndim else result.item()
    if isinstance(value,np.generic):
        return value.item()
    return value


class Controller:
    def __init__(self,args):
        self.args=args
        if args.policy!='eval':
            raise ValueError('This runtime supports frozen evaluation only')
        self.directory=Path(args.control_dir).resolve();self.directory.mkdir(parents=True,exist_ok=True)
        self.names={'none':(),'lb':('lb',),'mo':('mo',),'both':('lb','mo')}[args.mode]
        self.models={};self.model_paths={};self.model_hashes_before={};self.weight_hashes_before={}
        self.training_updates={};self.last_seq=-1;self.last_time=-1.;self.previous_updates={}
        self.finished=False;self.fixed_parameters=None;self.observations=0;self.started_wall=time.time()
        self.log_path=self.directory/'actions.jsonl'
        if self.log_path.exists():
            raise ValueError(f'Refusing to append another run to{self.log_path}')
        torch.set_num_threads(1)
        if self.names:
            from .models import QNetwork
            for name in self.names:
                path=Path(args.load_dir)/(name+'.pt')
                checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                if checkpoint.get('schema')!=CHECKPOINT_SCHEMA or checkpoint.get('app')!=name or checkpoint.get('state_dim')!=STATE_DIM:
                    raise ValueError(f'Checkpoint contract mismatch:{path}')
                if checkpoint.get('action_catalog')!=action_catalog(name):
                    raise ValueError(f'Checkpoint action domain mismatch:{path}')
                if int(checkpoint.get('gradient_updates',0))<=0:
                    raise ValueError(f'Checkpoint has no completed offline gradient updates:{path}')
                model=QNetwork(STATE_DIM,len(action_catalog(name)))
                model.load_state_dict(checkpoint['state_dict'],strict=True)
                model.eval();model.requires_grad_(False)
                self.models[name]=model;self.model_paths[name]=path
                self.model_hashes_before[name]=digest(path);self.weight_hashes_before[name]=weights_digest(model)
                self.training_updates[name]=int(checkpoint['gradient_updates'])
        atomic_json(self.directory/'controller_config.json',dict(**vars(args),state_schema=STATE_SCHEMA,
            online_learning=False,algorithm='Frozen offline model-based Double DQN',
            action_catalogs={name:action_catalog(name) for name in self.names},
            metrics_contract='Past90seconds only; three30s bins; same60-feature transform as training',
            long_delay_approximation='Goodput-weighted30s mean delays; exact raw90s delay separately audited',
            model_hashes=self.model_hashes_before,seed_role='Recorded only; evaluation uses no random action selection'))

    def process(self,payload):
        began=time.perf_counter();seq=int(payload['seq']);now=float(payload['time_s'])
        terminal=bool(payload.get('terminal',False))
        if seq<=self.last_seq or now<=self.last_time:
            raise ValueError('Sequence/time must strictly increase')
        if self.last_seq>=0 and seq!=self.last_seq+1:
            raise ValueError('Skipped bridge observation sequence')
        actual={k:float(v) for k,v in payload['parameters'].items()}
        for k,v in self.previous_updates.items():
            if k not in actual or abs(actual[k]-v)>1e-6:
                raise ValueError(f'Previous action not applied:{k} expected{v}, actual{actual.get(k)}')
        history,fixed,audit=read_history(payload['cell_kpi_path'],now)
        if self.fixed_parameters is None:
            self.fixed_parameters=fixed.copy()
        elif not np.allclose(fixed,self.fixed_parameters):
            raise ValueError('TxP/RET changed outside this isolated xApp experiment')
        params=np.concatenate([fixed,np.asarray([actual[k] for k in PARAMETERS[6:]])])[None]
        state=states(history[None],params,now)
        observed=metrics(history)
        observation_ms=1000*(time.perf_counter()-began)
        updates={};records={}
        for name in self.names:
            active=now>=getattr(self.args,name+'_start_s')
            record=dict(active=active,explored=False,decision_ms=0.)
            if active and not terminal:
                tick=time.perf_counter()
                mask=valid_mask(params,name)[0]
                with torch.inference_mode():
                    q=self.models[name](torch.from_numpy(state)).cpu().numpy()[0]
                if not np.isfinite(q).all():
                    raise ValueError('Nonfinite Q-network output')
                action_id=int(np.argmax(np.where(mask,q,-np.inf)))
                target=apply_actions(params,[action_id],name)[0]
                proposed={k:float(target[i]) for i,k in enumerate(PARAMETERS) if abs(target[i]-params[0,i])>1e-8}
                if updates.keys()&proposed.keys():
                    raise ValueError('Disjoint xApps unexpectedly share a control parameter')
                updates.update(proposed)
                record.update(action_id=action_id,label=action_catalog(name)[action_id]['label'],updates=proposed,
                    valid_action_count=int(mask.sum()),decision_ms=1000*(time.perf_counter()-tick),
                    chosen_q=float(q[action_id]),online_gradient_updates=0)
            records[name]=record
        response=dict(seq=seq,updates=updates)
        if not terminal:
            atomic_json(self.directory/'action.json',response)
        event=dict(seq=seq,time_s=now,terminal=terminal,parameters=dict(zip(PARAMETERS,params[0].tolist())),
            updates=updates,apps=records,metrics=json_metrics(observed),observation_audit=audit,
            observation_ms=observation_ms,controller_ms=1000*(time.perf_counter()-began),wall_time_unix=time.time(),
            online_learning=False,online_gradient_updates=0,online_replay_inserts=0,exploratory_actions=0)
        with self.log_path.open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(event,allow_nan=False)+'\n')
        self.previous_updates=updates;self.last_seq=seq;self.last_time=now
        self.observations+=1;self.finished=terminal
        atomic_json(self.directory/'summary.json',self.summary())
        return response

    def summary(self):
        after={name:digest(path) for name,path in self.model_paths.items()}
        weights_after={name:weights_digest(model) for name,model in self.models.items()}
        if after!=self.model_hashes_before or weights_after!=self.weight_hashes_before:
            raise RuntimeError('Frozen checkpoint/model changed during evaluation')
        return dict(status='complete' if self.finished else 'running',policy='eval',mode=self.args.mode,
            online_learning=False,model_hashes_before=self.model_hashes_before,model_hashes_after=after,
            weight_hashes_before=self.weight_hashes_before,weight_hashes_after=weights_after,
            online_gradient_updates=0,online_replay_inserts=0,exploratory_actions=0,
            offline_gradient_updates=self.training_updates,last_seq=self.last_seq,last_time_s=self.last_time,
            observations=self.observations,elapsed_wall_s=time.time()-self.started_wall,
            long_delay_state_approximation='goodput-weighted30s means',convergence_claim=False)

    def run(self):
        path=self.directory/'observation.json';last_new_wall=time.monotonic()
        while not self.finished:
            if path.exists():
                payload=json.loads(path.read_text(encoding='utf-8'))
                if int(payload['seq'])>self.last_seq:
                    self.process(payload);last_new_wall=time.monotonic()
                    print(f'seq={self.last_seq} t={self.last_time:g} frozen_eval complete={self.finished}',flush=True)
            if time.monotonic()-last_new_wall>self.args.timeout_s:
                raise TimeoutError('No new ns-3 observation within controller timeout')
            if not self.finished:
                time.sleep(self.args.poll_s)
        return self.summary()


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--control-dir',required=True)
    result.add_argument('--mode',choices=('none','lb','mo','both'),default='both')
    result.add_argument('--load-dir',required=True)
    result.add_argument('--lb-start-s',type=float,default=180)
    result.add_argument('--mo-start-s',type=float,default=540)
    result.add_argument('--policy',choices=('eval',),default='eval')
    result.add_argument('--seed',type=int,default=909)
    result.add_argument('--timeout-s',type=float,default=86400)
    result.add_argument('--poll-s',type=float,default=.01)
    return result


def main(argv=None):
    args=parser().parse_args(argv);controller=None
    try:
        controller=Controller(args);controller.run()
    except Exception as exc:
        directory=Path(args.control_dir);directory.mkdir(parents=True,exist_ok=True)
        atomic_json(directory/'controller_error.json',dict(type=type(exc).__name__,message=str(exc),
            last_seq=controller.last_seq if controller else -1,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
