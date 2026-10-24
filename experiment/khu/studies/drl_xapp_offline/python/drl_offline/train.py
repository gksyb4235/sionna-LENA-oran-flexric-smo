"""Model-based offline DDQN from the original training partition only.

Static traces identify conditional response, not action-change dynamics. Synthetic
transitions are explicitly labelled and are never called observed RL experience.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import (STATE_DIM, STATE_SCHEMA, CHECKPOINT_SCHEMA, Y_NAMES,
                     action_catalog, valid_mask, apply_actions, states, metrics, rewards)
from .models import QNetwork, ResponseNetwork, ResponseEnsemble, encode_targets

ROOT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factual_report(y, pred, train_y):
    output = {}
    for j, name in enumerate(Y_NAMES):
        if name in ('generated_mbps', 'mean_ues'):
            output[name] = dict(role='provided rollout context; copied input, not a predicted target')
            continue
        actual, estimated = y[:, j], pred[:, j]
        mse = np.mean((actual-estimated)**2)
        variance = np.mean((actual-actual.mean())**2)
        output[name] = dict(rmse=float(np.sqrt(mse)), mae=float(np.abs(actual-estimated).mean()),
                            r2=float(1-mse/variance) if variance > 1e-10 else None,
                            mean_baseline_rmse=float(np.sqrt(np.mean((actual-train_y[:, j].mean())**2))))
    truth = metrics(np.repeat(y[:, None], 3, axis=1))['short']
    estimate = metrics(np.repeat(pred[:, None], 3, axis=1))['short']
    for k in ('jain', 'pp_per_ho', 'delivery_ratio', 'rlf_per_1000_ue_s'):
        valid = truth['ho_attempts'] >= 10 if k == 'pp_per_ho' else np.ones(len(y), bool)
        a, b = truth[k][valid], estimate[k][valid]
        output[k] = dict(n=int(valid.sum()), mae=float(np.abs(a-b).mean()),
                         rmse=float(np.sqrt(np.mean((a-b)**2))))
    return output


def fit_response(data, out, device, epochs):
    p, exog, y = data['params'], data['exog'], data['y']
    fit, hold = data['inner_fit_idx'], data['inner_holdout_idx']
    # Exclude warm-up bins, and split by entire runs rather than overlapping windows.
    def flatten(ids):
        e = exog[ids, 3:].reshape(-1, 3)
        yy = y[ids, 3:].reshape(-1, 18)
        x = np.column_stack([np.repeat(p[ids], 27, axis=0), e]).astype(np.float32)
        return x, encode_targets(yy, e), yy, e
    xf, zf, yf, ef = flatten(fit)
    xv, zv, yv, ev = flatten(hold)
    xm, xs = xf.mean(0), np.maximum(xf.std(0), 1e-3)
    zm, zs = zf.mean(0), np.maximum(zf.std(0), .1)
    x = torch.tensor((xf-xm)/xs, device=device)
    z = torch.tensor((zf-zm)/zs, device=device)
    vx = torch.tensor((xv-xm)/xs, device=device)
    vz = torch.tensor((zv-zm)/zs, device=device)
    models, records = [], []
    for member in range(3):
        seed = 8100 + member
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        # Bootstrap whole runs to preserve temporal clustering.
        boot = rng.integers(0, len(fit), len(fit))[:, None]*27 + np.arange(27)[None]
        boot = torch.tensor(boot.ravel(), device=device)
        model = ResponseNetwork().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_epoch, best_state = float('inf'), 0, None
        for epoch in range(1, epochs+1):
            model.train()
            order = boot[torch.randperm(len(boot), device=device)]
            for ids in order.split(1024):
                loss = nn.functional.mse_loss(model(x[ids]), z[ids])
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val = float(nn.functional.mse_loss(model(vx), vz).item())
            if val < best-1e-5:
                best, best_epoch, best_state = val, epoch, copy.deepcopy(model.state_dict())
            if epoch % 25 == 0:
                print(json.dumps(dict(stage='response', member=member, epoch=epoch, val=val, best=best)), flush=True)
            if epoch-best_epoch >= 35:
                break
        model.load_state_dict(best_state); model.eval()
        models.append(model)
        records.append(dict(member=member, seed=seed, selected_epoch=best_epoch,
                            validation_standardized_mse=best))
    ensemble = ResponseEnsemble(models, xm, xs, zm, zs, device)
    prediction = ensemble.predict(np.repeat(p[hold], 27, axis=0), ev).mean(0)
    report = factual_report(yv, prediction, yf)
    # Count heads predict conditional log-scale centers. Record bias rather than
    # silently rescaling sparse events to manufacture a better response.
    for j in range(9, 15):
        report[Y_NAMES[j]]['actual_mean'] = float(yv[:, j].mean())
        report[Y_NAMES[j]]['predicted_mean'] = float(prediction[:, j].mean())
    torch.save(dict(schema='static-response-ensemble-v1', models=[m.cpu().state_dict() for m in models],
                    x_mean=xm.tolist(), x_std=xs.tolist(), z_mean=zm.tolist(), z_std=zs.tolist(),
                    metadata=records), out/'response_ensemble.pt')
    for m in models: m.to(device)
    save_json(out/'response_validation.json', dict(partition='144 inner-held-out runs inside original train720',
                fit_runs=len(fit), selection_runs=len(hold), records=records, metrics=report,
                limitation='Factual static response only; no observed action-change validation.'))
    return ensemble, report


def contexts(data, ids, rng, n, max_start=27):
    runs = rng.choice(ids, n)
    bins = rng.integers(3, max_start+1, n)
    history = data['y'][runs[:, None], bins[:, None]-np.arange(3, 0, -1)]
    return runs, bins, history.copy(), data['params'][runs].copy()


def support_distance(p, reference):
    scale = np.array([13]*3+[9]*3+[4]*3+[592, 4.5])
    p, reference = p/scale, reference/scale
    result = []
    for chunk in np.array_split(p, max(1, int(np.ceil(len(p)/1024)))):
        distance = np.maximum((chunk**2).sum(1)[:, None]+(reference**2).sum(1)[None]-2*chunk@reference.T, 0)
        result.append(np.sqrt(distance.min(1)/11))
    return np.concatenate(result)


def transition(ensemble, app, h, p, nextp, exog, reference, rng=None):
    ys = ensemble.predict(nextp, exog)
    member_histories = [np.concatenate([h[:, 1:], yy[:, None]], axis=1) for yy in ys]
    rs = np.stack([rewards(app, h, hh, p, nextp) for hh in member_histories])
    # Pessimism is only a heuristic, not a calibrated uncertainty guarantee.
    penalty = .75*rs.std(0) + .05*support_distance(nextp, reference)
    if rng is None:
        y = ys.mean(0)
    else:
        y = ys[rng.integers(0, len(ys), len(p)), np.arange(len(p))]
    nxt = np.concatenate([h[:, 1:], y[:, None]], axis=1)
    return nxt, rs.mean(0)-penalty, rs.std(0), penalty


def make_replay(data, ensemble, app, device, n, seed):
    rng = np.random.default_rng(seed)
    fit = data['inner_fit_idx']; reference=data['params'][fit]
    runs, bins, h, p = contexts(data, fit, rng, n)
    buffers = {k:[] for k in ('s','a','r','ns','done','mask')}
    action_counts = np.zeros(len(action_catalog(app)), dtype=int)
    for step in range(3):
        mask = valid_mask(p, app)
        ids = np.argmax(np.where(mask, rng.random(mask.shape), -1), axis=1)
        newp = apply_actions(p, ids, app)
        nh, reward, uncertainty, penalty = transition(ensemble, app, h, p, newp, data['exog'][runs, bins], reference, rng)
        buffers['s'].append(states(h, p, bins*30)); buffers['a'].append(ids)
        buffers['r'].append(reward); buffers['ns'].append(states(nh, newp, (bins+1)*30))
        buffers['done'].append(bins==29); buffers['mask'].append(valid_mask(newp, app))
        action_counts += np.bincount(ids, minlength=len(action_counts))
        h, p, bins = nh, newp, bins+1
    arrays = {k:np.concatenate(v) for k,v in buffers.items()}
    tensors = {k:torch.tensor(v, device=device, dtype=torch.long if k=='a' else torch.bool if k in ('mask','done') else torch.float32) for k,v in arrays.items()}
    return tensors, dict(transitions=len(arrays['a']), synthetic=True, rollout_steps=3,
                         action_counts=action_counts.tolist(), state_rows_observed_initial=n,
                         reward_mean=float(arrays['r'].mean()), reward_std=float(arrays['r'].std()))


@torch.inference_mode()
def policy_score(model, data, ensemble, app, device, n=2048):
    rng = np.random.default_rng(20261018)
    runs, bins, initial, p0 = contexts(data, data['inner_holdout_idx'], rng, n)
    reference = data['params'][data['inner_fit_idx']]
    scores, ends, action_histogram = {}, {}, np.zeros(len(action_catalog(app)), int)
    for name in ('hold', 'policy'):
        h, p = initial.copy(), p0.copy()
        total = np.zeros(n)
        for step in range(3):
            if name == 'hold':
                ids = np.zeros(n, int)
            else:
                s = torch.tensor(states(h, p, (bins+step)*30), device=device)
                q = model(s).cpu().numpy()
                ids = np.where(valid_mask(p, app), q, -np.inf).argmax(1)
                action_histogram += np.bincount(ids, minlength=len(action_histogram))
            newp = apply_actions(p, ids, app)
            h, r, _, _ = transition(ensemble, app, h, p, newp, data['exog'][runs, bins+step], reference)
            total += (.9**step)*r
            p = newp
        scores[name] = total
        m = metrics(h)['short']
        ends[name] = {k:float(m[k].mean()) for k in ('jain','pp_per_ho','ho_attempts','ho_successes','rlf','delivery_ratio')}
        ends[name]['pp_ratio_of_sums'] = float(m['pingpongs'].sum()/max(m['ho_attempts'].sum(),1e-9))
    difference = scores['policy']-scores['hold']
    return dict(policy_return=float(scores['policy'].mean()), hold_return=float(scores['hold'].mean()),
                improvement=float(difference.mean()), win_fraction=float((difference>0).mean()),
                action_counts=action_histogram.tolist(), final_metrics=ends,
                ho_success_retention=float(ends['policy']['ho_successes']/max(ends['hold']['ho_successes'],1e-9)),
                caution='PP reduction may include HO suppression; RLF and delivery must be checked in ns3.',
                label='synthetic inner-selection proxy; not actual ns3 policy performance')


def train_policy(data, ensemble, app, out, device, steps, replay_starts):
    replay, replay_record = make_replay(data, ensemble, app, device, replay_starts, 9201 if app=='lb' else 9202)
    candidates=[]; winner=None; winner_score=-float('inf')
    for seed in (42,43,44):
        torch.manual_seed(seed)
        model=QNetwork(STATE_DIM,len(action_catalog(app))).to(device)
        target=copy.deepcopy(model).eval()
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        losses=[]
        for step in range(1,steps+1):
            i=torch.randint(len(replay['a']),(512,),device=device)
            with torch.no_grad():
                next_ids=model(replay['ns'][i]).masked_fill(~replay['mask'][i],-torch.inf).argmax(1)
                next_q=target(replay['ns'][i]).gather(1,next_ids[:,None]).squeeze(1)
                expected=replay['r'][i]+.9*(~replay['done'][i]).float()*next_q
            actual=model(replay['s'][i]).gather(1,replay['a'][i,None]).squeeze(1)
            loss=nn.functional.smooth_l1_loss(actual,expected)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()))
            if step%200==0: target.load_state_dict(model.state_dict())
            if step%1000==0:
                print(json.dumps(dict(stage='ddqn',app=app,seed=seed,step=step,loss=float(np.mean(losses[-200:])))),flush=True)
        model.eval()
        score=policy_score(model,data,ensemble,app,device)
        record=dict(seed=seed,gradient_updates=steps,tail_td_loss=float(np.mean(losses[-200:])),selection=score)
        candidates.append(record)
        checkpoint=dict(schema=CHECKPOINT_SCHEMA,app=app,state_dim=STATE_DIM,state_schema=STATE_SCHEMA,
             action_catalog=action_catalog(app),state_dict={k:v.cpu() for k,v in model.state_dict().items()},
             gradient_updates=steps,metadata=dict(method='model-based offline Double DQN',seed=seed,
             train_source='original train720; inner576 fit /144 selection',
             data_arrays_sha256=sha(ROOT/'data_audit/data_arrays.npz'),
             response_ensemble_sha256=sha(out/'response_ensemble.pt'),
             synthetic_replay=replay_record,selection=score,online_learning=False,
             discount=.9,uncertainty_reward_penalty=.75,support_distance_penalty=.05,
             limitation='Fixed-configuration data do not identify action-change transients.'))
        torch.save(checkpoint,out/f'{app}_seed{seed}.pt')
        if score['improvement']>winner_score:
            winner_score=score['improvement'];winner=checkpoint
        print(json.dumps(dict(stage='selection',app=app,seed=seed,**score)),flush=True)
    selected=ROOT/'selected';selected.mkdir(exist_ok=True)
    torch.save(winner,selected/f'{app}.pt')
    report=dict(app=app,state_dim=STATE_DIM,action_count=len(action_catalog(app)),
                network_parameters=sum(v.numel() for v in winner['state_dict'].values()),
                replay=replay_record,candidates=candidates,selected_seed=winner['metadata']['seed'],
                selected_sha256=sha(selected/f'{app}.pt'))
    save_json(out/f'{app}_training.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--response-epochs',type=int,default=250)
    parser.add_argument('--policy-steps',type=int,default=6000)
    parser.add_argument('--replay-starts',type=int,default=16000)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    start=time.time();out=ROOT/'training';out.mkdir(exist_ok=True)
    if (ROOT/'selected/lb.pt').exists(): raise RuntimeError('Refusing to overwrite selected frozen policies')
    data=dict(np.load(ROOT/'data_audit/data_arrays.npz',allow_pickle=False))
    contract=json.loads((ROOT/'data_audit/arrays_contract.json').read_text())
    assert sha(ROOT/'data_audit/data_arrays.npz')==contract['arrays_sha256']
    assert len(data['params'])==720 and not set(data['inner_fit_idx'])&set(data['inner_holdout_idx'])
    manifest=dict(started_unix_s=start,device=args.device,
        torch=str(torch.__version__),numpy=str(np.__version__),
        input_hashes={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'data_audit').glob('*') if f.is_file()},
        code_hashes={str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.glob('*.py')},
        assumptions=['Static conditional response is reused after a parameter change; transients are unobserved.',
        'Future replayed aggregate generated-load and connected-UE profiles drive the synthetic environment only.',
        'Those profiles are not strictly control-independent; the actor never receives future realized KPI.',
        'Soft reward penalties are not hard safety constraints; actual ns3 outcomes decide performance.'])
    save_json(out/'provenance.json',manifest)
    ensemble,validation=fit_response(data,out,args.device,args.response_epochs)
    reports=[train_policy(data,ensemble,app,out,args.device,args.policy_steps,args.replay_starts) for app in ('lb','mo')]
    report=dict(status='complete',elapsed_s=time.time()-start,configuration=vars(args),
                original_train_runs=720,inner_fit_runs=576,inner_selection_runs=144,
                original_validation_or_test_used=False,actual_action_change_training_rows=0,
                response_validation=validation,policies=reports)
    save_json(out/'training_report.json',report)
    print(json.dumps(dict(stage='complete',elapsed_s=report['elapsed_s'])),flush=True)


if __name__=='__main__': main()
