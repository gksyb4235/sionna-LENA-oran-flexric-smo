"""Identical numpy state/action/reward contracts for offline training and runtime."""
import numpy as np

CELLS = ('5g','4g1','4g2')
PARAMETERS = tuple(f'{p}_{c}' for p in ('txp','ret','cio') for c in CELLS)+('ttt_ms','hys_db')
Y_NAMES = ('prb_5g','prb_4g1','prb_4g2','ues_5g','ues_4g1','ues_4g2',
           'goodput_5g','goodput_4g1','goodput_4g2','ho_attempts','ho_successes','ho_failures',
           'pingpongs','rlf','delay_ms','generated_mbps','mean_ues','goodput_mbps')
CIO_LEVELS = ((5.,6.,7.,8.,9.),(-2.,-1.,0.,1.,2.),(-2.,-1.,0.,1.,2.))
TTT_LEVELS = (128.,256.,340.,512.,720.)
HYS_LEVELS = tuple(np.arange(.5,5.01,.5).tolist())
STATE_DIM = 60
STATE_SCHEMA = 'offline-static30-history3-fullparams60-v1'
CHECKPOINT_SCHEMA = 'offline-response-ddqn-v1'


def action_catalog(app):
    result=[dict(id=0,app=app,label='hold',updates={})]
    if app=='lb':
        for c,levels in zip(CELLS,CIO_LEVELS):
            for v in levels:
                result.append(dict(id=len(result),app=app,label=f'CIO[{c}] = {v:g} dB',updates={'cio_'+c:v}))
    elif app=='mo':
        for t in TTT_LEVELS:
            for h in HYS_LEVELS:
                result.append(dict(id=len(result),app=app,label=f'TTT = {t:g} ms; HYS = {h:g} dB',updates={'ttt_ms':t,'hys_db':h}))
    else:
        raise ValueError(app)
    return result


def parameter_batch(p):
    p=np.atleast_2d(np.asarray(p,dtype=float))
    if p.shape[1]!=11 or not np.isfinite(p).all():
        raise ValueError('Parameters must be finite (N,11) in canonical order')
    return p


def valid_mask(p,app):
    p=parameter_batch(p)
    mask=np.ones((len(p),len(action_catalog(app))),dtype=bool)
    for a in action_catalog(app)[1:]:
        mask[:,a['id']]=np.any(np.column_stack([np.abs(p[:,PARAMETERS.index(k)]-v)>1e-8 for k,v in a['updates'].items()]),axis=1)
    return mask


def apply_actions(p,ids,app):
    p=parameter_batch(p).copy()
    ids=np.broadcast_to(np.asarray(ids,dtype=int).reshape(-1),(len(p),))
    catalog=action_catalog(app)
    if (ids<0).any() or (ids>=len(catalog)).any():
        raise ValueError('Invalid action ID')
    for a in catalog[1:]:
        selected=ids==a['id']
        for k,v in a['updates'].items():
            p[selected,PARAMETERS.index(k)]=v
    return p


def _ratio(n,d,default=0.):
    return np.divide(n,d,out=np.full_like(np.asarray(n,dtype=float),default),where=np.asarray(d)>0)


def _aggregate(h):
    """Aggregate 30s metric bins; long delay uses goodput-weighted approximation."""
    prb=h[:,:,:3].mean(1); ues=h[:,:,3:6].mean(1); goodput=h[:,:,6:9].mean(1)
    attempts,success,failures,pp,rlf=(h[:,:,i].sum(1) for i in range(9,14))
    generated=h[:,:,15].mean(1); total_gp=goodput.sum(1); total_ue=ues.sum(1)
    duration=float(30*h.shape[1])
    delay=_ratio((h[:,:,14]*h[:,:,17]).sum(1),h[:,:,17].sum(1))
    return dict(prb=prb,ues=ues,cell_goodput_mbps=goodput,
        jain=_ratio(prb.sum(1)**2,3*(prb**2).sum(1)),active_load=prb.sum(1)>1e-8,
        goodput_mbps=total_gp,generated_mbps=generated,mean_ues=total_ue,
        delivery_ratio=_ratio(total_gp,generated,1.),has_traffic=generated>0,
        ho_attempts=attempts,ho_successes=success,ho_failures=failures,
        ho_failure_ratio=_ratio(failures,attempts),pingpongs=pp,rlf=rlf,
        pp_per_ho=_ratio(pp,attempts),pp_posterior=(pp+1)/(attempts+10),
        ho_confidence=attempts/(attempts+10),ho_success_support=np.minimum(success,attempts)/(attempts+10),
        rlf_per_1000_ue_s=_ratio(1000*rlf,total_ue*duration),delay_ms=delay,
        duration_s=duration)


def metrics(history):
    history=np.asarray(history,dtype=float)
    if history.ndim==2:
        history=history[None]
    if history.ndim!=3 or history.shape[1:]!=(3,18) or not np.isfinite(history).all():
        raise ValueError('History must be finite (N,3,18), ordered oldest to newest')
    return dict(short=_aggregate(history[:,-1:]),long=_aggregate(history))


def _features(m):
    cells=np.stack([m['prb'],m['ues']/100,m['cell_goodput_mbps']/50],axis=2).reshape(-1,9)
    extra=np.column_stack([m['jain'],m['ho_attempts']/100,m['pingpongs']/30,
        m['pp_posterior'],m['ho_confidence'],m['rlf']/10,m['rlf_per_1000_ue_s']/5,
        m['delivery_ratio'],m['delay_ms']/100,m['goodput_mbps']/100,m['generated_mbps']/100,
        m['mean_ues']/300,m['active_load'],m['has_traffic'],m['ho_failure_ratio']])
    return np.column_stack([cells,extra])


def states(history,p,time_s):
    p=parameter_batch(p); m=metrics(history)
    if len(p)!=len(m['short']['jain']):
        raise ValueError('History/parameter batch mismatch')
    clock=np.broadcast_to(np.asarray(time_s,dtype=float).reshape(-1),(len(p),))/900
    controls=np.column_stack([p[:,6:9]/6,p[:,9]/1024,p[:,10]/6])
    fixed=np.column_stack([p[:,:3]/43,p[:,3:6]/12])
    state=np.column_stack([_features(m['short']),_features(m['long']),controls,clock,fixed])
    assert state.shape==(len(p),STATE_DIM)
    return np.clip(state,-10,10).astype(np.float32)


def rewards(app,before_history,after_history,oldp,newp,variant='local'):
    """Local objectives with minimum HO/RLF/delivery guards; no surrogate here."""
    if variant not in ('local','guarded'):
        raise ValueError(variant)
    p=metrics(before_history)['short']; c=metrics(after_history)['short']
    oldp=parameter_batch(oldp);newp=parameter_batch(newp)
    if app=='lb':
        objective=np.where(c['active_load'],2*(c['jain']-.7)+2*(c['jain']-p['jain']),0)
        movement=np.abs(newp[:,6:9]-oldp[:,6:9]).sum(1)/6
    elif app=='mo':
        confidence=np.minimum(p['ho_success_support'],c['ho_success_support'])
        objective=5*c['ho_success_support']*(.1-c['pp_posterior'])+2*confidence*(p['pp_posterior']-c['pp_posterior'])
        movement=(np.abs(newp[:,9]-oldp[:,9])/592+np.abs(newp[:,10]-oldp[:,10])/4.5)/2
    else:
        raise ValueError(app)
    rlf_penalty=(.8 if variant=='guarded' else .15)*c['rlf_per_1000_ue_s']
    delivery_penalty=(4 if variant=='guarded' else .5)*np.where(c['has_traffic'],np.maximum(0,.95-c['delivery_ratio']),0)
    failures=(2 if variant=='guarded' else .5)*c['ho_failures']/(c['ho_attempts']+10)
    cross=0
    if variant=='guarded':
        cross=3*c['ho_confidence']*np.maximum(0,c['pp_posterior']-.1) if app=='lb' else 2*np.where(c['active_load'],np.maximum(0,.9-c['jain']),0)
    return np.clip(objective-rlf_penalty-delivery_penalty-failures-cross-.02*movement,-5,5).astype(np.float32)
