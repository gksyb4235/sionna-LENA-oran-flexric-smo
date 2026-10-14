"""Four node types, explicit KPI propagation, dynamic relation weights, future state."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
def dense(n, h, out, dropout=0.):
    return nn.Sequential(nn.Linear(n,h), nn.SiLU(), nn.Dropout(dropout), nn.Linear(h,out))

def registry_tensors():
    rows=json.loads((Path(__file__).parent/'app_registry.json').read_text(encoding='utf-8'))['apps']
    features=torch.zeros(4,27);owner=torch.zeros(4,11);goals=torch.zeros(4,42)
    for i,row in enumerate(rows):
        features[i,i]=1
        features[i,4+torch.tensor(row['cell_indices'])]=1
        owner[i,row['rcp_indices']]=1
        if row['target_kpi_indices'] is not None:
            features[i,7+torch.tensor(row['target_kpi_indices'],dtype=torch.long)]=1
            features[i,21]=1
            for c in row['cell_indices']:
                goals[i,[14*c+k for k in row['target_kpi_indices']]]=1
        layer={None:0,'rApp':1,'xApp':2}[row['layer']];features[i,22+layer]=1
        if row['period_s'] is not None:
            features[i,25]=torch.log1p(torch.tensor(float(row['period_s'])))/10
            features[i,26]=1
    return features,owner,goals

class FourTypeBlock(nn.Module):
    def __init__(self,d,owner,goals):
        super().__init__()
        scope=torch.zeros(3,11)
        for c in range(3):scope[c,[c,c+3,c+6,9,10]]=1
        cell_kpi=torch.eye(3).repeat_interleave(14,dim=0)
        same=torch.tensor([[0.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
        cross=torch.tensor([[0.,1.,1.],[1.,0.,0.],[1.,0.,0.]])
        specs={'pa':('p','a',owner),'ap':('a','p',owner.T),'cp':('c','p',scope.T),
               'pc':('p','c',scope),'same':('c','c',same),'cross':('c','c',cross),
               'kc':('k','c',cell_kpi.T),'ck':('c','k',cell_kpi),
               'pk':('p','k',cell_kpi@scope),'kp':('k','p',(cell_kpi@scope).T),
               'ka':('k','a',goals),'ak':('a','k',goals.T)}
        self.ends={r:(s,t) for r,(s,t,a) in specs.items()}
        for r,(_,_,a) in specs.items():self.register_buffer(r,a/a.sum(-1,keepdim=True).clamp_min(1))
        self.maps=nn.ModuleDict({r:nn.Linear(d,d,bias=False) for r in specs})
        self.self_maps=nn.ModuleDict({t:nn.Linear(d,d) for t in 'apck'})
        self.norm=nn.ModuleDict({t:nn.LayerNorm(d) for t in 'apck'})
        self.gates=nn.ModuleDict({r:nn.Linear(2*d,1) for r in ('pk','same','cross')})
        self.drop=nn.Dropout(.05)
    def message(self,r,z,target):
        a=getattr(self,r)
        if r in self.gates:
            w=self.gates[r].weight
            target_score=F.linear(target,w[:,:target.shape[-1]],self.gates[r].bias)
            source_score=F.linear(z,w[:,target.shape[-1]:]).transpose(1,2)
            weights=a[None]*torch.sigmoid(target_score+source_score)
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            return torch.bmm(weights,self.maps[r](z))
        if a.shape[0]<a.shape[1]:
            return self.maps[r](torch.matmul(a,z))
        return torch.matmul(a,self.maps[r](z))
    def forward(self,c,p,a,k):
        z=dict(c=c,p=p,a=a,k=k)
        # Newly updated App -> RCP -> Cell -> KPI provides a complete control-to-KPI path.
        for target in 'apck':
            total=self.self_maps[target](z[target])
            for r,(source,dest) in self.ends.items():
                if dest==target:total=total+self.message(r,z[source],z[target])
            z[target]=self.norm[target](z[target]+self.drop(F.silu(total)))
        return z['c'],z['p'],z['a'],z['k']

class PaperGNN(nn.Module):
    def __init__(self,history=180,hidden=48,graph=True,temporal=True):
        super().__init__();assert graph and temporal
        self.history=history;self.hidden=hidden
        self.config=dict(history=history,hidden=hidden,graph=graph,temporal=temporal)
        features,owner,goals=registry_tensors()
        self.register_buffer('app_features',features);self.register_buffer('owner',owner/owner.sum(1,keepdim=True))
        self.register_buffer('cell_meta',torch.eye(3))
        typ=F.one_hot(torch.tensor([0,0,0,1,1,1,2,2,2,3,4]),5).float()
        scope=F.one_hot(torch.tensor([0,1,2,0,1,2,0,1,2,3,3]),4).float()
        self.register_buffer('parameter_meta',torch.cat((typ,scope),-1))
        self.register_buffer('observed_kpi',torch.tensor([1,2,3,4,5,6,7,8,9,10,0,0,11,0]))
        self.register_buffer('observed_kpi_mask',torch.tensor([1,1,1,1,1,1,1,1,1,1,0,0,1,0.]))
        self.cell_encoder=dense(28,hidden,hidden)
        self.parameter_encoder=dense(10,hidden,hidden)
        self.app_encoder=dense(28,hidden,hidden)
        self.kpi_embedding=nn.Embedding(14,8)
        self.kpi_encoder=dense(14,hidden,hidden) # value, mask, cell3, KPI8, clock1
        self.blocks=nn.ModuleList([FourTypeBlock(hidden,owner,goals) for _ in range(2)])
        self.cell_gru=nn.GRU(hidden,hidden,batch_first=True)
        self.kpi_gru=nn.GRU(hidden,hidden,batch_first=True)
        self.advance_cell=dense(hidden+2,hidden,hidden)
        self.advance_kpi=dense(hidden+2,hidden,hidden)
        self.future_state_decoder=dense(hidden,hidden,12)
        self.horizon_embedding=nn.Embedding(2,8)
        self.query=dense(hidden+10,hidden,hidden) # KPI state, horizon8, clock, delay
        self.kpi_decoder=dense(hidden,hidden,1)
        self.reference_encoder=dense(10,hidden,hidden)
        self.contribution_decoder=dense(3*hidden,hidden,1)
    def p_nodes(self,p):
        return self.parameter_encoder(torch.cat((p[...,None],self.parameter_meta[None].expand(len(p),-1,-1)),-1))
    def a_nodes(self,current,chosen):
        strength=torch.matmul((chosen-current).abs(),self.owner.T)
        return self.app_encoder(torch.cat((self.app_features[None].expand(len(current),-1,-1),strength[...,None]),-1))
    def encode_history(self,current,history,valid,clock):
        b,l=history.shape[:2];d=self.hidden
        x=torch.where(valid.bool(),history,torch.zeros_like(history))
        c=self.cell_encoder(torch.cat((x,valid,self.cell_meta[None,None].expand(b,l,-1,-1),
                                      clock[:,None,None].expand(-1,l,3,-1)),-1)).reshape(b*l,3,d)
        kv=x[...,self.observed_kpi]*self.observed_kpi_mask
        km=valid[...,self.observed_kpi]*self.observed_kpi_mask
        k=self.kpi_encoder(torch.cat((kv[...,None],km[...,None],
            self.cell_meta[None,None,:,None].expand(b,l,3,14,3),
            self.kpi_embedding.weight[None,None,None].expand(b,l,3,14,8),
            clock[:,None,None,None].expand(-1,l,3,14,-1)),-1)).reshape(b*l,42,d)
        p=self.p_nodes(current)[:,None].expand(-1,l,-1,-1).reshape(b*l,11,d)
        a=self.a_nodes(current,current)[:,None].expand(-1,l,-1,-1).reshape(b*l,4,d)
        for block in self.blocks:c,p,a,k=block(c,p,a,k)
        def temporal(z,n,gru):
            seq=z.reshape(b,l,n,d).permute(0,2,1,3).reshape(b*n,l,d)
            return gru(seq)[1][0].reshape(b,n,d)
        return temporal(c,3,self.cell_gru),temporal(k,42,self.kpi_gru)
    def advance(self,state,clock,delay):
        c,k=state;delta=delay[:,None]/180
        def evolve(z,net):
            n=z.shape[1];q=torch.cat((z,clock[:,None].expand(-1,n,-1),delta[:,None].expand(-1,n,-1)),-1)
            return z+delta[:,None]*net(q)
        return evolve(c,self.advance_cell),evolve(k,self.advance_kpi)
    def decode(self,state,current,clock,chosen,delay):
        c,k=state;p=self.p_nodes(chosen);a=self.a_nodes(current,chosen);b=len(chosen)
        for block in self.blocks:c,p,a,k=block(c,p,a,k)
        k=k.reshape(b,3,14,self.hidden)
        query=self.query(torch.cat((k[:,None].expand(-1,2,-1,-1,-1),
            self.horizon_embedding.weight[None,:,None,None].expand(b,-1,3,14,-1),
            ((clock+delay[:,None]/900)[:,None,None,None]).expand(-1,2,3,14,-1),
            (delay[:,None,None,None,None]/180).expand(-1,2,3,14,1)),-1))
        return self.kpi_decoder(query).squeeze(-1),query,p
    def backbone(self,current,history,valid,clock,summary,candidate=None,delay=None):
        delay=torch.zeros(len(current),device=current.device) if delay is None else delay
        state=self.advance(self.encode_history(current,history,valid,clock),clock,delay)
        return self.decode(state,current,clock,current if candidate is None else candidate,delay)
    def forward(self,*inputs,candidate=None,delay=None):
        return self.backbone(*inputs,candidate=candidate,delay=delay)[0]
    def multi(self,current,history,valid,clock,summary,candidates,delays=None,return_state=False):
        b,n=candidates.shape[:2];base=self.encode_history(current,history,valid,clock)
        ds=[0] if delays is None else delays
        def expand(z):return z[:,None].expand(b,n,*z.shape[1:]).reshape(b*n,*z.shape[1:])
        preds=[];states=[]
        for dt in ds:
            delay=torch.full((b,),float(dt),device=current.device);state=self.advance(base,clock,delay)
            states.append(self.future_state_decoder(state[0]))
            pred,_,_=self.decode(tuple(expand(z) for z in state),expand(current),expand(clock),
                candidates.reshape(b*n,11),expand(delay[:,None]).flatten())
            preds.append(pred.reshape(b,n,2,3,14))
        p=torch.stack(preds,1);s=torch.stack(states,1)
        if delays is None:p=p[:,0]
        return (p,s) if return_state else p
    def kpi_parameters(self):
        return [p for n,p in self.named_parameters() if not n.startswith(('reference_encoder.','contribution_decoder.'))]
