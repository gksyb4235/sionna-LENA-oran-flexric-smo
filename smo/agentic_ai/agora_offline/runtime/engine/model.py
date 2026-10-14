"""Temporal heterogeneous graph with declarative app capabilities and edge RCPs.

App identities describe the fixed schema, never the requesting user's intent.
Numeric goals, thresholds, SHAP and future observations are not model inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .schema import APPS, CORE, INPUTS, TARGETS


def mlp(n,h,out):
    return nn.Sequential(nn.Linear(n,h),nn.SiLU(),nn.Linear(h,out))


class RelationBlock(nn.Module):
    def __init__(self,hidden,k,apps,target_indices):
        super().__init__()
        ptypes = ['txp','ret','cio','ttt','hys']
        if apps == 5:
            ownership = torch.tensor([[float(p in a['controls']) for p in ptypes] for a in APPS])
            goals = torch.tensor([[float(TARGETS[j] in a['targets']) for j in target_indices] for a in APPS])
        else:
            ownership = torch.tensor([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]],dtype=torch.float32)
            goals = torch.zeros(4,k)
        cell_kpi = torch.eye(3).repeat_interleave(k,0)
        specs = {
            'pa':('p','a',ownership), 'ap':('a','p',ownership.T),
            'ka':('k','a',goals.repeat(1,3)), 'ak':('a','k',goals.repeat(1,3).T),
            'pc':('p','c',torch.ones(3,5)), 'cp':('c','p',torch.ones(5,3)),
            'pk':('p','k',torch.ones(3*k,5)), 'kp':('k','p',torch.ones(5,3*k)),
            'ck':('c','k',cell_kpi), 'kc':('k','c',cell_kpi.T),
            'same':('c','c',torch.tensor([[0,0,0],[0,0,1],[0,1,0]],dtype=torch.float32)),
            'cross':('c','c',torch.tensor([[0,1,1],[1,0,0],[1,0,0]],dtype=torch.float32)),
        }
        self.k=k; self.ends={r:(s,t) for r,(s,t,_) in specs.items()}
        for r,(_,_,a) in specs.items():
            self.register_buffer(r,a/a.sum(-1,keepdim=True).clamp_min(1))
        self.maps=nn.ModuleDict({r:nn.Linear(hidden,hidden,bias=False) for r in specs})
        self.own=nn.ModuleDict({t:nn.Linear(hidden,hidden) for t in 'apck'})
        self.norm=nn.ModuleDict({t:nn.LayerNorm(hidden) for t in 'apck'})
        self.gates=nn.ModuleDict({r:nn.Linear(hidden*2,1) for r in ['pk','same','cross']})

    def forward(self,c,p,a,k,edges):
        z=dict(c=c,p=p,a=a,k=k)
        for dest in 'apck':
            total=self.own[dest](z[dest])
            for r,(source,target) in self.ends.items():
                if target != dest: continue
                adj=getattr(self,r)
                if r in self.gates:
                    gate=self.gates[r]; d=z[source].shape[-1]
                    weights=adj[None]*torch.sigmoid(F.linear(z[dest],gate.weight[:,:d],gate.bias)+F.linear(z[source],gate.weight[:,d:]).transpose(1,2))
                    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
                    message=torch.bmm(weights,self.maps[r](z[source]))
                else:
                    message=adj@self.maps[r](z[source])
                if r == 'pc': message=message+self.maps[r](edges).mean(2)
                if r == 'pk':
                    local=self.maps[r](edges)[:, :,None].expand(-1,-1,self.k,-1,-1).reshape(-1,3*self.k,5,d)
                    message=message+(local*weights[...,None]).sum(2)
                total=total+message
            z[dest]=self.norm[dest](z[dest]+F.silu(total))
        return z['c'],z['p'],z['a'],z['k']


class TemporalGNN(nn.Module):
    def __init__(self,hidden=48,apps=5,target_indices=None):
        super().__init__()
        target_indices=list(CORE if target_indices is None else target_indices)
        self.config=dict(hidden=hidden,apps=apps,target_indices=target_indices)
        self.k=len(target_indices); self.hidden=hidden; self.apps=apps
        self.register_buffer('cell_id',torch.eye(3))
        self.register_buffer('parameter_id',torch.eye(5))
        self.register_buffer('app_id',torch.eye(apps))
        self.register_buffer('edge_indices',torch.tensor([[0,3,6,9,10],[1,4,7,9,10],[2,5,8,9,10]]))
        input_for_target={name:i for i,name in enumerate(INPUTS)}
        self.register_buffer('observed_index',torch.tensor([input_for_target.get(TARGETS[i],0) for i in target_indices]))
        self.register_buffer('observed_mask',torch.tensor([float(TARGETS[i] in input_for_target) for i in target_indices]))
        self.cell_encoder=mlp(2*len(INPUTS)+4,hidden,hidden)
        self.cell_gru=nn.GRU(hidden,hidden,batch_first=True)
        self.kpi_id=nn.Embedding(self.k,8)
        self.kpi_encoder=mlp(2+8+3,24,16)
        self.kpi_gru=nn.GRU(16,16,batch_first=True)
        self.kpi_project=nn.Linear(16,hidden)
        self.parameter_encoder=mlp(5,hidden,hidden)
        self.edge_encoder=mlp(3+5+3,hidden,hidden)
        self.app_encoder=mlp(apps,hidden,hidden)
        self.advance_c=mlp(hidden+2,hidden,hidden)
        self.advance_k=mlp(hidden+2,hidden,hidden)
        self.blocks=nn.ModuleList([RelationBlock(hidden,self.k,apps,target_indices) for _ in range(2)])
        self.decoder=mlp(2*hidden+8+3,hidden,1)
        self.summary_dim=3*len(INPUTS)*4
        self.skip=mlp(self.summary_dim+33+3,96,3*self.k)
        self.rlf_risk=mlp(hidden+3,32,1)
        self.network_rlf_risk=mlp(3*hidden+3,48,1)
        self.future_state=nn.Linear(hidden,len(INPUTS))

    def encode(self,x,m,current,clock):
        b,l=x.shape[:2]; x=torch.where(m.bool(),x,0)
        c=self.cell_encoder(torch.cat([x,m,self.cell_id[None,None].expand(b,l,-1,-1),clock[:,None,None].expand(b,l,3,-1)],-1))
        c=self.cell_gru(c.permute(0,2,1,3).reshape(b*3,l,self.hidden))[1][0].reshape(b,3,self.hidden)
        kv=x[...,self.observed_index]*self.observed_mask
        km=m[...,self.observed_index]*self.observed_mask
        k=self.kpi_encoder(torch.cat([kv[...,None],km[...,None],self.kpi_id.weight[None,None,None].expand(b,l,3,-1,-1),self.cell_id[None,None,:,None].expand(b,l,3,self.k,-1)],-1))
        k=self.kpi_gru(k.permute(0,2,3,1,4).reshape(b*3*self.k,l,16))[1][0]
        k=self.kpi_project(k).reshape(b,3*self.k,self.hidden)
        count=m.sum(1).clamp_min(1)
        mean=(x*m).sum(1)/count
        std=(((x-mean[:,None])**2*m).sum(1)/count).clamp_min(0).sqrt()
        summary=torch.cat([v.flatten(1) for v in [mean,std,x[:,-1],m.mean(1)]],-1)
        return c,k,summary

    def decode(self,state,current,chosen,clock,delay,horizon):
        c,k,summary=state; b=len(current)
        query=torch.cat([clock,delay[:,None]/180,horizon[:,None]/180],-1)
        dc=torch.cat([clock,delay[:,None]/180],-1)
        c=c+(delay[:,None,None]/180)*self.advance_c(torch.cat([c,dc[:,None].expand(-1,3,-1)],-1))
        k=k+(delay[:,None,None]/180)*self.advance_k(torch.cat([k,dc[:,None].expand(-1,3*self.k,-1)],-1))
        predicted_state=self.future_state(c)
        values=torch.stack([current[:,self.edge_indices],chosen[:,self.edge_indices],(chosen-current)[:,self.edge_indices]],-1)
        edges=self.edge_encoder(torch.cat([values,self.parameter_id[None,None].expand(b,3,-1,-1),self.cell_id[None,:,None].expand(b,-1,5,-1)],-1))
        p=self.parameter_encoder(self.parameter_id)[None].expand(b,-1,-1)
        a=self.app_encoder(self.app_id)[None].expand(b,-1,-1)
        for block in self.blocks: c,p,a,k=block(c,p,a,k,edges)
        k=k.reshape(b,3,self.k,self.hidden)
        features=torch.cat([k,c[:,:,None].expand(-1,-1,self.k,-1),self.kpi_id.weight[None,None].expand(b,3,-1,-1),query[:,None,None].expand(-1,3,self.k,-1)],-1)
        out=self.decoder(features).squeeze(-1)
        out=out+self.skip(torch.cat([summary,current,chosen,chosen-current,query],-1)).reshape(b,3,self.k)
        risk=self.rlf_risk(torch.cat([c,query[:,None].expand(-1,3,-1)],-1)).squeeze(-1)
        network_risk=self.network_rlf_risk(torch.cat([c.flatten(1),query],-1))
        risk=torch.cat([risk,network_risk],-1)
        return out,risk,predicted_state

    def forward(self,x,m,current,clock,chosen,delay,horizon):
        return self.decode(self.encode(x,m,current,clock),current,chosen,clock,delay,horizon)
