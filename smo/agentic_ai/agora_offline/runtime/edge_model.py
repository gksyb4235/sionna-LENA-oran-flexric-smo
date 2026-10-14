"""Five actual RCP type nodes; the 11 controls are encoded on scoped edges.

The same encoder and relation maps are shared by nodes and edge attributes.
Trainable weights and their seed-42 initial values match the v14 baseline.
There is no hidden bank of 11 evolving Parameter nodes.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .horizon_model import PaperGNN as Baseline
from .base_model import FourTypeBlock


class EdgeTypeBlock(FourTypeBlock):
    def __init__(self,d,owner,goals):
        super().__init__(d,owner,goals)
        scope=torch.ones(3,5)
        cell_kpi=torch.eye(3).repeat_interleave(14,dim=0)
        for name,a in dict(pc=scope,cp=scope.T,pk=cell_kpi@scope,kp=(cell_kpi@scope).T).items():
            setattr(self,name,a/a.sum(-1,keepdim=True))

    def attributed_message(self,r,z,target,edge):
        # edge: batch x cell3 x type5 x hidden. Values are edge features,
        # never updated as graph nodes. Preserve the original relation maps.
        a=getattr(self,r)
        mapped=self.maps[r](edge)
        if r=='pk':
            w=self.gates[r].weight;d=z.shape[-1]
            target_score=F.linear(target,w[:,:d],self.gates[r].bias)
            source_score=F.linear(z,w[:,d:]).transpose(1,2)
            edge_score=F.linear(edge,w[:,d:]).squeeze(-1).repeat_interleave(14,dim=1)
            weights=a[None]*torch.sigmoid(target_score+source_score+edge_score)
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            local=torch.matmul(weights.reshape(-1,3,14,5),mapped).reshape(-1,42,d)
            return torch.bmm(weights,self.maps[r](z))+local
        base=super().message(r,z,target)
        if r=='pc':
            extra=(mapped*a[None,:,:,None]).sum(2)
        elif r=='cp':
            extra=(mapped*a.T[None,:,:,None]).sum(1)
        elif r=='kp':
            # Each cell's attribute is used by its own 14 KPI edges.
            cell_weights=a.reshape(5,3,14).sum(-1).T
            extra=(mapped*cell_weights[None,:,:,None]).sum(1)
        else:
            raise ValueError(r)
        return base+extra

    def forward(self,c,p,a,k,edge):
        z=dict(c=c,p=p,a=a,k=k)
        for target in 'apck':
            total=self.self_maps[target](z[target])
            for r,(source,dest) in self.ends.items():
                if dest==target:
                    total=total+(self.attributed_message(r,z[source],z[target],edge)
                                 if r in ('pc','cp','pk','kp') else self.message(r,z[source],z[target]))
            z[target]=self.norm[target](z[target]+self.drop(F.silu(total)))
        return z['c'],z['p'],z['a'],z['k']


class PaperGNN(Baseline):
    def __init__(self,history=180,hidden=48,graph=True,temporal=True,
                 horizons_s=(60,90,150,180),parameter_representation='five_type_edge_attributes'):
        assert parameter_representation=='five_type_edge_attributes'
        super().__init__(history,hidden,graph,temporal,horizons_s)
        self.config['parameter_representation']=parameter_representation
        # Metadata on the 11 independent scalar controls is only for edges.
        self.register_buffer('edge_parameter_meta',self.parameter_meta.clone())
        del self.parameter_meta
        scope=torch.zeros(5,4);scope[3:,3]=1
        self.register_buffer('type_node_meta',torch.cat((torch.eye(5),scope),-1))
        self.register_buffer('edge_indices',torch.tensor([[0,3,6,9,10],[1,4,7,9,10],[2,5,8,9,10]]))
        owner=torch.tensor([[1.,0,0,0,0],[0,1.,0,0,0],[0,0,1.,0,0],[0,0,0,1.,1.]])
        goals=self.blocks[0].ka.clone()
        # Preserve both initial values and the RNG state used by training.
        rng=torch.random.get_rng_state();blocks=[]
        for old in self.blocks:
            new=EdgeTypeBlock(hidden,owner,goals)
            weights=dict(old.named_parameters())
            with torch.no_grad():
                for name,value in new.named_parameters():value.copy_(weights[name])
            blocks.append(new)
        self.blocks=nn.ModuleList(blocks);torch.random.set_rng_state(rng)

    def p_nodes(self,p):
        # Type identity and global/local scope only; numeric controls live on edges.
        features=torch.cat((torch.zeros(5,1,device=p.device,dtype=p.dtype),self.type_node_meta),-1)
        return self.parameter_encoder(features)[None].expand(len(p),-1,-1)

    def edge_values(self,p):
        return p[:,self.edge_indices]

    def encode_edges(self,p):
        features=torch.cat((p[...,None],self.edge_parameter_meta[None].expand(len(p),-1,-1)),-1)
        return self.parameter_encoder(features)[:,self.edge_indices]

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
        p=self.p_nodes(current)[:,None].expand(-1,l,-1,-1).reshape(b*l,5,d)
        a=self.a_nodes(current,current)[:,None].expand(-1,l,-1,-1).reshape(b*l,4,d)
        edge=self.encode_edges(current)[:,None].expand(-1,l,-1,-1,-1).reshape(b*l,3,5,d)
        for block in self.blocks:c,p,a,k=block(c,p,a,k,edge)
        def temporal(z,n,gru):
            seq=z.reshape(b,l,n,d).permute(0,2,1,3).reshape(b*n,l,d)
            return gru(seq)[1][0].reshape(b,n,d)
        return temporal(c,3,self.cell_gru),temporal(k,42,self.kpi_gru)

    def decode(self,state,current,clock,chosen,delay):
        c,k=state;p=self.p_nodes(chosen);a=self.a_nodes(current,chosen);b=len(chosen)
        edge=self.encode_edges(chosen)
        for block in self.blocks:c,p,a,k=block(c,p,a,k,edge)
        k=k.reshape(b,3,14,self.hidden)
        query=self.query(torch.cat((k[:,None].expand(-1,self.n_h,-1,-1,-1),
            self.horizon_embedding.weight[None,:,None,None].expand(b,-1,3,14,-1),
            ((clock+delay[:,None]/900)[:,None,None,None]).expand(-1,self.n_h,3,14,-1),
            (delay[:,None,None,None,None]/180).expand(-1,self.n_h,3,14,1)),-1))
        return self.kpi_decoder(query).squeeze(-1),query,p
