import torch
from torch import nn
from .base_model import PaperGNN as Base

class PaperGNN(Base):
    def __init__(self,history=180,hidden=48,graph=True,temporal=True,horizons_s=(60,90,150,180)):
        super().__init__(history,hidden,graph,temporal)
        self.n_h=len(horizons_s)
        self.horizon_embedding=nn.Embedding(self.n_h,8)
        self.config['horizons_s']=list(horizons_s)
    def decode(self,state,current,clock,chosen,delay):
        c,k=state;p=self.p_nodes(chosen);a=self.a_nodes(current,chosen);b=len(chosen)
        for block in self.blocks:c,p,a,k=block(c,p,a,k)
        k=k.reshape(b,3,14,self.hidden)
        query=self.query(torch.cat((k[:,None].expand(-1,self.n_h,-1,-1,-1),
            self.horizon_embedding.weight[None,:,None,None].expand(b,-1,3,14,-1),
            ((clock+delay[:,None]/900)[:,None,None,None]).expand(-1,self.n_h,3,14,-1),
            (delay[:,None,None,None,None]/180).expand(-1,self.n_h,3,14,1)),-1))
        return self.kpi_decoder(query).squeeze(-1),query,p
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
            preds.append(pred.reshape(b,n,self.n_h,3,14))
        p=torch.stack(preds,1);s=torch.stack(states,1)
        if delays is None:p=p[:,0]
        return (p,s) if return_state else p
