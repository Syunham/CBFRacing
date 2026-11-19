"""Learn future RGB-D occupancy as a probability-weighted mixture of three motion hypotheses."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import sys
import torch
from torch import nn
from torch.nn import functional as F


def group_fragments(motion, features):
    weight,curve=motion.fragments(*features)
    typical=(weight[:,:,None]*curve[:,:,9]).sum(1)
    progress=(curve[:,:,9]*F.normalize(typical,dim=-1)[:,None]).sum(-1)
    order=progress.argsort(dim=1)
    batches=torch.arange(len(curve))[:,None]
    centers=[];mass=[]
    for ids in (order[:,:5],order[:,5:11],order[:,11:]):
        w=weight[batches,ids]
        mass.append(w.sum(1))
        centers.append((w[:,:,None,None]*curve[batches,ids]).sum(1)/mass[-1][:,None,None])
    return torch.stack(centers,dim=2),torch.stack(mass,1)


class MixtureRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.decode=nn.Sequential(nn.Linear(20,48),nn.SiLU(),nn.Linear(48,5))
        nn.init.zeros_(self.decode[-1].weight)
        nn.init.zeros_(self.decode[-1].bias)

    def forward(self,key,centers,mass,lead):
        average=(mass[:,:,None]*centers).sum(1,keepdim=True)
        spread=(centers-average).norm(dim=-1)
        raw=self.decode(torch.cat((key[:,:13]/5.,mass,spread/2,lead[:,None]),dim=-1))
        weights=(mass.clamp_min(1.e-8).log()+raw[:,:3]).softmax(1)
        return weights,.42+.32*raw[:,3].sigmoid(),.045+.20*raw[:,4].sigmoid()


@torch.no_grad()
def data_for(block,motion,prior,old,seed):
    from scripts.training.calibrate_motion_risk import samples
    f,_,_=samples(block,motion,prior,seed)
    base=old.head(f)
    r=.35+.4*base[:,3:4].sigmoid();w=.04+.22*base[:,4:5].sigmoid()
    curves,mass=group_fragments(motion,block[:3])
    mean=motion(*block[:3])
    truth=block[3]
    n,h=truth.shape[:2]
    rng=torch.Generator().manual_seed(seed)
    unit=F.normalize(torch.randn(n,h,24,3,generator=rng),dim=-1)
    unit[:,:,::2,2]=0.
    unit=F.normalize(unit,dim=-1)
    q=truth[:,:,None]+torch.rand(n,h,24,1,generator=rng)*1.5*unit
    q[:,:,12:]+=mean[:,:,None]-truth[:,:,None]
    real=(q-truth[:,:,None]).norm(dim=-1).flatten(0,1)
    base_risk=((r-(q-mean[:,:,None]).norm(dim=-1).flatten(0,1))/w).sigmoid()
    dist=(q[:,:,:,None]-curves[:,:,None]).norm(dim=-1)
    keys=block[0][:,None].expand(-1,h,-1).flatten(0,1)
    return (keys,curves.flatten(0,1),mass[:,None].expand(-1,h,-1).flatten(0,1),
            torch.arange(h).repeat(n)*.1,dist.flatten(0,1),real,base_risk)


def scores(net,b):
    m,r,w=net(*b[:4])
    return (m[:,None]*((r[:,None,None]-b[4])/w[:,None,None]).sigmoid()).sum(-1)


def metrics(p,real):
    near,safe=real<.4,real>.85
    threshold=torch.quantile(p[safe],.95)
    return dict(brier=float((p-((.55-real)/.08).sigmoid()).square().mean()),
                near=float((p[near]>=.5).float().mean()),safe=float(p[safe].mean()),
                recall_at_fpr05=float((p[near]>threshold).float().mean()))


def train(name,source):
    from scripts.training.train_context_motion import ROOT, PRIOR, FlowPrior, LocalFutureField, extract, retrieval
    from scripts.perception.context_motion import ContextMotionPrior
    torch.set_num_threads(1);torch.manual_seed(1214)
    folder=ROOT/name;folder.mkdir()
    saved=torch.load(ROOT/source/'best.pt',weights_only=True,map_location='cpu')
    motion=ContextMotionPrior(PRIOR,saved).motion
    old=LocalFutureField();old.load_state_dict(saved['model'])
    prior=FlowPrior(PRIOR)
    banks,obs,_=extract('train',old,prior)
    block=tuple(torch.cat(a) for a in zip(*retrieval(obs,banks,True)))
    trainset=data_for(block,motion,prior,old,4111)
    _,vobs,_=extract('validation',old,prior)
    val=tuple(torch.cat(a) for a in zip(*[data_for(b,motion,prior,old,5111+i) for i,b in enumerate(retrieval(vobs,banks))]))
    previous=metrics(val[-1],val[-2]);best=float('inf');winner=None;history=[]
    for importance in (1.,3.,6.):
        net=MixtureRisk()
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        for epoch in range(1,16):
            for ix in torch.randperm(len(trainset[0])).split(384):
                b=tuple(a[ix] for a in trainset)
                p=scores(net,b)
                y=((.55-b[-2])/.08).sigmoid()
                loss=((p-y).square()*(1+importance*(b[-2]<.4))).mean()
                opt.zero_grad();loss.backward();opt.step()
            with torch.inference_mode():
                stat=metrics(scores(net,val),val[-2])
            history.append(dict(importance=importance,epoch=epoch,**stat))
            if stat['brier']<best and stat['near']>=previous['near']-.02 and stat['safe']<=previous['safe'] and stat['recall_at_fpr05']>=previous['recall_at_fpr05']-.01:
                best=stat['brier']
                winner={k:v.clone() for k,v in net.state_dict().items()}
                selected=history[-1]
    report=dict(source=source,baseline=previous,history=history,selected=None)
    if winner is not None and best<previous['brier']:
        net.load_state_dict(winner)
        report['selected']=selected
        saved['mixture']=winner
        torch.save(saved,folder/'best.pt')
        _,test,names=extract('test',old,prior)
        held={}
        for i,(n,b) in enumerate(zip(names,retrieval(test,banks))):
            d=data_for(b,motion,prior,old,6111+i)
            with torch.inference_mode():
                held[n]=dict(previous=metrics(d[-1],d[-2]),mixture=metrics(scores(net,d),d[-2]))
        report['test']=held
    (folder/'results.json').write_text(json.dumps(report,indent=2))
    print('base',previous,'selected',report['selected'],flush=True)


if __name__=='__main__':
    train(*sys.argv[1:])
