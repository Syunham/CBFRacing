"""Fit directional risk contours around a frozen RGB-D motion forecast."""

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


class DirectionalRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.adjust=nn.Sequential(nn.Linear(13,32),nn.SiLU(),nn.Linear(32,3))
        nn.init.zeros_(self.adjust[-1].weight)
        nn.init.zeros_(self.adjust[-1].bias)

    def forward(self,features,radius,width):
        change=self.adjust(features)
        cross=(radius+.22*change[...,0].tanh()).clamp(.40,.8)
        scale=(1.15*change[...,1].tanh()).exp()
        width=(width*(1.3*change[...,2].tanh()).exp()).clamp(.045,.33)
        return cross,scale,width


def probes(block,motion,prior,old,seed):
    from scripts.training.calibrate_motion_risk import samples
    features,_,_=samples(block,motion,prior,seed)
    features=features.reshape(-1,17,13)
    with torch.no_grad():
        pred=motion(*block[:3])
        axis=F.normalize(pred,dim=-1)
        truth=block[3]
        probabilities=old.head(features)
        radius=.35+.4*probabilities[...,3].sigmoid()
        width=.04+.22*probabilities[...,4].sigmoid()
        g=torch.Generator().manual_seed(seed)
        unit=F.normalize(torch.randn((*pred.shape[:2],24,3),generator=g),dim=-1)
        # Include queries explicitly to the sides as well as along the forecast motion.
        unit[:,:,:6]=F.normalize(unit[:,:,:6]-(unit[:,:,:6]*axis[:,:,None]).sum(-1,keepdim=True)*axis[:,:,None],dim=-1)
        unit[:,:,6:8]=axis[:,:,None]
        unit[:,:,8:10]=-axis[:,:,None]
        offset=unit*torch.rand((*pred.shape[:2],24,1),generator=g)*1.6
        q=truth[:,:,None]+offset
        q[:,:,12:]+=pred[:,:,None]-truth[:,:,None]
        dq=q-pred[:,:,None]
        parallel=(dq*axis[:,:,None]).sum(-1).square()
        perp=(dq.square().sum(-1)-parallel).clamp_min(0.)
        real=(q-truth[:,:,None]).norm(dim=-1)
        return tuple(a.flatten(0,1) for a in (features,radius,width,parallel,perp,real))


def forecast(net,b):
    f,r0,w0,para,perp,real=b
    radius,scale,width=net(f,r0,w0)
    d=(para/scale[:,None].square()+perp+1.e-10).sqrt()
    prob=((radius[:,None]-d)/width[:,None]).sigmoid()
    return prob,((.55-real)/.08).sigmoid(),scale


@torch.no_grad()
def stats(net,b):
    p,y,scale=forecast(net,b)
    real=b[-1]
    inner=real<.4
    safe=real>.85
    return dict(brier=float((p-y).square().mean()),inside_recall=float((p[inner]>=.5).float().mean()),
                safe_score=float(p[safe].mean()),safe_false=float((p[safe]>=.5).float().mean()),scale=float(scale.median()))


def train(name,source):
    from scripts.training.train_context_motion import ROOT, PRIOR, FlowPrior, LocalFutureField, extract, retrieval
    from scripts.perception.context_motion import ContextMotionPrior
    torch.set_num_threads(1);torch.manual_seed(5123)
    out=ROOT/name;out.mkdir()
    saved=torch.load(ROOT/source/'best.pt',map_location='cpu',weights_only=True)
    old=LocalFutureField().eval();old.load_state_dict(saved['model'])
    motion=ContextMotionPrior(PRIOR,saved).motion
    prior=FlowPrior(PRIOR)
    banks,obs,_=extract('train',old,prior)
    trainblock=tuple(torch.cat(a) for a in zip(*retrieval(obs,banks,True)))
    trainset=probes(trainblock,motion,prior,old,50123)
    _,vobs,_=extract('validation',old,prior)
    val=tuple(torch.cat(a) for a in zip(*[probes(b,motion,prior,old,60123+i) for i,b in enumerate(retrieval(vobs,banks))]))
    net=DirectionalRisk()
    original=stats(net,val)
    best=original['brier']
    best_epoch=0
    weights={k:v.clone() for k,v in net.state_dict().items()}
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    for epoch in range(1,26):
        for ix in torch.randperm(len(trainset[0])).split(256):
            batch=tuple(a[ix] for a in trainset)
            p,y,scale=forecast(net,batch)
            # Give close physical encounters additional weight to preserve recall.
            loss=((p-y).square()*(1+2*(batch[-1]<.4))).mean()+.002*scale.log().square().mean()
            opt.zero_grad();loss.backward();opt.step()
        met=stats(net,val)
        if (met['brier']<best and met['inside_recall']>=original['inside_recall']-.02
            and met['safe_score']<original['safe_score'] and met['safe_false']<=original['safe_false']+.005):
            best,best_epoch=met['brier'],epoch
            weights={k:v.clone() for k,v in net.state_dict().items()}
    net.load_state_dict(weights)
    _,tobs,tnames=extract('test',old,prior)
    held={}
    base=DirectionalRisk()
    for i,(flight,b) in enumerate(zip(tnames,retrieval(tobs,banks))):
        data=probes(b,motion,prior,old,70123+i)
        held[flight]=dict(previous=stats(base,data),directional=stats(net,data))
    report=dict(source=source,epoch=best_epoch,validation=dict(previous=original,directional=stats(net,val)),test=held)
    saved['directional']=weights
    torch.save(saved,out/'best.pt')
    (out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    train(*sys.argv[1:])
