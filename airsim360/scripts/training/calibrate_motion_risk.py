"""Fit the spatial risk profile with the future-center prediction held fixed."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import torch
from torch.nn import functional as F
from scripts.perception.local_future_field import LocalFutureField


@torch.no_grad()
def samples(block, motion, prior, seed):
    keys,differences,curves,truth=block[:4]
    pred=motion(keys,differences,curves)
    original=prior.displacement(keys[:,:3].numpy(),keys[:,6:9].numpy()/.09)
    count=len(keys)
    h=torch.arange(17).expand(count,-1)*.1
    features=torch.cat((keys[:,None,3:6].expand(-1,17,-1)/(.55*6),keys[:,None,6:9].expand(-1,17,-1)/(.09*15),
                        torch.as_tensor(original,dtype=torch.float32)/6,keys[:,None,9:12].expand(-1,17,-1)/(.13*15),h[:,:,None]),dim=-1)
    g=torch.Generator().manual_seed(seed)
    shape=(count,17,20,3)
    unit=F.normalize(torch.randn(shape,generator=g),dim=-1)
    # Equal exposure to queries around the true and predicted future locations.
    # Half the directions lie in the horizontal plane to sample side-by-side clearance.
    unit[:,:,::2,2]=0
    unit=F.normalize(unit,dim=-1)
    r=torch.rand(shape[:-1],generator=g)*1.5
    q=truth[:,:,None]+r[:,:,:,None]*unit
    q[:,:,10:]+=pred[:,:,None]-truth[:,:,None]
    actual=(q-truth[:,:,None]).norm(dim=-1)
    predicted=(q-pred[:,:,None]).norm(dim=-1)
    return features.flatten(0,1),predicted.flatten(0,1),actual.flatten(0,1)


def scores(model,b):
    feat,d,actual=b
    raw=model.head(feat)
    radius=.35+.4*raw[:,3:4].sigmoid()
    width=.04+.22*raw[:,4:5].sigmoid()
    prob=((radius-d)/width).sigmoid()
    target=((.55-actual)/.08).sigmoid()
    return prob,target,radius,width


@torch.no_grad()
def assessment(model,b):
    p,y,r,w=scores(model,b)
    inside=b[2]<.4
    safe=b[2]>.85
    return dict(brier=float((p-y).square().mean()),inside_above_half=float((p[inside]>=.5).float().mean()),
                safe_above_half=float((p[safe]>=.5).float().mean()),safe_mean=float(p[safe].mean()),
                radius_m=float(r.median()),width_m=float(w.median()))


def fit_profile(saved,motion,prior,data,val):
    model=LocalFutureField(0.)
    model.load_state_dict(saved['model'])
    train=samples(data,motion,prior,22031)
    devblocks=[samples(b,motion,prior,32031+i) for i,b in enumerate(val)]
    dev=tuple(torch.cat(b) for b in zip(*devblocks))
    before=assessment(model,dev)
    best=before['brier']
    best_epoch=0
    best_weights={k:v.clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.head.parameters(),lr=.001)
    for epoch in range(1,19):
        for ix in torch.randperm(len(train[0])).split(256):
            batch=tuple(a[ix] for a in train)
            p,y,r,w=scores(model,batch)
            # Keep the metric footprint anchored to the original 0.55 m safety label.
            loss=(p-y).square().mean()+.15*(r-.55).square().mean()+.08*(w-.08).square().mean()
            opt.zero_grad();loss.backward();opt.step()
        stats=assessment(model,dev)
        if stats['brier']<best and stats['inside_above_half']>=before['inside_above_half']-.02:
            best,best_epoch=stats['brier'],epoch
            best_weights={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_weights)
    return best_weights,dict(epoch=best_epoch,previous=before,calibrated=assessment(model,dev))


def calibrate_width(name,source):
    """Calibrate only the decay of risk outside the learned radius. Preserve its 0.5 contour."""
    import json
    from scripts.training.train_context_motion import ROOT, PRIOR, FlowPrior, extract, retrieval
    from scripts.perception.context_motion import ContextMotionPrior
    torch.set_num_threads(1);torch.manual_seed(1213)
    folder=ROOT/name
    folder.mkdir()
    saved=torch.load(ROOT/source/'best.pt',map_location='cpu',weights_only=True)
    motion=ContextMotionPrior(PRIOR,saved)
    model=LocalFutureField(0.);model.load_state_dict(saved['model'])
    old=LocalFutureField().eval()
    old.load_state_dict(torch.load(ROOT/'cf21_local_future01/best.pt',map_location='cpu',weights_only=True)['model'])
    prior=FlowPrior(PRIOR)
    banks,obs,_=extract('train',old,prior)
    data=tuple(torch.cat(c) for c in zip(*retrieval(obs,banks,True)))
    _,vobs,_=extract('validation',old,prior)
    train=samples(data,motion.motion,prior,22031)
    devsets=[samples(b,motion.motion,prior,32031+i) for i,b in enumerate(retrieval(vobs,banks))]
    dev=tuple(torch.cat(b) for b in zip(*devsets))
    before=assessment(model,dev)
    with torch.no_grad():
        hidden=model.head[:-1](train[0])
        radius=.35+.4*model.head[-1](hidden)[:,3:4].sigmoid()
        target=((.55-train[2])/.08).sigmoid()
    layer=model.head[-1]
    weight=torch.nn.Parameter(layer.weight[4:5].detach().clone())
    bias=torch.nn.Parameter(layer.bias[4:5].detach().clone())
    opt=torch.optim.Adam([weight,bias],lr=.01)
    best,epoch=before['brier'],0
    best_w,best_b=weight.detach().clone(),bias.detach().clone()
    for n in range(1,26):
        for ix in torch.randperm(len(hidden)).split(384):
            width=.04+.22*F.linear(hidden[ix],weight,bias).sigmoid()
            p=((radius[ix]-train[1][ix])/width).sigmoid()
            loss=F.mse_loss(p,target[ix])
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            layer.weight[4:5].copy_(weight);layer.bias[4:5].copy_(bias)
            score=assessment(model,dev)['brier']
            if score<best:
                best,epoch=score,n
                best_w,best_b=weight.detach().clone(),bias.detach().clone()
    with torch.no_grad():
        layer.weight[4:5].copy_(best_w);layer.bias[4:5].copy_(best_b)
    previous=LocalFutureField(0.);previous.load_state_dict(saved['model'])
    _,tobs,tnames=extract('test',old,prior)
    held={}
    for i,(flight,b) in enumerate(zip(tnames,retrieval(tobs,banks))):
        test=samples(b,motion.motion,prior,42031+i)
        held[flight]=dict(previous=assessment(previous,test),fitted=assessment(model,test))
    saved['model']=model.state_dict()
    torch.save(saved,folder/'best.pt')
    report=dict(source=source,epoch=epoch,validation=dict(previous=before,fitted=assessment(model,dev)),held_out=held)
    (folder/'calibration.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    import sys
    calibrate_width(*sys.argv[1:])
