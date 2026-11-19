"""Train and compare metric displacement predictors, with whole-flight splits and leave-one-out retrieval."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F

from scripts.perception.context_motion import FragmentMotion, MotionAttention, motion_keys, extend_keys, history_keys
from scripts.perception.local_future_field import LocalFutureField, observe, supports
from scripts.perception.occupancy_flow import FlowPrior
from scripts.training.train_local_future import MANIFEST, PRIOR, ROOT
from scripts.training.train_rgbd_field import Windows

TRAIN_RUNS = ('cf21_flow_recover_01','cf21_flow_w5600_01','cf21_full350_01','cf21_passagehead01_run1','cf21_gain175_01')
VALID_RUNS = ('cf21_rollhead01_run1',)
TEST_RUNS = ('cf21_bendhead03_run2','cf21_local_future_two_laps01')
FEEDBACK_RUNS = ('cf21_motion_attention02_race1','cf21_flow_buffer_01','cf21_flow_transient_01','cf21_flow_w0_control01')
STEPS = np.arange(17)*5


def runs(split, recorded=None, history=False):
    if recorded is None:
        entries = [e for e in json.loads(MANIFEST.read_text())['episodes'] if e['split']==split]
        for entry,ep in zip(entries,Windows(MANIFEST,split,1,.25,lazy_windows=True).episodes):
            t,x,rgb,depth=ep[:4]
            yield entry['episode_id'], t,x,lambda ids:(rgb[ids],depth[ids])
        recorded={'train':TRAIN_RUNS,'validation':VALID_RUNS,'test':TEST_RUNS}[split]
        if history and split=='train':
            recorded=(*recorded,*FEEDBACK_RUNS)
    for name in recorded:
        path=ROOT/name
        with np.load(path/'flight/execution.npz') as f:
            t,x=f['time'],f['state']
        def read(ids):
            return tuple(np.stack([np.load(path/d/f'{k*20:06d}.npy') for k in ids]) for d in ('rgb','depth'))
        yield name,t,x,read


def extract(split, old, prior, recorded=None, history=False):
    banks, observations, names=[],[],[]
    for name,t,x,read in runs(split,recorded,history):
        names.append(name)
        keys,curves,rows=[],[],[]
        for ego in (0,1):
            for k in range(0,len(t)-80,2):
                peer=x[k,1-ego,:3]
                vel=(peer-x[max(0,k-3),1-ego,:3])/max(.02,t[k]-t[max(0,k-3)])
                key=motion_keys(peer[None],vel[None],x[k,ego],t[k])
                if history:
                    ids=np.maximum(0,np.arange(k-3,k+1))
                    key=extend_keys(key,vel[None],x[ids,ego],t[ids])
                keys.append(key[0])
                curves.append(x[k+STEPS,1-ego,:3]-peer)
        banks.append((np.array(keys),np.array(curves,'f')))
        frames=np.unique(np.r_[np.arange(0,min(48,len(t)-80),4),np.arange(0,len(t)-80,12)])
        for k in frames:
            ids=np.maximum(0,np.arange(k-3,k+1)); rgb,depth=read(ids)
            for ego in (0,1):
                pose=x[k,ego]; peer=x[k,1-ego,:3]
                flow=observe(prior,t[ids],x[ids,ego],rgb[:,ego],depth[:,ego])
                _,_,_,selected=supports(flow,pose,.4)
                if not len(selected):continue
                errors=np.linalg.norm(flow.xyz[selected]-peer,axis=1)
                if errors.min()>=.12:continue
                i=selected[errors.argmin()]; point=flow.xyz[i]
                key=motion_keys(point[None],flow.flow[i:i+1],pose,t[k])[0]
                if history:
                    key=history_keys(flow,x[ids,ego],t[ids])[i]
                truth=x[k+STEPS,1-ego,:3]-point
                # Same visible return for both predictors, use the deployed old center.
                leads=STEPS*.02
                d=flow.displacements[i]
                feat=np.column_stack((np.tile((point-pose[:3])/6,(17,1)),np.tile(flow.flow[i]/15,(17,1)),d/6,
                                      np.tile(pose[3:6]/15,(17,1)),leads)).astype('f')
                with torch.inference_mode():
                    previous=old.kernels(torch.tensor(feat),torch.tensor(d,dtype=torch.float32))[0].numpy()
                    at88=supports(flow,pose,.88)
                    center88=old.kernels(torch.from_numpy(at88[0]),torch.from_numpy(at88[1]))[0][errors.argmin()].numpy()
                rows.append((key,truth,previous,t[k],x[k+44,1-ego,:3]-point,center88-point))
        observations.append(tuple(np.asarray(a,'f') for a in zip(*rows)))
        print(split,name,'seen',len(rows),'of',len(frames)*2,flush=True)
    return banks,observations,names


def retrieval(obs,banks,leave_out=False):
    parts=[]
    for j,block in enumerate(obs):
        key=block[0]
        sources=[b for i,b in enumerate(banks) if not leave_out or i!=j]
        keys,shifts=(np.concatenate(c) for c in zip(*sources))
        _,idx=cKDTree(keys).query(key,k=16)
        parts.append((key,keys[idx]-key[:,None],shifts[idx],*block[1:]))
    return [tuple(torch.tensor(a) for a in block) for block in parts]


def metrics(pred,b,new88=None):
    _,_,_,truth,old,stamp,truth88,old88=b
    if new88 is None:
        new88=.2*pred[:,8]+.8*pred[:,9]
    results={}
    for label,ix in [('all',torch.ones(len(stamp),dtype=torch.bool)),('launch',stamp<=.8)]:
        if not ix.any():continue
        groups=[]
        for new,base,y in [(pred[:,4],old[:,4],truth[:,4]),(new88,old88,truth88)]:
            groups.append([float((val[ix]-y[ix]).norm(dim=-1).mean()) for val in (base,new)])
        results[label]={'n':int(ix.sum()),'0.4':groups[0],'0.88':groups[1]}
    return results


def train(out, fragment=False, history_keys_enabled=False):
    torch.set_num_threads(1);torch.manual_seed(1112)
    out.mkdir()
    saved=torch.load(ROOT/'cf21_local_future01/best.pt',map_location='cpu',weights_only=True)
    old=LocalFutureField().eval();old.load_state_dict(saved['model'])
    base=FlowPrior(PRIOR)
    banks,obs,train_names=extract('train',old,base,history=history_keys_enabled)
    data=tuple(torch.cat(a) for a in zip(*retrieval(obs,banks,True)))
    if fragment:
        # Use all recorded training positions to supervise how nearby fragments differ,
        # alongside the actual RGB-D returns. Exclude each query flight from its lookup.
        examples=[(k[::2],v[::2]) for k,v in banks]
        augment=tuple(torch.cat(a) for a in zip(*retrieval(examples,banks,True)))
    _,vobs,vnames=extract('validation',old,base,history=history_keys_enabled)
    val=retrieval(vobs,banks)
    net=FragmentMotion(data[0].shape[1]) if fragment else MotionAttention()
    opt=torch.optim.AdamW(net.parameters(),lr=.0015)
    best=float('inf');history=[]
    bk,bd=(np.concatenate(a) for a in zip(*banks))
    for epoch in range(41):
        if epoch:
            for idx in torch.randperm(len(data[0])).split(64):
                batch=[a[idx] for a in data[:4]]
                if fragment:
                    extra=torch.randint(len(augment[0]),(len(idx),))
                    batch=[torch.cat((v,a[extra])) for v,a in zip(batch,augment)]
                pred=net(*batch[:3])
                err=F.smooth_l1_loss(pred[:,1:],batch[3][:,1:],reduction='none').sum(-1)
                loss=err[:,:12].mean()+.25*err.mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
        with torch.inference_mode():
            stats={n:metrics(net(*b[:3]),b) for n,b in zip(vnames,val)}
        score=np.mean([s[g][h][1] for s in stats.values() for g in ('all','launch') for h in ('0.4','0.88')])
        history.append(float(score))
        if score<best and (epoch or fragment):
            best=score
            torch.save(dict(model=saved['model'],correction_speed=0.,prior=str(PRIOR),keys=torch.from_numpy(bk),shifts=torch.from_numpy(bd),
                            epoch=epoch,motion=net.state_dict(),
                            architecture='fragment_transfer' if fragment else 'attention',use_history=history_keys_enabled),out/'best.pt')
        if epoch%10==0:print('epoch',epoch,'validation',score,flush=True)
    chosen=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(chosen['motion'])
    profile=None
    if fragment and not history_keys_enabled:
        from scripts.training.calibrate_motion_risk import fit_profile
        calibrated,profile=fit_profile(saved,net,base,data,val)
        if profile['epoch']:
            torch.save(chosen,out/'before_calibration.pt')
            chosen['model']=calibrated
            torch.save(chosen,out/'best.pt')
        print('risk profile',profile,flush=True)
    _,tobs,tnames=extract('test',old,base,history=history_keys_enabled)
    with torch.inference_mode():
        results={n:metrics(net(*b[:3]),b) for n,b in zip(vnames+tnames,val+retrieval(tobs,banks))}
    report=dict(train=train_names,validation=vnames,test=tnames,epoch=chosen['epoch'],validation_history=history,risk_profile=profile,
                units='mean 3D endpoint error in meters, [previous, candidate], matched visible returns only',flights=results)
    (out/'results.json').write_text(json.dumps(report,indent=2));print('RESULTS',json.dumps(results),flush=True)


if __name__=='__main__':
    train(ROOT/sys.argv[1], '--fragment' in sys.argv[2:] or '--history' in sys.argv[2:], '--history' in sys.argv[2:])
