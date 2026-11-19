"""Fit a localized dynamic field on eight recordings with out-of-flight transport features."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.special import expit
from torch.nn import functional as F

from scripts.perception.local_future_field import LocalFutureField, VisualMotionPrior, observe, supports
from scripts.perception.occupancy_flow import FlowPrior
from scripts.training.train_occupancy_flow import motion_examples
from scripts.training.train_rgbd_field import Windows

ROOT = Path("reports/spatial_potential")
MANIFEST = Path("captures/cf21_512_double_50hz/manifest.json")
PRIOR = ROOT/"cf21_flow_field_01/flow.npz"
HORIZONS = (0, 10, 20, 40, 80)


def examples(episodes, priors, stride, rng):
    records = []
    seen = 0
    for ep, prior in zip(episodes, priors):
        times, state, rgb, depth, _, gates, _ = ep
        for k in range(3,len(times)-80,stride):
            h = slice(k-3,k+1)
            for ego in (0,1):
                pose = state[k,ego]
                flow = observe(prior,times[h],state[h,ego],rgb[h,ego],depth[h,ego])
                seen += 1
                if not len(flow.xyz):
                    continue
                peer = state[:,1-ego,:3]
                gate_centers = gates.centers.reshape(-1,4,3).mean(1)
                gate = gate_centers[np.argmin(np.linalg.norm(gate_centers-pose[:3],axis=1))]
                for step in HORIZONS:
                    f, pred, valid, ids = supports(flow,pose,step*.02)
                    matched = np.zeros(32,bool)
                    matched[:len(ids)] = np.linalg.norm(flow.xyz[ids]-peer[k],axis=1) < .12
                    future = peer[k+step]
                    direction = rng.normal(size=(16,3))
                    direction /= np.linalg.norm(direction,axis=1,keepdims=True)
                    q = np.vstack((future, future + direction[:4], future + direction[4:]*rng.uniform(.05,1.3,(12,1)),
                                   gate+rng.uniform(-.4,.4,(5,3)), peer[k], pose[:3]+rng.uniform(-5,5,(3,3))))
                    y = expit((.55-np.linalg.norm(q-future,axis=1))/.08)
                    records.append((f,pred,valid,q.astype('f'),y.astype('f'),future.astype('f'),matched))
        print('extracted',seen,'observations',len(records),'horizon examples',flush=True)
    return tuple(torch.from_numpy(np.stack(a)) for a in zip(*records)), seen


def loss_for(net, data, idx):
    feat, p, valid, q, target, future, matched = (arr[idx] for arr in data)
    logits, centers = net(feat,p,valid,q)
    occupancy = F.binary_cross_entropy_with_logits(logits.clamp(min=-18),target)
    margin = F.softplus(2 - logits[:,:1] + logits[:,1:5]).mean()
    error = F.smooth_l1_loss(centers, future[:,None].expand_as(centers), reduction='none').sum(-1)
    transport = (error*matched).sum()/matched.sum().clamp_min(1)
    return occupancy + .12*margin + .5*transport


def leave_one_out_priors(train_eps, base):
    blocks = []
    for times,state,*_ in train_eps:
        both = [motion_examples(times,state[:,1-ego,:3]) for ego in (0,1)]
        blocks.append(tuple(np.concatenate(part) for part in zip(*both)))
    if not np.allclose(np.concatenate([b[0] for b in blocks]),base.tree.data):
        raise ValueError("Transport samples do not match eight training trajectories")
    priors = []
    for i in range(len(blocks)):
        pr = copy.copy(base)
        keys, shifts = zip(*(b for j,b in enumerate(blocks) if j!=i))
        pr.tree,pr.delta = cKDTree(np.concatenate(keys)),np.concatenate(shifts)
        priors.append(pr)
    return priors


def train(out, visual_only):
    torch.manual_seed(11012)
    torch.set_num_threads(1)
    rng = np.random.default_rng(11012)
    out.mkdir()
    base = VisualMotionPrior(PRIOR) if visual_only else FlowPrior(PRIOR)
    train_eps = Windows(MANIFEST,"train",1,.25,lazy_windows=True).episodes
    priors = [base]*len(train_eps) if visual_only else leave_one_out_priors(train_eps,base)
    data, n = examples(train_eps,priors,10,rng)
    dev_eps = Windows(MANIFEST,"validation",1,.25,lazy_windows=True).episodes
    dev, ndev = examples(dev_eps,[base]*len(dev_eps),10,rng)
    net = LocalFutureField(correction_speed=15. if visual_only else 3.)
    optimizer = torch.optim.AdamW(net.parameters(),lr=.001)
    best, history = np.inf,[]
    for epoch in range(1,31):
        net.train()
        for idx in torch.randperm(len(data[0])).split(96):
            loss = loss_for(net,data,idx)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5)
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            batches = torch.arange(len(dev[0])).split(128)
            val = sum(float(loss_for(net,dev,idx))*len(idx) for idx in batches)/len(dev[0])
        history.append(val)
        if val < best:
            best = val
            torch.save(dict(model=net.state_dict(),epoch=epoch,prior=str(PRIOR),visual_only=visual_only,
                            correction_speed=net.correction_speed),out/'best.pt')
        if epoch%5 == 0:
            print('epoch',epoch,'validation',val,flush=True)
    entries=json.loads(MANIFEST.read_text())['episodes']
    (out/'fit.json').write_text(json.dumps(dict(train=[e['episode_id'] for e in entries if e['split']=='train'],
        observations=n,training_examples=len(data[0]),observations_with_candidates=len(data[0])//len(HORIZONS),
        validation_observations=ndev,validation_loss=history,best_epoch=int(np.argmin(history)+1),
        horizons_s=[h*.02 for h in HORIZONS],transport_training='visual surface flow' if visual_only else 'leave one flight out',seed=11012),indent=2))


if __name__ == '__main__':
    destination = ROOT/sys.argv[1]
    train(destination, '--visual-only' in sys.argv[2:])
    if '--evaluate' in sys.argv[2:]:
        from scripts.evaluation.evaluate_local_future import main as evaluate
        evaluate(destination)
