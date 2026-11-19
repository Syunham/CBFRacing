"""Fit compact-surface appearance and local occupancy transport on training flights."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scripts.perception.occupancy_flow import compact_surface
from scripts.training.train_rgbd_field import Windows


def motion_examples(times, xyz):
    if len(times) < 85 or not np.allclose(np.diff(times),.02,rtol=0,atol=1e-5):
        raise ValueError("Expected at least 85 consecutive 50 Hz states")
    frames = np.arange(4,len(times)-80,2)
    velocity = (xyz[frames]-xyz[frames-4])/(times[frames]-times[frames-4])[:,None]
    keys = np.column_stack((xyz[frames],.10*velocity))
    displacement = xyz[frames[:,None]+np.arange(0,81,5)]-xyz[frames,None]
    return keys, displacement.astype(np.float32)


def extend_motion(base, flights):
    with np.load(base) as saved:
        result = dict(saved)
    keys, delta = [result["keys"]], [result["delta"]]
    for folder in flights:
        with np.load(folder/"flight/execution.npz") as data:
            times, states = data["time"], data["state"]
        for ego in (0,1):
            block, shifts = motion_examples(times,states[:,ego,:3])
            keys.append(block)
            delta.append(shifts)
    result.update(keys=np.concatenate(keys),delta=np.concatenate(delta))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out",type=Path)
    parser.add_argument("--base",type=Path,help="Retain the fitted surface network and extend its transport samples")
    parser.add_argument("--rollouts",type=Path,nargs="+",help="Closed-loop development directories with recorded states")
    args = parser.parse_args()
    if bool(args.base) != bool(args.rollouts):
        parser.error("--base and --rollouts must be provided together")
    if args.base:
        bundle = extend_motion(args.base,args.rollouts)
        args.out.mkdir(exist_ok=False)
        np.savez_compressed(args.out/"flow.npz",**bundle)
        (args.out/"fit.json").write_text(json.dumps(dict(base=str(args.base),rollouts=list(map(str,args.rollouts)),
                                                     transport_samples=len(bundle["keys"])),indent=2))
        return
    args.out.mkdir(exist_ok=False)
    manifest = Path("captures/cf21_512_double_50hz/manifest.json")
    train = Windows(manifest,"train",1,.25,lazy_windows=True)
    rng = np.random.default_rng(6106)
    torch.manual_seed(6106)
    torch.set_num_threads(1)
    features, labels, keys, displacement = [], [], [], []
    for times,states,rgb,depth,valid,_,_ in train.episodes:
        for ego in (0,1):
            peer = states[:,1-ego,:3]
            block, shifts = motion_examples(times,peer)
            keys.extend(block)
            displacement.extend(shifts)
            for k in range(0,len(times),5):
                points, f = compact_surface(rgb[k,ego],depth[k,ego],states[k,ego],True)
                distance = np.linalg.norm(points[:,:3]-peer[k],axis=1)
                for label,mask in ((1,distance<.085),(0,distance>.18)):
                    ids = np.flatnonzero(mask)
                    ids = rng.choice(ids,min(len(ids),32),replace=False)
                    features.append(f[ids])
                    labels.append(np.full(len(ids),label,dtype=np.float32))
        print("processed",len(keys),flush=True)
    x = torch.from_numpy(np.concatenate(features).astype(np.float32))
    y = torch.from_numpy(np.concatenate(labels))
    net = torch.nn.Sequential(torch.nn.Linear(12,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    opt = torch.optim.Adam(net.parameters(),lr=.003)
    for _ in range(18):
        for idx in torch.randperm(len(x)).split(512):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x[idx])[:,0],y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    np.savez_compressed(args.out/"flow.npz",keys=keys,delta=np.asarray(displacement,dtype=np.float32),
                        w0=net[0].weight.detach().numpy().T,b0=net[0].bias.detach().numpy(),
                        w1=net[2].weight.detach().numpy().T,b1=net[2].bias.detach().numpy())
    (args.out/"fit.json").write_text(json.dumps(dict(manifest=str(manifest),split="train",
        flights=len(train.episodes),points=len(y),positive=int(y.sum()),transport_samples=len(keys)),indent=2))

if __name__=="__main__":
    main()
