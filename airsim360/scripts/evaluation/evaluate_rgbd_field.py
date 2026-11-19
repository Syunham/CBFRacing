"""Evaluate matched checkpoints on the held-out flight and probe the gate plane."""

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
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.training.train_rgbd_field import RGBDField, Windows, image_history, threat_labels


def interpolate(times, path, t):
    return np.stack([np.interp(t, times, path[:,k]) for k in range(3)], axis=-1)


def camera_queries(world, state):
    return Rotation.from_quat(state[[7,8,9,6]]).inv().apply(world-state[:3])*[1,-1,-1]


def draw_planes(path, truth, predicted, time):
    size = 315
    canvas = Image.new("RGB", (720,430), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    colors = np.array([[31,45,82],[43,111,144],[116,185,152],[245,222,117],[208,75,54]])
    def colorize(arr):
        x = np.clip(arr,0,1)
        return np.stack([np.interp(x,np.linspace(0,1,5),c) for c in colors.T],-1).astype("uint8")
    for left,values,title in [(25,truth,"参考风险"),(380,predicted,"模型预测 · 第 1 轮")]:
        draw.text((left,10),title,fill="black",font=font)
        canvas.paste(Image.fromarray(colorize(values)).resize((size,size)),(left,45))
    gradient = np.tile(np.linspace(0,1,265),(14,1))
    canvas.paste(Image.fromarray(colorize(gradient)), (76,390))
    draw.text((25,380),"低 0",fill="black",font=font)
    draw.text((350,380),"1 高",fill="black",font=font)
    draw.text((435,380),f"t={time:.2f} s   τ=0.40 s",fill="black",font=font)
    canvas.save(path)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    model=RGBDField().cuda().eval()
    metrics={}
    ds=Windows(args.manifest,"test",1024,.25)
    ds.seed=11010
    for name in ("best","last"):
        checkpoint=torch.load(args.run/(name+".pt"),map_location="cuda",weights_only=True)
        model.load_state_dict(checkpoint["model"])
        total=tp=fp=fn=n=0
        for imgs,q,t,y in DataLoader(ds,batch_size=8):
            logits=model.query_logits(model.encode(imgs.cuda()),q.cuda(),t.cuda()).cpu()
            total+=F.binary_cross_entropy_with_logits(logits,y,reduction="sum").item()
            pos=y>=.5
            pred=logits>=0
            tp+=(pos&pred).sum().item()
            fp+=(~pos&pred).sum().item()
            fn+=(pos&~pred).sum().item()
            n+=y.numel()
        metrics[name]=dict(epoch=checkpoint["epoch"]+1, queries=n, BCE=total/n,
                           recall=tp/(tp+fn), precision=tp/(tp+fp), FPR=fp/(n-tp-fn),
                           TP=tp,FP=fp,FN=fn)
    model.load_state_dict(torch.load(args.run/"best.pt",weights_only=True)["model"])
    times,states,rgb,depth,valid,gates,eligible=ds.episodes[0]
    boxes=gates.centers.reshape(-1,4,3)
    centers=boxes.mean(axis=1)
    scores={"clear_opening":[],"bars":[],"neighbour_away_from_frame":[]}
    separations=[]
    view_frame=int(np.argmin(np.abs(times-.74)))
    for frame in eligible:
        for ego in (0,1):
            state=states[frame,ego]
            gate=int(np.argmin(np.linalg.norm(centers-state[:3],axis=1)))
            history=slice(frame-3,frame+1)
            h=image_history(rgb[history,ego],depth[history,ego],valid[history,ego])
            latent=model.encode(h[None].cuda())
            def score(points,t):
                q=camera_queries(points,state)
                tensor=torch.as_tensor(q,dtype=torch.float32,device="cuda")[None]
                return model.query_logits(latent,tensor,torch.full_like(tensor[:,:,:1],t)).sigmoid()[0].cpu().numpy()
            for tau in (0.,.4,.8,1.6):
                future=interpolate(times,states[:,1-ego,:3],times[frame]+tau)
                points=np.vstack((centers[gate],boxes[gate],future))
                pred=score(points,tau)
                if np.linalg.norm(points[0]-state[:3])<=6 and np.linalg.norm(points[0]-future)>=1.1:
                    scores["clear_opening"].append(pred[0])
                    scores["bars"].extend(pred[1:5])
                    separations.append(np.mean(pred[1:5]>pred[0]))
                if gates.distances(future[None])[0]>.75:
                    scores["neighbour_away_from_frame"].append(pred[-1])
            if frame==view_frame and ego==0:
                side=boxes[0,1]-boxes[0,0]
                side/=np.linalg.norm(side)
                y,x=np.mgrid[1.5:-1.501:-.025,-1.5:1.501:.025]
                plane=centers[0]+x[...,None]*side+y[...,None]*[0,0,1]
                points=plane.reshape(-1,3)
                future=interpolate(times,states[:,1-ego,:3],times[frame]+.4)
                target=threat_labels(points,future,gates,.25)
                draw_planes(args.run/"gate_plane.png",target.reshape(x.shape),
                            score(points,.4).reshape(x.shape),times[frame])
    metrics["probes"]={k:dict(n=len(v),median=float(np.median(v)),high_fraction=float(np.mean(np.array(v)>=.5))) for k,v in scores.items()}
    metrics["paired_bar_above_opening"]=float(np.mean(separations))
    metrics["test_flight_frames"]=len(times)
    metrics["random_seed"]=ds.seed
    (args.run/"test_metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    main()
