"""Matched gate and advance-warning tests for nested flight-count training runs."""

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
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
from scripts.training.train_rgbd_field import RGBDField, ProjectedRGBDField, Windows, image_history, threat_labels


def trajectory_at(times, positions, t):
    return np.stack([np.interp(t, times, v) for v in positions.T], axis=-1)


def make_probes(episode, events):
    times, states, _, _, _, gates, eligible = episode
    bars = gates.centers.reshape(-1, 4, 3)
    centers = bars.mean(axis=1)
    openings = []
    warnings = []
    for ego in (0, 1):
        neighbour = states[:,1-ego,:3]
        for frame in eligible:
            p = states[frame,ego,:3]
            g = np.argmin(np.linalg.norm(centers-p,axis=1))
            if np.linalg.norm(centers[g]-p)>6:
                continue
            for lead in (0.,.4,.8,1.6):
                if np.linalg.norm(trajectory_at(times,neighbour,times[frame]+lead)-centers[g])>=1.1:
                    openings.append((frame,ego,lead,np.vstack((centers[g], bars[g]))))
        for event in events[1-ego]:
            g = event["gate"]-1
            center=centers[g]
            side=bars[g,1]-bars[g,0]
            side/=np.linalg.norm(side)
            # All probes inside the opening, beyond the 0.25 m frame margin.
            grid=np.array([center+x*side+[0,0,z] for x in (-.48,0,.48) for z in (-.48,0,.48)])
            dist_now=np.linalg.norm(neighbour[:,None]-grid,axis=2).min(axis=1)
            for horizon in (.2,.4,.8,1.6):
                k=int(np.argmin(abs(times-(event["time"]-horizon))))
                lead=event["time"]-times[k]
                if k<3 or not 0<lead<=1.6+1e-9 or abs(lead-horizon)>.011 or dist_now[k]<=.55:
                    # At the maximum horizon use the next observation if rounding exceeded 1.6 s.
                    if horizon!=1.6 or lead<=1.6 or k+1>=len(times):
                        continue
                    k+=1
                    lead=event["time"]-times[k]
                    if k<3 or not 1.58-1e-8<=lead<=1.6 or dist_now[k]<=.55:
                        continue
                hit=trajectory_at(times,neighbour,event["time"])
                if np.linalg.norm(grid-hit,axis=1).min()>.55:
                    continue
                future=trajectory_at(times,neighbour,times+lead)
                empty=np.linalg.norm(future[:,None]-grid,axis=2).min(axis=1)>1.1
                empty &= (dist_now>.55)&(times+lead<=times[-1])&(np.arange(len(times))>=3)
                options=np.flatnonzero(empty)
                if not len(options):
                    continue
                # Match on observer position and orientation; never match using model outputs.
                pose=states[k,ego]
                distance=np.linalg.norm(states[options,ego,:3]-pose[:3],axis=1)
                dot=np.abs(states[options,ego,6:10]@pose[6:10]).clip(0,1)
                j=options[np.argmin(distance+2*np.arccos(dot))]
                warnings.append((horizon,event["gate"],event["time"],(k,ego,lead,grid),(j,ego,lead,grid)))
    return openings,warnings


@torch.inference_mode()
def evaluate(model,ds,openings,warnings,event_log=None,check_direction=False,direction_leads=(.04,.4)):
    times,states,rgb,depth,valid,gates,_=ds.episodes[0]
    device=next(model.parameters()).device
    batch_size=4 if device.type=="cpu" else 16
    caches=[]
    for ego in (0,1):
        histories=[]
        for k in range(3,len(times),batch_size):
            batch=[image_history(rgb[j-3:j+1,ego],depth[j-3:j+1,ego],valid[j-3:j+1,ego]) for j in range(k,min(k+batch_size,len(times)))]
            encoded=model.encode(torch.stack(batch).to(device))
            histories.append(encoded if isinstance(encoded,tuple) else (encoded,))
        caches.append(tuple(torch.cat(component) for component in zip(*histories)))
    cache=tuple(torch.stack(component) for component in zip(*caches))

    def scores(probes):
        values=[]
        for k in range(0,len(probes),128):
            items=probes[k:k+128]
            frame,ego,tau,q=map(np.asarray,zip(*items))
            poses=states[frame,ego]
            rot=Rotation.from_quat(poses[:,[7,8,9,6]]).as_matrix()
            camera=np.einsum("bqi,bij->bqj",q-poses[:,None,:3],rot)*[1,-1,-1]
            t=torch.tensor(tau,device=device,dtype=torch.float32)[:,None,None].expand(-1,q.shape[1],1)
            z=tuple(component[ego,frame-3] for component in cache)
            if len(z)==1:
                z=z[0]
            values.extend(model.query_logits(z,torch.tensor(camera,dtype=torch.float32,device=device),t).sigmoid().cpu().numpy())
        return np.array(values)

    s=scores(openings)
    gate=dict(pairs=int(s.shape[0]*4),bar_above_center=float((s[:,1:]>s[:,0:1]).mean()),
              bar_recall=float((s[:,1:]>=.5).mean()),opening_false_alarm=float((s[:,0]>=.5).mean()))
    warn={}
    for h in sorted({w[0] for w in warnings}):
        rows=[w for w in warnings if w[0]==h]
        pos=scores([r[3] for r in rows]).max(axis=1)
        neg=scores([r[4] for r in rows]).max(axis=1)
        now=scores([(k,e,0.,grid) for _,_,_,(k,e,_,grid),_ in rows]).max(axis=1)
        if event_log is not None and h==.4:
            for i,(_,gate_id,event_time,(k,ego,lead,_),(j,_,_,_)) in enumerate(rows):
                event_log.append(dict(gate=gate_id,observer=int(ego),event_s=float(event_time),
                                      observation_s=float(times[k]),empty_observation_s=float(times[j]),
                                      horizon_s=float(lead),risk_now=float(now[i]),
                                      risk_at_entry=float(pos[i]),risk_empty=float(neg[i])))
        warn[str(h)]=dict(events=len(rows),hits=int((pos>=.5).sum()),empty_alarms=int((neg>=.5).sum()),
                          occupied_above_empty=float((pos>neg).mean()),
                          now_low_future_high=int(((now<.5)&(pos>=.5)).sum()),
                          median_future_minus_now=float(np.median(pos-now)))
    total=0.
    for images,q,t,y in DataLoader(ds,batch_size=batch_size):
        logits=model.query_logits(model.encode(images.to(device)),q.to(device),t.to(device)).cpu()
        total+=binary_cross_entropy_with_logits(logits,y,reduction="sum").item()
    result=dict(test_bce=total/(32*len(ds)),gate=gate,warning=warn)
    if check_direction:
        direction={}
        for lead in direction_leads:
            probes=[]
            for ego in (0,1):
                peer=states[:,1-ego,:3]
                future=trajectory_at(times,peer,times+lead)
                mirror=2*peer-future
                # Omit ambiguous negatives and points close to a gate frame.
                keep=((times+lead<=times[-1]) & (np.arange(len(times))>=3) &
                      (np.linalg.norm(future-mirror,axis=1)>.55) &
                      (gates.distances(future)>.5) & (gates.distances(mirror)>.5))
                probes.extend((k,ego,lead,np.stack([future[k],mirror[k]])) for k in np.flatnonzero(keep))
            if probes:
                values=scores(probes)
                now=scores([(k,e,0.,q) for k,e,_,q in probes])
                direction[str(lead)]=dict(pairs=len(probes),
                    actual_above_mirror=float(np.mean(values[:,0]>values[:,1])),
                    median_forward_minus_mirror=float(np.median(values[:,0]-values[:,1])),
                    median_time_effect_at_actual=float(np.median(values[:,0]-now[:,0])))
        result["direction"]=direction
    return result


@torch.inference_mode()
def draw_example(output, episode, warning, models, titles=None):
    times,states,rgb,depth,valid,gates,_=episode
    device=next(models[0].parameters()).device
    horizon,gate_number,_,occupied,empty=warning
    bars=gates.centers.reshape(-1,4,3)[gate_number-1]
    side=bars[1]-bars[0]
    side/=np.linalg.norm(side)
    z,x=np.mgrid[1.3:-1.301:-.02,-1.3:1.301:.02]
    points=(bars.mean(axis=0)+x[...,None]*side+z[...,None]*[0,0,1]).reshape(-1,3)
    colors=np.array([[31,45,82],[43,111,144],[116,185,152],[245,222,117],[208,75,54]])
    def colorize(values):
        return np.stack([np.interp(values,np.linspace(0,1,5),c) for c in colors.T],-1).astype("uint8")
    canvas=Image.new("RGB",(120+225*(1+len(models)),565),"white")
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",19)
    for col,title in enumerate(["参考风险",*(titles or ["1 条训练航次","4 条训练航次","8 条训练航次"])]):
        draw.text((125+col*225,10),title,font=font,fill="black")
    for row,(k,ego,lead,_) in enumerate((occupied,empty)):
        future=trajectory_at(times,states[:,1-ego,:3],times[k]+lead)
        values=[threat_labels(points,future,gates,.25)]
        h=slice(k-3,k+1)
        images=image_history(rgb[h,ego],depth[h,ego],valid[h,ego])[None].to(device)
        pose=states[k,ego]
        q=Rotation.from_quat(pose[[7,8,9,6]]).inv().apply(points-pose[:3])*[1,-1,-1]
        tensor=torch.tensor(q,dtype=torch.float32,device=device)[None]
        tau=torch.full_like(tensor[:,:,:1],lead)
        for net in models:
            values.append(net(images,tensor,tau)[0].cpu().numpy())
        draw.text((12,132+row*228),["邻机将进门","空门对照"][row],font=font,fill="black")
        for col,v in enumerate(values):
            tile=Image.fromarray(colorize(v.reshape(x.shape))).resize((215,215))
            canvas.paste(tile,(120+col*225,42+row*228))
    legend=colorize(np.tile(np.linspace(0,1,200),(12,1)))
    canvas.paste(Image.fromarray(legend),(178,517))
    draw.text((120,506),"低 0",font=font,fill="black")
    draw.text((390,506),"1 高",font=font,fill="black")
    draw.text((485,506),f"{gate_number} 号门 · 提前 {horizon:g} 秒 · 同一色标",font=font,fill="black")
    canvas.save(output)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment",type=Path,required=True)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline",type=Path)
    mode.add_argument("--projected-run",type=Path)
    mode.add_argument("--retrained-run",type=Path,help="Compare a newly fitted RGBDField to its original on one test flight")
    ap.add_argument("--original-run",type=Path)
    ap.add_argument("--manifest",type=Path)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=ap.parse_args()
    torch.set_num_threads(1)
    if args.device=="cpu":
        torch.set_num_interop_threads(1)
    manifest=args.manifest or args.experiment/"manifest_8.json"
    ds=Windows(manifest,"test",1024,.25,half_size=bool(args.retrained_run))
    ds.seed=11010
    entry=next(e for e in json.loads(manifest.read_text())["episodes"] if e["split"]=="test")
    events=json.loads((manifest.parent / entry["trajectory"]).with_name("events.json").read_text())
    openings,warnings=make_probes(ds.episodes[0],events)
    report={}
    models=[]
    event_logs={}
    if args.retrained_run:
        if args.original_run is None:
            ap.error("--retrained-run needs --original-run")
        runs={"original_large":args.original_run,"retrained_crazyflie":args.retrained_run}
        out=args.retrained_run
        titles=["原尺寸训练","Crazyflie 尺寸训练"]
    elif args.projected_run:
        runs={"global_8":args.experiment/"run_8","projected_8":args.projected_run}
        out=args.projected_run
        titles=["原 8 条模型","投影查询模型"]
    else:
        runs={1:args.baseline,4:args.experiment/"run_4",8:args.experiment/"run_8"}
        out=args.experiment
        titles=None
    for n,run in runs.items():
        ck=torch.load(run/"best.pt",weights_only=True,map_location="cpu")
        model=(ProjectedRGBDField() if ck.get("projected") else RGBDField()).to(args.device).eval()
        model.load_state_dict(ck["model"])
        models.append(model)
        event_log=[] if args.projected_run or args.retrained_run else None
        report[n]=dict(epoch=ck["epoch"]+1,validation_bce=ck["best"],
                       **evaluate(model,ds,openings,warnings,event_log,check_direction=bool(args.retrained_run)))
        if event_log is not None:
            event_logs[n]=event_log
        print(n,json.dumps(report[n]),flush=True)
    (out/"results.json").write_text(json.dumps(report,indent=2))
    if event_logs:
        (out/"events_0p4.json").write_text(json.dumps(event_logs,indent=2))
    draw_example(out/"gate_comparison.png",ds.episodes[0],next(w for w in warnings if w[0]==.4),models,titles)


if __name__=="__main__":
    main()
