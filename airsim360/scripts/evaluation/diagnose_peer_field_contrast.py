"""Audit local contrast of the frozen future risk field on the two recorded flights."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json,sys
from pathlib import Path
from collections import deque
import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt
from scripts.perception.spatial_potential import camera_points, rotation
from scripts.perception.panorama import spherical_rays as rays
from scripts.training.train_rgbd_field import GateFrames,threat_labels
from scripts.simulation.gate_geometry import gate_boxes
from scripts.perception.time_query_field import load_rgbd,encode_views

root=Path.cwd()/"reports/spatial_potential"
theta=np.arange(24)*2*np.pi/24
ring=np.c_[np.cos(theta),np.sin(theta),np.zeros(24)]
sx,fy=np.meshgrid(np.linspace(-6,6,49),np.linspace(-2,6,33))
rr=rays(256,512)
def occlude(rgb,dep,state):
    masked_rgb,masked_dep=rgb.copy(),dep.copy()
    ns=[]
    for i in (0,1):
        q=camera_points(state[1-i,:3],state[i])
        valid=np.isfinite(dep[i]) & (abs(dep[i]-np.linalg.norm(q))<.12)
        mask=np.zeros_like(valid)
        mask[valid]=np.linalg.norm(rr[valid]*dep[i][valid,None]-q,axis=1)<.12
        ns.append(int(mask.sum()))
        if not mask.any():continue
        pad=8
        m=binary_dilation(np.pad(mask,((0,0),(pad,pad)),mode="wrap"),iterations=2)
        y,x=np.where(m[:,pad:-pad])
        fill=distance_transform_edt(m,return_distances=False,return_indices=True)[:,y,x+pad]
        yy,xx=fill[0],(fill[1]-pad)%mask.shape[1]
        masked_rgb[i,y,x]=rgb[i,yy,xx];masked_dep[i,y,x]=dep[i,yy,xx]
    return masked_rgb,masked_dep,ns
@torch.inference_mode()
def run():
    torch.set_num_threads(1)
    rows=[]
    for n in (1,2):
        src=root/f"cf21_bendhead03_run{n}";cfg=json.loads((src/"experiment.json").read_text())
        model=load_rgbd(cfg["time_checkpoint"]);gates=GateFrames(gate_boxes(json.loads((src/"rendered_course.json").read_text())))
        with np.load(src/"flight/execution.npz") as f: x=f["state"]
        histories=[deque(maxlen=4) for _ in range(5)]
        for k in range(len(x)-20):
            rgb,dep=[np.load(src/d/f"{k*20:06d}.npy") for d in ("rgb","depth")]
            for h,v in zip(histories,(rgb,dep,*occlude(rgb,dep,x[k]))):
                h.append(v)
                while len(h)<4:h.appendleft(v)
            if k%4:continue
            a,b,c,d,pix=map(np.stack,histories)
            full=encode_views(model,a,b);masked=encode_views(model,c,d)
            for i in (0,1):
                own=x[k,i];peer=x[k+20,1-i,:3];proj=peer.copy();proj[2]=own[2]
                front=rotation(own).apply([1.,0,0])[:2];front/=np.linalg.norm(front)
                right=np.array([front[1],-front[0]])
                grid=np.broadcast_to(own[:3],(*sx.shape,3)).copy()
                grid[:,:,:2]+=sx[:,:,None]*right+fy[:,:,None]*front
                pts=np.vstack([peer,proj,peer+ring,grid.reshape(-1,3)])
                q=torch.from_numpy(camera_points(pts,own).astype("f"))[None]
                tau=torch.full((1,len(pts),1),.4)
                val=model.query_logits(full[i:i+1],q,tau).sigmoid().numpy().ravel()
                masked_val=model.query_logits(masked[i:i+1],q[:,:1],tau[:,:1]).sigmoid().item()
                target=threat_labels(pts,peer,gates,.25)
                empty=target[2:26]<.05
                safe=target[26:]<.05;v=val[26:]
                rows.append(dict(run=n,ego=i,t=k*.02,center=float(val[0]),projected=float(val[1]),dz=float(abs(proj[2]-peer[2])),baseline=float(target[0]),ring=float(np.mean(val[2:26][empty])) if empty.any() else None,ring_max=float(np.max(val[2:26][empty])) if empty.any() else None,
                    masked=float(masked_val),min_mask_pixels=int(pix[:,i].min()),clear_above_half=float(np.mean(v[safe]>=.5)),clear_above_08=float(np.mean(v[safe]>=.8)),clear_mean=float(v[safe].mean())))
    z={k:np.array([r[k] if r[k] is not None else np.nan for r in rows]) for k in rows[0]}
    contrast=z["center"]-z["ring"]
    usable=z["min_mask_pixels"]>0
    def quant(v):return np.nanpercentile(v,[5,50,95]).tolist()
    stats=dict(n=len(rows),center=quant(z["center"]),ring=quant(z["ring"]),contrast=quant(contrast),above_all=float(np.nanmean(z["center"]>z["ring_max"])),above_mean=float(np.nanmean(contrast>0)),valid_ring=int(np.isfinite(contrast).sum()),dz=quant(z["dz"]),abs_proj_diff=quant(abs(z["center"]-z["projected"])),clear_above_half=quant(z["clear_above_half"]),clear_above_08=quant(z["clear_above_08"]),mask_valid=int(usable.sum()),masked_center_delta=quant(abs(z["center"]-z["masked"])[usable]),example=[r for r in rows if abs(r["t"]-5.76)<.001])
    out=root/sys.argv[1];out.mkdir();(out/"diagnosis.json").write_text(json.dumps(dict(summary=stats,rows=rows),indent=2))
    print(json.dumps(stats))
if __name__=="__main__":run()
