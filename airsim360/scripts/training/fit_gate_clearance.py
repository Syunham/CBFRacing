"""Calibrate gate output weights from measured centre and frame clearance in development rollouts."""

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
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize
from scripts.simulation.gate_geometry import gate_boxes
from scripts.planning.utils.passage_calibration import bend_features

RESULTS=Path(__file__).resolve().parents[2]/"reports/spatial_potential"


def read(path):
    return json.loads(path.read_text())


def observe(name):
    root=RESULTS/name
    folder=root/"flight"
    config=read(root/"experiment.json")
    course=read(folder/"course.json")
    with np.load(config["passage_head"]) as checkpoint:
        gains=checkpoint["weight"]
        geometry=checkpoint.get("geometry_weight",np.zeros((2,2)))
    params=np.r_[gains,geometry.ravel()]
    centers,axes=(np.array(course[k]) for k in ("centers","coordinate_axes"))
    bends=bend_features(centers,axes)
    front=read(folder/"frontends.json")
    events=read(folder/"events.json")
    with np.load(folder/"execution.npz") as flight:
        t,x=flight["time"],flight["state"]
    all_times=np.arange(t[0],t[-1],.02)
    pos=CubicHermiteSpline(t,x[:,:,:3],x[:,:,3:6])(all_times)
    jac=[]
    for ego,rows in enumerate(events):
        if len(rows)!=14:
            raise ValueError("A completed development run is required")
        phi=[np.zeros((6,3))]
        times=[0.]
        for k,row in enumerate(rows):
            idx=k%len(axes)
            updates=[r for r in front if r["agent"]==ego and r["gate_index"]==k and r["success"] and r["time"]<=row["time"]]
            lateral=updates[-1]["allocation"][0]
            raw=(lateral-geometry[0,0]-geometry[0,1]*bends[idx,0])/gains[0]
            a,up=axes[idx],np.array([0.,0.,1.])
            phi.append([raw*a,raw*up,a,bends[idx,0]*a,up,bends[idx,1]*up])
            times.append(row["time"])
        phi=np.array(phi).reshape(-1,18)
        jac.append(np.stack([np.interp(all_times,times,col) for col in phi.T],axis=-1).reshape(-1,6,3))
    return pos,np.stack(jac,axis=1),params,course


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--inputs",nargs="+",required=True)
    p.add_argument("--minimum",type=float,default=.62)
    p.add_argument("--frame",type=float,default=.28)
    args=p.parse_args()
    separation_pos,separation_jac,bar_gap,bar_grad=[],[],[],[]
    bases=[]
    for name in args.inputs:
        pos,jac,base,course=observe(name)
        bases.append(base)
        for shift in (-1,0,1):
            ii=np.arange(1,len(pos)-1)
            delta=pos[ii,0]-pos[ii+shift,1]
            gradient=jac[ii,0]-jac[ii+shift,1]
            keep=np.linalg.norm(delta,axis=1)<1.35
            separation_pos.append(delta[keep]-np.einsum("nkd,k->nd",gradient[keep],base))
            separation_jac.append(gradient[keep])
        points=pos.reshape(-1,3)
        jac=jac.reshape(-1,6,3)
        for bar in gate_boxes(course):
            c,s=np.cos(bar["yaw"]),np.sin(bar["yaw"])
            rot=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            local=(points-bar["center"])@rot
            edge=abs(local)-np.array(bar["size"])*.5
            outside=np.maximum(0,edge)
            gap=np.linalg.norm(outside,axis=1)
            near=gap<.55
            if not near.any():
                continue
            if (gap[near]<1e-5).any():
                raise ValueError("Use non-contact calibration flights")
            normal=np.sign(local[near])*outside[near]/gap[near,None]
            plane=np.einsum("nkd,nd->nk",jac[near],normal@rot.T)
            bar_gap.append(gap[near]-plane@base)
            bar_grad.append(plane)
    sep,jsep,frame,jframe=map(np.concatenate,(separation_pos,separation_jac,bar_gap,bar_grad))
    base=np.mean(bases,axis=0)
    def clearance(p):
        delta=sep+np.einsum("nkd,k->nd",jsep,p)
        return np.r_[np.linalg.norm(delta,axis=1)-args.minimum,frame+jframe@p-args.frame]
    def gradient(p):
        delta=sep+np.einsum("nkd,k->nd",jsep,p)
        first=np.einsum("nkd,nd->nk",jsep,delta/np.linalg.norm(delta,axis=1,keepdims=True))
        return np.r_[first,jframe]
    # A common slack also exposes incompatible safety targets in these local models.
    scale=np.array([.21,.21,1.,1.,1.,1.])
    def loss(v):
        return np.sum(((v[:6]-base)*scale)**2)+150*v[-1]**2
    def grad(v):
        return np.r_[2*scale**2*(v[:6]-base),300*v[-1]]
    def margin(v):
        return clearance(v[:6])+v[-1]
    def dmargin(v):
        g=gradient(v[:6]);return np.c_[g,np.ones(len(g))]
    bounds=[(.25,2.15),(-2.1,2.1)]+[(-.14,.14)]*4+[(0,.5)]
    initial=np.r_[base,max(0,-min(clearance(base)))]
    fitted=minimize(loss,initial,jac=grad,bounds=bounds,method="SLSQP",
                    constraints=[dict(type="ineq",fun=margin,jac=dmargin)],options=dict(maxiter=100,ftol=1e-10))
    if not fitted.success:
        raise ValueError(fitted.message)
    w=fitted.x[:6]
    args.output.mkdir(exist_ok=False)
    np.savez(args.output/"head.npz",weight=w[:2],geometry_weight=w[2:].reshape(2,2))
    residual_min=clearance(w)
    report=dict(inputs=args.inputs,weights=w.tolist(),target_minimum=args.minimum,target_frame=args.frame,
                surrogate_min_m=float(residual_min[:len(sep)].min()+args.minimum),
                surrogate_frame_m=float(residual_min[len(sep):].min()+args.frame),slack_m=float(fitted.x[-1]))
    (args.output/"fit.json").write_text(json.dumps(report,indent=2))
    print(report)

if __name__=="__main__":
    main()
