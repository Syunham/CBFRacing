"""Fit a shared two-output visual gate calibration on recorded development flights."""

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
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[2]/"reports/spatial_potential"

def read(path):
    return json.loads(path.read_text())

def examples(name, jitter):
    folder=ROOT/name/"flight"
    with np.load(folder/"execution.npz") as file:
        ts, states=file["time"],file["state"]
    course=read(folder/"course.json")
    events=read(folder/"events.json")
    front=read(folder/"frontends.json")
    axes=np.asarray(course["coordinate_axes"])
    # Jacobian surrogate: interpolate changes at the actual gate crossings.
    # The replay software never uses this interpolation to control an agent.
    jac=[]
    for ego,passes in enumerate(events):
        stamps=[0.]
        vectors=[np.zeros((2,3))]
        for i, event in enumerate(passes):
            accepted = [r for r in front if r["agent"]==ego and r["gate_index"]==i and r["time"]<event["time"] and r["success"]]
            raw=accepted[-1]["allocation"][0]
            stamps.append(event["time"])
            vectors.append(raw*np.stack([axes[i%len(axes)],[0,0,1]]))
        flat=np.array(vectors).reshape(-1,6)
        jac.append(np.stack([np.interp(ts,stamps,col) for col in flat.T],axis=1).reshape(-1,2,3))
    upto=min(events[0][-1]["time"],events[1][-1]["time"])
    keep=np.flatnonzero(ts<=upto)[::2]
    shifts=range(-jitter,jitter+1,2)
    distances=[]; changes=[]
    for shift in shifts:
        ii=keep[(keep+shift>=0)&(keep+shift<len(ts))]
        distances.append(states[ii,0,:3]-states[ii+shift,1,:3])
        changes.append(jac[0][ii]-jac[1][ii+shift])
    return np.concatenate(distances),np.concatenate(changes)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--inputs",nargs="+",required=True)
    p.add_argument("--clearance",default=.72,type=float)
    p.add_argument("--jitter-steps",default=4,type=int)
    args=p.parse_args()
    pos,jac=map(np.concatenate,zip(*(examples(n,args.jitter_steps) for n in args.inputs)))
    def gap(w):
        delta=pos+np.einsum("nkd,k->nd",jac,w)
        return (delta*delta).sum(axis=1)-args.clearance**2
    def deriv(w):
        return 2*np.einsum("nd,nkd->nk",pos+np.einsum("nkd,k->nd",jac,w),jac)
    bound=[(-.8,1.05),(-1.9,1.9)]
    choices=[]
    for init in ([0,1.5],[0,-1.5],[.8,1],[.8,-1]):
        fit=minimize(lambda w:w@w,init,jac=lambda w:2*w,method="SLSQP",bounds=bound,
                     constraints=[dict(type="ineq",fun=gap,jac=deriv)],options=dict(maxiter=120,ftol=1e-10))
        if min(gap(fit.x))>=-1e-6:
            choices.append(fit)
    if not choices:
        raise ValueError("No feasible calibrated gate outputs on this surrogate")
    best=min(choices,key=lambda r:r.fun)
    weights=best.x+np.array([1.,0.])
    args.output.mkdir(exist_ok=False)
    np.savez(args.output/"head.npz",weight=weights)
    report=dict(inputs=args.inputs,weights=weights.tolist(),surrogate_clearance_m=args.clearance,
                phase_perturbation_s=args.jitter_steps*.02,training_pairs=len(pos),fitted_min_m=float(np.sqrt(gap(best.x).min()+args.clearance**2)))
    (args.output/"fit.json").write_text(json.dumps(report,indent=2))
    print(report)

if __name__=="__main__":
    main()
