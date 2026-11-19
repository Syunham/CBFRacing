"""Fit a shared visual gate output using the measured response of closed-loop development flights."""

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
from scipy.interpolate import CubicHermiteSpline, RBFInterpolator
from scipy.optimize import differential_evolution
from scripts.simulation.gate_geometry import gate_boxes

RESULTS = Path(__file__).resolve().parents[2]/"reports/spatial_potential"
SAMPLES_PER_LEG = 75


def read(path):
    return json.loads(path.read_text())


def load_run(name):
    root = RESULTS/name
    config = read(root/"experiment.json")
    if config.get("local_flow") or config.get("flow_prior") or config["weight"] != 5600:
        raise ValueError(f"Incompatible perception or cost in {name}")
    head = config.get("passage_head")
    weights = np.load(head)["weight"] if head else np.array([config.get("passage_gain", 1.), 0.])
    events = read(root/"flight/events.json")
    data = np.load(root/"flight/execution.npz")
    poses, durations = [], []
    for ego, rows in enumerate(events):
        if len(rows) < 14:
            raise ValueError(f"{name} has an incomplete development trajectory")
        stamps = np.r_[0., [p["time"] for p in rows[:14]]]
        durations.append(np.diff(stamps))
        u = np.arange(14*SAMPLES_PER_LEG+1)/SAMPLES_PER_LEG
        times = np.interp(u, np.arange(15), stamps)
        poses.append(CubicHermiteSpline(data["time"], data["state"][:,ego,:3], data["state"][:,ego,3:6])(times))
    return weights, np.array(durations), np.array(poses)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--minimum", type=float, default=.59)
    ap.add_argument("--frame", type=float, default=.26)
    ap.add_argument("--vertical-range", nargs=2, type=float, default=[-.1,2.1])
    ap.add_argument("--lateral-range", nargs=2, type=float, default=[0.,2.1])
    args = ap.parse_args()
    weights, durations, poses = map(np.stack, zip(*(load_run(n) for n in args.inputs)))
    if np.linalg.matrix_rank(np.c_[np.ones(len(weights)),weights]) != 3:
        raise ValueError("Need independent horizontal and vertical perturbations")
    # Interpolate measured trajectories in the parameter plane. Progress here
    # labels each vehicle's own data for offline regression only.
    blend = RBFInterpolator(weights, np.eye(len(weights)), kernel="thin_plate_spline", smoothing=.0001)
    course = read(RESULTS/args.inputs[0]/"flight/course.json")
    bars = gate_boxes(course)
    center = np.array([b["center"] for b in bars])
    half = np.array([b["size"] for b in bars])*.5
    yaw = np.array([b["yaw"] for b in bars])
    anchors = poses.mean(0)[:,::2]
    closest_bars = np.argsort(np.linalg.norm(anchors[:,:,None,:]-center,axis=3),axis=2)[:,:,:6]
    bar_center, bar_half = center[closest_bars], half[closest_bars]
    cos, sin = np.cos(yaw[closest_bars]),np.sin(yaw[closest_bars])
    stamps_axis = np.arange(14*SAMPLES_PER_LEG+1)/SAMPLES_PER_LEG
    def measure(w):
        coefficients = blend(np.array(w)[None,:])[0]
        leg = np.einsum("i,ijk->jk", coefficients,durations)
        if leg.min() <= 0:
            return 10.,10.,0.,0.
        event = np.c_[np.zeros(2),np.cumsum(leg,axis=1)]
        pos = np.einsum("i,ijkl->jkl", coefficients, poses)
        stamps = [np.interp(stamps_axis, np.arange(15), e) for e in event]
        t = np.arange(0., min(event[:,-1]),.014)
        def at(ego, times):
            return np.stack([np.interp(times,stamps[ego],v) for v in pos[ego].T],axis=1)
        first = at(0,t)
        closest = min(np.linalg.norm(first-at(1,t+delay),axis=1).min() for delay in (-.04,0,.04))
        gap = abs(event[0,1:]-event[1,1:])
        rel = pos[:,::2,None,:]-bar_center
        dx,dy,dz = np.moveaxis(rel,-1,0)
        edge = abs(np.stack([cos*dx+sin*dy,cos*dy-sin*dx,dz],axis=-1))-bar_half
        reserve = np.min(np.linalg.norm(np.maximum(0,edge),axis=-1)+np.minimum(0,edge.max(axis=-1)))
        return float(gap.mean()),float(gap.max()),float(closest),float(reserve)
    def loss(w):
        avg,peak,closest,frame = measure(w)
        return avg + .3*peak + 10*max(0,args.minimum-closest) + 4*max(0,args.frame-frame)
    model = differential_evolution(loss,[args.lateral_range,args.vertical_range],seed=18,popsize=9,maxiter=60,tol=.0006)
    fitted = measure(model.x)
    args.dest.mkdir(exist_ok=False)
    np.savez(args.dest/"head.npz",weight=model.x)
    report = dict(inputs=args.inputs,observed_weights=weights.tolist(),weights=model.x.tolist(),
                  target_clearance_m=args.minimum,target_gate_reserve_m=args.frame,
                  predicted_mean_gap_s=fitted[0],predicted_max_gap_s=fitted[1],
                  predicted_minimum_center_m=fitted[2],predicted_gate_reserve_m=fitted[3])
    (args.dest/"fit.json").write_text(json.dumps(report,indent=2))
    print(report)

if __name__=="__main__":
    main()
