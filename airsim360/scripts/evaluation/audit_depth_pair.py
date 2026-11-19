"""Replay depth-only measurements; ground truth enters only the scoring path."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
from pathlib import Path
import sys

import numpy as np
from scripts.perception.depth_pair import DepthPair
from scripts.perception.dsp_peer_speed import DspPeerSpeed


def stats(values):
    a=np.abs(values)
    return dict(n=len(a),mae=float(a.mean()),p95=float(np.percentile(a,95)),maximum=float(a.max()))


def main(run,depths):
    cfg=json.loads((run/"experiment.json").read_text())
    from scripts.simulation.dynamics import select_vehicle
    select_vehicle(cfg.get("vehicle"))
    from scripts.control.pair_cbf import PairCBF
    from scripts.simulation.dynamics import MODEL
    from scripts.perception.visual_peer import visual_pair_states
    cbf=PairCBF()
    tracker=DepthPair(json.loads((run/"flight/course.json").read_text()),cfg["separation"])
    log=json.loads((run/"flight/planning_log.json").read_text())
    original={(round(r["time"],6),r["observer"]):r["peer"] for r in
              map(json.loads,(run/"visual_peer.jsonl").read_text().splitlines())}
    with np.load(run/"flight/execution.npz") as saved:
        times,states=saved["time"],saved["state"]
    cases=[]
    forces=np.full((2,4),MODEL["mass"]*MODEL["gravity"]/4)
    dsp=DspPeerSpeed(7)
    for k,t in enumerate(times):
        if k:
            prior_controls=log[2*k-2:2*k]
            forces+=(t-times[k-1])*np.array([r["control"][:4] for r in prior_controls])
        with np.load(depths/f"depth_{k:04d}.npz") as d:tracker.update(t,d["depth"])
        count=sum(tracker.hits)
        dsp_velocity=dsp.update_groups(t,[(tracker.rel,count)] if count else [])
        row=dict(t=t,valid=tracker.valid,hits=tracker.hits,
                 rel=tracker.rel.tolist() if tracker.rel is not None else None,
                 odometry=(tracker.ego[1].p-tracker.ego[0].p).tolist(),
                 dv_odom=(tracker.ego[1].v-tracker.ego[0].v).tolist())
        cases.append(row)
        if not tracker.valid:continue
        actual=states[k]
        peers=[original[round(t,6),i] for i in (0,1)]
        if all(p is None for p in peers):continue
        base=visual_pair_states(actual,peers)
        estimate=tracker.barrier_state(forces)
        controls=log[2*k:2*k+2]
        assert len(controls)==2 and all(abs(r["time"]-t)<1e-8 for r in controls)
        u=np.array([r.get("nominal_control",r["control"])[:4] for r in controls]).ravel()
        def score(x):
            r=x[0,:3]-x[1,:3];v=x[0,3:6]-x[1,3:6]
            _,_,_,res,a=cbf.barrier(x.ravel(),u)
            a=np.asarray(a).ravel()
            force=x[:,13:17].ravel()
            low=np.maximum(-200,(MODEL["thrust_min"]-force)/.02)
            high=np.minimum(200,(MODEL["thrust_max"]-force)/.02)
            return dict(closing=float(-r@v/np.linalg.norm(r)),distance=float(np.linalg.norm(r)),
                        residual=float(res),best=float(res)+a@(np.where(a>=0,high,low)-u))
        row.update(depth=score(estimate),old=score(base),truth=score(actual),
                   rel_error=np.linalg.norm(tracker.rel-(actual[1,:3]-actual[0,:3])),
                   ego_error=[np.linalg.norm(c.p-actual[i,:3]) for i,c in enumerate(tracker.ego)])
        # Counterfactual: retain the original CBF positions, attitude, rates and thrust.
        # Only swap relative velocity so we can isolate its effect on feasibility.
        only_velocity=base.copy()
        mean_velocity=base[:,3:6].mean(axis=0)
        only_velocity[:,3:6]=[mean_velocity-tracker.dv/2,mean_velocity+tracker.dv/2]
        row["only_velocity"]=score(only_velocity)
        if dsp_velocity is not None:
            for label,velocity in (("cluster_difference",dsp.raw),("cluster_particles",dsp_velocity)):
                variant=estimate.copy()
                variant[1,3:6]=variant[0,3:6]+velocity
                row[label]=score(variant)
    valid=[r for r in cases if "truth" in r]
    summary=dict(total=len(times),valid=len(valid),last=cases[-1],metrics={})
    for mode in ("depth","old","only_velocity","cluster_difference","cluster_particles"):
        selected=[r for r in valid if mode in r]
        summary["metrics"][mode]=dict(closing=stats([r[mode]["closing"]-r["truth"]["closing"] for r in selected]),
              range=stats([r[mode]["distance"]-r["truth"]["distance"] for r in selected]),
              infeasible=[round(r["t"],3) for r in selected if r[mode]["best"]<0])
    (depths/"depth_velocity_audit.json").write_text(json.dumps(dict(summary=summary,frames=cases),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main(*map(Path,sys.argv[1:3]))
