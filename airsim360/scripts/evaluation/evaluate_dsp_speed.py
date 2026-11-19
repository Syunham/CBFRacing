"""Paired same-observation velocity and CBF audit at fixed executed states/controls."""

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
from scripts.perception.dsp_peer_speed import DspPeerSpeed, clusters
from scripts.perception.visual_peer import VisualPeer, visual_pair_states


def stats(values):
    x=np.asarray(values)
    if not len(x):return None
    return dict(n=len(x),mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max()))


def main(folder,out):
    cfg=json.loads((folder/"experiment.json").read_text())
    from scripts.simulation.dynamics import select_vehicle
    select_vehicle(cfg.get("vehicle"))
    from scripts.control.pair_cbf import PairCBF
    from scripts.simulation.dynamics import MODEL
    cbf=PairCBF()
    logs=json.loads((folder/"flight/planning_log.json").read_text())
    recorded={(round(r["time"],6),r["observer"]):r for r in
              map(json.loads,(folder/"visual_peer.jsonl").read_text().splitlines())}
    with np.load(folder/"flight/execution.npz") as f:
        times,states=f["time"],f["state"]
    tracks=[VisualPeer(),VisualPeer()]
    particles=[DspPeerSpeed(i+7) for i in range(2)]
    names=("baseline","difference","particles")
    errors={name:[[],[]] for name in names}
    radial={name:[] for name in names}
    bad={name:[] for name in names}
    frames=[]
    def evaluate(x):
        r=x[0,:3]-x[1,:3]
        rates=nom[:,:4].ravel()
        _,_,_,res,jac=cbf.barrier(x.ravel(),rates)
        a=np.array(jac).ravel()
        force=x[:,13:17].ravel()
        lower=np.maximum(-200,(MODEL["thrust_min"]-force)/dt)
        upper=np.minimum(200,(MODEL["thrust_max"]-force)/dt)
        return dict(closing=float(-r@(x[0,3:6]-x[1,3:6])/np.linalg.norm(r)),
                    residual=float(res),best=float(res)+a@(np.where(a>=0,upper,lower)-rates))
    with np.load(out/"clouds.npz") as clouds:
        for k,(t,x) in enumerate(zip(times,states)):
            dt=times[1]-times[0] if k==0 else t-times[k-1]
            rows=logs[2*k:2*k+2]
            assert all(abs(r["time"]-t)<1e-8 for r in rows)
            nom=np.array([r.get("nominal_control",r["control"]) for r in rows])
            estimates={name:[None,None] for name in names}
            for i in (0,1):
                xyz=clouds[f"{k}_{i}"]
                tracks[i].update(t,xyz)
                base=tracks[i].estimate(t)
                new=particles[i].update(t,xyz)
                if base is None or new is None:continue
                for name,v in zip(names,[base[3:],particles[i].raw,new]):
                    estimates[name][i]=np.r_[base[:3],v]
                    errors[name][i].append(float(np.linalg.norm(v-x[1-i,3:6])))
            truth=evaluate(x)
            results={}
            # Same two observers available for every method, same positions, same nominal control.
            if all(v is not None for v in estimates["particles"]):
                for name,peers in estimates.items():
                    m=evaluate(visual_pair_states(x,peers))
                    radial[name].append(abs(m["closing"]-truth["closing"]))
                    if m["best"]<0:bad[name].append(float(t))
                    results[name]=m
            frames.append(dict(t=t,estimates=estimates,truth=truth,metrics=results))
    rec=[np.array(recorded[round(times[-1],6),i]["peer"]) for i in (0,1)]
    failure=dict(truth=frames[-1]["truth"],original=evaluate(visual_pair_states(states[-1],rec)))
    for name,est in frames[-1]["estimates"].items():
        if all(v is not None for v in est):
            failure[name]=evaluate(visual_pair_states(states[-1],[np.r_[r[:3],e[3:]] for r,e in zip(rec,est)]))
    report=dict(source=folder.name,frames=len(times),errors={
        n:dict(pooled=stats(errors[n][0]+errors[n][1]),per_view=[stats(e) for e in errors[n]],
               closing_abs=stats(radial[n]),infeasible_s=bad[n]) for n in names},
        last_fixed_original_positions=failure,last_replay=frames[-1]["metrics"])
    # This separate test reuses the original selected cluster medians and counts.
    # It isolates the temporal filter; original per-point returns were not logged.
    seeds=[]
    for seed in (7,17,27,37,47):
        pair=[DspPeerSpeed(seed+i) for i in (0,1)]
        for t in times:
            speeds=[]
            for i,tracker in enumerate(pair):
                row=recorded[round(t,6),i]
                groups=[(np.array(row["peer"][:3]),row["cluster_points"])] if row["cluster_points"]>=2 else []
                speeds.append(tracker.update_groups(t,groups))
        if all(v is not None for v in speeds):
            peers=[np.r_[r[:3],v] for r,v in zip(rec,speeds)]
            seeds.append(dict(seed=seed,**evaluate(visual_pair_states(states[-1],peers))))
    report["last_original_cluster_history"]=seeds
    (out/"speed_comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (out/"paired_speed_frames.json").write_text(json.dumps(frames,default=lambda a:a.tolist()),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main(*map(Path,sys.argv[1:3]))
