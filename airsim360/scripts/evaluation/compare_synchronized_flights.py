"""Compare gate timings against the original 5600 RGB-D baseline and audit the swept trajectory."""

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
from scripts.evaluation.audit_spatial_flight import audit

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/"reports/spatial_potential"
BASELINE="cf21_time8_w5600_nostop_01"

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def relative_geometry(folder,crossings):
    if not all(crossings):
        return {}
    with np.load(folder/"flight/execution.npz") as raw:
        t,x=raw["time"],raw["state"]
    path=CubicHermiteSpline(t,x[:,:,:3],x[:,:,3:6])
    grid=np.arange(t[0],min(times[-1] for times in crossings),.005)
    if not grid.size:
        return {}
    pos,velocity=path(grid),path(grid,1)
    delta=pos[:,1]-pos[:,0]
    direction=velocity.mean(1)[:,:2]
    speed=np.linalg.norm(direction,axis=1)
    direction=direction/np.maximum(speed[:,None],1e-9)
    along=(delta[:,:2]*direction).sum(1)
    lateral=delta[:,0]*direction[:,1]-delta[:,1]*direction[:,0]
    moving=(speed>1)&((velocity[:,0]*velocity[:,1]).sum(1)>0)
    near=np.linalg.norm(delta,axis=1)<=1.5
    n=min(map(len,crossings))
    pts=[path(times[:n])[:,ego,:] for ego,times in enumerate(crossings)]
    normals=read(folder/"flight/course.json")["normals"]
    normal=np.take(normals,np.arange(n)%len(normals),axis=0)
    diff=pts[0]-pts[1]
    return dict(horizontal_side_by_side_fraction=float((near & (abs(lateral)>abs(along)))[moving].mean()) if moving.any() else None,
                gate_height_difference_m=abs(diff[:,2]).tolist(),
                gate_lateral_difference_m=abs((diff*np.cross([0,0,1],normal)).sum(1)).tolist())

def inspect(name):
    folder=RESULTS/name
    result=audit(folder/"flight",strict_minimum=.5)
    (folder/"flight/spatial_audit.json").write_text(json.dumps(result,indent=2))
    config=read(folder/"experiment.json")
    run=read(folder/"flight/summary.json")
    crossings=result.get("ordered_crossing_times_s",[[],[]])
    gaps=np.array([(a-b)*1000 for a,b in zip(*crossings)])
    log=ROOT/"logs"/(name+"-telemetry.jsonl")
    telemetry=[json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines() if line] if log.exists() else []
    return dict(name=name,config=config,success=run["success"],reason=run["stop_reason"],
                ordered_crossings=result.get("ordered_crossings"),audited=result.get("passed",False),
                min_distance_m=result.get("minimum_separation_m"),minimum_frame_clearance_m=result.get("minimum_frame_clearance_m"),
                time_of_min_s=result.get("closest_approach_time_s"),duration_s=run["duration"],peak_speed_mps=run["max_speed"],
                stop_distance_m=run.get("stop_separation_m"),manual_waiting=run["manual_waiting"],
                physical_latency_simulated=run["physical_latency_simulated"],gaps_ms=gaps.tolist(),
                median_ms=float(np.median(abs(gaps))) if len(gaps) else None,
                maximum_ms=float(max(abs(gaps))) if len(gaps) else None,within_50ms=int((abs(gaps)<=50).sum()),
                cpu_max_c=max((r["cpu_c"] for r in telemetry),default=None),
                whea_records=sorted({r["whea_record"] for r in telemetry}),whea_baseline=telemetry[0]["whea_baseline"] if telemetry else None,
                **relative_geometry(folder,crossings))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials",nargs="+")
    args=parser.parse_args()
    baseline=inspect(BASELINE)
    values=[inspect(name) for name in args.trials]
    initial=np.load(RESULTS/BASELINE/"flight/execution.npz")["state"][0]
    for v in values:
        current=np.load(RESULTS/v["name"]/"flight/execution.npz")["state"][0]
        assert np.array_equal(current,initial)
        assert v["config"]["separation"]==1.2 and v["stop_distance_m"] is None
        n=len(v["gaps_ms"])
        v["mean_absolute_gap_difference_from_baseline_ms"]=float(np.mean(abs(np.abs(v["gaps_ms"])-np.abs(baseline["gaps_ms"][:n])))) if n else None
        print(v["name"],v["reason"],v["min_distance_m"],v["minimum_frame_clearance_m"],v["median_ms"],v["maximum_ms"],v["within_50ms"])
    out=RESULTS/args.trials[-1]/"timing_comparison.json"
    out.write_text(json.dumps(dict(baseline=baseline,trials=values),indent=2))
