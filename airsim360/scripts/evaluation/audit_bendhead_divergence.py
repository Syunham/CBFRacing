"""Audit stored gate targets, frontend ETAs, commands and controlled launch replays."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/"reports/spatial_potential"
OUT=ROOT/"cf21_bendhead03_comparison"
def read(folder,name):
    return json.loads((folder/name).read_text())
def commands(folder):
    return np.asarray([r["control"] for r in read(folder,"flight/planning_log.json")]).reshape(-1,2,5)
def report():
    runs=[ROOT/f"cf21_bendhead03_run{i}" for i in (1,2)]
    paths=[]; plans=[]; events=[]; gates=[]; configs=[]
    for folder in runs:
        audit=read(folder,"flight/spatial_audit.json")
        crossings=np.array(audit["ordered_crossing_times_s"])
        events.append(crossings)
        rows=read(folder,"flight/frontends.json")
        plans.append(rows)
        with np.load(folder/"flight/execution.npz") as raw:
            t,x=raw["time"],raw["state"]
        paths.append(x)
        cfg=read(folder,"experiment.json")
        cfg.pop("name")
        configs.append(cfg)
        stats=[]
        for k in range(14):
            endpoints=[]; lengths=[]; speeds=[]; final_times=[]
            for ego in (0,1):
                updates=[r for r in rows if r["success"] and r["agent"]==ego and r["gate_index"]==k]
                endpoints.append([r["time"]+r["duration"][0] for r in (updates[0],updates[-1])])
                final_times.append(updates[-1]["time"])
                a=crossings[ego,k-1] if k else 0.
                b=crossings[ego,k]
                times=np.r_[a,t[(t>a)&(t<b)],b]
                length=np.trapezoid(np.interp(times,t,np.linalg.norm(x[:,ego,3:6],axis=1)),times)
                lengths.append(length);speeds.append(length/(b-a))
            eta=np.asarray(endpoints)
            stats.append(dict(lap=k//7+1,gate=k%7+1,crossing_s=crossings[:,k].tolist(),
                signed_ms=1000*(crossings[0,k]-crossings[1,k]),frontend_first_last_ETA_s=eta.tolist(),
                first_final_planned_gap_ms=(1000*(eta[0]-eta[1])).tolist(),final_plan_origin_s=final_times,
                leg_length_m=lengths,mean_speed_mps=speeds))
        gates.append(stats)
    assert configs[0]==configs[1] and np.array_equal(paths[0][0],paths[1][0])
    checks=[]
    for j in range(8):
        data=[np.load(next((p/"flight/plans").glob(f"{j:03d}_*.npz"))) for p in runs]
        rows=[f[j] for f in plans]
        assert rows[0]["time"]==rows[1]["time"] and rows[0]["agent"]==rows[1]["agent"]
        grid=np.arange(1001)*.001
        curves=[np.stack([np.interp(grid,z["reference_time"],v) for v in z["reference_states"][:,:3].T],axis=1) for z in data]
        checks.append(dict(time=rows[0]["time"],agent=rows[0]["agent"],
            gate_point_differences_m=np.linalg.norm(data[1]["gates"]-data[0]["gates"],axis=1).tolist(),
            first_gate_planned_duration_delta_ms=1000*(data[1]["durations"][0]-data[0]["durations"][0]),
            reference_1s_max_position_delta_m=float(np.linalg.norm(curves[1]-curves[0],axis=1).max()),
            reference_states_identical=np.array_equal(data[1]["reference_states"],data[0]["reference_states"])))
        for d in data:d.close()
    actual=[commands(p) for p in runs]
    u=[v[13,1,:4] for v in actual]
    swaps=[]
    for prefix in ("bend03_launchswap01","bend03_refswap01"):
        for history in (1,2):
            for field in (1,2):
                folder=ROOT/f"{prefix}_h{history}_v{field}"
                if not folder.exists():continue
                test=commands(folder)
                control=test[13,1,:4]
                stop=10 if prefix=="bend03_refswap01" else 13
                swaps.append(dict(source=folder.name,history=history,donor=field,changed="reference at 0.20s" if stop==10 else "embedding at 0.26s",
                    unchanged_prefix_max_abs_error=float(abs(test[:stop]-actual[history-1][:stop]).max()),
                    motor_rate=control.tolist(),own_baseline_l2=float(np.linalg.norm(control-u[history-1]))))
    risks=[[r["node_risk"][0] for r in map(json.loads,(p/"vision.jsonl").read_text().splitlines()[:2])] for p in runs]
    data=dict(config_identical=True,initial_states_identical=True,initial_node_risk=risks,
              signed_definition="1000*(Drone0 crossing time - Drone1 crossing time); positive means Drone1 leads",
              gate_events=gates,early_reference_comparison=checks,first_control= [a[0].tolist() for a in actual],
              launch_control_l2=float(np.linalg.norm(u[0]-u[1])),launch_counterfactuals=swaps)
    OUT.mkdir(exist_ok=True)
    (OUT/"attribution.json").write_text(json.dumps(data,indent=2))
    return data

if __name__=="__main__":
    report()
