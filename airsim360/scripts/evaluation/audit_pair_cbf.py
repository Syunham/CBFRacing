"""Replay both flights' executed inputs and compare measured centre-to-centre minima."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def audit(folder):
    config = json.loads((folder/"experiment.json").read_text())
    from scripts.simulation.dynamics import select_vehicle
    select_vehicle(config.get("vehicle"))
    from scripts.simulation.dynamics import model_functions
    from scripts.control.pair_cbf import PairCBF
    from scipy.optimize import minimize_scalar
    cbf = PairCBF()
    _, step = model_functions(8)
    summary = json.loads((folder/"flight/summary.json").read_text())
    logs = json.loads((folder/"flight/planning_log.json").read_text())
    with np.load(folder/"flight/execution.npz") as data:
        times, states = data["time"], data["state"]
    closest = (float(np.linalg.norm(states[0,0,:3]-states[0,1,:3])), float(times[0]))
    max_pos_err = 0.
    residuals, corrections, flagged_times = [], [], []
    for k, dt in enumerate(np.diff(times)):
        x = states[k]
        rows = logs[2*k:2*k+2]
        if [r["agent"] for r in rows] != [0,1] or abs(rows[0]["time"]-times[k])>1e-8:
            raise ValueError("Controller and execution times differ")
        u = np.array([r["control"] for r in rows])
        nominal = np.array([r.get("nominal_control",r["control"]) for r in rows])
        if config["cbf_study"]=="filtered":
            projected, _ = cbf.project(x, nominal, dt)
            np.testing.assert_allclose(u, projected, atol=1e-6)
        raw = float(cbf.barrier(x.ravel(),nominal[:,:4].ravel())[3])
        applied = float(cbf.barrier(x.ravel(),u[:,:4].ravel())[3])
        if raw < 0:
            flagged_times.append(float(times[k]))
        residuals.append(applied)
        corrections.append(float(np.linalg.norm(u-nominal)))
        def positions(offset):
            return np.stack([np.asarray(step(x[i],u[i],offset)).ravel()[:3] for i in (0,1)])
        def distance(offset):
            p = positions(offset)
            return float(np.linalg.norm(p[0]-p[1]))
        end = positions(dt)
        max_pos_err = max(max_pos_err, float(abs(end-states[k+1,:,:3]).max()))
        # Refine the original 2.5 ms audit near the closest pass.
        if rows[0]["min_actual_distance"] <= summary["min_distance"]+.02:
            t = minimize_scalar(distance, bounds=(0.,dt), method="bounded", options={"xatol":1e-9}).x
            closest = min(closest, *((distance(z),float(times[k]+z)) for z in (0.,t,dt)))
    if max_pos_err>1e-5 or abs(closest[0]-summary["min_distance"])>.001:
        raise ValueError("Executed trajectory did not reproduce the recorded distance")
    return dict(run=folder.name, mode=config["cbf_study"], success=summary["success"],
                gates=summary["gates_passed"], duration_s=summary["duration"],
                min_distance_m=closest[0], closest_time_s=closest[1],
                dense_min_m=summary["min_distance"], ticks=len(corrections),
                nominal_cbf_violations=len(flagged_times), violation_times_s=flagged_times,
                corrected_ticks=sum(v>1e-8 for v in corrections), min_applied_residual=min(residuals),
                max_correction_norm_N_per_s=max(corrections), max_replay_position_error_m=max_pos_err)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nominal",type=Path)
    parser.add_argument("filtered",type=Path)
    args=parser.parse_args()
    report=[audit(folder) for folder in (args.nominal,args.filtered)]
    result=json.dumps(report,indent=2)
    (args.filtered/"cbf_comparison.json").write_text(result)
    print(result)
