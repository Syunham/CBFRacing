"""Offline accuracy, coverage, and actuator-feasibility checks for a depth-pair flight."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import sys
from pathlib import Path

import numpy as np


def closing(x):
    relative = x[1, :3] - x[0, :3]
    distance = np.linalg.norm(relative)
    return distance, -relative @ (x[1, 3:6] - x[0, 3:6]) / distance


def main(folder):
    config = json.loads((folder/"experiment.json").read_text())
    from scripts.simulation.dynamics import select_vehicle
    select_vehicle(config.get("vehicle"))
    from scripts.simulation.dynamics import MODEL
    from scripts.control.pair_cbf import PairCBF
    barrier = PairCBF().barrier
    log = json.loads((folder/"flight/planning_log.json").read_text())
    controls = {(round(r["time"], 6), r["agent"]): r for r in log}
    observations = [json.loads(line) for line in (folder/"depth_pair.jsonl").read_text().splitlines()]
    with np.load(folder/"flight/execution.npz") as saved:
        truth = dict(zip(saved["time"].round(6), saved["state"]))
    frames = []
    for capture in observations:
        t = round(capture["time"], 6)
        if t not in truth or capture["state"] is None:
            continue
        measured = np.array(capture["state"])
        actual = truth[t]
        distance, speed = closing(measured)
        true_distance, true_speed = closing(actual)
        row = dict(time=t, hits=capture["hits"], valid=capture["velocity_valid"],
                   estimated_distance=distance, true_distance=true_distance,
                   estimated_closing=speed, true_closing=true_speed)
        if all((t, i) in controls for i in (0, 1)):
            entries = [controls[t,i] for i in (0,1)]
            command = np.array([e.get("nominal_control", e["control"])[:4] for e in entries]).ravel()
            for label, x in (("depth", measured), ("truth", actual)):
                h, _, _, nominal, jac = barrier(x.ravel(), command)
                a = np.asarray(jac).ravel()
                force = x[:,13:17].ravel()
                rate = MODEL.get("thrust_rate_max", 200.)
                low = np.maximum(-rate, (MODEL["thrust_min"]-force)/.02)
                high = np.minimum(rate, (MODEL["thrust_max"]-force)/.02)
                row[label] = dict(h=float(h), nominal=float(nominal),
                                  best=float(nominal)+a@(np.where(a>=0,high,low)-command))
        frames.append(row)
    scored = [r for r in frames if r["valid"]]
    error = np.array([abs(r["estimated_closing"]-r["true_closing"]) for r in scored])
    telemetry = folder.parents[2]/"logs"/(folder.name+"-telemetry.jsonl")
    health = [json.loads(line) for line in telemetry.read_text(encoding="utf-8-sig").splitlines()]
    summary = dict(observations=len(observations), velocity_frames=len(scored),
                   bilateral=sum(all(r["hits"]) for r in scored),
                   one_view=sum(sum(bool(n) for n in r["hits"])==1 for r in scored),
                   dead_reckoning=sum(not any(r["hits"]) for r in scored),
                   mae=float(error.mean()), p95=float(np.percentile(error,95)), maximum=float(error.max()),
                   corrections=sum(r.get("pair_cbf",{}).get("correction_norm",0)>1e-8
                                   for r in log if r["agent"]==0),
                   infeasible=[r["time"] for r in frames if r.get("depth",{}).get("best",0)<0],
                   last=frames[-1], peak_cpu_c=max(h["cpu_c"] for h in health),
                   new_whea=any(h["whea_record"]!=h["whea_baseline"] for h in health))
    (folder/"depth_flight_audit.json").write_text(json.dumps(dict(summary=summary,frames=frames),indent=2))
    print(json.dumps(summary, indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]).resolve())
