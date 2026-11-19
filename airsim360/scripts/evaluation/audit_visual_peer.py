"""Compare visual peer observations, gate crossings and executed CBF corrections."""

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
from scripts.perception.visual_peer import visual_pair_states


def audit(folder):
    flight = folder / "flight"
    cfg = json.loads((folder/"experiment.json").read_text())
    summary = json.loads((flight/"summary.json").read_text())
    from scripts.simulation.dynamics import select_vehicle
    select_vehicle(cfg.get("vehicle"))
    from scripts.control.pair_cbf import PairCBF
    from scripts.simulation.dynamics import MODEL, model_functions
    from scipy.optimize import minimize_scalar

    with np.load(flight/"execution.npz") as data:
        times, states = data["time"], data["state"]
    events = json.loads((flight/"events.json").read_text())
    logs = json.loads((flight/"planning_log.json").read_text())
    pair = PairCBF() if cfg["cbf_study"] == "filtered" else None
    frames = [json.loads(line) for line in (folder/"visual_peer.jsonl").read_text().splitlines()]
    indexed = {(round(f["time"], 6), f["observer"]): f for f in frames}
    coverage = []
    for i in (0, 1):
        measured, propagated, missing, errors = 0, 0, 0, []
        for t, snapshot in zip(times, states):
            f = indexed.get((round(t, 6), i))
            if f is None:
                continue
            if f["peer"] is None:
                missing += 1
            elif f["cluster_points"] == 0:
                propagated += 1
            else:
                measured += 1
                errors.append(np.linalg.norm(f["peer"][:3]-snapshot[1-i,:3]))
        coverage.append(dict(observer=i, measured=measured, extrapolated=propagated, missing=missing,
                             measured_error_p50_m=float(np.median(errors)) if errors else None,
                             measured_error_p95_m=float(np.percentile(errors,95)) if errors else None))
    corrected = unchecked = replayed = 0
    intervals = []
    for k, dt in enumerate(np.diff(times)):
        rows = logs[2*k:2*k+2]
        if [r["agent"] for r in rows] != [0,1] or abs(rows[0]["time"]-times[k]) > 1e-8:
            raise ValueError("Execution and control logs are misaligned")
        intervals.append(rows[0]["min_actual_distance"])
        if pair is None:
            continue
        peers = [indexed[round(times[k], 6), i]["peer"] for i in (0, 1)]
        nominal = np.array([r["nominal_control"] for r in rows])
        if any(peer is not None for peer in peers):
            observed = visual_pair_states(states[k], peers)
            applied, record = pair.project(observed, nominal, dt)
            np.testing.assert_allclose(record["h"], rows[0]["pair_cbf"]["h"], atol=1e-8)
            corrected += record["correction_norm"] > 1e-8
        else:
            applied = nominal
            unchecked += 1
        np.testing.assert_allclose(applied, [r["control"] for r in rows], atol=1e-6)
        replayed += 1
    gaps = [1000*(a["time"]-b["time"]) for a,b in zip(*events)]
    failed_qp = None
    if pair is not None and "cbf_infeasible" in logs[-1]:
        rows = logs[-2:]
        t, actual = times[-1], states[-1]
        peers = [indexed[round(t,6), i]["peer"] for i in (0,1)]
        observed = visual_pair_states(actual, peers)
        rates = np.array([r["control"][:4] for r in rows]).ravel()
        forces = actual[:,13:17].ravel()
        dt = times[-1]-times[-2]
        lo = np.maximum(-200., (MODEL["thrust_min"]-forces)/dt)
        hi = np.minimum(200., (MODEL["thrust_max"]-forces)/dt)
        failed_qp = {}
        for name, x in (("visual",observed), ("truth_offline",actual)):
            _, _, _, residual, jac = pair.barrier(x.ravel(),rates)
            a = np.asarray(jac).ravel()
            failed_qp[name] = dict(distance_m=float(np.linalg.norm(x[0,:3]-x[1,:3])),
                                   nominal_residual=float(residual),
                                   max_feasible_residual=float(residual)+a@(np.where(a>=0,hi,lo)-rates))
    # Refine the dense 2.5 ms distance audit around the closest pass using applied inputs.
    _, step = model_functions(8)
    closest = (float(np.linalg.norm(states[0,0,:3]-states[0,1,:3])), times[0])
    for k in np.flatnonzero(np.asarray(intervals) <= summary["min_distance"]+.02):
        dt = times[k+1]-times[k]
        controls = [r["control"] for r in logs[2*k:2*k+2]]
        def distance(offset):
            p = [np.asarray(step(states[k,i], controls[i], offset)).ravel()[:3] for i in (0,1)]
            return float(np.linalg.norm(p[0]-p[1]))
        t = minimize_scalar(distance, bounds=(0., dt), method="bounded").x
        closest = min(closest, *((distance(s), times[k]+s) for s in (0., t, dt)))
    telemetry = folder.parents[2]/"logs"/(folder.name+"-telemetry.jsonl")
    health = [json.loads(line) for line in telemetry.read_text(encoding="utf-8-sig").splitlines()]
    return dict(run=folder.name, success=summary["success"], gates=summary["gates_passed"],
                stop_reason=summary["stop_reason"], min_distance_m=closest[0], closest_time_s=closest[1],
                dense_min_m=summary["min_distance"],
                closest_tick_start_s=float(times[np.argmin(intervals)]) if intervals else None,
                ticks_below_radius=int(np.count_nonzero(np.asarray(intervals)<.55)),
                duration_s=summary["duration"], signed_gate_gaps_ms=gaps,
                mean_abs_gate_gap_ms=float(np.mean(np.abs(gaps))) if gaps else None, vision=coverage,
                ticks=len(times)-1, cbf_replayed_ticks=replayed, cbf_corrected_ticks=int(corrected),
                cbf_unavailable_ticks=unchecked if pair else None,
                failed_qp=failed_qp,
                peak_cpu_c=max(row["cpu_c"] for row in health),
                whea_new=any(row["whea_record"]>row["whea_baseline"] for row in health))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=Path, nargs="+")
    args = parser.parse_args()
    for folder in args.folders:
        report = json.dumps(audit(folder), indent=2)
        (folder/"visual_comparison.json").write_text(report)
        print(report)
