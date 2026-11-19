"""Independent whole-flight swept sphere clearance and ordered gate audit."""

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
from scripts.simulation.gate_geometry import gate_boxes
from scripts.training.train_rgbd_field import GateFrames


def minimum_spline_separation(times, states):
    """Check endpoints and all stationary points of squared relative distance."""
    relative = states[:,0,:6]-states[:,1,:6]
    spline = CubicHermiteSpline(times, relative[:,:3], relative[:,3:6])
    best = (float("inf"), float(times[0]))
    for i, dt in enumerate(np.diff(times)):
        # Express each cubic in the unit interval for numerical conditioning.
        coeff = spline.c[::-1,i,:] * (dt**np.arange(4))[:,None]
        squared = sum(np.convolve(axis,axis) for axis in coeff.T)
        roots = np.polynomial.polynomial.polyroots(np.arange(1,7)*squared[1:])
        inside = roots.real[(abs(roots.imag)<1e-8) & (roots.real>0) & (roots.real<1)]
        u = np.r_[0., inside, 1.]
        distances = np.linalg.norm(np.polynomial.polynomial.polyval(u,coeff),axis=0)
        j = distances.argmin()
        best = min(best,(float(distances[j]),float(times[i]+dt*u[j])))
    return best


def audit(folder, strict_minimum=None):
    with np.load(folder/"execution.npz") as f:
        t, x = f["time"], f["state"]
    summary = json.loads((folder/"summary.json").read_text())
    stop_separation = summary.get("stop_separation_m", .5)
    vehicle_geometry = folder.parent/"vehicle_geometry.json"
    body_diameter = 2 * json.loads(vehicle_geometry.read_text())["body_radius_m"] if vehicle_geometry.exists() else None
    geometry = folder.parent/"rendered_course.json"
    c = json.loads((geometry if geometry.exists() else folder/"course.json").read_text())
    boxes = gate_boxes(c)
    bars = GateFrames(boxes)
    centers, normals = np.array(c["centers"]), np.array(c["normals"])
    if len(t) < 2:
        return dict(passed=False, stop_reason=summary["stop_reason"])
    grid = np.linspace(t[0], t[-1], (len(t)-1)*20+1)
    p = CubicHermiteSpline(t, x[:, :, :3], x[:, :, 3:6])(grid)
    clearance = np.stack([bars.distances(p[:, ego]) for ego in (0, 1)], axis=1)
    minimum, approach_time = minimum_spline_separation(t,x)
    completed, passes, speed_stats = [], [], []
    for ego in (0, 1):
        idx = 0
        crossing_times = []
        for step, (a, b) in enumerate(zip(p[:-1, ego], p[1:, ego])):
            g = idx % len(centers)
            before, after = (np.stack((a, b))-centers[g])@normals[g]
            if before < 0 <= after:
                crossing = a + (b-a)*(-before/(after-before))-centers[g]
                side = np.cross([0, 0, 1], normals[g])
                if max(abs(crossing@side), abs(crossing[2])) <= .75:
                    idx += 1
                    crossing_times.append(float(grid[step] + (grid[step+1]-grid[step])*(-before/(after-before))))
        completed.append(idx)
        passes.append(crossing_times)
        finish = crossing_times[13] if idx >= 14 else t[-1]
        sample_t = np.r_[t[t < finish], finish]
        speeds = np.interp(sample_t, t, np.linalg.norm(x[:, ego, 3:6], axis=1))
        speed_stats.append(dict(mean=float(np.sum(.5*(speeds[1:]+speeds[:-1])*np.diff(sample_t))/(finish-t[0])),
                                peak=float(speeds.max())))
    where = np.unravel_index(clearance.argmin(), clearance.shape)
    nearest = p[where[0], where[1]][None, :]
    part = min(boxes, key=lambda box: GateFrames([box]).distances(nearest)[0])
    return dict(passed=bool(summary["success"] and min(completed) >= 14 and
                            clearance.min() >= .25 and
                            (stop_separation is None or minimum >= stop_separation) and
                            (strict_minimum is None or minimum > strict_minimum) and
                            (body_diameter is not None or stop_separation is not None) and
                            (body_diameter is None or minimum > body_diameter)),
                success=summary["success"], stop_reason=summary["stop_reason"], ordered_crossings=completed,
                minimum_frame_clearance_m=clearance.min(axis=0).tolist(),
                closest_approach_time_s=approach_time, strict_minimum_m=strict_minimum,
                bounding_sphere_gap_m=minimum-body_diameter if body_diameter is not None else None,
                minimum_separation_m=minimum, closest_frame_time_s=float(grid[where[0]]),
                closest_frame_agent=int(where[1]), closest_frame_part=part["name"],
                ordered_crossing_times_s=passes, speed_until_two_laps_mps=speed_stats,
                frame_reserve_m=.25, required_separation_m=stop_separation, sample_interval_s=float(grid[1]-grid[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("flight", type=Path)
    parser.add_argument("--strict-minimum",type=float,help="Independent strict separation requirement in metres")
    args = parser.parse_args()
    result = json.dumps(audit(args.flight,args.strict_minimum), indent=2)
    (args.flight/"spatial_audit.json").write_text(result)
    print(result)
