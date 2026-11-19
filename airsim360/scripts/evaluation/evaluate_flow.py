"""Evaluate the online field on causal RGB-D histories paired by gate and pose."""

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
from scipy.spatial import cKDTree
from scipy.special import expit
from scripts.evaluation.evaluate_flight_count import make_probes
from scripts.evaluation.evaluate_resolution import operating_point
from scripts.perception.occupancy_flow import FlowPrior, OccupancyFlow
from scripts.perception.panorama import spherical_rays
from scripts.perception.spatial_potential import camera_points
from scripts.training.train_rgbd_field import Windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("split", choices=["test","validation"])
    p.add_argument("out", type=Path)
    p.add_argument("--prior", type=Path)
    args = p.parse_args()
    manifest = Path("captures/cf21_512_double_50hz/manifest.json")
    entry, = [e for e in json.loads(manifest.read_text())["episodes"] if e["split"]==args.split]
    events = json.loads((manifest.parent / entry["trajectory"]).with_name("events.json").read_text())
    episode, = Windows(manifest,args.split,1,.25,lazy_windows=True).episodes
    times, states, rgb, depth = episode[:4]
    _, probes = make_probes(episode, events)
    rows = []
    prior = FlowPrior(args.prior) if args.prior else None
    for h,gate,stamp,occupied,empty in probes:
        if h != .4:
            continue
        row = dict(gate=gate,event_s=stamp)
        for (frame,ego,lead,grid), key in ((occupied,"risk_at_entry"),(empty,"risk_empty")):
            flow = OccupancyFlow(prior)
            for k in range(max(0,frame-4),frame+1):
                flow.observe(times[k],rgb[k,ego],depth[k,ego],states[k,ego])
            ranges = depth[frame,ego]
            mask = np.isfinite(ranges) & (ranges>.08) & (ranges<18)
            cloud = spherical_rays(*ranges.shape)[mask]*ranges[mask,None]
            distance,_ = cKDTree(cloud).query(camera_points(grid,states[frame,ego]))
            scores = np.maximum(expit((.20-distance)/.055),flow.score(grid,lead))
            row[key] = float(scores.max())
            if key=="risk_at_entry":
                row["observer"]=int(ego)
                row["observation_s"]=float(times[frame])
                row["peer_distance_m"]=float(np.linalg.norm(states[frame,0,:3]-states[frame,1,:3]))
        rows.append(row)
    empty = sorted((r["risk_empty"] for r in rows),reverse=True)
    report = dict(split=args.split,episode=entry["episode_id"],rows=rows,
                  at_half=operating_point(rows,.5),
                  matched=[operating_point(rows,np.nextafter(empty[n],np.inf)) for n in (2,7) if n<len(rows)])
    args.out.write_text(json.dumps(report,indent=2))
    print(report["at_half"],report["matched"])

if __name__=="__main__":
    main()
