"""Fit the full-scale course to interior free space measured by static ERP depth."""

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
from scipy.ndimage import distance_transform_edt, minimum_filter
from scipy.spatial.transform import Rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=Path, nargs="+", required=True)
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    scans = [dict(np.load(path)) for path in args.scans]
    step = .2
    origin = scans[0]["position"] - [24., 24., 9.]
    grid = np.indices((241,241,91), dtype="float32")
    world = np.moveaxis(grid,0,-1).reshape(-1,3)*step + origin
    free = np.zeros(len(world), dtype=bool)
    for scan in scans:
        depth = scan["depth"].copy()
        depth[(depth<=0) | (depth>100) | ~np.isfinite(depth)] = 0
        depth = minimum_filter(depth,size=3,mode=("nearest","wrap"))
        h,w = depth.shape
        orientation = Rotation.from_quat(scan["orientation_xyzw"]).inv()
        for start in range(0,len(world),100000):
            sl = slice(start,start+100000)
            direction = orientation.apply(world[sl]-scan["position"])
            x,y,z = direction.T
            u = np.floor((np.arctan2(y,x)/(2*np.pi)+.5)*w).astype(int)%w
            v = np.floor((np.arctan2(z,np.hypot(x,y))/np.pi+.5)*h).astype(int).clip(0,h-1)
            free[sl] |= np.linalg.norm(direction,axis=1) < depth[v,u]-.15
    free = free.reshape(grid.shape[1:])
    for axis in range(3):
        edges = np.moveaxis(free,axis,0)
        edges[[0,-1]]=False
    clearance = distance_transform_edt(free)*step
    flights = {}
    for path in sorted(args.trajectories.glob("*/execution.npz")):
        with np.load(path) as f:
            flights[path.parent.name] = f["state"][:,:,:3].reshape(-1,3)
    course = json.loads(next(args.trajectories.glob("*/course.json")).read_text())
    bars = []
    for center,axis,normal,size in zip(course["centers"], course["coordinate_axes"],course["normals"],course["sizes"]):
        for x in [-size[0]/2,size[0]/2]:
            for y in (-1,0,1):
                for z in (-1,0,1):
                    bars.append(center + np.asarray(axis)*y*(size[1]/2+.12) +
                                np.asarray(normal)*x + [0,0,z*(size[2]/2+.12)])
    points = np.concatenate([*flights.values(),np.array(bars)])
    def transform(p, angle):
        return Rotation.from_euler("z",angle,degrees=True).apply(p*[1,-1,-1])
    def at(p):
        idx = np.rint((p-origin)/step).astype(int)
        idx = np.clip(idx,[0,0,0],np.array(clearance.shape)-1)
        return clearance[tuple(idx.T)]
    # Candidate placements span the court around the survey, with no scale change.
    c = scans[0]["position"]
    offsets = np.array([[x,y,z] for x in np.arange(-10,10.1,1.) for y in np.arange(-10,10.1,1.)
                        for z in np.arange(-1,6,.5)])+c
    sparse = np.concatenate([*[v[::18] for v in flights.values()], bars])
    candidates=[]
    for angle in range(0,180,15):
        rotated=transform(sparse,angle)
        score=[]
        for block in np.array_split(offsets,32):
            d=at((block[:,None,:]+rotated).reshape(-1,3)).reshape(len(block),-1)
            score.extend(d.min(axis=1)+.01*d.mean(axis=1))
        for i in np.argsort(score)[-8:]:
            candidates.append((angle,offsets[i]))
    best = None
    for angle, offset in candidates:
        values=at(transform(points,angle)+offset)
        score=values.min()+.01*values.mean()
        if best is None or score>best[0]:
            best=(score,angle,offset)
    _,angle,offset=best
    margin=np.sqrt(3)*step
    def minimum(p):
        return round(float(at(transform(p,angle)+offset).min()-margin),3)
    report=dict(heading_ned_degrees=angle, translation_ned_m=offset.tolist(), grid_step_m=step,
                scans=[str(p) for p in args.scans], course_scale=1, flight_samples=len(points)-len(bars),
                method="Distance to boundary of depth-observed free space. Three-pixel minimum filter, 0.15 m radial margin and sqrt(3) grid-cell deduction. Finite resolution estimate.",
                gate_sample_clearance_m=minimum(np.array(bars)),
                flight_center_clearance_m={k:minimum(v) for k,v in flights.items()})
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
