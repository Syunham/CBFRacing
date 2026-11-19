"""Targeted synthetic time-allocation study for the stacked Split-S gates."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import torch

from scripts.training.synthetic_snap_teacher import MODEL, cost, course, fixed_solver, gate_pair, label


BASE = Path(__file__).resolve().parents[2] / "reports/spatial_potential/synthetic_snap_times01"
FAILED = BASE.parent / "cf21_snap_learned_time01"


def transition_problems(seed_offset=0):
    # Independent Latin hypercubes: the same geometry strata occur in every split.
    for split_id, (split, count) in enumerate([("train", 144), ("val", 36), ("test", 48)]):
        for stratum in range(3):
            gate = 3 if stratum == 0 else 4
            centers, normals, _, axes = gate_pair(gate, 5)
            samples = qmc.LatinHypercube(22, seed=19371 + seed_offset + 51*split_id + stratum).random(count//3)
            for i, q in enumerate(samples):
                depth = .002 + .85*q[0]**2
                dz = (-.58 + 1.26*q[1]) if gate == 3 else (1.25 + 1.63*q[1])
                forward = (.3 + 4*q[2]) if gate == 3 else (-3.8 + (3.8 if stratum == 1 else 6.4)*q[2])
                x = np.zeros(19)
                x[:3] = centers[0]-depth*normals[0]+(.85*q[3]-.425)*axes[0]+[0, 0, dz]
                x[3:6] = forward*normals[0]+(4*q[4]-2)*axes[0]+[0, 0, -1.2 + 5*q[5]]
                x[6:10] = np.roll(Rotation.from_euler("ZYX", [360*q[6]-180, 100*q[7]-50, 120*q[8]-60],
                                                        degrees=True).as_quat(), 1)
                x[10:13] = 5*q[9:12]-2.5
                x[13:17] = np.clip(1.55 + 4.6*q[12] + 1.3*(q[13:17]-.5), .1, 7)
                offsets = .9*(q[17:22]-.5)
                if i%4 == 0:
                    offsets[:] = 0
                yield dict(own=x, gates=centers+offsets[:, None]*axes, normals=normals, offsets=offsets,
                           half=np.tile([.075, .1], (5, 1)), gate=gate, mode=stratum, kind="transition",
                           split=split, group=10000+1000*split_id+100*stratum+i)


def generate(folder):
    dst = folder/"labels"
    dst.mkdir(parents=True, exist_ok=True)
    solver = fixed_solver()
    for i, row in enumerate(transition_problems()):
        path = dst/f"{row['group']}_transition.npz"
        if path.exists():
            continue
        answer = label(row, solver)
        temp = path.with_suffix(".partial.npz")
        np.savez_compressed(temp, **row, **answer)
        temp.replace(path)
        if (i+1)%12 == 0:
            print(f"optimizer labels {i+1}/228", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["transition-label", "transition-train"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "transition-label":
        generate(args.output)
    else:
        from scripts.training.train_transition_times import study
        study(args.output)
