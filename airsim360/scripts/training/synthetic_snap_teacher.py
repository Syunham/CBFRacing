"""Generate independent Split-S planning problems and label them with offline Minimum Snap."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from scripts.paths import FLIGHT_ASSETS
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
from scripts.simulation.course import MODEL, configuration, course, gate_pair
from scripts.planning.utils.allocation import load_allocator, observation
from scripts.planning.minsnap.minimum_snap import MinimumSnapPlanner
from scripts.planning.minsnap.time_optimized_snap import VariableTimeSnap
import torch


def fixed_solver():
    os.environ.update(SNAP_KINEMATICS_ONLY="1", SNAP_THRUST_SOC="0", SNAP_SOFT_ENVELOPE="0",
                      SNAP_TIME_SCALE="1", SNAP_SHORT_TIME_SCALE="1", SNAP_NODES="17",
                      SNAP_INITIAL_JERK="1", SNAP_PIECES="1", SNAP_PATH_ENERGY_WEIGHT="0")
    return MinimumSnapPlanner(configuration(), course())


def cost(plan):
    d = 840*np.diff(plan["coefficients"], n=4, axis=1)
    energy = np.einsum("eic,ij,ejc->e", d, VariableTimeSnap.gram, d)
    return float(plan["durations"].sum() + np.sum(energy/plan["coefficient_durations"]**7)/VariableTimeSnap.time_weight)


def generate_group(number, gate, allocator):
    rng = np.random.default_rng(83811 + 373*number + gate)
    centers, normals, _, axes = gate_pair(gate, 6)
    normal, side = normals[0], axes[0]
    mode = (number + gate) % 5
    distance = rng.uniform(*[(3, 6), (.12, .95), (1, 4), (2, 6), (.6, 3)][mode])
    x = np.zeros(19)
    box = np.array(course()["world_box"])
    x[:3] = centers[0]-distance*normal + side*rng.uniform(-.72,.72) + [0,0,rng.uniform(-.45,.45)]
    x[:3] = np.clip(x[:3], box[:3]+.15, box[3:]-.15)
    if mode:
        x[3:6] = normal*rng.uniform(*[(0,0),(2,11),(2,7),(6,13),(1,9)][mode]) + side*rng.uniform(-1.8,1.8) + [0,0,rng.uniform(-1.7,1.7)]
    yaw = np.arctan2(normal[1],normal[0]) + rng.uniform(-.45,.45)
    q = Rotation.from_euler("zyx",[yaw,*rng.uniform(-.32,.32,2)]).as_quat()
    x[6:10] = np.roll(q,1)
    x[10:13] = rng.uniform(-1.8,1.8,3)
    x[13:17] = MODEL["mass"]*MODEL["gravity"]/4 * rng.uniform(.78,1.25,4)
    peer = x.copy()
    peer[:3] += rng.uniform(-1.4,1.4,3)
    peer[3:6] += rng.uniform(-3,3,3)
    with torch.no_grad():
        obs = np.array([observation(x,peer,centers[i],normals[i],axes[i],centers[i+1],normals[i+1]) for i in range(5)])
        allocated = allocator(torch.tensor(obs)).numpy()
    spread = (qmc.LatinHypercube(5, seed=5330+gate).random(64)[number%64]-.5)*.9
    pattern = (number*7+gate)%33
    extreme = np.zeros(5) if pattern==32 else .45*np.array([1 if pattern & (1<<i) else -1 for i in range(5)])
    for kind, offsets in zip(("allocator","stratified","edge"),(allocated,spread,extreme)):
        yield dict(group=number*7+gate, gate=gate, mode=mode, kind=kind, own=x, offsets=offsets,
                   gates=centers[:5]+offsets[:,None]*axes[:5], normals=normals[:5],
                   half=np.tile([.075,.10],(5,1)))


def label(problem, solver):
    args = [problem[k] for k in ("own","gates","normals","half")]
    x,g,n,_ = args
    length = np.linalg.norm(np.diff(np.vstack([x[:3],g]),axis=0),axis=1)
    guess = np.maximum(.45,length/8.)
    guess[0] = np.clip(length[0]/max(3.,x[3:6]@n[0]),.07,2.)
    limits = [(.02,8.)]+[(.4,8.)]*4
    best = None
    scores = []
    stats = []
    cache = {}

    def evaluate(t):
        nonlocal best
        key = tuple(t)
        if key not in cache:
            try:
                plan = solver.solve(*args,t)
                j = cost(plan)
                if best is None or j < best[0]:
                    best = (j,t.copy())
                cache[key] = j
            except RuntimeError:
                cache[key] = 1e8
        return cache[key]

    for scale in (.87,1.42):
        seed = np.clip(guess*scale,[lo for lo,_ in limits],8.)
        res = minimize(evaluate, seed, method="COBYQA", bounds=limits,
                       options=dict(initial_tr_radius=.16,final_tr_radius=.003,maxfev=125))
        scores.append(float(res.fun))
        stats.append([res.status,res.nfev])
    if best is None:
        raise ValueError("No audited feasible candidate")
    # Tighten the best basin when restarts disagree or both exhaust the budget.
    if max(scores)-min(scores) > .01*best[0] or all(s[0] != 0 for s in stats):
        res = minimize(evaluate,best[1],method="COBYQA",bounds=limits,
                       options=dict(initial_tr_radius=.06,final_tr_radius=.001,maxfev=145))
        scores.append(float(res.fun))
        stats.append([res.status,res.nfev])
    plan = solver.solve(*args,best[1])
    if not np.isclose(cost(plan),best[0],rtol=2e-6):
        raise ValueError("Offline teacher replay mismatch")
    return dict(times=best[1],teacher_cost=best[0],coefficients=plan["coefficients"],
                passages=plan["optimized_passages"],eq=plan["audit"]["equality_error"],
                inequality=plan["audit"]["linear_inequality_error"],scores=scores,search=stats)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    p.add_argument("--groups-per-gate",type=int,default=36)
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--holdout",action="store_true")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    dst=a.output/"labels"
    dst.mkdir(exist_ok=True)
    solver=fixed_solver()
    allocator=load_allocator(FLIGHT_ASSETS/"models/allocation.pt")
    # Three paired waypoint layouts always stay on the same side of the split.
    for number in range(a.start,a.start+a.groups_per_gate):
        for gate in range(7):
            for prob in generate_group(number,gate,allocator):
                path=dst/f"{prob['group']:04d}_{prob['kind']}.npz"
                if path.exists():
                    continue
                try:
                    answer=label(prob,solver)
                except ValueError as e:
                    with (a.output/"rejected.jsonl").open("a") as f:
                        f.write(json.dumps(dict(group=prob["group"],kind=prob["kind"],reason=str(e)))+"\n")
                    continue
                split=("val" if number%6==2 else "test" if number%6==4 else "train")
                if a.holdout:
                    split="external"
                temp=path.with_suffix(".partial.npz")
                np.savez_compressed(temp,**prob,**answer,split=split)
                temp.replace(path)
        print(f"completed {number-a.start+1}/{a.groups_per_gate} group batches, {len(list(dst.glob('*.npz')))} labels",flush=True)


if __name__=="__main__":
    main()
