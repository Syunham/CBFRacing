"""Compare frozen and retrained time regressors on identical fixed-time QPs."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import copy
import json
import math
from shutil import copyfile
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.planning.minsnap.snap_time_network import FixedTimePredictor, TimeNet, features
from scripts.training.snap_transition_study import BASE, FAILED
from scripts.training.synthetic_snap_teacher import course, fixed_solver
from scripts.training.train_snap_times import fit, inputs, load_rows


def wide_fit(rows, train, valid, path):
    """Expand the pretrained MLP, retaining its initial function on all inputs."""
    torch.manual_seed(1973)
    old = FixedTimePredictor(BASE/"best.pt").net
    widths = [320, 320, 256, 192]
    model = TimeNet(len(old.center), widths)
    source = [m for m in old.layers if isinstance(m, nn.Linear)]
    dest = [m for m in model.layers if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for name, buffer in old.named_buffers():
            getattr(model, name).copy_(buffer)
        previous = torch.arange(len(old.center))
        for layer, expanded in zip(source, dest):
            current = torch.arange(expanded.out_features) % layer.out_features
            multiplicity = torch.bincount(previous)[previous]
            weights = layer.weight[current][:, previous]/multiplicity
            expanded.weight.copy_(weights)
            expanded.bias.copy_(layer.bias[current])
            previous = current
        x = torch.from_numpy(np.stack([features(*inputs(r), "ballistic") for r in rows]))
        torch.testing.assert_close(model(x[:32]), old(x[:32]), atol=2e-5, rtol=2e-5)
        # Break replica symmetry before fine-tuning.
        for layer in dest[:-1]:
            layer.weight.add_(.0005*torch.randn_like(layer.weight))
    y = torch.tensor(np.stack([r["times"] for r in rows]), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=.00035, weight_decay=.003)
    transition = torch.tensor([str(r["kind"]) == "transition" for r in rows])
    best, epoch_best = float("inf"), 0
    for epoch in range(650):
        for ids in torch.tensor(train)[torch.randperm(len(train))].split(80):
            pred = model(x[ids])
            log_loss = F.smooth_l1_loss(pred, y[ids].log(), beta=.22, reduction="none")
            sec_loss = F.smooth_l1_loss(pred.clamp(max=2.08).exp(), y[ids], beta=.16, reduction="none")
            losses = log_loss + .55*sec_loss
            loss = (losses.mean(1)+.45*losses[:, 0]).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        if epoch in (80, 200, 400):
            for group in opt.param_groups:
                group["lr"] *= .45
        if epoch%5 == 0:
            with torch.no_grad():
                err = (model.durations(x[valid])-y[valid]).abs()
                def score(e):
                    return e.mean()+.55*e[:, 0].mean()+.15*torch.quantile(e[:, 0], .9)
                local = transition[valid]
                metric = float((score(err[local])+score(err[~local]))/2)
            if metric < best:
                best, epoch_best = metric, epoch
                state = copy.deepcopy(model.state_dict())
            if epoch-epoch_best >= 100:
                break
    torch.save(dict(geometry="ballistic", dimension=len(old.center), widths=widths, state=state), path)
    return dict(best_epoch=epoch_best, validation_score=best)


@lru_cache(maxsize=1)
def crossing_geometry():
    matrix = np.zeros((8, 8))
    for k in range(8):
        for j in range(k, 8):
            matrix[j, k] = math.comb(7,k)*math.comb(7-k,j-k)*(-1)**(j-k)
    return course(), matrix


def early_crossings(coeff, gate, durations):
    # Exact polynomial roots within each leg, excluding the designated endpoint.
    geom, matrix = crossing_geometry()
    events = []
    for leg, control in enumerate(coeff):
        index = (gate+leg)%7
        n = np.asarray(geom["normals"][index])
        poly = matrix@(control-geom["centers"][index])
        plane = np.polynomial.Polynomial(poly@n)
        for root in plane.roots():
            u = root.real
            if abs(root.imag)>1e-7 or not 1e-7<u<1-1e-6 or plane.deriv()(u)<=0:
                continue
            delta = np.polynomial.polynomial.polyval(u, poly)
            xy = float(delta@np.cross([0,0,1], n))
            events.append(dict(leg=leg, gate=index+1, seconds=float(u*durations[leg]),
                               height=float(delta[2]+geom["centers"][index][2]), offset=max(abs(xy),abs(delta[2]))))
    return events


def evaluate_case(row, predictor, qp):
    args = inputs(row)
    times = predictor.predict(*args)
    record = dict(times=times.tolist(), teacher=row["times"].tolist(), group=int(row["group"]))
    try:
        plan = qp.solve(*args, times)
        np.testing.assert_allclose(times, plan["durations"], rtol=0, atol=1e-10)
        record.update(solved=True, crossings=early_crossings(plan["coefficients"], int(row["gate"]), times))
    except RuntimeError as exc:
        record.update(solved=False, error=str(exc))
    return record


def summarize(records):
    err = np.abs(np.array([r["times"] for r in records])-np.array([r["teacher"] for r in records]))
    solved = [r for r in records if r["solved"]]
    return dict(count=len(records), solved=len(solved), mae=float(err.mean()), first_mae=float(err[:,0].mean()),
                first_p95=float(np.percentile(err[:,0],95)),
                aperture_075=sum(any(e["offset"]>.75 for e in r["crossings"]) for r in solved),
                aperture_045=sum(any(e["offset"]>.45 for e in r["crossings"]) for r in solved))


def study(folder):
    torch.set_num_threads(1)
    rows = load_rows(BASE)+load_rows(folder)
    sets = {s:np.array([i for i,r in enumerate(rows) if str(r["split"])==s])
            for s in ("train","val","test","external")}
    groups = [set(int(rows[i]["group"]) for i in ids) for ids in sets.values()]
    assert all(a.isdisjoint(b) for i,a in enumerate(groups) for b in groups[i+1:])
    assert sum(str(r["kind"])=="transition" for r in rows)==228
    info = {}
    info["dense_original_size"] = fit(rows,sets["train"],sets["val"],[160,160,128,96],"ballistic",folder/"dense.pt")
    print("trained original-size control", flush=True)
    info["dense_wide"] = wide_fit(rows,sets["train"],sets["val"],folder/"wide.pt")
    copyfile(folder/"wide.pt",folder/"best.pt")
    print("trained expanded model", flush=True)
    qp = fixed_solver()
    execution = np.load(FAILED/"flight/execution.npz")
    counter = json.loads((FAILED/"gate45_time_comparison.json").read_text())
    records, report = {}, dict(training=info, splits={s:len(i) for s,i in sets.items()}, models={})
    teacher_checks = []
    for file in sorted((folder/"labels").glob("*.npz")):
        with np.load(file) as r:
            if str(r["split"]) == "test":
                teacher_checks.append(early_crossings(r["coefficients"],int(r["gate"]),r["times"]))
    report["transition_teacher_crossings"] = {
        str(h):sum(any(e["offset"]>h for e in ev) for ev in teacher_checks) for h in (.45,.75)}
    for name,path in [("old",BASE/"best.pt"),("dense_original_size",folder/"dense.pt"),("dense_wide",folder/"wide.pt")]:
        pred = FixedTimePredictor(path)
        subsets = {}
        stored = {}
        for subset, ids in [("original",sets["test"]),("external",sets["external"]),("transition",sets["test"])]:
            ids = [i for i in ids if (str(rows[i]["kind"])=="transition")==(subset=="transition")]
            stored[subset] = [evaluate_case(rows[i],pred,qp) for i in ids]
            subsets[subset] = summarize(stored[subset])
        replay = {}
        for plan_id, stamp, ix in [(79,7.64,3),(81,7.72,4)]:
            artifact = np.load(FAILED/f"flight/plans/{plan_id:03d}_agent1_event{ix}.npz")
            row = dict(own=execution["state"][abs(execution["time"]-stamp).argmin(),1],
                       gates=artifact["learned_gates"], normals=artifact["normals"], half=artifact["half"],
                       group=-1, gate=ix, times=np.array(counter[str(plan_id)]["teacher_times"]))
            replay[str(stamp)] = evaluate_case(row,pred,qp)
        report["models"][name]=dict(parameters=sum(v.numel() for v in pred.net.parameters()),subsets=subsets,replay=replay)
        records[name]=stored
        print(name,subsets,flush=True)
    (folder/"test_records.json").write_text(json.dumps(records))
    (folder/"results.json").write_text(json.dumps(report,indent=2))
