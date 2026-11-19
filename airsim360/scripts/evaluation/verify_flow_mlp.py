"""Replace only OccupancyFlow's displacement lookup and audit the matched 0.4 s events."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
from copy import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

from scripts.evaluation.evaluate_flight_count import make_probes, trajectory_at
from scripts.perception.occupancy_flow import FlowPrior, OccupancyFlow, compact_surface
from scripts.perception.panorama import spherical_rays
from scripts.perception.spatial_potential import camera_points
from scripts.training.train_occupancy_flow import motion_examples
from scripts.training.train_rgbd_field import Windows


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/"captures/cf21_512_double_50hz/manifest.json"
BASE = ROOT/"reports/spatial_potential/cf21_flow_field_01"
PEER_RADIUS = .085  # positive surface label used by the frozen appearance classifier
STEPS = np.arange(17)*5


def split_episodes(split):
    entries = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))["episodes"]
    entries = [row for row in entries if row["split"] == split]
    episodes = Windows(MANIFEST, split, 1, .25, lazy_windows=True).episodes
    return list(zip(entries, episodes, strict=True))


def measured_examples(records, prior):
    inputs, curves, counts = [], [], {}
    rng = np.random.default_rng(6106)
    for entry, episode in records:
        times, states, rgb, depth = episode[:4]
        start = len(inputs)
        for ego in (0, 1):
            previous = None
            peer = states[:, 1-ego, :3]
            for frame in range(0, len(times)-80, 4):
                own = states[frame, ego]
                pts, features = compact_surface(rgb[frame, ego], depth[frame, ego], own, True)
                pts = pts[prior.visible(features) >= .5]
                # The production matcher uses the oldest of the previous four 50 Hz
                # frames. Intermediate frames do not enter its velocity estimate.
                if previous is not None:
                    flow = OccupancyFlow()
                    flow.observe_points(times[frame-4], previous, states[frame-4, ego])
                    flow.observe_points(times[frame], pts, own)
                    ids = np.flatnonzero(np.linalg.norm(flow.xyz-peer[frame], axis=1) < PEER_RADIUS)
                    for i in rng.choice(ids, min(len(ids), 4), replace=False):
                        inputs.append(np.r_[flow.xyz[i], flow.flow[i]])
                        curves.append(peer[frame+STEPS]-peer[frame])
                previous = pts
        counts[entry["episode_id"]] = len(inputs)-start
        print("RGB-D motion examples", entry["episode_id"], counts[entry["episode_id"]], flush=True)
    return torch.tensor(np.array(inputs), dtype=torch.float32), torch.tensor(np.array(curves), dtype=torch.float32), counts


def train(output, training, validation, base):
    x, y, counts = measured_examples(training, base)
    vx, vy, val_counts = measured_examples(validation, base)
    with np.load(BASE/"flow.npz") as pack:
        clean_x = np.array(pack["keys"])
        clean_y = pack["delta"]
        appearance = {k: pack[k] for k in ("w0", "b0", "w1", "b1")}
    # Verify the clean teacher curves and their provenance against the eight train flights.
    clean = [motion_examples(ep[0], ep[1][:, 1-ego, :3]) for _, ep in training for ego in (0, 1)]
    np.testing.assert_allclose(clean_x, np.concatenate([c[0] for c in clean]))
    np.testing.assert_allclose(clean_y, np.concatenate([c[1] for c in clean]))
    clean_x[:, 3:] /= .10
    cx, cy = torch.tensor(clean_x, dtype=torch.float32), torch.from_numpy(clean_y)
    combined = torch.cat((x, cx))
    mean, std = combined.mean(0), combined.std(0).clamp_min(1e-4)
    x, cx, vx = [(a-mean)/std for a in (x, cx, vx)]
    net = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 48))
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    best, best_epoch = np.inf, 0
    epochs = []
    for epoch in range(1, 151):
        for ids in torch.randperm(len(x)).split(128):
            clean_ids = torch.randint(len(cx), (len(ids),))
            batch = torch.cat((x[ids], cx[clean_ids]))
            target = torch.cat((y[ids, 1:], cy[clean_ids, 1:]))
            pred = net(batch).reshape(-1, 16, 3).cumsum(1)
            loss = F.smooth_l1_loss(pred, target) + F.smooth_l1_loss(pred[:, 3], target[:, 3])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite displacement loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            estimate = net(vx).reshape(-1, 16, 3).cumsum(1)[:, 3]
            error = (estimate-vy[:, 4]).norm(dim=1).mean().item()
        epochs.append(error)
        if error < best:
            best, best_epoch = error, epoch
            weights = dict(appearance, motion_mean=mean.numpy(), motion_std=std.numpy())
            for i, layer in enumerate((net[0], net[2], net[4])):
                weights[f"motion_w{i}"] = layer.weight.detach().numpy().T
                weights[f"motion_b{i}"] = layer.bias.detach().numpy()
            np.savez_compressed(output/"flow_mlp.npz", **weights)
            # Check the actual NumPy deployment path before selecting this checkpoint.
            loaded = FlowPrior(output/"flow_mlp.npz")
            raw = (vx[:24]*std+mean).numpy()
            deployed = loaded.displacement(raw[:, :3], raw[:, 3:])
            np.testing.assert_allclose(deployed[:, 4], estimate[:24].numpy(), atol=2e-5)
        if epoch-best_epoch >= 25:
            break
    report = dict(seed=6106, architecture=[6,64,64,48], clean_training_curves=len(cx),
                  measured_train=counts, measured_validation=val_counts, best_epoch=best_epoch,
                  validation_mean_displacement_error_04_m=best, validation_error_history=epochs)
    (output/"fit.json").write_text(json.dumps(report, indent=2))
    print("selected epoch", best_epoch, "validation error", best, flush=True)


def score_observation(ep, probe, base, mlp):
    frame, ego, lead, grid = probe
    times, states, rgb, depth = ep[:4]
    baseline = OccupancyFlow(base)
    for k in range(max(0, frame-4), frame+1):
        baseline.observe(times[k], rgb[k, ego], depth[k, ego], states[k, ego])
    learned = copy(baseline)
    learned.prior = mlp
    if len(learned.xyz):
        learned.displacements = mlp.displacement(learned.xyz, learned.flow)
    peer = states[:, 1-ego, :3]
    distances = np.linalg.norm(baseline.xyz-peer[frame], axis=1)
    matched = distances < PEER_RADIUS
    ranges = depth[frame, ego]
    valid = np.isfinite(ranges) & (ranges>.08) & (ranges<18.)
    cloud = spherical_rays(*ranges.shape)[valid]*ranges[valid, None]
    peer_cam = camera_points(peer[frame:frame+1], states[frame, ego])[0]
    raw_hits = (np.linalg.norm(cloud-peer_cam, axis=1) < PEER_RADIUS)
    compact = compact_surface(rgb[frame, ego], ranges, states[frame, ego])
    compact_hits = np.linalg.norm(compact[:, :3]-peer[frame], axis=1) < PEER_RADIUS
    clearance, _ = cKDTree(cloud).query(camera_points(grid, states[frame, ego]))
    static = float(expit((.20-clearance)/.055).max())
    row = dict(observation_s=float(times[frame]), lead_s=float(lead), points=len(distances),
               peer_points=int(matched.sum()), closest_point_m=float(distances.min()) if len(distances) else None,
               raw_peer_points=int(raw_hits.sum()), compact_peer_points=int(compact_hits.sum()), static=static)
    if not row["raw_peer_points"] >= row["compact_peer_points"] >= row["peer_points"]:
        raise AssertionError("Inconsistent raw / compact / classified peer masks")
    for name, flow in (("retrieval", baseline), ("mlp", learned)):
        dynamic = float(flow.score(grid, lead).max())
        score = dict(future=max(static, dynamic), dynamic=dynamic, now=max(static, float(flow.score(grid, 0.).max())))
        if matched.any():
            # Attribute risk to verified peer returns without changing operational scores.
            positive_only = copy(flow)
            positive_only.xyz = flow.xyz[matched]
            positive_only.displacements = flow.displacements[matched]
            score["peer_only"] = float(positive_only.score(grid, lead).max())
            idx = distances.argmin()
            predicted = flow.xyz[idx]+flow.displacements[idx, 4]
            score["peer_center_error_04_m"] = float(np.linalg.norm(predicted-trajectory_at(times, peer, times[frame]+.4)))
        row[name] = score
    return row


def events(records, base, mlp, split):
    rows = []
    for entry, ep in records:
        event_file = (MANIFEST.parent/entry["trajectory"]).with_name("events.json")
        _, probes = make_probes(ep, json.loads(event_file.read_text()))
        for horizon, gate, stamp, occupied, empty in probes:
            if horizon == .4:
                row = dict(flight=entry["episode_id"], gate=gate, event_s=stamp, ego=occupied[1])
                row["entry"] = score_observation(ep, occupied, base, mlp)
                row["empty"] = score_observation(ep, empty, base, mlp)
                rows.append(row)
    # Event-for-event check against the published baseline, including the empty pair.
    old = json.loads((BASE/f"{split}.json").read_text())["rows"]
    for row, ref in zip(rows, old, strict=True):
        assert (row["gate"], row["ego"], row["event_s"]) == (ref["gate"], ref["observer"], ref["event_s"])
        np.testing.assert_allclose([row[k]["retrieval"]["future"] for k in ("entry", "empty")],
                                   [ref["risk_at_entry"], ref["risk_empty"]], rtol=1e-6, atol=1e-7)
    return rows


def summarize(rows, thresholds):
    seen = [r for r in rows if r["entry"]["peer_points"]]
    summary = dict(events=len(rows), raw_peer_seen=sum(r["entry"]["raw_peer_points"]>0 for r in rows),
                   compact_peer_seen=sum(r["entry"]["compact_peer_points"]>0 for r in rows), seen=len(seen))
    hits = {}
    for model, threshold in thresholds.items():
        def hit(row, key="entry"):
            return row[key][model]["future"] >= threshold
        hits[model] = [hit(r) for r in seen]
        summary[model] = dict(threshold=threshold, hits=sum(hit(r) for r in rows),
            empty_alarms=sum(hit(r, "empty") for r in rows), hits_when_seen=sum(hits[model]),
            true_peer_alarms=sum(r["entry"][model]["peer_only"] >= threshold for r in seen),
            newly_high=sum(hit(r) and r["entry"][model]["now"]<threshold for r in rows),
            mean_center_error_04_m=float(np.mean([r["entry"][model]["peer_center_error_04_m"] for r in seen])))
    summary["seen_hits_gained"] = sum(b and not a for a,b in zip(hits["retrieval"], hits["mlp"]))
    summary["seen_hits_lost"] = sum(a and not b for a,b in zip(hits["retrieval"], hits["mlp"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(6106)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = FlowPrior(BASE/"flow.npz")
    training, validation = split_episodes("train"), split_episodes("validation")
    train(args.out, training, validation, base)
    model = FlowPrior(args.out/"flow_mlp.npz")
    val = events(validation, base, model, "validation")
    names = ("retrieval", "mlp")
    thresholds = {n: float(np.nextafter(sorted((r["empty"][n]["future"] for r in val), reverse=True)[2], np.inf))
                  for n in names}
    test = events(split_episodes("test"), base, model, "test")
    (args.out/"events.json").write_text(json.dumps(dict(validation=val, test=test), indent=2))
    report = {}
    for name, cuts in (("fixed_half", dict.fromkeys(names, .5)), ("validation_2_empty", thresholds)):
        report[name] = dict(validation=summarize(val, cuts), test=summarize(test, cuts))
    (args.out/"comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["validation_2_empty"]["test"]), flush=True)


if __name__ == "__main__":
    main()
