"""Train an aperture-specific 0.4 s crossing alert alongside the frozen point field."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import functional as F

from scripts.evaluation.evaluate_flight_count import make_probes, trajectory_at
from scripts.training.train_rgbd_field import GateFrames, image_history


def load_split(manifest, split):
    entries = json.loads(manifest.read_text(encoding="utf-8-sig"))["episodes"]
    frames, targets, pairs, counts = [], [], [], {}
    for entry in entries:
        if entry["split"] != split:
            continue
        trajectory = manifest.parent / entry["trajectory"]
        with np.load(trajectory) as f:
            times, states = f["time"], f["state"]
        events = json.loads(trajectory.with_name("events.json").read_text())
        gates = GateFrames(entry["gate_boxes"])
        episode = (times, states, None, None, None, gates, [])
        _, probes = make_probes(episode, events)
        with np.load(manifest.parent / entry["observations"]) as f:
            assert np.allclose(f["time"], times, atol=1e-4, rtol=0)
            rgb, depth, valid = f["rgb"], f["depth"], f["valid"]
        cache = {}
        count_before = len(targets)
        for horizon, gate, event, positive, negative in probes:
            if horizon != .4:
                continue
            for offset in (range(-2, 3) if split == "train" else (0,)):
                k, ego, lead, grid = positive
                j = negative[0]
                k, j = k + offset, j + offset
                if min(k, j) < 3 or max(k, j) >= len(times) or times[j] + lead > times[-1]:
                    continue
                if not (valid[k-3:k+1, ego].all() and valid[j-3:j+1, ego].all()):
                    continue
                neighbour = states[:, 1-ego, :3]
                def clearance(pos):
                    return np.linalg.norm(pos - grid, axis=-1).min()
                if min(clearance(neighbour[k]), clearance(neighbour[j])) <= .55:
                    continue
                if clearance(trajectory_at(times, neighbour, times[j] + lead)) <= 1.1:
                    continue
                # Reject a second crossing of this gate anywhere inside the label tolerance.
                if any(e["gate"] == gate and abs(e["time"] - times[j] - .4) <= .051
                       for e in events[1-ego]):
                    continue
                for idx, target in ((k, 1.), (j, 0.)):
                    pose = states[idx, ego]
                    center = Rotation.from_quat(pose[[7, 8, 9, 6]]).inv().apply(
                        grid[4] - pose[:3]) * [1, -1, -1]
                    # Keep the entire panorama. Apply one common roll to all four
                    # images so that motion is preserved and the gate is at mid-azimuth.
                    roll = -int(np.rint(np.arctan2(center[1], center[0]) / (2*np.pi) * 128))
                    if (idx, ego) not in cache:
                        sl = slice(idx-3, idx+1)
                        images = image_history(rgb[sl, ego], depth[sl, ego], valid[sl, ego])
                        cache[idx, ego] = F.avg_pool2d(images, 2).half()
                    frames.append(cache[idx, ego].roll(roll, -1))
                    targets.append(target)
                pairs.append(dict(flight=entry["episode_id"], gate=gate, observer=ego,
                                  event_s=float(event), observation_s=float(times[k]),
                                  empty_observation_s=float(times[j]), horizon_s=float(event-times[k])))
        counts[entry["episode_id"]] = (len(targets)-count_before)//2
        print(split, entry["episode_id"], counts[entry["episode_id"]], "pairs", flush=True)
        del rgb, depth, cache
    if not all(counts.values()):
        raise ValueError(f"Missing training or evaluation data: {counts}")
    return torch.stack(frames), torch.tensor(targets), pairs, counts


class GateWarning(nn.Module):
    """Inputs are four gate-centered RGB-D/validity frames at 50 Hz."""
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout in ((20, 24), (24, 32), (32, 48), (48, 64)):
            layers.extend((nn.Conv2d(cin, cout, 3, 2, 1), nn.GroupNorm(4, cout), nn.SiLU()))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(),
                                  nn.Linear(512, 64), nn.SiLU(), nn.Dropout(.15), nn.Linear(64, 1))

    def forward(self, images, static=False):
        latest = images[:, -1]
        differences = images.diff(dim=1).flatten(1, 2)
        if static:
            differences = torch.zeros_like(differences)
        features = torch.cat((latest, (differences * 4).clamp(-1, 1)), dim=1)
        return self.head(self.features(features)).flatten()


@torch.inference_mode()
def predict(model, data, static):
    model.eval()
    return torch.cat([model(x.cuda().float(), static).cpu() for x in data.split(32)])


def fit(train, validation, static, output, epochs, seed):
    torch.manual_seed(seed)
    model = GateWarning().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0006, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.00006)
    best = float("inf")
    history = []
    name = "last_frame" if static else "temporal"
    for epoch in range(1, epochs+1):
        model.train()
        total = 0.
        for idx in torch.randperm(len(train[1])).split(32):
            images, labels = train[0][idx].cuda().float(), train[1][idx].cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(images, static), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()
        logits = predict(model, validation[0], static)
        val = F.binary_cross_entropy_with_logits(logits, validation[1]).item()
        history.append(dict(epoch=epoch, train_bce=total/len(train[1]), validation_bce=val))
        if val < best:
            best = val
            torch.save(dict(model=model.state_dict(), epoch=epoch, validation_bce=val,
                            static=static), output / f"{name}.pt")
        if epoch == 1 or epoch % 5 == 0:
            print(name, history[-1], flush=True)
    (output / f"{name}_history.json").write_text(json.dumps(history, indent=2))


def metrics(pos, neg):
    return dict(events=len(pos), hits=int((pos >= .5).sum()), empty_alarms=int((neg >= .5).sum()),
                detection_rate=float((pos >= .5).mean()), empty_false_alarm_rate=float((neg >= .5).mean()),
                occupied_above_empty=float((pos > neg).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/flight_count_comparison/manifest_8.json"))
    parser.add_argument("--baseline", type=Path, default=Path("reports/projected_rgbd_8"))
    parser.add_argument("--output", type=Path, default=Path("reports/temporal_gate_warning_8"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    seed = 6106
    train = load_split(args.manifest, "train")
    validation = load_split(args.manifest, "validation")
    result = dict(manifest=str(args.manifest.resolve()), threshold=.5, seed=seed,
                  epochs=args.epochs, train_pairs=train[3], validation_pairs=validation[3],
                  frozen_field=str((args.baseline / "best.pt").resolve()),
                  field_sha256=hashlib.sha256((args.baseline / "best.pt").read_bytes()).hexdigest())
    for static in (False, True):
        fit(train, validation, static, args.output, args.epochs, seed)
    del train
    images, labels, events, result["test_pairs"] = load_split(args.manifest, "test")
    baseline = json.loads((args.baseline / "events_0p4.json").read_text())["projected_8"]
    assert len(events) == len(baseline)
    for event, old in zip(events, baseline):
        for key in ("gate", "observer", "event_s", "observation_s", "empty_observation_s", "horizon_s"):
            assert abs(event[key]-old[key]) < 1e-8, key
        event["projected_8"] = [old["risk_at_entry"], old["risk_empty"]]
    result["gate"] = json.loads((args.baseline / "results.json").read_text())["projected_8"]["gate"]
    result["metrics"] = {"projected_8": metrics(*np.array([e["projected_8"] for e in events]).T)}
    for name, checkpoint, static in (("temporal", "temporal", False), ("repeat_latest", "temporal", True),
                                      ("last_frame", "last_frame", True)):
        saved = torch.load(args.output / f"{checkpoint}.pt", weights_only=True)
        model = GateWarning().cuda()
        model.load_state_dict(saved["model"])
        scores = predict(model, images, static).sigmoid().reshape(-1, 2).numpy()
        assert np.isfinite(scores).all() and torch.all(labels.reshape(-1, 2) == torch.tensor([1., 0.]))
        result["metrics"][name] = dict(**metrics(*scores.T), epoch=saved["epoch"],
                                       validation_bce=saved["validation_bce"])
        for event, pair in zip(events, scores):
            event[name] = pair.tolist()
    (args.output / "events_0p4.json").write_text(json.dumps(events, indent=2))
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
