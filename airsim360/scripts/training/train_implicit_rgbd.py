"""Train and compare continuous RGB-D occupancy fields on disjoint recorded flights."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.evaluation.evaluate_flight_count import make_probes, trajectory_at
from scripts.perception.implicit_rgbd import HISTORIES, HORIZON, RADIUS, ImplicitRGBD, camera_matrices, history_indices
from scripts.training.train_rgbd_field import ProjectedRGBDField, Windows, image_history


class ProjectedBaseline(ProjectedRGBDField):
    """Original weights layout and latest-frame projection, using the shared dynamic targets."""
    def forward(self, images, origins, rotations, times, world, tau):
        camera = (world-origins[:, -1, None]) @ rotations[:, -1]
        return self.query_logits(self.encode(images), camera, tau)


def gate_examples(episode):
    times, states, _, _, _, gates, eligible = episode
    bars = gates.centers.reshape(-1, 4, 3)
    pools = {key: [] for key in ("arrival", "departure", "empty", "side")}
    for ego in (0, 1):
        peer = states[:, 1-ego, :3]
        now = peer[eligible]
        future = trajectory_at(times, peer, times[eligible]+.4)
        for frame_bars in bars:
            center = frame_bars.mean(0)
            axis = frame_bars[1]-frame_bars[0]
            axis /= np.linalg.norm(axis)
            near = np.linalg.norm(states[eligible, ego, :3]-center, axis=1) < 7
            # Use a fixed location in the aperture for each pair of time labels.
            for key, target in (("arrival", future), ("departure", now), ("empty", None), ("side", None)):
                q = np.broadcast_to(center, now.shape).copy()
                if target is not None:
                    q += np.clip((target-center) @ axis, -.48, .48)[:, None]*axis
                    q[:, 2] += np.clip(target[:, 2]-center[2], -.48, .48)
                d0 = np.linalg.norm(now-q, axis=1)
                d1 = np.linalg.norm(future-q, axis=1)
                if key == "arrival":
                    selected = (d0 > .65) & (d1 < .45)
                elif key == "departure":
                    selected = (d0 < .45) & (d1 > .65)
                elif key == "empty":
                    selected = (d0 > 1.35) & (d1 > 1.35)
                else:
                    lateral = abs((future-center) @ axis)
                    selected = (d0 > .65) & (d1 > .65) & (d1 < 1.35) & (lateral > .55)
                pools[key].extend((eligible[i], ego, q[i]) for i in np.flatnonzero(selected & near))
    return pools


def observation(episode, frame, ego, history):
    times, states, rgb, depth, valid, _, _ = episode
    ids = history_indices(times, times[frame], HISTORIES[history])
    poses = states[ids, ego]
    images = image_history(rgb[ids, ego], depth[ids, ego], valid[ids, ego])
    return (images, *(torch.as_tensor(v, dtype=torch.float32) for v in
                      (poses[:, :3], camera_matrices(poses), times[ids])))


class PeerWindows(Windows):
    def __init__(self, manifest, split, samples, history):
        super().__init__(manifest, split, samples, .25, lazy_windows=True)
        self.history = history
        self.pools = [gate_examples(ep) for ep in self.episodes]
        if any(ep[2].shape[-3:-1] != (256, 512) for ep in self.episodes):
            raise ValueError("Expected native 512 x 256 panorama recordings")

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed+self.samples*self.epoch+i)
        e = i % len(self.episodes)
        episode = self.episodes[e]
        times, states, _, _, _, _, eligible = episode
        # Keep broad spatial queries even in targeted windows.
        pool = self.pools[e].get(("general", "arrival", "departure", "empty", "side")[
            (i//len(self.episodes)+self.epoch) % 5])
        if pool:
            frame, ego, center = pool[rng.integers(len(pool))]
        else:
            frame, ego = rng.choice(eligible), rng.integers(2)
            center = states[frame, 1-ego, :3]
        peer = states[:, 1-ego, :3]
        tau = rng.uniform(0, HORIZON, 32)
        tau[:12] = np.tile([0., .4], 6)
        future = trajectory_at(times, peer, times[frame]+tau)
        direction = rng.normal(size=(32, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        q = future + direction * rng.uniform(0, 1.25, (32, 1))
        q[-4:] = states[frame, ego, :3]+rng.uniform(-5, 5, (4, 3))
        paired = center+rng.uniform(-.035, .035, (6, 3))
        paired[0] = center
        q[:12] = np.repeat(paired, 2, axis=0)
        labels = np.linalg.norm(q-future, axis=1) <= RADIUS
        return (*observation(episode, frame, ego, self.history),
                *(torch.as_tensor(v, dtype=torch.float32) for v in (q, tau[:, None], labels)))


def speed_bound(episodes):
    speeds = []
    for times, states, *_ in episodes:
        delta = np.diff(states[:, :, :3], axis=0)/np.diff(times)[:, None, None]
        speeds.extend(np.linalg.norm(delta, axis=2).ravel())
    return float(np.quantile(speeds, .995))


def train(args, variant, train_set, val_set, speed):
    folder = args.output/variant
    folder.mkdir()
    torch.manual_seed(args.seed)
    model = (ProjectedBaseline() if variant == "P" else ImplicitRGBD(variant, speed)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    best = float("inf")
    for epoch in range(args.epochs):
        train_set.epoch = epoch
        metrics = {}
        for training, dataset in ((True, train_set), (False, val_set)):
            model.train(training)
            total = 0.
            with torch.set_grad_enabled(training):
                for batch in DataLoader(dataset, batch_size=args.batch_size):
                    *inputs, labels = (v.to(args.device) for v in batch)
                    logits = model(*inputs)
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite occupancy loss")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                        optimizer.step()
                    total += loss.item()*len(labels)
                    if args.rest_ms:
                        time.sleep(args.rest_ms/1000)
            metrics["train_bce" if training else "val_bce"] = total/len(dataset)
        print(variant, epoch+1, metrics, flush=True)
        with (folder/"epochs.jsonl").open("a") as stream:
            print(json.dumps(dict(epoch=epoch+1, **metrics)), file=stream)
        if metrics["val_bce"] < best:
            best = metrics["val_bce"]
            path = folder/"best.pt"
            tmp = path.with_suffix(".tmp")
            checkpoint_format = "projected_dynamic_baseline_v1" if variant == "P" else "implicit_rgbd_v1"
            torch.save(dict(format=checkpoint_format, model=model.state_dict(), variant=variant,
                            history=args.history, radius=RADIUS, speed_bound=speed, epoch=epoch+1,
                            val_bce=best, manifest=str(args.manifest.resolve())), tmp)
            tmp.replace(path)
    model.load_state_dict(torch.load(folder/"best.pt", weights_only=True, map_location=args.device)["model"])
    return model.eval()


def event_sets(manifest, split):
    ds = Windows(manifest, split, 1, .25, lazy_windows=True)
    entries = [e for e in json.loads(manifest.read_text(encoding="utf-8-sig"))["episodes"] if e["split"] == split]
    groups = []
    for episode, entry in zip(ds.episodes, entries):
        path = (manifest.parent/entry["trajectory"]).with_name("events.json")
        _, probes = make_probes(episode, json.loads(path.read_text()))
        groups.append((entry["episode_id"], episode, [p for p in probes if p[0] == .4]))
    return groups


@torch.no_grad()
def event_scores(model, groups, args):
    rows = []
    for flight, episode, probes in groups:
        times, states = episode[:2]
        def score(frame, ego, grid, lead):
            inputs = [v[None].to(args.device) for v in observation(episode, frame, ego, args.history)]
            grid = torch.as_tensor(grid, dtype=torch.float32, device=args.device)[None]
            q = torch.cat((grid, grid), dim=1)
            tau = torch.zeros_like(q[..., :1])
            tau[:, grid.shape[1]:] = lead
            now, later = model(*inputs, q, tau).sigmoid()[0].chunk(2)
            return float(now.max()), float(later.max())
        for _, gate, event, (frame, ego, lead, grid), (empty, _, _, _) in probes:
            now, arrival = score(frame, ego, grid, lead)
            empty_now, clear = score(empty, ego, grid, lead)
            row = dict(flight=flight, gate=gate, ego=ego, event_s=event, arrival=arrival,
                       now=now, empty=clear, empty_now=empty_now)
            leaving = int(np.abs(times-event).argmin())
            if times[leaving]+.4 <= times[-1]:
                peer = states[:, 1-ego, :3]
                future = trajectory_at(times, peer, times[leaving]+.4)
                if (np.linalg.norm(grid-peer[leaving], axis=1).min() <= RADIUS and
                        np.linalg.norm(grid-future, axis=1).min() > RADIUS):
                    row["depart_now"], row["depart_future"] = score(leaving, ego, grid, .4)
            rows.append(row)
    if not rows:
        raise ValueError("No matched 0.4 s gate events")
    return rows


def operating_point(validation, test, fpr, frozen):
    pos, neg = ("now", "empty_now") if frozen else ("arrival", "empty")
    alarms = sorted((v[neg] for v in validation), reverse=True)
    allowed = int(fpr*len(alarms))
    threshold = float(np.nextafter(alarms[allowed], np.inf))
    def counts(rows):
        departed = [r for r in rows if "depart_future" in r]
        return dict(events=len(rows), hits=sum(r[pos] >= threshold for r in rows),
                    empty_alarms=sum(r[neg] >= threshold for r in rows),
                    now_low_future_high=sum(r["now"] < threshold <= r[pos] for r in rows),
                    departures=len(departed), departure_alarms=sum(
                        r["depart_now" if frozen else "depart_future"] >= threshold for r in departed))
    return dict(threshold=threshold, validation=counts(validation), test=counts(test))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("captures/cf21_512_double_50hz/manifest.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variants", nargs="+", choices=list("PABC"), default=list("ABC"),
                   help="P is the original ProjectedRGBDField baseline; A/B/C share the lightweight encoder")
    p.add_argument("--history", choices=HISTORIES, default="short")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--val-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--rest-ms", type=int, default=20)
    p.add_argument("--seed", type=int, default=6106)
    p.add_argument("--fpr", type=float, default=.10)
    args = p.parse_args()
    if min(args.epochs, args.samples, args.val_samples, args.batch_size) < 1 or not 0 <= args.fpr < 1 or args.rest_ms < 0:
        p.error("Invalid sample counts, rest duration or false alarm rate")
    if len(set(args.variants)) != len(args.variants):
        p.error("Duplicate variant")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    train_set, val_set = [PeerWindows(args.manifest, split, n, args.history) for split, n in
                          (("train", args.samples), ("validation", args.val_samples))]
    train_set.seed = args.seed
    speed = speed_bound(train_set.episodes)
    validation, test = [event_sets(args.manifest, split) for split in ("validation", "test")]
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), train_flights=len(train_set.episodes), speed_bound_mps=speed,
                  gate_pool_counts=[{k: len(v) for k, v in pools.items()} for pools in train_set.pools])
    (args.output/"config.json").write_text(json.dumps(config, default=str, indent=2))
    comparison = {}
    for variant in args.variants:
        model = train(args, variant, train_set, val_set, speed)
        val_rows, test_rows = [event_scores(model, group, args) for group in (validation, test)]
        (args.output/variant/"events.json").write_text(json.dumps(dict(validation=val_rows, test=test_rows), indent=2))
        comparison[variant] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                   forecast=operating_point(val_rows, test_rows, args.fpr, False),
                                   frozen=operating_point(val_rows, test_rows, args.fpr, True))
        (args.output/"comparison.json").write_text(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
