"""Fit the spatial-only risk potential using disjoint RGB-D flights."""

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
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from scripts.perception.spatial_potential import PotentialNet, SurfaceObservation, camera_points
from scripts.training.train_rgbd_field import GateFrames


def dataset(manifest, split, stride, paired_manifest=None):
    rng = np.random.default_rng(6106 if split == "train" else 39)
    xs, ds, ys, groups = [], [], [], []
    meta = json.loads(manifest.read_text(encoding="utf-8-sig"))
    geometry = meta.get("vehicle_geometry")
    sampling = meta.get("trajectory_stride", 1)
    if stride % sampling:
        raise ValueError("Requested image history cadence unavailable in dataset")
    gate_margin, peer_margin = .25, .55
    if geometry:
        radius, buffer = geometry["body_radius_m"], geometry["label_buffer_m"]
        gate_margin, peer_margin = radius + buffer, 2*radius + buffer
    query_sigma = .7*peer_margin if geometry else .45
    entries = meta["episodes"]
    paired = {} if paired_manifest is None else {
        e["episode_id"]: e for e in json.loads(paired_manifest.read_text(encoding="utf-8-sig"))["episodes"]}
    for entry in entries:
        if entry["split"] != split:
            continue
        with np.load(manifest.parent / entry["trajectory"]) as f:
            states, times = f["state"], f["time"]
        path = manifest.parent / entry["observations"]
        take = 1
        if paired_manifest is not None:
            src = paired[entry["episode_id"]]
            if src["split"] != split or src["trajectory_sha256"] != entry["trajectory_sha256"]:
                raise ValueError("Paired render uses different ground truth")
            path, take = paired_manifest.parent/src["observations"], sampling
        with np.load(path) as f:
            rgb, depth, valid = (f[key][::take] for key in ("rgb","depth","valid"))
            np.testing.assert_allclose(times[::sampling], f["time"][::take], atol=1e-4, rtol=0)
        gates = GateFrames(entry["gate_boxes"])
        bars = gates.centers.reshape(-1, 4, 3)
        step = 8 if split == "train" else (5 if sampling == 1 else sampling)
        for k in range(3*stride, len(times)-7, step):
            for ego in range(2):
                h = slice((k-3*stride)//sampling, k//sampling+1, stride//sampling)
                if not valid[h, ego].all():
                    continue
                pose = states[k, ego]
                other = states[k:k+7, 1-ego, :3]
                g = int(np.linalg.norm(bars.mean(1)-pose[:3], axis=1).argmin())
                center = bars[g].mean(0)
                direction = bars[g, 1]-bars[g, 0]
                direction /= np.linalg.norm(direction)
                offsets = rng.uniform(-.55, .55, (16, 2))
                opening = center + offsets[:, :1]*direction + offsets[:, 1:]*[0, 0, 1]
                q = np.vstack((gates.sample(rng, pose[:3], 64),
                               other[rng.integers(len(other), size=64)] + rng.normal(0, query_sigma, (64, 3)),
                               pose[:3]+rng.uniform(-5, 5, (24, 3)), bars[g], opening))
                gate_distance = gates.distances(q)
                neighbor_distance = np.linalg.norm(q[:, None]-other, axis=-1).min(1)
                # One field pools a fixed 120 ms swept volume for reaction reserve.
                # Only labels access the simulator trajectory or gate boxes.
                logit = np.maximum((gate_margin-gate_distance)/(.045 if geometry else .075),
                                   (peer_margin-neighbor_distance)/(.05 if geometry else .10))
                group = np.full(len(q), -1, np.int8)
                group[-20:-16] = 1
                group[-16:][(neighbor_distance[-16:] > 1.1) & (gate_distance[-16:] > .32)] = 0
                peer = neighbor_distance[64:128] < (peer_margin-.03 if geometry else .45)
                if geometry:
                    peer &= gate_distance[64:128] > gate_margin+.10
                group[64:128][peer] = 2
                obs = SurfaceObservation(rgb[h, ego], depth[h, ego])
                dist, idx = obs.neighbors(camera_points(q, pose))
                xs.append(obs.features[idx].astype(np.float16))
                ds.append(dist.astype(np.float32))
                ys.append((1/(1+np.exp(-logit.clip(-70, 70)))).astype(np.float32))
                groups.append(group)
        print(split, entry["episode_id"], flush=True)
    return [torch.from_numpy(np.concatenate(a)) for a in (xs, ds, ys, groups)]


@torch.inference_mode()
def evaluate(model, data):
    x, d, y, group = data
    device = next(model.parameters()).device
    preds = torch.cat([model.logits(a.float().to(device), b.to(device)).cpu()
                       for a, b in zip(x.split(4096), d.split(4096))])
    metrics = dict(bce=binary_cross_entropy_with_logits(preds, y).item(), queries=len(y))
    for i, name in enumerate(("clear_opening", "frame", "neighbor")):
        sel = group == i
        metrics[name] = dict(queries=sel.sum().item(), alarm_rate=(preds[sel]>=0).float().mean().item())
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("reports/flight_count_comparison/manifest_8.json"))
    parser.add_argument("--out", type=Path, required=True, help="New checkpoint directory")
    parser.add_argument("--stride", type=int, choices=(1, 4), default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--paired-manifest", type=Path, help="Original-scale 50 Hz observations for a matched test")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    metadata=json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    geometry=metadata.get("vehicle_geometry")
    resolution=metadata.get("resolution",[256,128])
    torch.manual_seed(6106)
    torch.set_num_threads(2)
    train = dataset(args.manifest, "train", args.stride)
    val = dataset(args.manifest, "validation", args.stride)
    model = PotentialNet().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    best = float("inf")
    history = []
    x, d, y = [a.to(args.device) for a in train[:3]]
    for epoch in range(30):
        model.train()
        for idx in torch.randperm(len(y), device=args.device).split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = binary_cross_entropy_with_logits(model.logits(x[idx].float(), d[idx]), y[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        result = evaluate(model, val)
        history.append(result)
        if result["bce"] < best:
            best = result["bce"]
            torch.save(dict(model=model.state_dict(), epoch=epoch+1, validation=result,
                            manifest=str(args.manifest.resolve()), train_queries=len(y), stride=args.stride,
                            vehicle_geometry=geometry,resolution=resolution),
                       args.out / "best.pt")
        if epoch % 5 == 0:
            print(epoch+1, result, flush=True)
    ck = torch.load(args.out / "best.pt", weights_only=True)
    model.load_state_dict(ck["model"])
    test_data = dataset(args.manifest, "test", args.stride)
    test = evaluate(model, test_data)
    comparison = {}
    if geometry:
        previous = "crazyflie21" if resolution == [512,256] else "sensor12hz"
        old_path = Path(__file__).resolve().parents[2]/"reports/spatial_potential"/previous/"best.pt"
        old = PotentialNet().to(args.device)
        old.load_state_dict(torch.load(old_path, map_location=args.device, weights_only=True)["model"])
        comparison["previous_model_on_new_test"] = evaluate(old.eval(), test_data)
        if args.paired_manifest:
            large = dataset(args.manifest,"test",args.stride,args.paired_manifest)
            if not torch.equal(large[2],test_data[2]) or not torch.equal(large[3],test_data[3]):
                raise ValueError("Paired labels or groups differ")
            comparison["previous_model_on_large_test"] = evaluate(old,large)
    (args.out / "training.json").write_text(json.dumps(dict(history=history, epoch=ck["epoch"],
        test=test, train_queries=len(y), validation=ck["validation"], **comparison), indent=2))
    print("test", test, flush=True)


if __name__ == "__main__":
    main()
