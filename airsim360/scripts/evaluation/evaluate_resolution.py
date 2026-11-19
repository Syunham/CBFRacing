"""Compare 0.4 s warnings at equal empirical empty-gate false-alarm rates."""

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

from scripts.evaluation.evaluate_flight_count import draw_example, evaluate, make_probes
from scripts.training.train_rgbd_field import RGBDField, Windows


def operating_point(rows, threshold):
    pos = np.array([r["risk_at_entry"] for r in rows])
    neg = np.array([r["risk_empty"] for r in rows])
    distance = np.array([r["peer_distance_m"] for r in rows])
    bands = {}
    for low, high in ((0, 1), (1, 2), (2, np.inf)):
        mask = (distance >= low) & (distance < high)
        bands[f"[{low},{high})"] = dict(events=int(mask.sum()), hits=int((pos[mask] >= threshold).sum()))
    return dict(threshold=float(threshold), events=len(rows), hits=int((pos >= threshold).sum()),
                empty_alarms=int((neg >= threshold).sum()), distance_bands_m=bands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "native", "baseline"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sampling-comparison", action="store_true")
    parser.add_argument("--width-comparison", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ds = Windows(args.manifest, "test", 1024, .25, lazy_windows=True)
    ds.seed = 11010
    entry, = [e for e in json.loads(args.manifest.read_text())["episodes"] if e["split"] == "test"]
    trajectory = args.manifest.parent / entry["trajectory"]
    events = json.loads(trajectory.with_name("events.json").read_text())
    episode, = ds.episodes
    times, states = episode[:2]
    openings, warnings = make_probes(episode, events)
    ck = torch.load(args.native / "best.pt", map_location="cpu", weights_only=True)
    if ck["half_size"] or ck["projected"] or episode[2].shape[-3:-1] != (256, 512):
        raise ValueError("Expected native 512x256 RGBDField")
    model = RGBDField(wide=ck.get("wide",False)).eval()
    model.load_state_dict(ck["model"])
    native = []
    leads = (.03,.04,.4) if args.width_comparison else (.04,.4)
    metrics = evaluate(model, ds, openings, warnings, native, check_direction=True,direction_leads=leads)
    native_baseline = args.sampling_comparison or args.width_comparison
    baseline_key = "512x256" if native_baseline else "retrained_crazyflie"
    previous_report = "resolution_comparison.json" if native_baseline else "results.json"
    old_name, new_name = ("original_sampling", "gate_pairs") if args.sampling_comparison else ("256x128", "512x256")
    if args.width_comparison:
        old_name, new_name = "original_512", "wide_512"
    baseline = json.loads((args.baseline / "events_0p4.json").read_text())[baseline_key]
    if len(native) != len(baseline) or not native:
        raise ValueError("Different numbers of held-out events")
    event_keys = ("gate", "observer", "event_s", "observation_s", "empty_observation_s", "horizon_s")
    for current, previous in zip(native, baseline):
        if any(current[k] != previous[k] for k in event_keys):
            raise ValueError("Mismatched held-out event")
        frame = np.abs(times - current["observation_s"]).argmin()
        distance = float(np.linalg.norm(states[frame, 0, :3] - states[frame, 1, :3]))
        current["peer_distance_m"] = previous["peer_distance_m"] = distance
    report = {"test_flight":entry["episode_id"], "baseline_run":str(args.baseline),
              old_name:json.loads((args.baseline / previous_report).read_text())[baseline_key],
              new_name:dict(epoch=ck["epoch"] + 1, validation_bce=ck["best"], **metrics)}
    if args.width_comparison:
        small = RGBDField().eval()
        small.load_state_dict(torch.load(args.baseline/"best.pt",map_location="cpu",weights_only=True)["model"])
        matched = evaluate(small,ds,openings,warnings,check_direction=True,direction_leads=leads)
        assert abs(matched["test_bce"]-report[old_name]["test_bce"])<1e-5
        report[old_name].update(matched)
        for name,net in ((old_name,small),(new_name,model)):
            report[name]["parameters"]=sum(v.numel() for v in net.parameters())
        example=next(w for w in warnings if w[0]==.4)
        titles=["原网络 · 512×256","扩容网络 · 512×256"]
        draw_example(args.native/"gate_comparison.png",episode,example,[small,model],titles)
        _,gate,event,(_,ego,_,grid),(empty,_,_,_)=example
        frame=int(np.argmin(abs(times-(event-.03))))
        def grid_distance(k):
            future=np.array([np.interp(times[k]+.03,times,v) for v in states[:,1-ego,:3].T])
            return np.linalg.norm(grid-future,axis=1).min()
        if grid_distance(frame)<=.55 and grid_distance(empty)>1.1:
            short=(.03,gate,event,(frame,ego,.03,grid),(empty,ego,.03,grid))
            draw_example(args.native/"gate_30ms.png",episode,short,[small,model],titles)
            report["example_30ms"]=dict(gate=gate,observer=ego,entry_frame_s=float(times[frame]),empty_frame_s=float(times[empty]))
    baseline_alarms = report[old_name]["warning"]["0.4"]["empty_alarms"]
    for name, rows in ((old_name, baseline), (new_name, native)):
        empty = sorted((r["risk_empty"] for r in rows), reverse=True)
        points = {"threshold_0.5": operating_point(rows, .5)}
        for allowed in sorted({len(rows)//10, len(rows)//4, baseline_alarms}):
            threshold = np.nextafter(empty[allowed], np.inf) if allowed < len(rows) else 0.
            points[f"at_most_{allowed}_empty_alarms"] = operating_point(rows, threshold)
        report[name]["operating_points"] = points
    (args.native / "events_0p4.json").write_text(json.dumps({old_name:baseline, new_name:native}, indent=2))
    result_name = "sampling_comparison.json" if args.sampling_comparison else "resolution_comparison.json"
    if args.width_comparison:
        result_name="width_comparison.json"
    (args.native / result_name).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
