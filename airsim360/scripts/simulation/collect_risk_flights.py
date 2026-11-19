"""Collect complete planner trajectories with resumable, synchronized frame pairs."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.simulation.airsim360_replay import create_renderer
from scripts.simulation.gate_geometry import gate_boxes
from scripts.simulation.preview_split_s import place_course


FLIGHTS = {"distance_dev12": "train", "learned_cbf_dev12_v2": "validation",
           "threat_dev12_w100000": "test"}


def collect(source, output, placement_path=None, flight_paths=None, profile=None, sample_every=1, resolution=(256,128)):
    placement = json.loads(placement_path.read_text()) if placement_path else None
    course = json.loads(Path(placement["course"]).read_text()) if placement else None
    entries = []
    if flight_paths is None:
        flight_paths = [(flight, split, source / flight / "execution.npz") for flight, split in FLIGHTS.items()]
    for flight, split, trajectory in flight_paths:
        folder = output / flight
        folder.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(trajectory.read_bytes()).hexdigest()
        with np.load(trajectory) as archive:
            times, states = archive["time"][::sample_every], archive["state"][::sample_every]
        frames = folder / "frames"
        frames.mkdir(exist_ok=True)
        provenance = dict(trajectory_sha256=digest, count=len(times), scene="FactoryExteriorInteriorBundle",
                          width=resolution[0], height=resolution[1], translation_ned_m=[8, 1, -5], axis_signs=[1, -1, -1],
                          appearance="Quadrotor1", time_step_s=float(np.median(np.diff(times))))
        meta = folder / "metadata.json"
        if profile:
            provenance.update(vehicle_geometry=profile, trajectory_stride=sample_every)
        if placement:
            provenance.update(translation_ned_m=placement["translation_ned_m"],
                              placement_sha256=hashlib.sha256(placement_path.read_bytes()).hexdigest(),
                              gate_boxes=gate_boxes(course))
        if meta.exists() and json.loads(meta.read_text()) != provenance:
            raise ValueError(f"Source or capture settings changed for {flight}")
        meta.write_text(json.dumps(provenance, indent=2))
        final = folder / "observations.npz"
        if not final.exists():
            os.environ["RISK_REPLAY_LOG"] = str(folder / f"poses-{time.time_ns()}.jsonl")
            with ExitStack() as stack:
                renderer = stack.enter_context(create_renderer(resolution=resolution, single=bool(profile), scale=profile["scale_xyz"] if profile else (1.,1.,1.)))
                if placement:
                    renderer.translation = np.array(placement["translation_ned_m"])
                    renderer.heading = np.deg2rad(placement["heading_ned_deg"])
                    actors = set(renderer.client.simListSceneObjects())
                    for actor in placement.get("removed_yard_fixtures", []):
                        if actor in actors and not renderer.client.simDestroyObject(actor):
                            raise RuntimeError(f"Failed to clear {actor}")
                    parts = place_course(renderer.client, course, stack, placement_path.with_name("gate_texture.png"),
                                         renderer.translation, renderer.heading)
                    if len(parts) != len(placement["bars"]) or any(
                            not np.allclose(p["position"], ref["position"], atol=.005)
                            or p["textured"] != ref["textured"] for p,ref in zip(parts,placement["bars"])):
                        raise RuntimeError("Rendered gates differ from the reviewed placement")
                for k, (t, state) in enumerate(zip(times, states)):
                    path = frames / f"{k:04}.npz"
                    if path.exists():
                        continue
                    result = renderer.capture(float(t), state)
                    tmp = path.with_suffix(".tmp")
                    with tmp.open("wb") as stream:
                        np.savez(stream, **result)
                    tmp.replace(path)
                    if k % 50 == 0:
                        print(f"{flight} {k+1}/{len(times)}", flush=True)
            rgb = np.empty((len(times), 2, resolution[1], resolution[0], 3), dtype=np.uint8)
            depth = np.empty(rgb.shape[:-1], dtype=np.float32)
            valid = np.empty((len(times), 2), dtype=bool)
            for k,t in enumerate(times):
                with np.load(frames / f"{k:04}.npz") as obs:
                    if obs["time"].item() != t:
                        raise ValueError(f"Frame time mismatch at {flight}/{k}")
                    rgb[k], depth[k], valid[k] = obs["rgb"], obs["depth"], obs["valid"]
            if not valid.all() or not np.isfinite(depth).any(axis=(-2,-1)).all():
                raise ValueError(f"Missing observation at {flight}")
            tmp = final.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.savez(stream, time=times, rgb=rgb, depth=depth, valid=valid)
            tmp.replace(final)
            print(f"Completed {flight}", flush=True)
        entries.append(dict(episode_id=flight, split=split, trajectory=Path(os.path.relpath(trajectory,output)).as_posix(),
                            observations=f"{flight}/observations.npz", trajectory_sha256=digest,
                            gate_boxes=gate_boxes(course) if placement else []))
    manifest = dict(episodes=entries, resolution=list(resolution))
    if profile:
        manifest.update(vehicle_geometry=profile, trajectory_stride=sample_every)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=Path("assets/flight/trajectories/baseline"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--placement", type=Path, help="Saved preview placement for gated captures")
    parser.add_argument("--flight-list", type=Path, help="JSON list of episode_id, split and absolute trajectory path")
    args = parser.parse_args()
    selected = None
    if args.flight_list:
        selected = [(e["episode_id"], e["split"], (args.flight_list.parent / e["trajectory"]))
                    for e in json.loads(args.flight_list.read_text(encoding="utf-8-sig"))]
    collect(args.source, args.output, args.placement, selected)
