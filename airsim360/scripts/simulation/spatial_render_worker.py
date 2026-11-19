"""Serve frozen-time, online RGB-D pairs from the isolated AirSim Python runtime."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

import numpy as np
from scripts.simulation.airsim360_replay import create_renderer
from scripts.simulation.preview_split_s import place_course


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--resolution", type=int, nargs=2, default=(256,128))
    p.add_argument("--scale", type=float, nargs=3, default=(1.,1.,1.))
    p.add_argument("--placement", type=Path, default=Path("reports/split_s_open_yard/placement.json"))
    p.add_argument("--gate-offset", nargs=3, type=float, metavar=("GATE", "SIDE", "UP"))
    p.add_argument("--legacy-single-trigger", action="store_true", help="Replay pre-refresh recordings")
    p.add_argument("--depth-only", action="store_true", help="Return only panorama ranges on the sensor pipe")
    args = p.parse_args()
    os.environ["RISK_REPLAY_LOG"] = str(args.log)
    pipe = sys.stdout.buffer
    sys.stdout = sys.stderr
    placement = json.loads(args.placement.read_text())
    with ExitStack() as stack:
        renderer = stack.enter_context(create_renderer(single=os.getenv("SINGLE_FLIGHT_CAMERA") == "1",
            scale=args.scale, resolution=args.resolution, refresh_on_switch=not args.legacy_single_trigger))
        renderer.translation = np.array(placement["translation_ned_m"])
        renderer.heading = np.deg2rad(placement["heading_ned_deg"])
        actors = set(renderer.client.simListSceneObjects())
        for name in placement["removed_yard_fixtures"]:
            if name in actors and not renderer.client.simDestroyObject(name):
                raise RuntimeError(f"Could not clear {name}")
        course = json.loads(Path(placement["course"]).read_text())
        if args.gate_offset:
            num, side, up = args.gate_offset
            index = int(num)-1
            if num != int(num) or not 0 <= index < len(course["centers"]):
                raise ValueError("Unknown gate for perturbation")
            center = course["centers"][index] + side*np.array(course["coordinate_axes"][index])
            course["centers"][index] = (center + [0., 0., up]).tolist()
        (args.log.parent/"rendered_course.json").write_text(json.dumps(course, indent=2))
        parts = place_course(renderer.client, course,
                             stack, args.placement.with_name("gate_texture.png"), renderer.translation, renderer.heading)
        if len(parts) != 28 or not all(p["textured"] for p in parts):
            raise RuntimeError("Race gates not placed")
        pipe.write(b"READY\n")
        pipe.flush()
        for request in sys.stdin:
            t, state = json.loads(request)
            obs = renderer.capture(t, np.array(state))
            if os.getenv("SAVE_FLIGHT_VIEWS") == "1":
                folder = args.log.parent / "rgb"
                folder.mkdir(exist_ok=True)
                np.save(folder / f"{round(1000*t):06d}.npy", obs["rgb"])
                depth_folder = args.log.parent / "depth"
                depth_folder.mkdir(exist_ok=True)
                np.save(depth_folder / f"{round(1000*t):06d}.npy", obs["depth"])
            for value in ((obs["depth"],) if args.depth_only else (obs["rgb"], obs["depth"])):
                pipe.write(value.tobytes(order="C"))
            pipe.flush()


if __name__ == "__main__":
    main()
