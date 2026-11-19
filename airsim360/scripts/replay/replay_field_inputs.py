"""Replay archived sensor poses to reconstruct missing final-window depth images."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    folder, out = map(Path, sys.argv[1:])
    out.mkdir(parents=True, exist_ok=False)
    geometry = json.loads((folder/"vehicle_geometry.json").read_text())
    trajectory = np.load(folder/"flight/execution.npz")
    inputs = sorted((folder/"rgb").glob("*.npy"))
    frames = []
    latest = inputs[-4:]
    _, h, w, _ = np.load(inputs[0], mmap_mode="r").shape
    command = [sys.executable, "-u", "-m","scripts.simulation.spatial_render_worker", str(out/"sensor.jsonl"),
               "--resolution", str(w), str(h), "--scale", *map(str,geometry["scale_xyz"])]
    env = dict(os.environ, SAVE_FLIGHT_VIEWS="0")
    with (folder/"sensor.jsonl").open() as sensor_log:
        if not json.loads(next(sensor_log)).get("discarded_sensor_ns"):
            command.append("--legacy-single-trigger")
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env) as child:
        try:
            if child.stdout.readline().strip() != b"READY":
                raise RuntimeError("Renderer initialization failed")
            for f in inputs:
                ms = int(f.stem)
                k = int(abs(trajectory["time"]-ms/1000).argmin())
                t = trajectory["time"][k]
                assert abs(t-ms/1000)<1e-7
                child.stdin.write((json.dumps([t,trajectory["state"][k].tolist()])+"\n").encode())
                child.stdin.flush()
                rgb = np.frombuffer(child.stdout.read(2*h*w*3),np.uint8).reshape(2,h,w,3)
                depth = np.frombuffer(child.stdout.read(2*h*w*4),"<f4").reshape(2,h,w)
                if f in latest:
                    err = np.abs(rgb.astype(float)-np.load(f))
                    frames.append((ms, depth, err.mean(axis=(1,2,3))))
        finally:
            child.stdin.close()
        if child.wait(timeout=45):
            raise RuntimeError("Renderer exited with error")
    times,depth,errors = zip(*frames)
    np.savez_compressed(out/"replay_depth.npz",ms=times,depth=depth,rgb_mae=errors)


if __name__=="__main__":
    main()
