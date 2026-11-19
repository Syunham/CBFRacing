"""Check the recorded second exposures against the executed peer poses offline."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from scripts.perception.spatial_potential import PotentialNet, SpatialField, camera_points


def audit(folder):
    sensors = [json.loads(line) for line in (folder/"sensor.jsonl").read_text().splitlines()]
    experiment = json.loads((folder/"experiment.json").read_text())
    checkpoint = Path(experiment["checkpoint"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == experiment["checkpoint_sha256"]
    net = PotentialNet().eval()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    torch.set_num_threads(1)
    trace = np.load(folder/"flight/execution.npz")
    rgb, depth = deque(maxlen=4), deque(maxlen=4)
    frames = []
    for sensor in sensors:
        t = sensor["time"]
        assert len(sensor["discarded_sensor_ns"]) == len(sensor["sensor_ns"]) == 2
        name = f"{round(t*1000):06d}.npy"
        rgb.append(np.load(folder/"rgb"/name))
        depth.append(np.load(folder/"depth"/name))
        while len(rgb)<4:
            rgb.appendleft(rgb[0])
            depth.appendleft(depth[0])
        k = np.argmin(abs(trace["time"]-t))
        assert abs(trace["time"][k]-t)<1e-7
        states = trace["state"][k]
        counts, values = [], []
        for ego in (0,1):
            field = SpatialField(net, np.stack([im[ego] for im in rgb]), np.stack([d[ego] for d in depth]))
            peer = camera_points(states[None,1-ego,:3], states[ego])
            counts.append(len(field.observation.tree.query_ball_point(peer[0],.15)))
            values.append(float(field.query(peer)[0]))
        frames.append(dict(time_s=t,separation_m=float(np.linalg.norm(states[0,:3]-states[1,:3])),
                           peer_surface_points=counts, U_at_peer=values))
    assert frames
    counts = np.array([row["peer_surface_points"] for row in frames])
    return dict(radius_m=.15, frames=len(frames), seen_frames_per_agent=(counts>0).sum(axis=0).tolist(),
                mutually_seen_frames=int((counts>0).all(axis=1).sum()), observations=frames)


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    result = audit(folder)
    (folder/"peer_observations.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
