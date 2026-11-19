"""Compute t+0.4 s surface forecasts in the flight model's conda runtime."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
from pathlib import Path
import sys

import numpy as np
from scripts.perception.occupancy_flow import FlowPrior, OccupancyFlow


def main():
    folder = Path(sys.argv[1]).resolve()
    config = json.loads((folder/"experiment.json").read_text())
    prior = FlowPrior(config["flow_prior"])
    if prior.uncertainty_aware:
        raise ValueError("This replay targets the saved unmodified kernel prior")
    with np.load(folder/"flight/execution.npz") as data:
        times, states = data["time"], data["state"]
    streams = [OccupancyFlow(prior), OccupancyFlow(prior)]
    channel = sys.stdin.buffer
    wire = sys.stdout.buffer
    for line in iter(channel.readline, b""):
        k = int(line)
        rgb = np.frombuffer(channel.read(2*256*512*3), np.uint8).reshape(2,256,512,3)
        depth = np.frombuffer(channel.read(2*256*512*4), "<f4").reshape(2,256,512)
        rows = []
        for i, stream in enumerate(streams):
            stream.observe(times[k],rgb[i],depth[i],states[k,i])
            pred = stream.xyz+stream.displacements[:,4] if len(stream.xyz) else stream.xyz
            if len(pred):
                _, radius, width = stream.support(pred[:1], .4)
                np.testing.assert_allclose([radius[0],width[0]],[.70,.13])
            rows.append(pred.tolist())
        wire.write((json.dumps(rows)+"\n").encode())
        wire.flush()


if __name__ == "__main__":
    main()
