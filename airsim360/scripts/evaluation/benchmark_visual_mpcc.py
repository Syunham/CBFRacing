"""Time a single agent's RGB-D processing and MPCC command in a recorded closed loop."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from scripts.control.adapter import ROOT
from scripts.perception.live_vision import LiveVision
from scripts.control.run_spatial_mpcc import main as run_experiment
from scripts.perception.spatial_potential import SpatialField
from scripts.training.train_rgbd_field import image_history

name = sys.argv[1]
source = ROOT/"reports/spatial_potential/cf21_bendhead03_run2"
output = source.parent/name
config = json.loads((source/"experiment.json").read_text())
timing = []


class TimedVision(LiveVision):
    def start_renderer(self, checkpoint_geometry, folder, gate_offset):
        self.single_fields = [None,None]

    def capture(self, stamp, states):
        # Disk I/O and the canonical dual-agent encoding finish before the command timer.
        fname = f"{round(1000*stamp):06d}.npy"
        return tuple(np.load(source/channel/fname) for channel in ("rgb","depth"))

    def close(self):
        self.log.close()

    @torch.inference_mode()
    def benchmark_agent(self, ego, tick):
        start = perf_counter()
        refreshed = abs(tick*.02-self.stamp) < 1e-7
        encoded = start
        if refreshed:
            rgb = np.stack([frame[ego] for frame in self.fast_rgb])
            depth = np.stack([frame[ego] for frame in self.fast_depth])
            images = image_history(rgb,depth,np.ones(4,bool))
            images = torch.cat([F.interpolate(images[:,:3],(128,256),mode="area"),
                                F.interpolate(images[:,3:],(128,256),mode="nearest-exact")],dim=1)
            ctx = self.time_model.encode(images.unsqueeze(0))
            first = self.time_model.decoder[0]
            bias = ctx @ first.weight[:,:64].T+first.bias
            encoded = perf_counter()
            # Retain each agent's spatial index between refreshes, as in the live frontend.
            self.single_fields[ego] = SpatialField(self.network,np.stack([frame[ego] for frame in self.rgb]),
                                                   np.stack([frame[ego] for frame in self.depth]))
            done = perf_counter()
            # Keep the canonical batched values in the controller to reproduce the saved flight.
            if not np.allclose(self.image_bias[ego],bias.numpy()[0],rtol=1e-5,atol=2e-6):
                raise AssertionError("Single-vehicle temporal encoding differs from the live model")
        else:
            done = perf_counter()
        timing.append((tick,ego,refreshed,(encoded-start)*1000,(done-encoded)*1000))




def distribution(values):
    q=np.percentile(values,[50,95,99])
    return dict(count=len(values),mean_ms=float(np.mean(values)),median_ms=float(q[0]),p95_ms=float(q[1]),
                p99_ms=float(q[2]),max_ms=float(np.max(values)),within_20ms=int(np.sum(values<=20)))


if __name__=="__main__":
    options = ["--name",name,"--separation","1.2","--weight","5600","--guarded-fast-fly",
              "--rti-cold-start","--recover-frontend","--time-checkpoint",config["time_checkpoint"],
              "--checkpoint",config["checkpoint"],"--passage-head",config["passage_head"]]
    run_experiment(options, vision_class=TimedVision)
    read=lambda p: json.loads((p/"flight/planning_log.json").read_text())
    rows=read(output)
    original=read(source)
    assert len(rows)==len(timing)==len(original)
    max_diff=max(float(np.max(np.abs(np.subtract(r["control"],s["control"])))) for r,s in zip(rows,original))
    assert max_diff<1e-6,max_diff
    measured=[]
    for row,(k,i,refresh,encode_ms,surface_ms) in zip(rows,timing):
        assert round(row["time"]/.02)==k and row["agent"]==i
        ms=1000*(row["controller_seconds"]-row["observation_processing_seconds"])
        measured.append((ms,refresh,encode_ms,surface_ms,k,i))
    measured=np.array(measured)
    # Report steady flight; both cold-start calls are preserved separately.
    total,refresh,encode,surface,_,_=measured[measured[:,4]>0].T
    mask=refresh.astype(bool)
    report=dict(source=source.name,device="CPU, one PyTorch/BLAS thread; affinity 0x300, BelowNormal",
                initial_command_ms=measured[:2,0].tolist(),max_control_difference=max_diff,
                single_agent=dict(all=distribution(total),cached=distribution(total[~mask]),refreshed=distribution(total[mask])),
                temporal_encoder_ms=distribution(encode[mask]),spatial_network_and_surface_index_ms=distribution(surface[mask]),
                temporal_plus_mpcc_excluding_frontend_surface_ms=dict(all=distribution(total-surface),refreshed=distribution((total-surface)[mask])))
    (output/"latency.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
