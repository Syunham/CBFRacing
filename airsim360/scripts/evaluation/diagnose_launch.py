"""Replay the launch with fixed world-space RGB-D observations and independently vary MPCC inputs."""

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
from scripts.control.adapter import ROOT
from scripts.perception.spatial_potential import camera_points, rotation
from scripts.perception.live_vision import LiveVision
from scripts.control.run_spatial_mpcc import main as run_experiment


class RecordedVision(LiveVision):
    def start_renderer(self, checkpoint_geometry, folder, gate_offset):
        with np.load(self.source / "flight/execution.npz") as saved:
            self.times = saved["time"]
            self.states = saved["state"]
        self.flow_guidance = self.guide
        if self.mode == "wide_lane":
            self.lane_limit = .42
        self.errors = []
        self.output = folder

    def close(self):
        self.log.close()
        (self.output / "prediction_errors.json").write_text(json.dumps(self.errors))

    def capture(self, stamp, states):
        frame = f"{round(stamp*1000):06d}.npy"
        return tuple(np.load(self.source / channel / frame) for channel in ("rgb", "depth"))

    def install(self, solver, od):
        super().install(solver, od)
        if self.mode == "static_only":
            ego = solver.agent
            pose = self.poses[ego]
            observation = self.fields[ego].observation
            _, idx = observation.tree.query(camera_points(solver.X[:,:3], pose))
            od[:,16:19] = rotation(pose).apply(observation.points[idx]*[1,-1,-1])+pose[:3]
            od[:,19:21] = [.20,.055]
        elif self.mode == "dynamic_only":
            from scripts.control.rti_problem import DT
            field = self.flow_fields[solver.agent]
            lead = self.control_stamp-field.stamp+np.arange(len(od))*DT
            center, radius, width = field.support(solver.X[:,:3],lead,for_control=True)
            od[:,16:19], od[:,19], od[:,20] = center, radius, width

    def observe(self, stamp, states):
        if self.capture_stamp == stamp:
            return
        k = round(stamp/.02)
        assert abs(self.times[k]-stamp)<1e-7
        super().observe(stamp, self.states[k])
        for ego, field in enumerate(self.flow_fields):
            peer = self.states[k:,1-ego,:3]
            future = peer[np.arange(0,81,5)]
            delta = future - peer[0]
            valid = np.linalg.norm(field.xyz-peer[0],axis=1)<.12
            if valid.any():
                err = np.linalg.norm(field.displacements[valid,4]-delta[4],axis=1)
                forward = np.asarray(self.nominal["normals"][0])
                signed = (field.displacements[valid,4]-delta[4]) @ forward
                self.errors.append(dict(t=stamp,ego=ego,points=int(valid.sum()),total=len(valid),
                                        error_m=float(np.median(err)),along_error_m=float(np.median(signed))))
            else:
                self.errors.append(dict(t=stamp,ego=ego,points=0,total=len(valid)))
            if self.mode == "true_motion" and len(field.xyz):
                # Replace transport only. Keep the appearance detector, omissions and support kernel intact.
                field.displacements = np.zeros((len(valid),17,3))
                field.displacements[valid] = delta
            elif self.mode == "true_peer":
                # Diagnostic oracle for both visibility and motion, still using the same threat kernel.
                field.xyz = peer[:1].copy()
                field.displacements = delta[None,:,:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("name")
    p.add_argument("weight",type=float)
    p.add_argument("guidance",type=int,choices=(0,1))
    p.add_argument("mode",choices=("learned","true_motion","true_peer","static_only","dynamic_only","wide_lane"))
    p.add_argument("--seconds",type=float,default=1.36)
    p.add_argument("--live", action="store_true", help="Render the two interacting agents online")
    a = p.parse_args()
    torch.set_num_threads(1)
    source_config = json.loads((a.source/"experiment.json").read_text())
    RecordedVision.source = a.source
    RecordedVision.guide = bool(a.guidance)
    RecordedVision.mode = a.mode
    if a.live:
        if a.mode != "learned":
            p.error("The oracle needs a recorded future trajectory")
        class LiveAblation(LiveVision):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.flow_guidance = bool(a.guidance)
        vision_class = LiveAblation
    else:
        vision_class = RecordedVision
    options = ["--name",a.name,"--separation","1.2","--weight",str(a.weight),
                "--guarded-fast-fly","--rti-cold-start","--recover-frontend","--flow-prior",source_config["flow_prior"],
                "--checkpoint",source_config["checkpoint"],"--max-time",str(a.seconds)]
    if a.guidance:
        options.append("--flow-guidance")
    run_experiment(options, vision_class=vision_class)
    folder = ROOT/"reports/spatial_potential"/a.name
    (folder/"replay.json").write_text(json.dumps(dict(source=str(a.source), weight=a.weight,
        guidance=bool(a.guidance),field=a.mode,seconds=a.seconds,
        scope="live reciprocal RGB-D" if a.live else "fixed recorded world observations, two independent ego rollouts"),indent=2))


if __name__ == "__main__":
    main()
