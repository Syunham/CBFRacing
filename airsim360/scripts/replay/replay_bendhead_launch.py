"""Exchange only the temporal RGB-D embedding at 0.26 s in two recorded closed-loop launches."""

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
from scripts.control.adapter import ROOT, controller
from scripts.perception.live_vision import LiveVision
from scripts.control.run_spatial_mpcc import main as run_experiment
from scripts.perception.time_query_field import encode_views


class OfflineLaunch(LiveVision):
    def start_renderer(self,checkpoint_geometry,folder,gate_offset):
        pass

    def close(self):
        self.log.close()

    def capture(self,stamp,states):
        return load_pair(source,stamp)

    def install(self,solver,od):
        if abs(self.control_stamp-.26)<1e-8 and args.history!=args.embedding:
            stamps=np.maximum(0,self.stamp+np.arange(-3,1)*.02)
            rgb,depth=map(np.stack,zip(*(load_pair(donor,t) for t in stamps)))
            self.context=encode_views(self.time_model,rgb,depth)
            first=self.time_model.decoder[0]
            with torch.inference_mode():
                self.image_bias=(self.context @ first.weight[:,:64].T+first.bias).numpy()
        super().install(solver,od)


def load_pair(folder,stamp):
    name=f"{round(stamp*1000):06d}.npy"
    return tuple(np.load(folder/c/name) for c in ("rgb","depth"))


def use_recorded_reference(adapter):
    validate = controller(adapter)
    from scripts.planning.fast_fly.guarded_fast_fly import GuardedFastFly
    normal_solve = GuardedFastFly.solve
    def solve(planner,own,other,index):
        plan = normal_solve(planner,own,other,index)
        if abs(planner.current_time-.20)<1e-8:
            ego=planner.active_agent
            path=reference_source/"flight/plans"/f"{2+ego:03d}_agent{ego}_event0.npz"
            with np.load(path) as fixed:
                plan.update({key:fixed[key] for key in fixed.files})
        return plan
    GuardedFastFly.solve=solve
    return validate


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history",choices=("1","2"))
    parser.add_argument("embedding",choices=("1","2"))
    parser.add_argument("name")
    parser.add_argument("--reference",choices=("1","2"),help="Override the 0.20 s Fast-fly plan from this recording")
    args=parser.parse_args()
    root=ROOT/"reports/spatial_potential"
    source=root/("cf21_bendhead03_run"+args.history)
    donor=root/("cf21_bendhead03_run"+args.embedding)
    cfg=json.loads((source/"experiment.json").read_text())
    if args.reference:
        reference_source=root/("cf21_bendhead03_run"+args.reference)
    options = ["--name",args.name,"--separation","1.2","--weight","5600",
              "--guarded-fast-fly","--rti-cold-start","--recover-frontend","--time-checkpoint",cfg["time_checkpoint"],
              "--checkpoint",cfg["checkpoint"],"--passage-head",cfg["passage_head"],"--max-time",".281"]
    run_experiment(options, vision_class=OfflineLaunch, load_controller=use_recorded_reference if args.reference else controller)
    def controls(folder):
        return np.array([r["control"] for r in json.loads((folder/"flight/planning_log.json").read_text())]).reshape(-1,2,5)
    actual=controls(source)
    replay=controls(root/args.name)
    assert len(replay)==14
    prefix=10 if args.reference else 13
    error=float(np.max(abs(actual[:prefix]-replay[:prefix])))
    error_last=float(np.max(abs(actual[13]-replay[13])))
    print("prefix error",error,"treatment max delta",error_last,flush=True)
    assert error<1e-5 and (args.reference or args.history!=args.embedding or error_last<1e-5)
