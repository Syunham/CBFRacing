"""Isolate post-gate-2 control changes on fixed recorded RGB-D world fields."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from scripts.evaluation.diagnose_launch import RecordedVision
from scripts.control.adapter import ROOT
from scripts.perception.live_vision import LiveVision
from scripts.control.run_spatial_mpcc import main as run_experiment

class TurnReplay(RecordedVision):
    def observe(self, stamp, states):
        if self.capture_stamp == stamp:
            return
        recorded = self.states[round(stamp/.02)]
        LiveVision.observe(self,stamp,recorded)
        if treatment == "remove_gate3_dynamic" and stamp >= switch_s-1e-8:
            for ego, field in enumerate(self.flow_fields):
                if not len(field.xyz):
                    continue
                near_frame = self.gate3.distances(field.xyz) <= .08
                far_from_peer = np.linalg.norm(field.xyz-recorded[1-ego,:3],axis=1) > .18
                keep = ~(near_frame & far_from_peer)
                self.errors.append(dict(t=stamp,ego=ego,removed=int((~keep).sum())))
                field.xyz = field.xyz[keep]
                field.flow = field.flow[keep]
                field.displacements = field.displacements[keep]

    def install(self,solver,od):
        post = self.control_stamp >= switch_s-1e-8
        # The static-only branch preserves the original static depth potential.
        self.mode = "static_only" if post and treatment in ("no_dynamic_cost","no_dynamic_response") else "learned"
        self.flow_guidance = not (post and treatment in ("no_guidance","no_dynamic_response"))
        super().install(solver,od)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("name")
    parser.add_argument("treatment", choices=("no_dynamic_cost","no_guidance","no_dynamic_response","remove_gate3_dynamic"))
    parser.add_argument("--switch",type=float,default=2.02)
    args = parser.parse_args()
    source,name,treatment,switch_s=args.source,args.name,args.treatment,args.switch
    if treatment == "remove_gate3_dynamic":
        from scripts.simulation.gate_geometry import gate_boxes
        from scripts.training.train_rgbd_field import GateFrames
        boxes = gate_boxes(json.loads((Path(source)/"rendered_course.json").read_text()))
        TurnReplay.gate3 = GateFrames([b for b in boxes if b["name"].startswith("SplitS_G3_")])
    TurnReplay.source=Path(source)
    TurnReplay.guide=True
    TurnReplay.mode="learned"
    torch.set_num_threads(1)
    cfg=json.loads((Path(source)/"experiment.json").read_text())
    options = ["--name",name,"--separation","1.2","--weight","350","--guarded-fast-fly","--rti-cold-start","--flow-guidance","--recover-frontend","--flow-prior",cfg["flow_prior"],"--checkpoint",cfg["checkpoint"],"--max-time","3.30"]
    run_experiment(options, vision_class=TurnReplay)
    out=ROOT/"reports/spatial_potential"/name/"flight"
    original=np.load(Path(source)/"flight/execution.npz")["state"]
    replay=np.load(out/"execution.npz")["state"]
    end=round(switch_s/.02)+1
    err=float(np.max(np.abs(original[:end]-replay[:end])))
    events=json.loads((out/"events.json").read_text())
    assert err<1e-7 and min(map(len,events))>=3,(err,[len(e) for e in events])
    row=dict(treatment=treatment,source=source, switch_s=switch_s,max_pre_switch_state_error=err,g3_times_s=[e[2]["time"] for e in events])
    (out.parent/"turn_ablation.json").write_text(json.dumps(row,indent=2))
