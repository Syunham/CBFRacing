"""Replay a recorded 350-weight flight and export the final MPCC risk kernels at gates 3-5."""

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
from scipy.special import expit
from scripts.control.adapter import ROOT
from scripts.perception.spatial_potential import camera_points
from scripts.perception.live_vision import LiveVision
from scripts.control.run_spatial_mpcc import main as run_experiment
from scripts.evaluation.diagnose_launch import RecordedVision


class GateInspection(RecordedVision):
    def start_renderer(self, checkpoint_geometry, folder, gate_offset):
        super().start_renderer(checkpoint_geometry, folder, gate_offset)
        self.records = {}
        self.planes = {}
        self.nodes = {}
        self.wrapped = set()
        self.targets = {}
        self.axis = np.linspace(-1.05, 1.05, 85)
        crossing = json.loads((self.source / "flight/spatial_audit.json").read_text())["ordered_crossing_times_s"]
        for ego in range(2):
            for segment in (2,3,4,9,10,11):
                for lead in (0.,.4):
                    event = crossing[ego][segment]
                    tick = int(np.floor((event-lead)/.02))
                    self.targets[tick,ego] = (segment,lead,event)

    def observe(self, stamp, states):
        # The original recorded own pose calibrates each recorded image.
        LiveVision.observe(self,stamp,self.states[round(stamp/.02)])

    def install(self, solver, od):
        super().install(solver,od)
        if solver.agent in self.wrapped:
            return
        self.wrapped.add(solver.agent)
        original = solver.step
        def inspect_step():
            result = original()
            if result["status"] == 0:
                self.measure(solver)
            return result
        solver.step = inspect_step

    def measure(self, solver):
        from scripts.control.rti_problem import DT
        x = solver.X[:,:3]
        od = solver.od
        center, radius, width = od[:,16:19], od[:,19], od[:,20]
        d = np.sqrt(np.sum((x-center)**2,axis=1)+1e-10)
        phi = expit((radius-d)/width)
        dynamic = radius > .21
        ego = solver.agent
        tick = round(self.control_stamp/.02)
        field = self.flow_fields[ego]
        self.nodes[tick,ego] = np.column_stack([np.arange(len(phi))*DT,x,center,radius,width,dynamic,phi,350*phi**2])
        if (tick,ego) not in self.targets:
            return
        seg,lead,event = self.targets[tick,ego]
        gate = seg%7
        lap = seg//7+1
        normal = np.array(self.nominal["normals"][gate])
        origin = np.array(self.nominal["centers"][gate])
        side = np.cross([0,0,1],normal)
        key = f"lap{lap}_gate{gate+1}_ego{ego}_" + ("forecast" if lead else "cross")
        j = round(lead/DT)
        peer = self.states[tick,1-ego,:3]
        peer_dist = np.linalg.norm(field.xyz-peer,axis=1)
        self.records[key] = dict(t_s=self.control_stamp, crossing_s=event, lead_s=lead, ego=ego, lap=lap, gate=gate+1,
            field_pixels=len(field.xyz), peer_surface_pixels_within_0p12m=int((peer_dist<.12).sum()),
            actual_separation_m=float(np.linalg.norm(self.states[tick,ego,:3]-peer)),
            own_phi=float(phi[0]), node=int(j), node_phi=float(phi[j]), node_cost=float(350*phi[j]**2),
            node_source="dynamic" if dynamic[j] else "depth",
            peak_phi=float(phi.max()), peak_lead_s=float(np.argmax(phi)*DT), peak_source="dynamic" if dynamic[np.argmax(phi)] else "depth",
            predicted_node_world_m=x[j].tolist(), kernel_center_world_m=center[j].tolist(),
            node_signed_distance_to_gate_plane_m=float((x[j]-origin)@normal), node_gate_side_up_m=[float((x[j]-origin)@side),float(x[j,2]-origin[2])])
        if key in self.planes:
            return
        u,v = np.meshgrid(self.axis,self.axis)
        q = origin + u[...,None]*side + v[...,None]*np.array([0.,0.,1.])
        q = q.reshape(-1,3)
        obs = self.fields[ego].observation
        dist, _ = obs.tree.query(camera_points(q,self.poses[ego]))
        static = expit((.20-dist)/.055)
        moving = []
        for part in np.array_split(q,16):
            c,r,w=field.support(part,self.control_stamp-field.stamp+lead,for_control=True)
            moving.extend(expit((r-np.linalg.norm(part-c,axis=1))/w))
        moving = np.asarray(moving)
        self.planes[key] = np.stack((static,moving,np.maximum(static,moving))).reshape(3,len(self.axis),len(self.axis))

    def close(self):
        self.log.close()
        (self.output/"gate_samples.json").write_text(json.dumps(self.records,indent=2))
        np.savez_compressed(self.output/"gate_planes.npz", axis_m=self.axis, **self.planes)
        keys=sorted(self.nodes)
        np.savez_compressed(self.output/"node_history.npz", tick_ego=np.array(keys), nodes=np.stack([self.nodes[k] for k in keys]),
            columns=["lead_s","x","y","z","support_x","support_y","support_z","radius_m","width_m","dynamic","phi","weighted_cost"])


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("name")
    a=p.parse_args()
    cfg=json.loads((a.source/"experiment.json").read_text())
    assert cfg["weight"]==350 and cfg["flow_guidance"] and cfg["recover_frontend"] and cfg["rti_cold_start"]
    torch.set_num_threads(1)
    GateInspection.source=a.source
    GateInspection.guide=True
    GateInspection.mode="learned"
    events=json.loads((a.source/"flight/spatial_audit.json").read_text())["ordered_crossing_times_s"]
    end=max(e[11] for e in events)+.04
    options = ["--name",a.name,"--separation","1.2","--weight","350","--guarded-fast-fly","--rti-cold-start",
              "--recover-frontend","--flow-guidance","--flow-prior",cfg["flow_prior"],"--checkpoint",cfg["checkpoint"],"--max-time",str(end)]
    run_experiment(options, vision_class=GateInspection)
    output=ROOT/"reports/spatial_potential"/a.name
    orig=np.load(a.source/"flight/execution.npz")
    replay=np.load(output/"flight/execution.npz")
    xx=replay["state"]
    err=float(np.max(np.abs(xx-orig["state"][:len(xx)])))
    records=json.loads((output/"gate_samples.json").read_text())
    assert len(records)==24 and err<1e-7, (len(records),err)
    (output/"verification.json").write_text(json.dumps(dict(source=str(a.source),samples=len(records),max_state_error=err,
        fields=["static_depth","dynamic","combined"],weight=350),indent=2))
