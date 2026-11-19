"""RGB-D risk fields and passage selection for the comparison controllers."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from scripts.simulation.renderer_process import RendererProcess
from scripts.perception.spatial_potential import PotentialNet, SpatialField, camera_points, rotation

ROOT = Path(__file__).resolve().parents[2]


class LiveVision(RendererProcess):
    def __init__(self, checkpoint, folder, gate_offset=None, refine=False, passage_weight=9., lane_limit=.42,
                 time_checkpoint=None, flow_prior=None, flow_guidance=False, implicit_checkpoint=None):
        self.network = PotentialNet().eval()
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.network.load_state_dict(ck["model"])
        self.period = .02*ck.get("stride", 1)
        self.width, self.height = ck.get("resolution", [256,128])
        self.rgb, self.depth = deque(maxlen=4), deque(maxlen=4)
        self.time_model = None
        self.future_model = None
        self.flow_fields = None
        self.peer_tracks = None
        self.local_flow = None
        self.implicit_fields = None
        if implicit_checkpoint:
            from scripts.perception.implicit_rgbd import CachedField, load_field
            if (self.width, self.height) != (512, 256):
                raise ValueError("Implicit RGB-D requires a 512x256 spatial checkpoint")
            model, history = load_field(implicit_checkpoint)
            self.implicit_fields = [CachedField(model, history) for _ in range(2)]
        self.flow_guidance = flow_guidance
        if flow_prior:
            from scripts.perception.occupancy_flow import FlowPrior, OccupancyFlow
            transport = FlowPrior(flow_prior)
            self.flow_fields = [OccupancyFlow(transport) for _ in range(2)]
        if time_checkpoint:
            from scripts.perception.time_query_field import load_rgbd
            self.time_model = load_rgbd(time_checkpoint)
            self.fast_rgb, self.fast_depth = deque(maxlen=4), deque(maxlen=4)
        self.capture_stamp = None
        self.stamp = None
        self.preference = [None, None]
        self.passage_weight = passage_weight
        self.lane_limit = lane_limit
        placement = json.loads((ROOT/"reports/split_s_open_yard/placement.json").read_text())
        self.nominal = json.loads(Path(placement["course"]).read_text())
        self.refine = refine
        self.corrections, self.updated = {}, {}
        self.log = (folder / "vision.jsonl").open("w", buffering=1)
        self.start_renderer(checkpoint_geometry=ck.get("vehicle_geometry"), folder=folder, gate_offset=gate_offset)

    def capture(self, stamp, states):
        self.worker.stdin.write((json.dumps([stamp, states.tolist()])+"\n").encode())
        self.worker.stdin.flush()
        pixels = 2*self.height*self.width
        color = self.worker.stdout.read(3*pixels)
        depth = self.worker.stdout.read(4*pixels)
        if len(color) != 3*pixels or len(depth) != 4*pixels:
            raise RuntimeError("Incomplete RGB-D update from renderer")
        frame = np.frombuffer(color, np.uint8).reshape(2, self.height, self.width, 3)
        ranges = np.frombuffer(depth,"<f4").reshape(2,self.height,self.width)
        return frame, ranges

    def observe(self, stamp, states):
        self.control_stamp = stamp
        new_field = self.stamp is None or stamp-self.stamp >= self.period-1e-8
        future_model = getattr(self, "future_model", None)
        implicit_fields = getattr(self, "implicit_fields", None)
        if (not new_field and self.time_model is None and self.flow_fields is None and future_model is None and implicit_fields is None) or self.capture_stamp == stamp:
            return
        frame, ranges = self.capture(stamp, states)
        self.capture_stamp = stamp
        if implicit_fields is not None:
            for i, field in enumerate(implicit_fields):
                field.observe(stamp, states[i], frame[i], ranges[i])
            if not new_field:
                return
        if future_model is not None:
            from scripts.perception.local_future_field import observe, static_tree
            self.future_history.append((stamp,states.copy(),frame,ranges))
            if not new_field:
                return
            while len(self.future_history)<4:
                self.future_history.appendleft(self.future_history[0])
            times, poses, colors, depths = map(np.stack,zip(*self.future_history))
            self.future_views = []
            for i in (0,1):
                flow = observe(self.future_prior,times,poses[:,i],colors[:,i],depths[:,i])
                self.future_views.append((flow,static_tree(ranges[i],states[i],flow.xyz)))
        if self.local_flow is not None:
            self.local_states = states.copy()
            for i, field in enumerate(self.local_flow):
                field.observe(stamp,frame[i],ranges[i],states[i])
        if self.flow_fields is not None:
            for i, field in enumerate(self.flow_fields):
                field.observe(stamp,frame[i],ranges[i],states[i])
                if self.peer_tracks is not None:
                    track = self.peer_tracks[i]
                    hits = track.update(stamp, field.xyz)
                    peer = track.estimate(stamp)
                    print(json.dumps(dict(time=stamp, observer=i, surface_points=len(field.xyz),
                                          cluster_points=hits, peer=None if peer is None else peer.tolist())),
                          file=self.peer_log)
            if not new_field:
                return
        if self.time_model is not None:
            from scripts.perception.time_query_field import encode_views
            self.fast_rgb.append(frame)
            self.fast_depth.append(ranges)
            while len(self.fast_rgb)<4:
                self.fast_rgb.appendleft(frame)
                self.fast_depth.appendleft(ranges)
            if not new_field:
                return
            self.context = encode_views(self.time_model,np.stack(self.fast_rgb),np.stack(self.fast_depth))
            first = self.time_model.decoder[0]
            with torch.inference_mode():
                self.image_bias = (self.context @ first.weight[:,:64].T + first.bias).numpy()
        self.rgb.append(frame)
        self.depth.append(ranges)
        while len(self.rgb) < 4:
            self.rgb.appendleft(self.rgb[0])
            self.depth.appendleft(self.depth[0])
        rgb, d = np.stack(self.rgb), np.stack(self.depth)
        self.fields = [SpatialField(self.network, rgb[:, i], d[:, i]) for i in range(2)]
        self.poses = states.copy()
        self.stamp = stamp

    def allocation_peer(self, agent, own):
        peer = self.peer_tracks[agent].estimate(self.control_stamp)
        # Identical inputs yield exactly zero offset in the antisymmetric allocator.
        return own[:6] if peer is None else peer

    def visual_projection(self, pair_filter, states, nominal, dt):
        from scripts.perception.visual_peer import visual_pair_states
        peers = [track.estimate(self.control_stamp) for track in self.peer_tracks]
        if all(peer is None for peer in peers):
            return nominal, dict(visual_coverage=False, correction_norm=0.)
        observed = visual_pair_states(states, peers)
        applied, audit = pair_filter.project(observed, nominal, dt)
        audit["visual_coverage"] = True
        return applied, audit

    def passage(self, agent, gate, normal, axis):
        field, pose = self.fields[agent], self.poses[agent]
        # Choose the initial passing side from risk on either side of the camera.
        # Carry that choice through the continuous track axes to avoid switching
        # sides while side by side. Both vehicles run the identical rule.
        if self.preference[agent] is None:
            probe = pose[:3]+np.array([-1, 1])[:, None, None]*np.linspace(.25, 1., 8)[None, :, None]*axis
            score = field.query(camera_points(probe.reshape(-1, 3), pose)).reshape(2, -1).mean(1)
            preferred_offset = min(.36, self.lane_limit)
            self.preference[agent] = float([-preferred_offset, preferred_offset][score.argmin()])
            print("visual passing side", agent, score, self.preference[agent], flush=True)
        offsets = np.linspace(-self.lane_limit, self.lane_limit, 15)
        points = gate + offsets[:, None]*axis
        potential = field.query(camera_points(points, pose))
        cost = potential + self.passage_weight*(offsets-self.preference[agent])**2
        offset = float(offsets[cost.argmin()])
        head = getattr(self, "passage_head", None)
        if head is None:
            return offset*getattr(self, "passage_gain", 1.)
        target = offset*head
        if self.passage_geometry is not None:
            bias, gain = self.passage_geometry.T
            target += bias+gain*self.gate_bends[tuple(gate)]
        return target

    def install(self, solver, od):
        ego = solver.agent
        pose = self.poses[ego]
        if getattr(self,"future_model",None) is not None:
            from scripts.control.rti_problem import DT
            from scripts.perception.local_future_field import control_support, metric_distance
            flow, tree = self.future_views[ego]
            nodes = solver.X[:,:3]
            tau = np.clip(self.control_stamp-self.stamp + np.arange(len(nodes))*DT,0.,1.6)
            if hasattr(self.future_prior,'mixture'):
                from scripts.perception.local_future_field import mixture_control
                centers,mass,radius,width,moving,risk=mixture_control(flow,tree,pose,tau,nodes)
                od[:,16:19],od[:,19],od[:,20]=centers[:,0],radius,width
                od[:,21:28]=0.
                solver.mixture_data=np.column_stack((centers.reshape(len(nodes),9),mass))
                print(json.dumps(dict(time=self.stamp,decision_s=self.control_stamp,ego=ego,flow_pixels=len(flow.xyz),
                                      dynamic_nodes=int(moving.sum()),peak=float(risk.max()),node_risk=risk.tolist())),file=self.log)
                return
            centers,radius,width,moving,axis,coeff = control_support(self.future_model,flow,tree,pose,tau,nodes,with_metric=True)
            od[:,16:19], od[:,19], od[:,20] = centers,radius,width
            od[:,21:28] = 0.
            od[:,21:24],od[:,24]=axis,coeff
            risk = .5+.5*np.tanh((radius-metric_distance(nodes-centers,axis,coeff))/(2*width))
            print(json.dumps(dict(time=self.stamp,decision_s=self.control_stamp,ego=ego,flow_pixels=len(flow.xyz),
                                  dynamic_nodes=int(moving.sum()),peak=float(risk.max()),node_risk=risk.tolist())),file=self.log)
            return
        points = camera_points(solver.X[:, :3], pose)
        if self.flow_fields is not None:
            from scripts.control.rti_problem import DT
            obs = self.fields[ego].observation
            distance, idx = obs.tree.query(points)
            static = rotation(pose).apply(obs.points[idx]*[1,-1,-1])+pose[:3]
            field = self.flow_fields[ego]
            lead = self.control_stamp-field.stamp+np.arange(len(points))*DT
            moving, radius, width = field.support(solver.X[:,:3],lead,for_control=True)
            dynamic_distance = np.linalg.norm(solver.X[:,:3]-moving,axis=1)
            active = (radius-dynamic_distance)/width > (.20-distance)/.055
            shift = np.zeros_like(static)
            if self.flow_guidance and len(field.xyz):
                # Let the tracking reference move away from a visually occupied path.
                # Displace across the local racing tangent to preserve progress.
                reference = od[:,:3].copy()
                obstacle, _, _ = field.support(reference,lead)
                away = reference-obstacle
                distance_ref = np.linalg.norm(away,axis=1)
                # A parallel pass can stay close for several frames while the longer
                # forecast puts the surface too far ahead. Preserve the gradient of
                # the current observation through the short reaction interval.
                own = solver.X[0,:3]
                current, _, _ = field.support(own[None,:],0.)
                current_away = own-current[0]
                current_distance = np.linalg.norm(current_away)
                present_risk = np.clip((1.18-current_distance)/.5,0.,1.)*np.exp(-lead/.65)
                forecast_risk = np.clip((1.10-distance_ref)/.55,0.,1.)*np.exp(-lead/.65)
                away = np.where((present_risk>forecast_risk)[:,None],current_away,away)
                tangent = od[:,3:6]
                tangent = tangent / np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6)
                lateral = away-np.sum(away*tangent,axis=1,keepdims=True)*tangent
                strength = np.maximum(present_risk,forecast_risk)
                shift = .32*strength[:,None]*lateral/np.maximum(.15,np.linalg.norm(lateral,axis=1,keepdims=True))
                # Respect the gate-frame reserve when adapting a near-plane reference.
                gate_centers = np.asarray(self.nominal["centers"])
                nearest = np.linalg.norm(reference[:,None]-gate_centers,axis=-1).argmin(axis=1)
                normal = np.asarray(self.nominal["normals"])[nearest]
                side = np.cross([0.,0.,1.],normal)
                local = reference-gate_centers[nearest]
                near = (np.abs(np.sum(local*normal,axis=1)) < .65) & (np.linalg.norm(local,axis=1)<1.15)
                for axis in (side,np.broadcast_to([0.,0.,1.],side.shape)):
                    base = np.sum(local*axis,axis=1)
                    amount = np.sum(shift*axis,axis=1)
                    bound = np.maximum(.37,np.abs(base))
                    shift += (near*(np.clip(base+amount,-bound,bound)-base-amount))[:,None]*axis
                od[:,:3] += shift
            od[:,16:19] = np.where(active[:,None],moving,static)
            od[:,19] = np.where(active,radius,.20)
            od[:,20] = np.where(active,width,.055)
            od[:,21:28] = 0.
            print(json.dumps(dict(time=self.control_stamp,ego=ego,flow_pixels=len(field.xyz),
                                  dynamic_nodes=int(active.sum()),max_shift_m=float(np.linalg.norm(shift,axis=1).max()))),file=self.log)
            return
        if self.time_model is not None:
            from scripts.control.rti_problem import DT
            age = self.control_stamp-self.stamp
            tau = np.minimum(age+np.arange(len(points))*DT,1.6)
            od[:,16:19] = pose[:3]
            world_to_frd = np.diag([1.,-1.,-1.]) @ rotation(pose).as_matrix().T
            od[:,19:28] = world_to_frd.ravel(order="F")
            solver.time_field_data = np.column_stack((np.broadcast_to(self.image_bias[ego],(len(points),96)),tau))
            if self.local_flow is not None:
                # In the translating ego frame the common racing velocity cancels.
                # Each visible surface sample has its own measured scene flow.
                local = self.local_flow[ego]
                own = self.local_states[ego]
                leads = np.arange(len(points))*DT
                anchor = solver.X[:,:3]
                centers = np.full_like(anchor,1000.)
                if len(local.xyz):
                    relative = local.xyz-own[:3]
                    velocity = local.flow-own[3:6]
                    samples = relative[None,:,:]+leads[:,None,None]*velocity[None,:,:]
                    nearest = np.linalg.norm(samples,axis=2).argmin(axis=1)
                    centers = anchor+samples[np.arange(len(points)),nearest]
                solver.time_field_data = np.column_stack((solver.time_field_data,centers,np.exp(-leads/.4)))
            with torch.inference_mode():
                q = torch.tensor(points,dtype=torch.float32)[None]
                leads = torch.tensor(tau,dtype=torch.float32)[None,:,None]
                values = self.time_model.query_logits(self.context[ego:ego+1],q,leads).sigmoid()[0]
                now = self.time_model.query_logits(self.context[ego:ego+1],q,torch.zeros_like(leads)).sigmoid()[0]
            print(json.dumps(dict(time=self.stamp,decision_s=self.control_stamp,ego=ego,lead_s=tau.tolist(),
                                  node_risk=values.tolist(),same_nodes_zero_lead=now.tolist(),peak=float(values.max()))),
                  file=self.log)
            return
        centers, radius, width = self.fields[ego].support(points)
        od[:, 16:19] = rotation(pose).apply(centers*[1, -1, -1]) + pose[:3]
        od[:, 19] = radius
        od[:, 20] = width
        od[:, 21:28] = 0.
        implicit_log = {}
        if getattr(self, "implicit_fields", None) is not None:
            from scripts.control.rti_problem import DT
            nodes = solver.X[:, :3]
            node_times = self.control_stamp + np.arange(len(nodes))*DT
            logits, gradients = self.implicit_fields[ego].linearize(nodes, node_times)
            od[:, 21:24], od[:, 24], od[:, 25:28] = gradients, logits, nodes
            implicit_log = dict(decision_s=self.control_stamp, vision_s=self.capture_stamp,
                                dynamic_risk=(.5+.5*np.tanh(logits/2)).tolist(),
                                world_logit_gradient=gradients.tolist())
        correction = np.zeros(3)
        if self.refine:
            correction = self.update_reference(ego, pose, od)
        peak = self.fields[ego].query(points).max()
        print(json.dumps(dict(time=self.stamp, ego=ego, peak=float(peak),
                              min_width=float(width.min()), reference_shift=correction.tolist(), **implicit_log)), file=self.log)

    def update_reference(self, ego, pose, od):
        centers = np.asarray(self.nominal["centers"])
        gate = int(np.linalg.norm(centers-od[0, 34:37], axis=-1).argmin())
        center = centers[gate]
        side = np.array(self.nominal["coordinate_axes"][gate])
        up = np.array([0., 0., 1.])
        key = ego, gate
        shift = self.corrections.get(key, np.zeros(3))
        distance = np.linalg.norm(center-pose[:3])
        if 1.2 < distance < 5. and self.updated.get(key) != self.stamp:
            self.updated[key] = self.stamp
            offsets = np.linspace(-.36, .36, 19)
            delta = np.zeros(3)
            for axis, cross_axis in ((side, up), (up, side)):
                # Each physical bar is a ridge in the learned potential.
                # Require both opposite ridges to support the same correction.
                lines = center + np.array([-.81, .81])[:, None, None]*axis
                lines = lines + np.linspace(-.42, .42, 7)[None, :, None]*cross_axis
                samples = lines[:, None, :, None, :] + (offsets[:, None] +
                          np.array([-.35, 0., .35]))[None, :, None, :, None]*axis
                vals = self.fields[ego].query(camera_points(samples.reshape(-1, 3), pose)).reshape(2, 19, 7, 3)
                contrast = (vals[..., 1] - .5*(vals[..., 0]+vals[..., 2])).mean(axis=2)
                ids = (contrast-.08*np.abs(offsets)).argmax(axis=1)
                best = offsets[ids]
                gain = (contrast[np.arange(2), ids]-contrast[:, 9]).mean()
                if abs(best[0]-best[1]) <= .16 and gain > .055 and abs(best.mean()) >= .10:
                    delta += axis*best.mean()
            shift = .55*shift + .45*delta
            self.corrections[key] = shift
        weight = np.exp(-.5*(np.linalg.norm(od[:, :3]-center, axis=-1)/1.7)**2)
        od[:, :3] += weight[:, None]*shift
        return shift


class ZeroPotential:
    """The same side and offset minimizations with all visual scores set to zero."""
    def __init__(self, replay=None):
        self.allocations = {}
        if replay:
            centers = json.loads((replay/"course.json").read_text())["centers"]
            for update in json.loads((replay/"frontends.json").read_text()):
                for j, offset in enumerate(update["allocation"]):
                    gate = centers[(update["gate_index"]+j) % len(centers)]
                    key = (update["agent"], tuple(gate))
                    if key in self.allocations and not np.isclose(self.allocations[key], offset):
                        raise ValueError("Passage varied between replans in the source trial")
                    self.allocations[key] = offset
            if len(self.allocations) != 2*len(centers):
                raise ValueError("Source trial does not cover all passages")

    def observe(self, stamp, states):
        pass

    def passage(self, agent, gate, normal, axis):
        if self.allocations:
            return self.allocations[agent, tuple(gate)]
        # Exact zero-field tie in the ordered [-.36, +.36] candidate search.
        return -.36

    def install(self, solver, od):
        od[:, 16:28] = 0.
        od[:, 20] = 1.

    def close(self):
        pass
