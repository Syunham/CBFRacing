"""Depth-only observations for the joint CBF and gate allocator."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import numpy as np
from scripts.simulation.renderer_process import RendererProcess


class DepthBarrierVision(RendererProcess):
    """Cooperative depth observations for allocation and the centralized pair QP."""
    def __init__(self, geometry, resolution, folder, separation):
        from scripts.simulation.dynamics import MODEL
        self.width, self.height = resolution
        self.separation = separation
        self.tracker = None
        self.stamp = None
        self.forces = np.full((2, 4), MODEL["mass"] * MODEL["gravity"] / 4)
        self.log = (folder/"depth_pair.jsonl").open("w", buffering=1)
        self.start_renderer(geometry, folder, None, depth_only=True)

    def observe(self, stamp, states):
        if self.stamp == stamp:
            return
        from scripts.simulation.course import course
        from scripts.perception.depth_pair import DepthPair
        if self.tracker is None:
            self.tracker = DepthPair(course(), self.separation)
        # Only the renderer receives the simulated camera transforms.
        poses = np.zeros_like(states)
        poses[:, :3], poses[:, 6:10] = states[:, :3], states[:, 6:10]
        self.worker.stdin.write((json.dumps([stamp, poses.tolist()])+"\n").encode())
        self.worker.stdin.flush()
        data = self.worker.stdout.read(2*self.width*self.height*4)
        ranges = np.frombuffer(data, "<f4").reshape(2, self.height, self.width)
        self.tracker.update(stamp, ranges)
        self.stamp = stamp
        x = self.tracker.barrier_state(self.forces) if self.tracker.rel is not None else None
        print(json.dumps(dict(time=stamp, hits=self.tracker.hits, velocity_valid=self.tracker.valid,
                              state=x.tolist() if x is not None else None)), file=self.log)

    def allocation_own(self, agent):
        cam = self.tracker.ego[agent]
        return np.r_[cam.p, cam.v]

    def allocation_peer(self, agent, own):
        return self.tracker.peer(agent)

    def visual_projection(self, pair_filter, states, nominal, dt):
        if self.tracker.rel is None or (self.stamp > 0 and not self.tracker.valid):
            if pair_filter.continue_infeasible:
                return nominal, dict(measurement_unavailable=True, correction_norm=0., policy="nominal")
            raise ValueError("Depth pair measurement unavailable")
        observed = self.tracker.barrier_state(self.forces)
        return pair_filter.project(observed, nominal, dt)

    def record_applied_control(self, controls, dt):
        self.forces += dt * controls[:, :4]

    def install(self, solver, od):
        od[:, 16:28] = 0.
        od[:, 20] = 1.
