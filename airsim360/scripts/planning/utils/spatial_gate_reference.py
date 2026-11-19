"""Arc-length Split-S approaches for the original MPCC rigid-body solver."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

from types import SimpleNamespace
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def transport(v, a, b):
    cross = np.cross(a, b)
    angle = np.arctan2(np.linalg.norm(cross), a @ b)
    return Rotation.from_rotvec(unit(cross)*angle).apply(v)


def lane_segment(a, b, na, nb, side_a, side_b, offset_a, offset_b):
    distance = np.linalg.norm(b-a)
    # Bezier-like turn with normal entry and exit, including the stacked reversal.
    tangent_scale = np.clip(.8*distance, 2.5, 7.)
    base = CubicHermiteSpline([0, 1], [a, b], [na*tangent_scale, nb*tangent_scale])
    u = np.linspace(0, 1, max(70, int(distance*16)))
    tangent = unit(base(u, 1))
    frame = [side_a]
    for old, new in zip(tangent, tangent[1:]):
        frame.append(transport(frame[-1], old, new))
    angle = np.arctan2(np.cross(frame[-1], side_b)@nb, frame[-1]@side_b)
    blend = u*u*(3-2*u)
    axes = Rotation.from_rotvec(tangent*(angle*blend)[:, None]).apply(frame)
    offset = offset_a + blend*(offset_b-offset_a)
    return base(u)+axes*offset[:, None]


class GateReference:
    def __init__(self, *args):
        from scripts.simulation.course import course
        self.course = course()
        self.c = dict(cells_per_leg=12)
        self.phase = SimpleNamespace(c=self.c)
        self.net = True
        self.paths = {}

    def build(self, own, agent):
        c = self.course
        centers, normals, axes = map(np.array, (c["centers"], c["normals"], c["coordinate_axes"]))
        center = np.array(c["start_position_center"])
        # Preserve exactly the physical initial position when entering the lane.
        observed_offset = float((own[:3]-center)@axes[0])
        normal, side = normals[0], axes[0]
        center = own[:3]-side*observed_offset
        parts = [own[None, :3]]
        gate_idx = []
        for seq in range(21):
            idx = seq % len(centers)
            gate, target_normal, target_side = centers[idx], normals[idx], axes[idx]
            offset = self.visual_passage(agent, gate, target_normal, target_side)
            # A straight normal corridor protects the whole 0.4 m frame thickness.
            entry, exit_point = gate-.65*target_normal, gate+.65*target_normal
            lane = lane_segment(center, entry, normal, target_normal, side, target_side, observed_offset, offset)
            parts.append(lane[1:])
            through = np.linspace(entry, exit_point, 41) + target_side*offset
            gate_idx.append(sum(map(len, parts))+19)
            parts.append(through[1:])
            center, normal, side, observed_offset = exit_point, target_normal, target_side, offset
        positions = np.concatenate(parts)
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
        return s, positions, np.array(gate_idx)

    def solve(self, own, unused, index):
        agent = self.active_agent
        if agent not in self.paths:
            self.paths[agent] = self.build(own, agent)
        s, positions, gates = self.paths[agent]
        lo = max(0, gates[index-1]-30) if index else 0
        hi = gates[index]
        begin = lo + np.linalg.norm(positions[lo:hi]-own[:3], axis=-1).argmin()
        selected = gates[index:index+5]
        end = selected[-1]+20
        path = positions[begin:end]
        local_s = s[begin:end]-s[begin]
        tangent = unit(np.gradient(path, axis=0))
        speed = 6.
        ids = np.arange(index, index+5)%7
        return dict(reference_states=np.c_[path, speed*tangent], reference_time=local_s/speed,
                    durations=np.diff(np.r_[0., (s[selected]-s[begin])/speed]),
                    gates=positions[selected], normals=np.array(self.course["normals"])[ids],
                    physical_gates=np.array(self.course["centers"])[ids],
                    allocation=np.sum((positions[selected]-np.array(self.course["centers"])[ids])*
                                      np.array(self.course["coordinate_axes"])[ids], axis=1),
                    kinematics_only=True, wall_seconds=0., seed_diagnostics={},
                    audit=dict(reference="geometric spatial corridor"))
