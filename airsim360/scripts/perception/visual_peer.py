"""Reduce classified RGB-D surface returns to a causal peer observation."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
from scipy.spatial import cKDTree


class VisualPeer:
    max_age = .20

    def __init__(self):
        self.time = None
        self.center = None
        self.velocity = np.zeros(3)

    def estimate(self, time):
        if self.time is None or time - self.time > self.max_age + 1e-8:
            return None
        return np.r_[self.center + (time-self.time)*self.velocity, self.velocity]

    def update(self, time, xyz):
        previous = self.estimate(time)
        if previous is not None:
            radius = .18 + 21*(time-self.time)
            xyz = xyz[np.linalg.norm(xyz-previous[:3], axis=1) <= radius]
        if not len(xyz):
            return 0
        # One local surface patch, so disconnected frame returns are not averaged together.
        tree = cKDTree(xyz)
        counts = tree.query_ball_point(xyz, .095, return_length=True)
        candidates = np.flatnonzero(counts == counts.max())
        index = candidates[0]
        if previous is not None:
            index = candidates[np.linalg.norm(xyz[candidates]-previous[:3], axis=1).argmin()]
        cluster = tree.query_ball_point(xyz[index], .095)
        center = np.median(xyz[cluster], axis=0)
        if previous is None:
            self.velocity = np.zeros(3)
        else:
            dt = time-self.time
            measured = (center-self.center)/dt
            measured *= min(1., 21/max(np.linalg.norm(measured), 1e-8))
            self.velocity += (1-np.exp(-dt/.04))*(measured-self.velocity)
        self.center, self.time = center, time
        return len(cluster)


def visual_pair_states(own_states, estimates):
    # Keep each aircraft's own attitude, rates and motor forces for the actuator
    # model. Relative translation and velocity must come from visual sightings.
    relative = []
    for i, peer in enumerate(estimates):
        if peer is not None:
            relative.append((1 if i == 0 else -1)*(own_states[i,:6]-peer))
    if not relative:
        raise ValueError("Both visual peer observations expired; CBF unavailable")
    delta = np.mean(relative, axis=0)
    states = own_states.copy()
    anchor = 0 if estimates[0] is not None else 1
    states[1-anchor,:6] = states[anchor,:6] - (1 if anchor == 0 else -1)*delta
    return states
