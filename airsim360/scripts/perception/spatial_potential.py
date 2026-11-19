"""A learned potential on metric RGB-D surfaces. All query arguments are spatial."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from torch import nn
from scripts.perception.panorama import spherical_rays as rays


def rotation(state):
    return Rotation.from_quat(state[[7, 8, 9, 6]])


def camera_points(world, state):
    return rotation(state).inv().apply(world - state[:3]) * [1, -1, -1]


class SurfaceObservation:
    def __init__(self, rgb, depth):
        if rgb.shape != (*depth.shape, 3) or len(depth) != 4:
            raise ValueError("Four consecutive RGB-D panoramas required")
        valid = np.isfinite(depth) & (depth > .08) & (depth < 18.)
        d = np.where(valid, depth, 18.)
        visible = valid[-1]
        self.points = (rays(*visible.shape)[visible]*depth[-1, visible, None]).astype(np.float32)
        if not len(self.points):
            raise ValueError("No usable depth surface")
        color = rgb[:, visible].astype(np.float32)/255.
        change = np.diff(color, axis=0).mean(axis=-1).T
        features = np.column_stack((color[-1], change, d[-1, visible]/18,
                                    np.diff(d[:, visible], axis=0).T.clip(-2, 2)/2,
                                    (np.roll(d[-1], 1, axis=1)-d[-1])[visible].clip(-1, 1),
                                    (np.roll(d[-1], 1, axis=0)-d[-1])[visible].clip(-1, 1)))
        self.features = features.astype(np.float32)
        self.tree = cKDTree(self.points)

    def neighbors(self, q):
        return self.tree.query(q, k=min(8, len(self.points)))


class PotentialNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 32),
                                     nn.SiLU(), nn.Linear(32, 2))

    def forward(self, features):
        raw = self.encoder(features).sigmoid()
        return .04 + 1.3*raw[..., 0], .035 + .25*raw[..., 1]

    def logits(self, features, distances):
        radius, width = self(features)
        return ((radius - distances)/width).amax(-1)


class SpatialField:
    """Encode once per sensor update; query camera-frame positions in metres."""
    def __init__(self, network, rgb, depth):
        self.observation = SurfaceObservation(rgb, depth)
        with torch.inference_mode():
            self.radius, self.width = (x.cpu().numpy() for x in network(
                torch.from_numpy(self.observation.features).to(next(network.parameters()).device)))

    def support(self, points):
        points = np.atleast_2d(points)
        dist, idx = self.observation.neighbors(points)
        if idx.ndim == 1:
            idx, dist = idx[:, None], dist[:, None]
        which = ((self.radius[idx]-dist)/self.width[idx]).argmax(-1)
        active = idx[np.arange(len(idx)), which]
        return self.observation.points[active], self.radius[active], self.width[active]

    def query(self, points):
        centers, radius, width = self.support(points)
        logit = (radius-np.linalg.norm(points-centers, axis=-1))/width
        return 1/(1+np.exp(-logit.clip(-70, 70)))
