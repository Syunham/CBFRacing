"""Five-leg duration regressor used by the Split-S fixed-time planner."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.paths import FLIGHT_ASSETS

DEFAULT_CHECKPOINT = FLIGHT_ASSETS / "models" / "segment_times_085.pt"
DEFAULT_HIDDEN_WIDTHS = (192, 192, 128)


def flight_features(own, gates):
    own = np.asarray(own, dtype=np.float64)
    gates = np.asarray(gates, dtype=np.float64)
    if own.shape != (19,) or gates.shape != (5, 3):
        raise ValueError("Expected a 19-value state and five world-frame xyz targets")
    x = np.r_[own[3:6], own[10:13], own[13:17], (gates - own[:3]).ravel()]
    if not np.isfinite(x).all():
        raise ValueError("Non-finite state or gate coordinates")
    return x.astype(np.float32)


class DurationMLP(nn.Module):
    def __init__(self, hidden_widths=DEFAULT_HIDDEN_WIDTHS):
        super().__init__()
        self.register_buffer("center", torch.zeros(45))
        self.register_buffer("scale", torch.ones(45))
        sizes = (45, *hidden_widths)
        layers = []
        for inputs, outputs in zip(sizes, sizes[1:]):
            layers.extend((nn.Linear(inputs, outputs), nn.SiLU()))
        self.layers = nn.Sequential(*layers, nn.Linear(sizes[-1], 5))

    @classmethod
    def from_checkpoint(cls, path):
        state = torch.load(path, weights_only=True, map_location="cpu")
        weights = [name for name in state if name.startswith("layers.") and name.endswith(".weight")]
        weights.sort(key=lambda name: int(name.split(".")[1]))
        model = cls(tuple(state[name].shape[0] for name in weights[:-1]))
        model.load_state_dict(state)
        return model

    @staticmethod
    def geometry(x):
        points = x[..., 10:].reshape(*x.shape[:-1], 5, 3)
        legs = torch.cat((points[..., :1, :], points[..., 1:, :] - points[..., :-1, :]), dim=-2)
        lengths = torch.linalg.vector_norm(legs, dim=-1)
        return torch.cat((x, legs.flatten(-2), lengths), dim=-1)

    def from_geometry(self, features):
        raw = self.layers((features - self.center) / self.scale)
        # Learn seconds per metre of straight-line displacement for each leg.
        return (features[..., -5:].clamp_min(1e-4) / 10) * nn.functional.softplus(raw) + 1e-6

    def forward(self, x):
        return self.from_geometry(self.geometry(x))


class SegmentTimeNetwork:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        self.checkpoint = Path(checkpoint)
        self.model = None

    @torch.inference_mode()
    def predict(self, own, gates, normals, half):
        if self.model is None:
            self.model = DurationMLP.from_checkpoint(self.checkpoint).eval()
        return self.model(torch.from_numpy(flight_features(own, gates))).numpy().astype(float)
