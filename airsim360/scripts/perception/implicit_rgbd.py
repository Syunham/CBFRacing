"""Continuous peer safety occupancy from native ERP frames and own camera poses."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
import torch.nn.functional as F

from scripts.training.train_rgbd_field import image_history


RADIUS = .55
HORIZON = 1.6
HISTORIES = {"short": (.06, .04, .02, 0.), "long": (.20, .08, .02, 0.)}


def history_indices(times, stamp, lags):
    """Choose the nearest available past observations, padding at startup."""
    times = np.asarray(times)
    end = np.searchsorted(times, stamp + 1e-8, side="right")
    if end == 0:
        raise ValueError("No past observation")
    wanted = stamp - np.asarray(lags)
    hi = np.minimum(np.searchsorted(times[:end], wanted), end-1)
    lo = np.maximum(hi-1, 0)
    return np.where(abs(times[lo]-wanted) < abs(times[hi]-wanted), lo, hi)


def camera_matrices(states):
    """Row-vector convention: (world - origin) @ matrix yields camera FRD."""
    return Rotation.from_quat(states[..., [7, 8, 9, 6]]).as_matrix() * [1., -1., -1.]


class ERPConv(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv2d(inputs, outputs, 3, stride=2), nn.GroupNorm(4, outputs), nn.SiLU())

    def forward(self, x):
        return super().forward(F.pad(F.pad(x, (1, 1, 0, 0), mode="circular"),
                                     (0, 0, 1, 1), mode="replicate"))


def interpolate_erp(feature, u, v, stride):
    """Bilinear ERP lookup with periodic longitude, stride-aligned CNN centres."""
    batch, channels, height, width = feature.shape
    x = (u / stride).remainder(width)
    y = (v / stride).clamp(0, height-1)
    floor_x = x.floor()
    ix, iy = floor_x.long() % width, y.floor().long()
    nx, ny = (ix+1) % width, (iy+1).clamp(max=height-1)
    index = torch.stack((iy*width+ix, iy*width+nx, ny*width+ix, ny*width+nx), -1)
    values = feature.flatten(2).gather(2, index.reshape(batch, 1, -1).expand(-1, channels, -1))
    dx, dy = x-floor_x, y-iy
    weights = torch.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), -1)
    return (values.reshape(batch, channels, -1, 4)*weights[:, None]).sum(-1).transpose(1, 2)


@dataclass
class Frame:
    raw: torch.Tensor
    fine: torch.Tensor
    coarse: torch.Tensor
    context: torch.Tensor
    origin: torch.Tensor
    rotation: torch.Tensor
    time: torch.Tensor


class ImplicitRGBD(nn.Module):
    def __init__(self, variant, speed_bound):
        super().__init__()
        if variant not in ("A", "B", "C") or not np.isfinite(speed_bound) or speed_bound <= 0:
            raise ValueError("Expected variant A/B/C and a positive speed bound in m/s")
        self.variant, self.speed_bound = variant, float(speed_bound)
        self.fine = ERPConv(5, 16)
        self.coarse = nn.Sequential(ERPConv(16, 24), ERPConv(24, 32))
        self.context = nn.Linear(32*2*4, 16)
        # Both depth and depth residual are masked by interpolated return validity.
        self.local = nn.Sequential(nn.Linear(16+32+4, 24), nn.SiLU())
        nframes = 1 if variant == "A" else 4
        self.base = nn.Sequential(nn.Linear(nframes*24+4*17+4, 64), nn.SiLU())
        if variant == "C":
            self.offset = nn.Linear(64, 6)
            self.evidence = nn.Sequential(nn.Linear(4*24, 24), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(64 + (48 if variant == "C" else 0), 32),
                                  nn.SiLU(), nn.Linear(32, 1))

    def encode_frame(self, image, origin, rotation, stamp):
        if image.shape[1:] != (5, 256, 512):
            raise ValueError("Expected native 5 x 256 x 512 RGB-D input")
        fine = self.fine(image)
        coarse = self.coarse(fine)
        context = self.context(F.adaptive_avg_pool2d(coarse, (2, 4)).flatten(1))
        return Frame(image[:, 3:], fine, coarse, context, origin, rotation, stamp)

    def read_point(self, frame, world):
        q = (world-frame.origin[:, None]) @ frame.rotation
        horizontal = torch.linalg.vector_norm(q[..., :2], dim=-1)
        has_longitude = horizontal > 1e-5
        x, y, z = q.unbind(-1)
        lon = torch.atan2(torch.where(has_longitude, y, 0.), torch.where(has_longitude, x, 1.))
        lat = torch.atan2(z, horizontal.clamp_min(1e-5))
        u, v = (lon/(2*torch.pi)+.5)*512-.5, (lat/torch.pi+.5)*256-.5
        dist = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        depth, valid = interpolate_erp(frame.raw, u, v, 1).split(1, -1)
        residual = ((dist*valid-50*depth)/4).clamp(-2, 2)
        features = torch.cat((interpolate_erp(frame.fine, u, v, 2),
                              interpolate_erp(frame.coarse, u, v, 8),
                              depth, valid, residual, dist/10), -1)
        # The ERP direction is undefined at the optical centre.
        fade = (dist/.10).clamp(max=1)
        return self.local(features) * (fade.square()*(3-2*fade))

    def query_logits(self, frames, world, tau):
        latest = frames[-1]
        frd = (world-latest.origin[:, None]) @ latest.rotation
        context = torch.cat([torch.cat((f.context, (f.time-latest.time)[:, None]), -1)
                             for f in frames], -1)
        local_frames = frames[-1:] if self.variant == "A" else frames
        local = [self.read_point(f, world) for f in local_frames]
        base = self.base(torch.cat([*local, context[:, None].expand(-1, world.shape[1], -1),
                                    frd/5, tau/HORIZON], -1))
        if self.variant == "C":
            offset = self.offset(base).reshape(*world.shape[:2], 2, 3).tanh()
            norm = torch.linalg.vector_norm(offset, dim=-1, keepdim=True).clamp_min(1)
            offset = offset/norm * (RADIUS + tau*self.speed_bound).unsqueeze(-1)
            # Offsets live in current FRD. Both hypotheses are sampled in one batch.
            delta = offset.flatten(1, 2) @ latest.rotation.transpose(1, 2)
            sites = (world[:, :, None]+delta.reshape_as(offset)).flatten(1, 2)
            evidence = self.evidence(torch.cat([self.read_point(f, sites) for f in frames], -1))
            base = torch.cat((base, evidence.reshape(*world.shape[:2], -1)), -1)
        return self.head(base).squeeze(-1)

    def forward(self, images, origins, rotations, times, world, tau):
        frames = [self.encode_frame(images[:, k], origins[:, k], rotations[:, k], times[:, k])
                  for k in range(4)]
        return self.query_logits(frames, world, tau)


class CachedField:
    """One agent; encode a new observation once and query all world-frame nodes."""
    def __init__(self, model, history):
        self.model = model.eval()
        self.lags = HISTORIES[history]
        self.frames = deque()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def observe(self, stamp, own, rgb, depth):
        if self.frames and stamp <= self.frames[-1][0]:
            if stamp == self.frames[-1][0]:
                return
            raise ValueError("Visual timestamps went backwards")
        image = image_history(rgb[None], depth[None], np.ones(1, bool)).to(self.device)
        args = [torch.as_tensor(v, dtype=torch.float32, device=self.device)
                for v in (own[None, :3], camera_matrices(own[None]), [stamp])]
        self.frames.append((stamp, self.model.encode_frame(image, *args)))
        while len(self.frames) > 1 and self.frames[1][0] <= stamp-max(self.lags):
            self.frames.popleft()

    @torch.enable_grad()
    def linearize(self, world, node_times):
        stamps, frames = zip(*self.frames)
        indices = history_indices(stamps, stamps[-1], self.lags)
        leads = np.asarray(node_times)-stamps[-1]
        if np.any(leads < -1e-7):
            raise ValueError("Node time precedes the visual observation")
        q = torch.tensor(np.asarray(world)[None], dtype=torch.float32, device=self.device, requires_grad=True)
        tau = torch.as_tensor(leads.clip(0, HORIZON), dtype=q.dtype, device=q.device)[None, :, None]
        logits = self.model.query_logits([frames[i] for i in indices], q, tau)
        grad, = torch.autograd.grad(logits.sum(), q)
        if not torch.isfinite(logits).all() or not torch.isfinite(grad).all():
            raise FloatingPointError("Nonfinite MPCC risk or gradient")
        return logits[0].detach().cpu().numpy(), grad[0].cpu().numpy()


def load_field(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["format"] != "implicit_rgbd_v1" or ck["radius"] != RADIUS:
        raise ValueError("Incompatible implicit RGB-D checkpoint")
    model = ImplicitRGBD(ck["variant"], ck["speed_bound"])
    model.load_state_dict(ck["model"])
    return model.eval(), ck["history"]
