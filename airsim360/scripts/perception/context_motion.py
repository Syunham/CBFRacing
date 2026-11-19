"""Learn transport of visible RGB-D returns using similar training-flight fragments."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from scripts.perception.occupancy_flow import FlowPrior


def motion_keys(xyz, velocity, own, stamp):
    count = len(xyz)
    return np.column_stack((xyz, .55*(xyz-own[:3]), .09*velocity,
                           np.broadcast_to(.13*own[3:6],(count,3)),
                           np.full(count, 4*min(stamp,1.8)))).astype('f')


def extend_keys(keys, surface_velocity, poses, times):
    dt=max(.02,times[-1]-times[0])
    acceleration=(poses[-1,3:6]-poses[0,3:6])/dt
    return np.column_stack((keys,.15*surface_velocity,np.tile(.07*acceleration,(len(keys),1)),
                            np.tile(.16*poses[-1,10:13],(len(keys),1)))).astype('f')


def history_keys(flow,poses,times):
    keys=motion_keys(flow.xyz,flow.flow,poses[-1],times[-1])
    _,old_pose,old=flow.history[0]
    dt=times[-1]-times[0]
    velocity=flow.flow.copy()
    if dt>.001 and len(old) and len(flow.xyz):
        # Match the compact RGB-D patches in ego-translating coordinates, then average
        # neighboring returns within a Crazyflie-sized surface to reduce pixel jitter.
        before,now=old[:,:3],flow.xyz
        tree=cKDTree(before)
        distance,match=tree.query(now-poses[-1,:3]+old_pose)
        groups=cKDTree(now).query_ball_point(now,.095)
        sources=tree.query_ball_point(before[match],.095)
        for j in np.flatnonzero(distance<.15+20*dt):
            raw=(now[groups[j]].mean(0)-before[sources[j]].mean(0))/dt
            velocity[j]=raw*min(1.,21/max(np.linalg.norm(raw),1.e-6))
    return extend_keys(keys,velocity,poses,times)


class MotionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.match = nn.Sequential(nn.Linear(26,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,1))
        self.adjust = nn.Sequential(nn.Linear(26,64),nn.SiLU(),nn.Linear(64,48),nn.SiLU(),nn.Linear(48,12))
        nn.init.zeros_(self.adjust[-1].weight)
        nn.init.zeros_(self.adjust[-1].bias)

    def forward(self, key, offsets, shifts):
        scaled = key / 5.
        context = scaled[:,None].expand_as(offsets)
        logits = self.match(torch.cat((context,offsets),dim=-1)).squeeze(-1)
        weights = (logits - 2*torch.log(.03+offsets.norm(dim=-1))).softmax(1)
        prior = (weights[:,:,None,None]*shifts).sum(1)
        difference = (weights[:,:,None]*offsets).sum(1)
        knots = self.adjust(torch.cat((scaled,difference),dim=1)).reshape(-1,4,3)
        # Residual velocity at four 0.4 s intervals. Integrating makes displacement continuous.
        changes = .1*knots.tanh().repeat_interleave(4,dim=1)*5.
        return prior + torch.cat((torch.zeros_like(changes[:,:1]), changes.cumsum(1)),dim=1)


class FragmentMotion(nn.Module):
    """Learn how changes in a fragment's initial conditions change its future displacement."""
    def __init__(self,dimensions=13):
        super().__init__()
        self.transfer = nn.Sequential(nn.Linear(dimensions,48),nn.SiLU(),nn.Linear(48,dimensions*12))
        self.log_metric = nn.Parameter(torch.zeros(dimensions))
        nn.init.zeros_(self.transfer[-1].weight)
        nn.init.zeros_(self.transfer[-1].bias)

    def fragments(self, key, offsets, shifts):
        metric = (.8*self.log_metric.tanh()).exp()
        distance = (offsets*metric).norm(dim=-1)
        weights = (1/(.03+distance).square())
        weights = weights/weights.sum(1,keepdim=True)
        jacobian = self.transfer(key/5.).reshape(-1,key.shape[-1],12)
        # The correction vanishes for an exact initial-condition match.
        residual = (-offsets @ jacobian).reshape(*offsets.shape[:2],4,3)
        steps = .1*(4*residual.tanh()).repeat_interleave(4,dim=2)
        corrected = shifts + torch.cat((torch.zeros_like(steps[:,:,:1]),steps.cumsum(2)),dim=2)
        return weights,corrected

    def forward(self,key,offsets,shifts):
        weights,corrected=self.fragments(key,offsets,shifts)
        return (weights[:,:,None,None]*corrected).sum(1)


class ContextMotionPrior(FlowPrior):
    def __init__(self, appearance, checkpoint):
        super().__init__(appearance)
        self.context_keys = checkpoint['keys'].numpy()
        self.index = cKDTree(self.context_keys)
        self.shifts = checkpoint['shifts'].numpy()
        self.use_history=bool(checkpoint.get('use_history'))
        self.motion = (FragmentMotion(self.context_keys.shape[1]) if checkpoint.get('architecture')=='fragment_transfer' else MotionAttention()).eval()
        self.motion.load_state_dict(checkpoint['motion'])
        if 'mixture' in checkpoint:
            from scripts.training.train_motion_mixture import MixtureRisk
            self.mixture=MixtureRisk().eval()
            self.mixture.load_state_dict(checkpoint['mixture'])

    def neighbors(self,key):
        _, near = self.index.query(key,k=16)
        return tuple(torch.from_numpy(a) for a in (key,self.context_keys[near]-key[:,None],self.shifts[near]))

    @torch.inference_mode()
    def hypotheses(self,key):
        from scripts.training.train_motion_mixture import group_fragments
        return tuple(a.numpy() for a in group_fragments(self.motion,self.neighbors(key)))

    @torch.inference_mode()
    def conditioned(self, xyz, velocity, own, stamp):
        key = motion_keys(xyz,velocity,own,stamp)
        return self.predict(key)

    @torch.inference_mode()
    def predict(self,key):
        return self.motion(*self.neighbors(key)).numpy()
