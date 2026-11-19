"""RGB-D surface transport and a query-conditioned, metric future occupancy potential."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.special import expit
from torch import nn

from scripts.perception.occupancy_flow import FlowPrior, OccupancyFlow
from scripts.perception.panorama import spherical_rays
from scripts.perception.spatial_potential import rotation


class VisualMotionPrior(FlowPrior):
    """Reuse only the RGB-D appearance classifier. Transport starts from observed surface flow."""
    def __init__(self, path):
        with np.load(path) as data:
            self.weight = [data['w0'],data['w1']]
            self.bias = [data['b0'],data['b1']]
        self.uncertainty_aware = False

    def displacement(self, xyz, flow):
        return flow[:,None,:] * (np.arange(17)[None,:,None]/10.)


def observe(prior, times, poses, rgb, depth):
    flow = OccupancyFlow(prior)
    for t, pose, color, ranges in zip(times, poses, rgb, depth):
        flow.observe(t, color, ranges, pose)
    if hasattr(prior,'conditioned') and len(flow.xyz):
        flow.original_shifts = flow.displacements
        if getattr(prior,'use_history',False):
            from scripts.perception.context_motion import history_keys
            flow.displacements=prior.predict(history_keys(flow,poses,times))
        else:
            flow.displacements = prior.conditioned(flow.xyz,flow.flow,poses[-1],times[-1])
        if hasattr(prior,'mixture'):
            from scripts.perception.context_motion import motion_keys
            flow.motion_keys=motion_keys(flow.xyz,flow.flow,poses[-1],times[-1])
            flow.hypotheses,flow.masses=prior.hypotheses(flow.motion_keys)
    return flow


def supports(flow, pose, lead):
    # Reserve distinct spatial cells before sampling additional returns on the same surface.
    xyz = flow.xyz
    _, first = np.unique(np.floor(xyz / .06).astype(int), axis=0, return_index=True)
    indices = np.r_[first, np.setdiff1d(np.arange(len(xyz)), first)][:32]
    count = len(indices)
    features = np.zeros((32, 13), np.float32)
    predicted = np.zeros((32, 3), np.float32)
    if count:
        step = np.clip(lead * 10, 0., 16.)
        k = min(int(step), 15)
        alpha = step - k
        delta = (1-alpha)*flow.displacements[indices,k] + alpha*flow.displacements[indices,k+1]
        predicted[:count] = xyz[indices] + delta
        if hasattr(flow,'original_shifts'):
            delta = (1-alpha)*flow.original_shifts[indices,k] + alpha*flow.original_shifts[indices,k+1]
        features[:count] = np.column_stack((
            (xyz[indices] - pose[:3])/6, flow.flow[indices]/15, delta/6,
            np.broadcast_to(pose[3:6]/15, (count,3)), np.full(count, lead)))
    return features, predicted, np.arange(32) < count, indices


class LocalFutureField(nn.Module):
    def __init__(self, correction_speed=3.):
        super().__init__()
        self.correction_speed = correction_speed
        self.head = nn.Sequential(nn.Linear(13, 48), nn.SiLU(), nn.Linear(48, 48),
                                  nn.SiLU(), nn.Linear(48, 5))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def kernels(self, features, predicted):
        raw = self.head(features)
        # Enforce zero motion correction at the observation time.
        centers = predicted + self.correction_speed*features[..., -1:] * raw[..., :3].tanh()
        radius = .35 + .4*raw[..., 3].sigmoid()
        width = .04 + .22*raw[..., 4].sigmoid()
        return centers, radius, width

    def forward(self, features, predicted, valid, queries):
        centers, radius, width = self.kernels(features, predicted)
        distances = torch.linalg.vector_norm(queries[:,:,None] - centers[:,None], dim=-1)
        logits = (radius[:,None] - distances)/width[:,None]
        return logits.masked_fill(~valid[:,None], -60.).amax(-1), centers


def static_tree(depth, pose, dynamic):
    mask = np.isfinite(depth) & (depth > .08) & (depth < 18.)
    cloud = spherical_rays(*depth.shape)[mask] * depth[mask,None]
    cloud = rotation(pose).apply(cloud * [1,-1,-1]) + pose[:3]
    if len(dynamic):
        dist, _ = cKDTree(dynamic).query(cloud, distance_upper_bound=.08)
        cloud = cloud[~np.isfinite(dist)]
    return cKDTree(cloud)


def kernel_geometry(model,flow,pose,leads):
    rows=[supports(flow,pose,float(t)) for t in leads]
    features,predicted,valid=(torch.from_numpy(np.stack(a)) for a in zip(*(row[:3] for row in rows)))
    centers,radius,width=model.kernels(features,predicted)
    axis=torch.zeros_like(centers)
    coefficient=torch.zeros_like(radius)
    if hasattr(model,'directional'):
        ids=rows[0][3]
        origin=torch.zeros_like(centers[0])
        origin[:len(ids)]=torch.as_tensor(flow.xyz[ids],dtype=centers.dtype)
        axis=torch.nn.functional.normalize(centers-origin,dim=-1,eps=1.e-6)
        radius,scale,width=model.directional(features,radius,width)
        coefficient=scale.reciprocal().square()-1.
    return tuple(a.numpy() for a in (centers,radius,width,valid,axis,coefficient))


def metric_distance(delta,axis,coefficient):
    return np.sqrt(np.maximum(0.,(delta*delta).sum(-1)+coefficient*(delta*axis).sum(-1)**2))


@torch.inference_mode()
def scores(model, flow, tree, pose, lead, queries):
    if hasattr(flow.prior,'mixture'):
        c,m,r,w=mixed_kernels(flow,pose,[lead])
        static=expit((.20-tree.query(queries)[0])/.055)
        if not c.shape[1]:
            return static,np.zeros(len(queries)),static
        dynamic=np.empty(len(queries))
        for start in range(0,len(queries),256):
            q=queries[start:start+256]
            distances=np.linalg.norm(q[:,None,None]-c[0],axis=-1)
            dynamic[start:start+len(q)]=(m[0]*expit((r[0,:,None]-distances)/w[0,:,None])).sum(-1).max(-1)
        return np.maximum(static,dynamic),dynamic,np.maximum(flow.score(queries,lead),static)
    c,r,w,valid,axis,coeff=kernel_geometry(model,flow,pose,[lead])
    distance=metric_distance(queries[:,None]-c,axis,coeff)
    logits=np.where(valid,(r-distance)/w,-60.).max(-1)
    static = expit((.20-tree.query(queries)[0])/.055)
    dynamic = expit(logits)
    return np.maximum(dynamic, static), dynamic, np.maximum(flow.score(queries,lead), static)


@torch.inference_mode()
def mixed_kernels(flow,pose,leads):
    indices=supports(flow,pose,0.)[3]
    n=len(indices);h=len(leads)
    if not n:
        return np.empty((h,0,3,3)),np.empty((h,0,3)),np.empty((h,0)),np.empty((h,0))
    t=np.clip(np.asarray(leads,dtype='f'),0,1.6)
    step=np.minimum((t*10).astype(int),15)
    a=(t*10-step).astype('f')[:,None,None,None]
    bank=flow.hypotheses[indices].transpose(1,0,2,3)
    shifts=(1-a)*bank[step]+a*bank[step+1]
    keys=np.tile(flow.motion_keys[indices],(h,1))
    mass=np.tile(flow.masses[indices],(h,1))
    tensors=(keys,shifts.reshape(-1,3,3),mass,np.repeat(t,n))
    m,r,w=flow.prior.mixture(*(torch.from_numpy(v) for v in tensors))
    return (shifts+flow.xyz[indices][None,:,None],m.numpy().reshape(h,n,3),
            r.numpy().reshape(h,n),w.numpy().reshape(h,n))


def mixture_control(flow,tree,pose,leads,queries):
    c,m,r,w=mixed_kernels(flow,pose,leads)
    dist,idx=tree.query(queries)
    static=tree.data[idx] if tree.n else np.broadcast_to([1e5,1e5,1e5],queries.shape)
    centers=np.repeat(static[:,None],3,axis=1)
    mass=np.zeros((len(queries),3));mass[:,0]=1
    radius=np.full(len(queries),.20);width=np.full(len(queries),.055)
    peak=expit((radius-dist)/width)
    moving=np.zeros(len(queries),bool)
    if c.shape[1]:
        distance=np.linalg.norm(queries[:,None,None]-c,axis=-1)
        value=(m*expit((r[:,:,None]-distance)/w[:,:,None])).sum(-1)
        best=value.argmax(1);rows=np.arange(len(best))
        moving=value[rows,best]>peak
        k=best[moving]
        centers[moving]=c[moving,k];mass[moving]=m[moving,k]
        radius[moving]=r[moving,k];width[moving]=w[moving,k]
        peak[moving]=value[moving,k]
    return centers,mass,radius,width,moving,peak


@torch.inference_mode()
def control_support(model, flow, tree, pose, leads, queries, with_metric=False):
    """Select the active metric kernel for each horizon node at the RTI linearization."""
    centers,radius,width,valid,axis,coeff=kernel_geometry(model,flow,pose,leads)
    logits=(radius-metric_distance(queries[:,None]-centers,axis,coeff))/width
    logits[~valid] = -np.inf
    nodes = np.arange(len(queries))
    best = np.argmax(logits,axis=1)
    distance, nearest = tree.query(queries)
    static = tree.data[nearest] if tree.n else np.broadcast_to([1e5,1e5,1e5],queries.shape)
    moving = logits[nodes,best] > (.20-distance)/.055
    result=(np.where(moving[:,None],centers[nodes,best],static),
            np.where(moving,radius[nodes,best],.20), np.where(moving,width[nodes,best],.055), moving)
    if with_metric:
        return (*result,np.where(moving[:,None],axis[nodes,best],0.),np.where(moving,coeff[nodes,best],0.))
    return result
