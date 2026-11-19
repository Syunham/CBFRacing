"""Short-horizon potential from compact RGB-D surfaces and local scene flow."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)


import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import expit
from scripts.perception.panorama import spherical_rays
from scripts.perception.spatial_potential import rotation


def compact_surface(rgb, depth, own, with_features=False):
    """Keep samples on small connected surfaces. Use only the current camera and ego pose."""
    mask = np.isfinite(depth) & (depth > .08) & (depth < 12.)
    row, col = np.nonzero(mask)
    if not len(row):
        return (np.empty((0,6)), np.empty((0,12))) if with_features else np.empty((0,6))
    d = depth[row, col].astype(float)
    index = np.full(depth.shape, -1, np.int32)
    index[row, col] = np.arange(len(row))
    edges = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        yy = row + dy
        xx = (col + dx) % depth.shape[1]
        neighbor = index[yy.clip(max=depth.shape[0]-1), xx]
        ok = (yy < depth.shape[0]) & (neighbor >= 0) & (abs(d-d[neighbor]) < .05+.034*d)
        edges.append((np.flatnonzero(ok), neighbor[ok]))
    left, right = (np.concatenate(parts) for parts in zip(*edges))
    graph = coo_matrix((np.ones(len(left), np.uint8), (left, right)), shape=(len(d), len(d)))
    n, ids = connected_components(graph, directed=False)
    xyz = spherical_rays(*depth.shape)[row, col]*d[:, None]
    low, high = np.full((n,3), np.inf), np.full((n,3), -np.inf)
    np.minimum.at(low, ids, xyz)
    np.maximum.at(high, ids, xyz)
    keep = (np.linalg.norm(high-low, axis=1) < .22)[ids]
    xyz = rotation(own).apply(xyz[keep]*[1,-1,-1])+own[:3]
    points = np.column_stack((xyz, rgb[row[keep],col[keep]].astype(float)/255.))
    if not with_features:
        return points
    rr, cc, rad = row[keep], col[keep], d[keep]
    scale = depth.shape[1]/(2*np.pi)
    contrast = []
    for diameter in (.12, .32):
        offset = np.maximum(1,np.rint(diameter*scale/rad).astype(int))
        offsets = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
        surround = np.stack([depth[(rr+dy*offset).clip(0,depth.shape[0]-1),(cc+dx*offset)%depth.shape[1]] for dy,dx in offsets])
        contrast.extend([(surround > rad+.25).mean(0), (surround < rad-.25).mean(0)])
    extent = (high-low)[ids[keep]]
    features = np.column_stack((points[:,3:],rad/12, np.sort(extent,axis=1)*8,
                               np.log1p(np.bincount(ids)[ids[keep]])/5,*contrast))
    return points,features


class FlowPrior:
    """A sampled Eulerian displacement field fitted on training flights."""
    def __init__(self, path):
        with np.load(path) as f:
            self.motion_layers = None
            if "motion_w0" in f:
                self.motion_layers = [(f[f"motion_w{i}"], f[f"motion_b{i}"]) for i in range(3)]
                self.motion_mean, self.motion_std = f["motion_mean"], f["motion_std"]
            else:
                self.tree = cKDTree(f["keys"])
                self.delta = f["delta"]
            self.uncertainty_aware = bool(f["uncertainty_aware"]) if "uncertainty_aware" in f else False
            self.transient_buffer = bool(f["transient_buffer"]) if "transient_buffer" in f else False
            self.weight = [f["w0"],f["w1"]]
            self.bias = [f["b0"],f["b1"]]

    def visible(self, features):
        hidden = np.maximum(0,features @ self.weight[0] + self.bias[0])
        return expit(hidden @ self.weight[1] + self.bias[1]).ravel()

    def displacement(self, xyz, flow):
        if self.motion_layers is not None:
            h = (np.column_stack((xyz, flow)).astype(np.float32)-self.motion_mean)/self.motion_std
            for weight, bias in self.motion_layers[:-1]:
                h = np.maximum(0, h @ weight + bias)
            weight, bias = self.motion_layers[-1]
            increments = (h @ weight + bias).reshape(len(xyz), 16, 3)
            return np.concatenate((np.zeros_like(increments[:, :1]), increments.cumsum(1)), axis=1)
        dist, idx = self.tree.query(np.column_stack((xyz, .10*flow)), k=12)
        weights = 1/(.03+dist)**2
        return np.einsum("nk,nktd->ntd",weights/weights.sum(1,keepdims=True),self.delta[idx])


class OccupancyFlow:
    def __init__(self, prior=None):
        self.prior = prior
        self.history = []
        self.xyz = self.flow = np.empty((0,3))

    def observe(self, stamp, rgb, depth, own):
        if self.prior:
            points, features = compact_surface(rgb,depth,own,True)
            points = points[self.prior.visible(features) >= .5]
        else:
            points = compact_surface(rgb, depth, own)
        self.observe_points(stamp, points, own)

    def observe_points(self, stamp, points, own):
        """Update from already filtered XYZ/RGB surface samples."""
        self.xyz = points[:,:3]
        flow = np.zeros_like(self.xyz)
        if len(points):
            # Match surface samples in an ego-translating frame, then transport their
            # world-space occupancy. No persistent identities or vehicle state vector.
            for old_t, old_own, old in self.history[:1]:
                dt = stamp-old_t
                if len(old) and dt > 1e-5:
                    feature = np.column_stack((self.xyz-own[:3], .12*points[:,3:]))
                    previous = np.column_stack((old[:,:3]-old_own, .12*old[:,3:]))
                    distance, idx = cKDTree(previous).query(feature)
                    velocity = (self.xyz-old[idx,:3])/dt
                    # Prefer a stationary correspondence if the same surface is still there.
                    _, static = cKDTree(old[:,:3]).query(self.xyz)
                    shift = np.linalg.norm(self.xyz-old[static,:3],axis=1)
                    usable = (distance < .15+20*dt) & (shift > .08)
                    flow[usable] = velocity[usable]
                    speed = np.linalg.norm(flow, axis=1)
                    flow *= np.minimum(1., 21/np.maximum(speed,1e-6))[:,None]
        self.flow = flow
        if self.prior and len(points):
            self.displacements = self.prior.displacement(self.xyz,flow)
        self.history.append((stamp, own[:3].copy(), points))
        self.history = self.history[-4:]
        self.stamp = stamp

    def support(self, query, lead, for_control=False):
        query = np.atleast_2d(query)
        if not len(self.xyz):
            return np.full_like(query, 1e3), np.full(len(query), .70), np.full(len(query), .13)
        t = np.asarray(lead).reshape(-1)
        if self.prior:
            step = np.clip(t*10,0,16)
            index = np.minimum(step.astype(int),15)
            fraction = (step-index)[:,None,None]
            shifts = (1-fraction)*self.displacements[:,index].transpose(1,0,2)
            shifts += fraction*self.displacements[:,index+1].transpose(1,0,2)
            if self.prior.uncertainty_aware:
                # Preserve the measured short-term motion. Relax its influence as
                # extrapolation becomes unreliable in the stacked-gate reversal.
                residual = self.flow-self.displacements[:,1]/.1
                shifts += (.085*(1-np.exp(-t/.085))*np.exp(-t/.35))[:,None,None]*residual
            predicted = self.xyz[None,:,:]+shifts
        else:
            predicted = self.xyz[None,:,:] + t[:,None,None]*self.flow
        idx = np.linalg.norm(query[:,None]-predicted,axis=-1).argmin(axis=1)
        center = predicted[np.arange(len(query)) if len(predicted)>1 else 0, idx]
        if for_control and self.prior and self.prior.uncertainty_aware:
            if self.prior.transient_buffer:
                # Inflate near-term costs without blocking the distant gate corridor.
                radius = .70 + .20*(1-np.exp(-t/.08))*np.exp(-t/.55)
                return center, np.broadcast_to(radius,(len(query),)), np.full(len(query),.15)
            return center, np.broadcast_to(.72+np.minimum(2.5*t,.43),(len(query),)), np.full(len(query),.21)
        return center, np.full(len(query), .70), np.full(len(query), .13)

    def score(self, query, lead):
        center, radius, width = self.support(query, lead)
        return expit((radius-np.linalg.norm(query-center,axis=-1))/width)
