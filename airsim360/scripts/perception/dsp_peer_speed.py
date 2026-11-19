"""Target-level 3-D particle velocity filter using DSP-style cluster associations."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def clusters(xyz):
    if not len(xyz):
        return []
    tree = cKDTree(xyz)
    graph = tree.sparse_distance_matrix(tree,.08,output_type="coo_matrix")
    count, ids = connected_components(graph,directed=False)
    groups = [xyz[ids==i] for i in range(count)]
    return [(g.mean(0),len(g)) for g in groups if len(g)>=2]


class DspPeerSpeed:
    """One target in each camera; callers must respect None for unknown velocities."""
    n = 768

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.last = []
        self.target = None
        self.last_time = None
        self.particles = None
        self.raw = None

    def update_groups(self, t, groups):
        dt = None if self.last_time is None else t-self.last_time
        self.raw = None
        if dt is not None and dt<=0:
            raise ValueError("Timestamps must increase")
        matches = {}
        if groups and self.last:
            old, counts = zip(*self.last)
            new, sizes = zip(*groups)
            distances = np.linalg.norm(np.array(old)[:,None,:]-np.array(new)[None,:,:],axis=2)
            valid = (distances <= .12+23*dt) & (abs(np.subtract.outer(counts,sizes))<=100)
            rows, cols = linear_sum_assignment(np.where(valid,distances,1e5))
            matches = {a:b for a,b in zip(rows,cols) if valid[a,b]}
        linked = self.target in matches
        current = matches[self.target] if linked else (max(range(len(groups)),key=lambda i:groups[i][1]) if groups else None)
        if linked:
            self.raw = (groups[current][0]-self.last[self.target][0])/dt
        else:
            self.particles = None
        self.last_time,self.last,self.target = t,groups,current
        if not linked:
            return None
        center = groups[current][0]
        fresh = self.n if self.particles is None else self.n//6
        if fresh < self.n:
            keep = self.particles[:self.n-fresh].copy()
            dv = self.rng.normal(0,35*dt,(len(keep),3))
            keep[:,:3] += (keep[:,3:]+.5*dv)*dt
            keep[:,3:] += dv
        else:
            keep = np.empty((0,6))
        # Mixture of static, flow-initialized and exploratory newborn hypotheses.
        v = self.raw+self.rng.normal(0,1.3,(fresh,3))
        v[:fresh//12] = 0.
        v[fresh//12:fresh//6] = self.rng.uniform(-18.,18.,(fresh//6-fresh//12,3))
        born = np.c_[center+self.rng.normal(0,.035,(fresh,3)), v]
        cloud = np.vstack([keep,born])
        logits = -.5*np.sum((cloud[:,:3]-center)**2,axis=1)/.035**2
        w = np.exp(logits-logits.max())
        w /= w.sum()
        velocity = w@cloud[:,3:]
        cdf = w.cumsum()
        cdf[-1] = 1.
        ids = cdf.searchsorted((np.arange(self.n)+self.rng.random())/self.n)
        self.particles = cloud[ids[self.rng.permutation(self.n)]]
        return velocity

    def update(self,t,xyz):
        return self.update_groups(t,clusters(xyz))
