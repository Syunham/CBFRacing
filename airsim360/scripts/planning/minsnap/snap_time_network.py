"""Predict five gate intervals for the local fixed-duration Minimum Snap QP."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn


def features(own, gates, normals, half, geometry):
    own, gates, normals, half = map(np.asarray, (own, gates, normals, half))
    q = own[6:10].copy()
    if q[0]<0:
        q *= -1
    base = [own[:6], q, own[10:17], (gates-own[:3]).ravel(), normals.ravel(), half.ravel()]
    if geometry:
        legs = np.diff(np.vstack((own[:3],gates)),axis=0)
        length = np.linalg.norm(legs,axis=1)
        unit = legs/np.maximum(length[:,None],1e-8)
        sides = np.cross([0,0,1],normals)
        side = sides[0]
        r = gates[0]-own[:3]
        v = own[3:6]
        forward = v@normals[0]
        depth = r@normals[0]
        rot = Rotation.from_quat(np.roll(q,-1)).as_matrix()
        bz = rot[:,2]
        base += [length, np.sum(unit*normals,axis=1),np.sum(unit*sides,axis=1),
                 np.sum(unit[1:]*unit[:-1],axis=1), bz,
                 [depth,r@side,r[2],forward,v@side,v[2],depth/max(forward,1.),
                  np.linalg.norm(v),np.sum(own[13:17])]]
        if geometry=="ballistic":
            # Under the baseline model the known initial acceleration includes body drag.
            from scripts.simulation.dynamics import MODEL
            a=bz*own[13:17].sum()/MODEL["mass"]-[0.,0.,MODEL["gravity"]]
            a-=rot@(np.asarray(MODEL["drag"])*(rot.T@v))/MODEL["mass"]
            t=np.clip(depth/max(forward,.35),.02,2.5)
            miss=r-v*t-.5*a*t*t
            base += [a,[t,miss@side,miss[2],np.linalg.norm(miss),r@v/max(v@v,.1),
                        np.linalg.norm(np.cross(r,v))/max(np.linalg.norm(v),.3)],
                     (gates[1:]-gates[:-1]).ravel()]
    return np.concatenate(base).astype(np.float32)


class TimeNet(nn.Module):
    def __init__(self, dimension, widths):
        super().__init__()
        self.register_buffer("center",torch.zeros(dimension))
        self.register_buffer("scale",torch.ones(dimension))
        self.register_buffer("log_center",torch.zeros(5))
        self.register_buffer("log_scale",torch.ones(5))
        self.register_buffer("lower",torch.tensor([.02,.4,.4,.4,.4]))
        layers=[]
        for width in widths:
            layers.extend([nn.Linear(dimension,width),nn.SiLU()])
            dimension=width
        self.layers=nn.Sequential(*layers,nn.Linear(dimension,5))

    def forward(self,x):
        raw=self.layers(((x-self.center)/self.scale).clamp(-10,10))
        return raw*self.log_scale+self.log_center

    def durations(self,x):
        return torch.maximum(self.lower,self(x).exp().clamp(max=8.))


class FixedTimePredictor:
    def __init__(self,path):
        ck=torch.load(path,map_location="cpu",weights_only=True)
        self.geometry=ck["geometry"]
        self.net=TimeNet(ck["dimension"],ck["widths"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()

    @torch.inference_mode()
    def predict(self,own,gates,normals,half):
        x=features(own,gates,normals,half,self.geometry)
        return self.net.durations(torch.from_numpy(x)).numpy().astype(float)

    def solve(self,qp,own,gates,normals,half):
        times=self.predict(own,gates,normals,half)
        return qp.solve(own,gates,normals,half,times),times
