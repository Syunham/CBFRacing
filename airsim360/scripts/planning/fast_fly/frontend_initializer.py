"""Supervised primal guess only; the original physical optimizer certifies output."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
import torch
from torch import nn

STATE_SCALE=np.array([10.]*6+[1.]*4+[10.]*3)
FEATURE_SIZE=59
TARGET_SIZE=13*61+4*60+5

def encode_features(state,forces,gates,normals,half,force_weight,smooth_weight):
    state=np.asarray(state)
    return np.r_[state/STATE_SCALE,np.asarray(forces)/7.,((np.asarray(gates)-state[:3])/10.).ravel(),
                 np.asarray(normals).ravel(),np.asarray(half).ravel()/.7,force_weight/.1,np.log10(max(smooth_weight,1e-8))/5.]

def encode_target(X,U,T,state):
    x=np.asarray(X).copy();x[:3]-=np.asarray(state)[:3,None];x/=STATE_SCALE[:,None]
    return np.r_[x.ravel(order='F'),(np.asarray(U)/7).ravel(order='F'),np.log(np.asarray(T))]

def decode_target(y,state,lower_times,upper_times):
    y=np.asarray(y);nx=13*61
    if not np.isfinite(y).all():raise ValueError('Nonfinite learned initializer')
    x=y[:nx].reshape(13,61,order='F').copy()*STATE_SCALE[:,None];x[:3]+=np.asarray(state)[:3,None]
    qnorm=np.linalg.norm(x[6:10],axis=0);bad=qnorm<1e-6
    x[6:10,bad]=np.asarray(state)[6:10,None];x[6:10]/=np.maximum(np.linalg.norm(x[6:10],axis=0),1e-8)
    x[:,0]=state
    u=np.clip(7*y[nx:nx+240],0.,7.)
    times=np.clip(np.exp(np.clip(y[-5:],-20,5)),lower_times,upper_times)
    return np.r_[x.ravel(order='F'),u,times]

class PrimalInitializer(nn.Module):
    def __init__(self,xmean,xscale,ymean,yscale):
        super().__init__()
        for name,value in [('xmean',xmean),('xscale',xscale),('ymean',ymean),('yscale',yscale)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(FEATURE_SIZE,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,TARGET_SIZE))
    def normalized(self,x):return self.net((x-self.xmean)/self.xscale)
    def forward(self,x):return self.normalized(x)*self.yscale+self.ymean

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=PrimalInitializer(**ck['normalization']);model.load_state_dict(ck['state_dict']);model.eval()
    return model,ck
