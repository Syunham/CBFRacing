"""Causal panorama-only relative kinematics for the pairwise barrier."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from collections import deque

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scripts.perception.panorama import spherical_rays
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def slope(history, quadratic=False):
    t, y = zip(*history)
    t = np.asarray(t)-t[-1]
    if len(t)<2:
        return np.zeros_like(y[0])
    design=np.c_[np.ones(len(t)),t]
    if quadratic and len(t)>=5:
        design=np.c_[design,t*t]
    return np.linalg.lstsq(design,np.array(y),rcond=None)[0][1]


def surfaces(depth):
    row,col=np.nonzero(np.isfinite(depth)&(depth>.08)&(depth<9))
    if not len(row):return []
    d=depth[row,col]
    index=np.full(depth.shape,-1,np.int32)
    index[row,col]=np.arange(len(row))
    sources,neighbors=[],[]
    for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
        yy=row+dy
        n=index[np.minimum(yy,len(depth)-1),(col+dx)%depth.shape[1]]
        ok=(yy<len(depth))&(n>=0)&(abs(d-d[n])<.05+.034*d)
        sources.append(np.flatnonzero(ok));neighbors.append(n[ok])
    a,b=np.concatenate(sources),np.concatenate(neighbors)
    graph=coo_matrix((np.ones(len(a),np.uint8),(a,b)),shape=(len(d),len(d)))
    count,ids=connected_components(graph,directed=False)
    xyz=spherical_rays(*depth.shape)[row,col]*d[:,None]*[1,-1,-1]
    lo,hi=np.full((count,3),np.inf),np.full((count,3),-np.inf)
    np.minimum.at(lo,ids,xyz);np.maximum.at(hi,ids,xyz)
    candidates=[]
    for j in np.flatnonzero(np.linalg.norm(hi-lo,axis=1)<.16):
        pts=xyz[ids==j]
        q=np.median(pts,axis=0)
        # Approximate center behind the near-facing surface of the 10.7 cm airframe.
        q+=.024*q/np.linalg.norm(q)
        if len(pts)>=2:
            candidates.append((q,len(pts)))
    return candidates


class DepthOdometry:
    """Frame-to-keyframe point-to-plane ICP, in the configured initial world frame."""
    def __init__(self,origin,orientation):
        self.p=origin.copy()
        self.R=orientation
        self.key=None
        self.history=deque(maxlen=9)
        self.v=np.zeros(3)
        self.angular_step=Rotation.identity()
        self.omega=np.zeros(3)

    def update(self,t,depth):
        ray=spherical_rays(*depth.shape)*[1,-1,-1]
        xyz=depth[...,None]*ray
        # Distributed background samples, with surface normals from neighboring rays.
        pts=xyz[3:-3:3,3:-3:3]
        dx=xyz[3:-3:3,4:-2:3]-xyz[3:-3:3,2:-4:3]
        dy=xyz[4:-2:3,3:-3:3]-xyz[2:-4:3,3:-3:3]
        n=np.cross(dx,dy)
        mag=np.linalg.norm(n,axis=2)
        valid=(np.isfinite(mag)&(mag>1e-8)&(np.linalg.norm(pts,axis=2)<45)
               &(np.linalg.norm(dx,axis=2)<1.2)&(np.linalg.norm(dy,axis=2)<1.2))
        pts,n=pts[valid],n[valid]/mag[valid,None]
        if self.key is not None:
            tree,ref,norm,key_time=self.key
            rot=self.R*self.angular_step
            pos=self.p+self.v*(t-self.history[-1][0])
            for _ in range(7):
                offset=rot.apply(pts)
                world=offset+pos
                d,idx=tree.query(world)
                normal=norm[idx]
                e=np.sum((world-ref[idx])*normal,axis=1)
                ok=(d<1.4)&(abs(e)<.35)&(np.abs(np.sum(rot.apply(n)*normal,axis=1))>.8)
                if ok.sum()<20:raise ValueError("Insufficient static geometry for depth odometry")
                w=np.minimum(1.,.045/(abs(e[ok])+1e-8))**.5
                J=np.c_[np.cross(offset[ok],normal[ok]),normal[ok]]
                step=np.linalg.lstsq(J*w[:,None],-e[ok]*w,rcond=1e-5)[0]
                rot=Rotation.from_rotvec(step[:3])*rot
                pos+=step[3:]
                if np.linalg.norm(step)<.00015:break
            self.angular_step=self.R.inv()*rot
            self.omega=self.angular_step.as_rotvec()/(t-self.history[-1][0])
            self.R,self.p=rot,pos
        self.history.append((t,self.p.copy()))
        self.v=slope(self.history,quadratic=True)
        if self.key is None or t-self.key[-1]>=.095:
            world=self.R.apply(pts)+self.p
            self.key=(cKDTree(world),world,self.R.apply(n),t)


class DepthPair:
    def __init__(self,course,separation):
        origin=np.asarray(course["start_position_center"])
        orientation=Rotation.from_quat(np.asarray(course["start_attitude"])[[1,2,3,0]])
        axis=np.asarray(course["coordinate_axes"][0])
        self.ego=[DepthOdometry(origin+(i-.5)*separation*axis,orientation) for i in (0,1)]
        self.history=deque(maxlen=8)
        self.rel=None
        self.dv=np.zeros(3)
        self.valid=False
        self.hits=[0,0]

    def update(self,t,depth):
        candidates=[]
        for i,(camera,d) in enumerate(zip(self.ego,depth)):
            camera.update(t,d)
            choices=surfaces(d)
            candidates.append([(camera.R.apply(p)*(1 if i==0 else -1),n) for p,n in choices])
        gap=t-self.history[-1][0] if self.history else 0
        prediction=None if self.rel is None else self.rel+self.dv*(t-self.time)
        options=[]
        for p,n in candidates[0]:
            for q,m in candidates[1]:
                err=np.linalg.norm(p-q)
                center=(p+q)*.5
                shift=np.linalg.norm(center-prediction) if prediction is not None else 0.
                if err<.17 and (prediction is None or shift<.09+7*gap):
                    options.append((err+shift-.005*np.log1p(min(n,m)),center,[n,m]))
        self.hits=[0,0]
        point=None
        if options:
            _,point,self.hits=min(options,key=lambda o:o[0])
        elif prediction is not None and gap<=.16:
            single=[(np.linalg.norm(p-prediction),p,i,n) for i,choices in enumerate(candidates) for p,n in choices]
            if single:
                dist,pt,i,n=min(single,key=lambda o:o[0])
                if dist<.12+8*gap:
                    point=pt;self.hits[i]=n
        if point is not None:
            if gap>.16:self.history.clear()
            self.history.append((t,point))
            self.rel=point
            self.dv=self.ego[1].v-self.ego[0].v
        elif prediction is not None:
            self.rel=prediction if gap<=.16 else None
        self.valid=self.rel is not None and len(self.history)>=2
        self.time=t

    def peer(self,i):
        cam=self.ego[i]
        if self.rel is None:return np.r_[cam.p,cam.v]
        return np.r_[cam.p,cam.v]+(1 if i==0 else -1)*np.r_[self.rel,self.dv]

    def translations(self):
        # Preserve binocular relative measurements and depth-derived common motion.
        common=np.mean([np.r_[c.p,c.v] for c in self.ego],axis=0)
        relative=np.r_[self.rel,self.dv]
        return np.stack([common-relative/2,common+relative/2])

    def barrier_state(self,forces):
        x=np.zeros((2,19))
        x[:,:6]=self.translations()
        for i,cam in enumerate(self.ego):
            x[i,6:10]=cam.R.as_quat()[[3,0,1,2]]
            x[i,10:13]=cam.omega
        x[:,13:17]=forces
        return x
