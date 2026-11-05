"""Finite geometric optimizer initializer at vertical/inverted configurations."""
from cbfracing.core.context import *
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

def finite_seed(state,gates,normals,durations,c):
    m=c['cells_per_leg'];times=np.r_[0,np.cumsum(durations)]
    speeds=np.clip(np.linalg.norm(np.diff(np.vstack([state[:3],gates]),axis=0),axis=1)/durations,.8,2.5)
    curve=CubicHermiteSpline(times,np.vstack([state[:3],gates]),np.vstack([state[3:6],normals*speeds[:,None]]))
    grid=np.r_[0.,np.concatenate([np.linspace(times[j],times[j+1],m+1)[1:] for j in range(len(durations))])]
    position=curve(grid);velocity=curve(grid,1);acceleration=curve(grid,2);force=acceleration+[0,0,9.81]
    magnitude=np.linalg.norm(force,axis=1);axis=force/np.maximum(magnitude[:,None],1e-8)
    axis[magnitude<1e-8]=[0,0,1];matrices=[]
    for z in axis:
        heading=np.array([1.,0.,0.])
        if abs(z@heading)>.97:heading=np.array([0.,1.,0.])
        y=np.cross(z,heading);y/=np.linalg.norm(y);x=np.cross(y,z);matrices.append(np.column_stack([x,y,z]))
    quat=Rotation.from_matrix(np.array(matrices)).as_quat()[:,[3,0,1,2]];quat[0]=state[6:10]
    for k in range(1,len(quat)):
        if quat[k]@quat[k-1]<0:quat[k]*=-1
    derivative=np.gradient(quat,grid,axis=0)
    omega=2*(quat[:,:1]*derivative[:,1:]-derivative[:,:1]*quat[:,1:]-np.cross(quat[:,1:],derivative[:,1:]))
    norm=np.linalg.norm(omega,axis=1);omega*=np.minimum(1.,8./np.maximum(norm,1e-8))[:,None]
    X=np.c_[position,velocity,quat,omega];X[0]=state
    vehicle=c['vehicle'];inertia=np.array(vehicle['inertia']);dw=np.diff(omega,axis=0)/np.diff(grid)[:,None];wm=(omega[1:]+omega[:-1])/2
    torque=dw*inertia+np.cross(wm,wm*inertia);collective=vehicle['mass']*(magnitude[:-1]+magnitude[1:])/2
    arm=vehicle['arm_length']/np.sqrt(2);tc=vehicle['torque_coefficient'];allocation=np.array([[1,1,1,1],[arm,arm,-arm,-arm],[-arm,arm,arm,-arm],[tc,-tc,tc,-tc]])
    U=np.c_[collective,torque]@np.linalg.inv(allocation).T;U=np.clip(U,vehicle['thrust_min'],vehicle['thrust_max'])
    result=np.r_[X.T.ravel(order='F'),U.T.ravel(order='F')];assert np.isfinite(result).all();return result
