"""Exact supplied track and the unchanged full rigid-body physical model."""
from pathlib import Path
import sys,os,json,time,hashlib
HERE=Path(__file__).resolve().parent
from model import np,ca,MODEL,save_json,model_functions
import torch
torch.set_num_threads(1)

POSITIONS=np.array([[-.6,-.86,3.68],[9.,6.45,1.05],[8.85,-3.8,1.05],[-4.3,-5.6,3.4],[-4.3,-5.6,1.42],[4.5,-.45,1.05],[-1.95,6.81,1.05]])
QUATERNIONS=np.array([[.9848078,0,0,-.1736482],[1.,0,0,0],[.4226183,0,0,-.9063078],[0.,0,0,-1.],[1.,0,0,0],[.7660444,0,0,.6427876],[.258819,0,0,-.9659258]])
START=np.array([-5.,4.7,.61]);ATTITUDE=np.array([.9238795,0,0,-.3826834]);WORLD=np.array([-10.,-8.,.05,14.,9.,6.])

def course():
    q=QUATERNIONS/np.linalg.norm(QUATERNIONS,axis=1)[:,None];w,x,y,z=q.T
    normals=np.c_[1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)];axes=[]
    # This defines coordinates only; no UAV identifier or side assignment is
    # inserted into the controller. Gate passage offsets are network outputs.
    for n in normals:
        a=np.cross([0.,0.,1.],n);a/=np.linalg.norm(a)
        if axes and a@axes[-1]<0:a=-a
        axes.append(a)
    return dict(name='Exact supplied Split-S',centers=POSITIONS.tolist(),normals=normals.tolist(),coordinate_axes=np.array(axes).tolist(),
        rotations_wxyz=QUATERNIONS.tolist(),sizes=np.tile([.4,1.5,1.5],(7,1)).tolist(),half_extents=np.full((7,2),.75).tolist(),
        start_position_center=START.tolist(),start_attitude=(ATTITUDE/np.linalg.norm(ATTITUDE)).tolist(),world_box=WORLD.tolist(),scale=1.,
        required_laps=2,required_events_per_agent=14,extra_waypoints=0,physical_gate_events_per_cpc=configuration()['preview_gates'])

def initial_states(separation=1.3,forward_offset=0.):
    c=course();q=np.array(c['start_attitude']);side=np.array(c['coordinate_axes'][0]);direction=POSITIONS[0]-START;direction/=np.linalg.norm(direction)
    out=np.zeros((2,19));out[:,6:10]=q;out[:,13:17]=MODEL['mass']*MODEL['gravity']/4
    for i in range(2):out[i,:3]=START+(i-.5)*(separation*side+forward_offset*direction)
    return out

def configuration():
    c=json.loads((HERE/'config.json').read_text(encoding='utf-8'));c['preview_gates']=int(os.environ.get('CPC_PREVIEW_GATES',c['online_planning']['preview_gates']));return c

def gate_pair(index,count=2):
    c=course();ids=np.arange(index,index+count)%7
    return tuple(np.array(c[k])[ids] for k in ['centers','normals','half_extents','coordinate_axes'])
