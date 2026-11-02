"""Full quadrotor physical dynamics and JSON serialization."""
from pathlib import Path
import os,json
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ.setdefault(key,"1")
os.environ.setdefault("MPLBACKEND","Agg")
import numpy as np
import casadi as ca
MODEL={'mass': 0.85, 'arm': 0.15, 'inertia': [0.0025, 0.0021, 0.0043], 'torque_coefficient': 0.022, 'gravity': 9.81, 'thrust_min': 0.0, 'thrust_max': 7.0, 'omega_max': 10.0, 'drag': [0.0, 0.0, 0.0]}

def save_json(path,data):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(data,default=default,ensure_ascii=False,indent=2),encoding='utf-8')


def rhs13(x,f):
    p=x[:3];v=x[3:6];q=x[6:10];w=x[10:13]
    qw,qx,qy,qz=[q[i] for i in range(4)]
    m=MODEL['mass'];arm=MODEL['arm']/np.sqrt(2);jx,jy,jz=MODEL['inertia']
    tq=ca.vertcat(arm*(f[0]+f[1]-f[2]-f[3]),arm*(-f[0]+f[1]+f[2]-f[3]),MODEL['torque_coefficient']*(f[0]-f[1]+f[2]-f[3]))
    thrust=ca.sum1(f)/m
    a=ca.vertcat(2*(qw*qy+qx*qz)*thrust,2*(qy*qz-qw*qx)*thrust,(1-2*(qx*qx+qy*qy))*thrust-MODEL['gravity'])
    dq=.5*ca.vertcat(-qx*w[0]-qy*w[1]-qz*w[2],qw*w[0]+qy*w[2]-qz*w[1],qw*w[1]+qz*w[0]-qx*w[2],qw*w[2]+qx*w[1]-qy*w[0])
    dw=ca.vertcat((tq[0]-(jz-jy)*w[1]*w[2])/jx,(tq[1]-(jx-jz)*w[2]*w[0])/jy,(tq[2]-(jy-jx)*w[0]*w[1])/jz)
    return ca.vertcat(v,a,dq,dw)


def model_functions(substeps=8,augmented=True):
    nx=19 if augmented else 13;nu=5 if augmented else 4
    x=ca.SX.sym('x',nx);u=ca.SX.sym('u',nu);h=ca.SX.sym('h')
    if augmented:
        dx=ca.vertcat(rhs13(x[:13],x[13:17]),u[:4],x[18],u[4])
    else:dx=rhs13(x,u)
    rhs=ca.Function('paper_rhs',[x,u],[dx]);nxt=x
    for _ in range(substeps):
        dt=h/substeps;k1=rhs(nxt,u);k2=rhs(nxt+dt/2*k1,u);k3=rhs(nxt+dt/2*k2,u);k4=rhs(nxt+dt*k3,u)
        nxt=nxt+dt/6*(k1+2*k2+2*k3+k4)
    return rhs,ca.Function('paper_step',[x,u,h],[nxt])
