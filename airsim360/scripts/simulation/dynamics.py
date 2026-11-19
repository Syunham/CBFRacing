"""Full quadrotor physical dynamics and JSON serialization."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from pathlib import Path
import os,json
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ.setdefault(key,"1")
os.environ.setdefault("MPLBACKEND","Agg")
import numpy as np
import casadi as ca
from scripts.paths import FLIGHT_ASSETS
BASE_CONFIG = json.loads((FLIGHT_ASSETS / "config.json").read_text(encoding="utf-8"))
MODEL = {}
_vehicle = None


def vehicle_parameters():
    return dict(BASE_CONFIG["vehicle"])


def select_vehicle(vehicle=None):
    global _vehicle
    _vehicle = vehicle
    vehicle = BASE_CONFIG["vehicle"] if vehicle is None else vehicle
    MODEL.update(mass=vehicle['mass'],arm=vehicle['arm_length'],inertia=vehicle['inertia'],
        torque_coefficient=vehicle['torque_coefficient'],gravity=vehicle['gravity'],
        thrust_rate_max=vehicle.get('thrust_rate_max', 200.),thrust_min=vehicle['thrust_min'],thrust_max=vehicle['thrust_max'],omega_max=BASE_CONFIG['body_rate_limit'],
        drag=vehicle['linear_drag_kg_s'],drag_units='kg/s',drag_frame='body',angular_constraint='Euclidean norm')

def planner_configuration():
    import copy
    config = copy.deepcopy(BASE_CONFIG)
    if _vehicle is not None:
        config.update(vehicle=dict(_vehicle), phase_cell_function=True, expand_solver_graph=False)
    return config


select_vehicle()


def save_json(path,data):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(data,default=default,ensure_ascii=False,indent=2),encoding='utf-8')


def rhs13(x,f,parameters=None):
    model=MODEL if parameters is None else parameters
    p=x[:3];v=x[3:6];q=x[6:10];w=x[10:13]
    qw,qx,qy,qz=[q[i] for i in range(4)]
    m=model['mass'];arm=model['arm']/np.sqrt(2);jx,jy,jz=model['inertia']
    tq=ca.vertcat(arm*(f[0]+f[1]-f[2]-f[3]),arm*(-f[0]+f[1]+f[2]-f[3]),model['torque_coefficient']*(f[0]-f[1]+f[2]-f[3]))
    thrust=ca.sum1(f)/m
    a=ca.vertcat(2*(qw*qy+qx*qz)*thrust,2*(qy*qz-qw*qx)*thrust,(1-2*(qx*qx+qy*qy))*thrust-model['gravity'])
    R=ca.vertcat(ca.horzcat(1-2*(qy*qy+qz*qz),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy)),
        ca.horzcat(2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-qw*qx)),
        ca.horzcat(2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)))
    a=a-ca.mtimes([R,ca.diag(ca.DM(model['drag'])),R.T,v])/m
    dq=.5*ca.vertcat(-qx*w[0]-qy*w[1]-qz*w[2],qw*w[0]+qy*w[2]-qz*w[1],qw*w[1]+qz*w[0]-qx*w[2],qw*w[2]+qx*w[1]-qy*w[0])
    dw=ca.vertcat((tq[0]-(jz-jy)*w[1]*w[2])/jx,(tq[1]-(jx-jz)*w[2]*w[0])/jy,(tq[2]-(jy-jx)*w[0]*w[1])/jz)
    return ca.vertcat(v,a,dq,dw)


def model_functions(substeps=8,augmented=True,parameters=None):
    nx=19 if augmented else 13;nu=5 if augmented else 4
    x=ca.SX.sym('x',nx);u=ca.SX.sym('u',nu);h=ca.SX.sym('h')
    if augmented:
        dx=ca.vertcat(rhs13(x[:13],x[13:17],parameters),u[:4],x[18],u[4])
    else:dx=rhs13(x,u,parameters)
    rhs=ca.Function('paper_rhs',[x,u],[dx]);nxt=x
    for _ in range(substeps):
        dt=h/substeps;k1=rhs(nxt,u);k2=rhs(nxt+dt/2*k1,u);k3=rhs(nxt+dt/2*k2,u);k4=rhs(nxt+dt*k3,u)
        nxt=nxt+dt/6*(k1+2*k2+2*k3+k4)
    return rhs,ca.Function('paper_step',[x,u,h],[nxt])
