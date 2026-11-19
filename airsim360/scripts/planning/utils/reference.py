"""Arc-length reference and MPCC online-data construction."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
from scipy.interpolate import CubicHermiteSpline,CubicSpline
import os

def reference_from_plan(plan,initial):
    x=plan['reference_states'];ds=np.linalg.norm(np.diff(x[:,:3],axis=0),axis=1)
    raw=np.r_[0,np.cumsum(ds)];keep=np.r_[True,ds>1e-8];raw=raw[keep];p=x[keep,:3]
    first_gate_s=float(np.interp(plan['durations'][0],plan['reference_time'],np.r_[0,np.cumsum(ds)]))
    required=np.unique(np.r_[0.,np.interp(np.cumsum(plan['durations']),plan['reference_time'],np.r_[0,np.cumsum(ds)]),raw[-1]])
    spacing=float(os.getenv('REFERENCE_SPACING','.15'))
    grid=np.arange(0.,raw[-1],spacing)
    grid=grid[np.min(abs(grid[:,None]-required[None,:]),axis=1)>spacing/6]
    grid=np.sort(np.r_[grid,required])
    physical_tangent=None
    if os.getenv('REFERENCE_RESAMPLING','linear')=='hermite':
        velocity=x[keep,3:6];speed=np.linalg.norm(velocity,axis=1)
        physical_tangent=velocity/np.maximum(speed[:,None],1e-8)
        fallback=np.gradient(p,raw,axis=0)
        fallback/=np.maximum(np.linalg.norm(fallback,axis=1)[:,None],1e-8)
        physical_tangent[speed<.01]=fallback[speed<.01]
        positions=CubicHermiteSpline(raw,p,physical_tangent,axis=0)(grid)
    else:
        positions=np.array([np.interp(grid,raw,p[:,j]) for j in range(3)]).T
    tangent=np.gradient(positions,grid,axis=0);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    if np.linalg.norm(initial[3:6])>.01:tangent[0]=initial[3:6]/np.linalg.norm(initial[3:6])
    boundary=((1,physical_tangent[0]),(1,physical_tangent[-1])) if physical_tangent is not None else 'not-a-knot'
    spline=(CubicSpline(grid,positions,axis=0,bc_type=boundary) if os.environ.get('PREVIEW_SPLINE','hermite')=='c2' else CubicHermiteSpline(grid,positions,tangent,axis=0))
    # Analytic continuation evaluates the terminal spline polynomial; it adds
    # no waypoint event and imposes no endpoint position constraint on CPC.
    return dict(arc=grid,position=positions,spline=spline,initial=np.array(initial),length=raw[-1],first_gate_s=first_gate_s,
        next_gate=plan['gates'][0],next_normal=plan['normals'][0])

def online_data(ref,progress,neighbor,config,acceleration=None):
    s=np.asarray(progress);sp=ref['spline'];out=np.zeros((len(s),44))
    for j in range(4):out[:,3*j:3*j+3]=sp(s,j)/[1,1,2,6][j]
    out[:,12]=s
    gates=np.array(config['gates']);dist=np.sum((sp(s)[:,None,:]-gates[None,:,:])**2,axis=2)
    out[:,13]=config.get('qnom',1.)+config.get('qwp',20000.)*np.exp(-.5*dist/config.get('sigma',1.)**2).sum(axis=1)
    out[:,14]=0.;out[:,15]=1.
    # Neighbor and ego predictions must refer to the same physical grid.
    prediction_dt=float(config['dt'])
    if not np.isfinite(prediction_dt) or prediction_dt<=0:
        raise ValueError('MPCC prediction dt must be positive and finite')
    t=np.arange(len(s))*prediction_dt
    a=np.zeros(3) if acceleration is None else np.array(acceleration)
    # Only present-time neighbor observations and a causal estimate enter here.
    decay=config.get('acceleration_decay',.25)
    a_time=a[None,:]*np.exp(-t[:,None]/decay)
    v_time=neighbor[3:6]+a[None,:]*decay*(1-np.exp(-t[:,None]/decay))
    p_time=neighbor[:3]+neighbor[3:6]*t[:,None]+a[None,:]*decay*(t[:,None]-decay*(1-np.exp(-t[:,None]/decay)))
    out[:,16:19]=p_time;out[:,19:22]=v_time;out[:,22:25]=a_time;out[:,25:28]=-a_time/decay
    active=bool(config.get('cbf',True)) and np.linalg.norm(ref.get('current_position',ref['initial'][:3])-neighbor[:3])<=config.get('cbf_radius',np.inf)
    out[:,28]=active;out[:,29]=config.get('clearance',.5)
    out[:,30]=config.get('cbf_rate',6.);out[:,31]=config.get('clearance_margin',.05)
    out[:,32]=config.get('max_safety_slack',0.);out[:,33]=ref['length']
    out[:,34:37]=ref['next_gate'];out[:,37:40]=ref['next_normal']
    out[:,40]=.68 if config.get('gate_funnel',False) else 1000.
    out[:,41]=config.get('gate_slope',2.)
    out[:,42]=ref['first_gate_s'];out[:,43]=config.get('gate_progress_slope',0.)
    return out
