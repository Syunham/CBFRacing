"""Two-gate free-duration phase initialization and reference audit."""
import numpy as np
import casadi as ca
import torch
from scipy.interpolate import CubicHermiteSpline
from seed import seed_states

def axes(normal):
    lateral = np.cross([0., 0., 1.], normal)
    lateral /= np.linalg.norm(lateral)
    return lateral, np.cross(normal, lateral)


def speed_bound_ok(states,c,tol=3e-5):
    """None explicitly removes the translational speed constraint."""
    limit=c.get('speed_limit')
    return bool(limit is None or np.linalg.norm(np.asarray(states)[:,3:6],axis=1).max()<=limit+tol)


def integrate_symbolic(x, u, duration, c, substeps):
    v = c['vehicle']; inertia = ca.DM(v['inertia'])
    arm = v['arm_length']/np.sqrt(2); tc = v['torque_coefficient']
    def rhs(s):
        qw,qx,qy,qz = [s[i] for i in range(6,10)]
        wx,wy,wz = [s[i] for i in range(10,13)]
        qd = .5*ca.vertcat(-qx*wx-qy*wy-qz*wz, qw*wx+qy*wz-qz*wy,
            qw*wy+qz*wx-qx*wz, qw*wz+qx*wy-qy*wx)
        axis = ca.vertcat(2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx), 1-2*(qx*qx+qy*qy))
        torque = ca.vertcat(arm*(u[0]+u[1]-u[2]-u[3]),
            arm*(-u[0]+u[1]+u[2]-u[3]), tc*(u[0]-u[1]+u[2]-u[3]))
        return ca.vertcat(s[3:6], ca.sum1(u)/v['mass']*axis-ca.DM([0,0,v['gravity']]),
            qd, (torque-ca.cross(s[10:13],inertia*s[10:13]))/inertia)
    h = duration/substeps; states = []
    for _ in range(substeps):
        a=rhs(x); b=rhs(x+h*a/2); d=rhs(x+h*b/2); e=rhs(x+h*d)
        x=x+h*(a+2*b+2*d+e)/6
        x=ca.vertcat(x[:6],x[6:10]/ca.sqrt(ca.sumsqr(x[6:10])),x[10:13])
        states.append(x)
    return x, states


class GatePlanner:
    """Fixed event durations are parameters; envelope AD differentiates optimal value.

    The mesh uses the configured cells per leg and ends at each physical gate event.
    No auxiliary spatial target, terminal-rest equality, or collision term exists.
    """
    def __init__(self, c, course, optimize_durations=False):
        self.c=c; self.course=course; m=c['cells_per_leg']; n=2*m; self.n=n
        self.optimize_durations=optimize_durations
        X=ca.MX.sym('X',13,n+1); U=ca.MX.sym('U',4,n)
        x0=ca.MX.sym('x0',13); G=ca.MX.sym('G',3,2); N=ca.MX.sym('N',3,2)
        H=ca.MX.sym('half',2,2); T=ca.MX.sym('durations',2)
        p=ca.vertcat(x0,ca.vec(G),ca.vec(N),ca.vec(H),*([] if optimize_durations else [T]))
        z=ca.vertcat(ca.vec(X),ca.vec(U),*([T] if optimize_durations else [])); expr=[]; lo=[]; hi=[]
        def add(a,l=0.,h=0.):
            expr.append(ca.vec(a));lo.extend([l]*a.numel());hi.extend([h]*a.numel())
        world=np.asarray(course['world_box'])
        def physical(s,initial=False):
            # Interior margins protect against between-checkpoint extrema;
            # the independently audited physical limits remain unchanged.
            margin=0. if initial else c['physical_position_margin']
            qm=0. if initial else c['physical_quaternion_margin']
            if c.get('speed_limit') is not None:
                add(ca.sumsqr(s[3:6]), -np.inf, (c['speed_limit']-margin*10)**2)
            add(ca.sumsqr(s[10:13]), -np.inf, (c['body_rate_limit']-margin*10)**2)
            add(s[2], c['height_bounds'][0]+margin, c['height_bounds'][1]-margin)
            add(s[6], c['quaternion_scalar_min']+qm, 1.)
            add(s[7:10], -c['quaternion_vector_max']+qm, c['quaternion_vector_max']-qm)
            for j in range(3): add(s[j], float(world[j])+margin, float(world[j+3])-margin)
        add(X[:,0]-x0)
        physical(X[:,0],initial=True); f=ca.sum1(T)
        hover=c['vehicle']['mass']*c['vehicle']['gravity']/4
        for k in range(n):
            h=T[k//m]/m
            nxt,dense=integrate_symbolic(X[:,k],U[:,k],h,c,c['integration_substeps'])
            add(X[:,k+1]-nxt)
            for s in dense: physical(s)
            f+=c['effort_weight']*h*ca.sumsqr(U[:,k]-hover)
            f+=c['angular_weight']*h*ca.sumsqr(X[10:13,k])
            if k:
                f+=c['thrust_difference_weight']/h*ca.sumsqr(U[:,k]-U[:,k-1])
        for j,k in enumerate([m,n]):
            normal=N[:,j]; lateral=ca.cross(ca.DM([0,0,1]),normal)
            lateral=lateral/ca.sqrt(ca.sumsqr(lateral)); vertical=ca.cross(normal,lateral)
            displacement=X[:3,k]-G[:,j]
            add(ca.dot(normal,displacement))
            for axis,hidx in [(lateral,0),(vertical,1)]:
                a=ca.dot(axis,displacement); bound=H[hidx,j]-c['aperture_reserve']
                add(a-bound,-np.inf,0);add(-a-bound,-np.inf,0)
            add(ca.dot(normal,X[3:6,k]),c['gate_forward_speed_min'],np.inf)
            for before in range(j*m,k):
                add(ca.dot(normal,X[:3,before]-G[:,j]),-np.inf,1e-8)
            add(ca.dot(normal,X[:3,k-1]-G[:,j]),-np.inf,-.005)
            if j==0: add(ca.dot(normal,X[:3,k+1]-G[:,j]),.005,np.inf)
        g=ca.vertcat(*expr)
        opts={'print_time':False,'error_on_fail':False,'expand':c.get('expand_solver_graph',False),'ipopt':{
            'print_level':0,'sb':'yes','tol':2e-7,'acceptable_tol':1e-6,
            'max_iter':600,'max_wall_time':25.,'bound_relax_factor':0.,
            'mu_strategy':'adaptive','linear_solver':'mumps'}}
        self.solver=ca.nlpsol('gate_events','ipopt',{'x':z,'p':p,'f':f,'g':g},opts)
        lam=ca.MX.sym('lam',g.numel())
        self.envelope=ca.Function('envelope',[z,p,lam],[ca.gradient(f+ca.dot(lam,g),T)])
        self.lbg=np.array(lo);self.ubg=np.array(hi)
        self.lbx=np.r_[np.full(X.numel(),-np.inf),np.full(U.numel(),c['vehicle']['thrust_min'])]
        self.ubx=np.r_[np.full(X.numel(),np.inf),np.full(U.numel(),c['vehicle']['thrust_max'])]
        if optimize_durations:
            self.lbx=np.r_[self.lbx,np.full(2,c['duration_min'])]
            self.ubx=np.r_[self.ubx,np.full(2,c['duration_max'])]
        sx=ca.SX.sym('s',13);su=ca.SX.sym('u',4);sh=ca.SX.sym('h')
        sn,_=integrate_symbolic(sx,su,sh,c,c['audit_substeps'])
        self.step=ca.Function('independent_step',[sx,su,sh],[sn])

    def seed(self,state,gates,normals,durations):
        m=self.c['cells_per_leg']; ts=np.r_[0,np.cumsum(durations)]
        speed=np.clip(np.linalg.norm(np.diff(np.vstack([state[:3],gates]),axis=0),axis=1)/durations,.8,2.5)
        spline=CubicHermiteSpline(ts,np.vstack([state[:3],gates]),np.vstack([state[3:6],normals*speed[:,None]]))
        grid=np.r_[np.linspace(0,ts[1],m+1),np.linspace(ts[1],ts[2],m+1)[1:]]
        positions=torch.tensor(spline(grid)[None],dtype=torch.float64)
        cfg=dict(self.c,dt=float(sum(durations)/(2*m)),terminal_rest=False)
        x,u=seed_states(torch.tensor(state[None],dtype=torch.float64),positions,cfg)
        return np.r_[x[0].numpy().ravel(order='F'),u[0].numpy().ravel(order='F')]

    def solve(self,state,gates,normals,half,durations,warm=None):
        durations=np.asarray(durations,dtype=float)
        if durations.shape!=(2,) or not np.isfinite(durations).all() or np.any(durations<=0):
            raise ValueError('Two positive finite flight durations are required.')
        par=np.r_[state,gates.T.ravel(order='F'),normals.T.ravel(order='F'),half.T.ravel(order='F')]
        if not self.optimize_durations:par=np.r_[par,durations]
        start=self.seed(state,gates,normals,durations) if warm is None else warm
        if self.optimize_durations and len(start)==13*(self.n+1)+4*self.n:start=np.r_[start,durations]
        import time
        tick=time.perf_counter()
        trace=[]
        for attempt in range(self.c.get('phase_solve_retries',0)+1):
            r=self.solver(x0=start,p=par,lbg=self.lbg,ubg=self.ubg,lbx=self.lbx,ubx=self.ubx)
            elapsed=time.perf_counter()-tick; stats=self.solver.stats()
            zz=np.asarray(r['x']).ravel();gg=np.asarray(r['g']).ravel()
            violation=max(np.maximum(self.lbg-gg,0).max(),np.maximum(gg-self.ubg,0).max(),
                np.maximum(self.lbx-zz,0).max(),np.maximum(zz-self.ubx,0).max())
            good=bool(stats['success'] and np.isfinite(zz).all() and violation<=2e-5)
            trace.append(dict(attempt=attempt,status=stats['return_status'],seconds=elapsed,iterations=int(stats['iter_count']),residual=float(violation)))
            self.last_trace=trace;self.last_iterate=zz.copy()
            if good:break
            if not np.isfinite(zz).all():break
            start=zz.copy()
        if not good:raise RuntimeError('Phase initialization failed: '+str(trace))
        ns=13*(self.n+1)
        if self.optimize_durations:durations=zz[-2:]
        result=dict(X=zz[:ns].reshape(13,self.n+1,order='F'),U=zz[ns:ns+4*self.n].reshape(4,self.n,order='F'),
            durations=durations.copy(),z=zz,value=float(r['f']),
            gradient=np.asarray(self.envelope(r['x'],par,r['lam_g'])).ravel(),
            seconds=elapsed,iterations=sum(v['iterations'] for v in trace),residual=float(violation),solve_trace=trace,
            gates=gates.copy(),normals=normals.copy(),half=half.copy())
        return result

    def audit(self,plan,state):
        c=self.c;m=c['cells_per_leg'];x=state.copy();xs=[x.copy()];times=[0.];t=0.
        cell_durations=plan.get('cell_durations',np.repeat(plan['durations']/m,m))
        event_nodes=list(plan.get('event_nodes',[m,2*m]))
        node_error=0.; gate_records=[]
        for k in range(self.n):
            h=cell_durations[k]
            for _ in range(8):
                x=np.asarray(self.step(x,plan['U'][:,k],h/8)).ravel(); t+=h/8
                xs.append(x.copy());times.append(t)
            node_error=max(node_error,float(np.abs(x-plan['X'][:,k+1]).max()))
            if k+1 in event_nodes:
                j=event_nodes.index(k+1);normal=plan['normals'][j];a,b=axes(normal);d=x[:3]-plan['gates'][j]
                gate_records.append(dict(plane_error=float(abs(d@normal)),
                    aperture_coordinates=[float(d@a),float(d@b)],
                    aperture_passed=bool(abs(d@a)<=plan['half'][j,0] and abs(d@b)<=plan['half'][j,1]),
                    forward_speed=float(x[3:6]@normal)))
        xs=np.asarray(xs); world=np.asarray(self.course['world_box']);tol=3e-5
        bounds=bool(np.all(xs[:,:3]>=world[:3]-tol) and np.all(xs[:,:3]<=world[3:]+tol)
            and xs[:,2].min()>=c['height_bounds'][0]-tol and xs[:,2].max()<=c['height_bounds'][1]+tol
            and xs[:,6].min()>=c['quaternion_scalar_min']-tol
            and abs(xs[:,7:10]).max()<=c['quaternion_vector_max']+tol
            and speed_bound_ok(xs,c,tol)
            and np.linalg.norm(xs[:,10:13],axis=1).max()<=c['body_rate_limit']+tol)
        good=bool(bounds and node_error<.002 and all(g['plane_error']<.002 and g['aperture_passed']
            and g['forward_speed']>=c['gate_forward_speed_min']-tol for g in gate_records))
        return dict(passed=good,physical_bounds_passed=bounds,max_node_error=node_error,
            gate_records=gate_records,min_height=float(xs[:,2].min()),min_qw=float(xs[:,6].min()),
            max_qvector=float(abs(xs[:,7:10]).max()),max_height=float(xs[:,2].max()),
            max_speed=float(np.linalg.norm(xs[:,3:6],axis=1).max()),
            max_body_rate=float(np.linalg.norm(xs[:,10:13],axis=1).max()),
            position_min=xs[:,:3].min(0).tolist(),position_max=xs[:,:3].max(0).tolist()), np.asarray(times),xs
