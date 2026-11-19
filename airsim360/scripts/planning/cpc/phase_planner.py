"""Configurable gate-preview phase planner with free durations and dense audit."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
import casadi as ca
import torch
from scipy.interpolate import CubicHermiteSpline
from scripts.planning.cpc.seed import seed_states

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
        R=ca.vertcat(ca.horzcat(1-2*(qy*qy+qz*qz),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy)),
            ca.horzcat(2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-qw*qx)),
            ca.horzcat(2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)))
        drag=ca.mtimes([R,ca.diag(ca.DM(v['linear_drag_kg_s'])),R.T,s[3:6]])/v['mass']
        return ca.vertcat(s[3:6], ca.sum1(u)/v['mass']*axis-ca.DM([0,0,v['gravity']])-drag,
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
        self.c=c; self.course=course; m=c['cells_per_leg']; M=c.get('preview_gates',2); n=M*m; self.n=n; self.M=M
        self.optimize_durations=optimize_durations
        X=ca.MX.sym('X',13,n+1); U=ca.MX.sym('U',4,n)
        x0=ca.MX.sym('x0',13); G=ca.MX.sym('G',3,M); N=ca.MX.sym('N',3,M)
        H=ca.MX.sym('half',2,M); T=ca.MX.sym('durations',M)
        p=ca.vertcat(x0,ca.vec(G),ca.vec(N),ca.vec(H),*([] if optimize_durations else [T]))
        self.base_parameter_count=int(p.numel())
        self.initial_force_weight=float(c.get('phase_initial_force_weight',0.))
        if not np.isfinite(self.initial_force_weight) or self.initial_force_weight<0:raise ValueError('Initial-force weight must be finite and nonnegative')
        self.initial_forces=None
        if self.initial_force_weight:
            if not c.get('fast_fly_phase',False):raise ValueError('Initial-force penalty is an experimental Fast-fly option')
            initial_force=ca.MX.sym('initial_force',4);p=ca.vertcat(p,initial_force)
        z=ca.vertcat(ca.vec(X),ca.vec(U),*([T] if optimize_durations else [])); expr=[]; lo=[]; hi=[]
        self.cell_rows=[];self.gate_rows=[];dynamics=[]
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
        if self.initial_force_weight:f+=self.initial_force_weight*ca.sumsqr(U[:,0]-initial_force)
        hover=c['vehicle']['mass']*c['vehicle']['gravity']/4
        cell_function=None
        if c.get('phase_cell_function',False) or c.get('phase_mapped_cells',False):
            cs=ca.SX.sym('cell_state',13);cu=ca.SX.sym('cell_control',4);ct=ca.SX.sym('cell_time')
            cn,cd=integrate_symbolic(cs,cu,ct,c,c['integration_substeps'])
            cell_function=ca.Function('phase_cell',[cs,cu,ct],[cn,ca.horzcat(*cd)])
        mapped=None
        if c.get('phase_mapped_cells',False):
            mapped=cell_function.map(n)(X[:,:n],U,ca.horzcat(*[T[k//m]/m for k in range(n)]))
        for k in range(n):
            row_start=len(lo)
            h=T[k//m]/m
            if mapped is not None:
                nxt=mapped[0][:,k]
                dense=[mapped[1][:,k*c['integration_substeps']+j] for j in range(c['integration_substeps'])]
            elif cell_function is None:
                nxt,dense=integrate_symbolic(X[:,k],U[:,k],h,c,c['integration_substeps'])
            else:
                nxt,dd=cell_function(X[:,k],U[:,k],h)
                dense=[dd[:,j] for j in range(c['integration_substeps'])]
            add(X[:,k+1]-nxt)
            dynamics.append(X[:,k+1]-nxt)
            for s in dense: physical(s)
            self.cell_rows.append(slice(row_start,len(lo)))
            if not c.get('fast_fly_phase',False):
                f+=c['effort_weight']*h*ca.sumsqr(U[:,k]-hover)
                f+=c['angular_weight']*h*ca.sumsqr(X[10:13,k])
                if k:
                    f+=c['thrust_difference_weight']/h*ca.sumsqr(U[:,k]-U[:,k-1])
            elif k and c.get('phase_thrust_smooth',0)>0:
                f+=c['phase_thrust_smooth']/h*ca.sumsqr(U[:,k]-U[:,k-1])
        for j,k in enumerate(range(m,n+1,m)):
            row_start=len(lo)
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
            if j<M-1: add(ca.dot(normal,X[:3,k+1]-G[:,j]),.005,np.inf)
            self.gate_rows.append(slice(row_start,len(lo)))
        g=ca.vertcat(*expr)
        opts={'print_time':False,'error_on_fail':False,'expand':c.get('expand_solver_graph',False),'ipopt':{
            'print_level':0,'sb':'yes','tol':2e-7,'acceptable_tol':1e-6,
            'max_iter':600,'max_wall_time':25.,'bound_relax_factor':0.,
            'mu_strategy':'adaptive','linear_solver':'mumps'}}
        if c.get('fast_fly_phase',False):
            opts['ipopt'].update(warm_start_init_point='yes',warm_start_bound_push=1e-6,
                                warm_start_mult_bound_push=1e-6,warm_start_slack_bound_push=1e-6)
        opts['ipopt'].update(c.get('phase_ipopt_options',{}))
        if c.get('phase_native_library'):
            from pathlib import Path
            import json
            library=Path(c['phase_native_library']).resolve()
            generated=json.loads((library.parent/'native_generation.json').read_text())['config']
            keys=['body_rate_limit','height_bounds','quaternion_scalar_min','quaternion_vector_max',
                  'gate_forward_speed_min','aperture_reserve','physical_position_margin','physical_quaternion_margin',
                  'integration_substeps','speed_limit','preview_gates']
            # The generated shooting model uses thrust directly. Motor slew
            # applies to the MPCC actuator state and is absent in older metadata.
            vehicle_fields = ("mass", "inertia", "arm_length", "torque_coefficient",
                              "gravity", "thrust_min", "thrust_max", "linear_drag_kg_s")
            same_vehicle = all(c['vehicle'][k] == generated['vehicle'][k] for k in vehicle_fields)
            if (not optimize_durations or m!=generated['phase_cells_per_leg'] or not same_vehicle or
                    any(c[k]!=generated[k] for k in keys)):
                raise ValueError('Native derivatives do not match this planner configuration')
            opts.update(grad_f=ca.external('nlp_grad_f',str(library)),
                        jac_g=ca.external('nlp_jac_g',str(c.get('phase_jacobian_library',library))),
                        hess_lag=ca.external('nlp_hess_l',str(c.get('phase_hessian_library',library))))
            if self.initial_force_weight:
                # Only the objective has new parameters. Preserve the compiled
                # physical constraints and their exact derivatives unchanged.
                native_p=p[:self.base_parameter_count]
                native_g=ca.external('nlp_g',str(library))
                self.constraint_value=ca.Function('handoff_constraints',[z,p],[native_g(z,native_p)])
                opts['jac_g']=ca.Function('handoff_jacobian',[z,p],list(opts['jac_g'](z,native_p)))
                native_hessian=opts['hess_lag']
                hs=ca.MX.sym('sigma');hl=ca.MX.sym('multipliers',g.numel())
                opts['hess_lag']=ca.Function('handoff_constraint_hessian',[z,p,hs,hl],[native_hessian(z,native_p,hs,hl)])
            if c.get('phase_thrust_smooth',0)>0 or self.initial_force_weight:
                sigma=ca.MX.sym('objective_scale');multipliers=ca.MX.sym('constraint_multipliers',g.numel())
                constraint_hessian=opts['hess_lag'](z,p,sigma,multipliers)
                opts['grad_f']=ca.Function('regularized_gradient',[z,p],[f,ca.gradient(f,z)])
                opts['hess_lag']=ca.Function('regularized_hessian',[z,p,sigma,multipliers],
                    [constraint_hessian+sigma*ca.triu(ca.hessian(f,z)[0])])
        self.solver=ca.nlpsol('gate_events','ipopt',{'x':z,'p':p,'f':f,'g':g},opts)
        self.nlp_definition=dict(x=z,p=p,f=f,g=g)
        self.solver_options=opts
        self.symbols=dict(X=X,U=U,T=T,p=p,x0=x0,G=G,dynamics=ca.vertcat(*dynamics))
        lam=ca.MX.sym('lam',g.numel())
        self.envelope=ca.Function('envelope',[z,p,lam],[ca.gradient(f+ca.dot(lam,g),T)])
        self.lbg=np.array(lo);self.ubg=np.array(hi)
        self.lbx=np.r_[np.full(X.numel(),-np.inf),np.full(U.numel(),c['vehicle']['thrust_min'])]
        self.ubx=np.r_[np.full(X.numel(),np.inf),np.full(U.numel(),c['vehicle']['thrust_max'])]
        if optimize_durations:
            self.lbx=np.r_[self.lbx,np.full(M,c['duration_min'])]
            self.ubx=np.r_[self.ubx,np.full(M,c['duration_max'])]
        sx=ca.SX.sym('s',13);su=ca.SX.sym('u',4);sh=ca.SX.sym('h')
        sn,_=integrate_symbolic(sx,su,sh,c,c['audit_substeps'])
        self.step=ca.Function('independent_step',[sx,su,sh],[sn])
        self.rollout_batch=self.step.mapaccum('phase_audit_rollout',n*8)

    def parameters(self,state,gates,normals,half,durations=None):
        values=np.r_[state,gates.T.ravel(order='F'),normals.T.ravel(order='F'),half.T.ravel(order='F')]
        if not self.optimize_durations:values=np.r_[values,durations]
        if self.initial_force_weight:
            forces=np.asarray(self.initial_forces,dtype=float)
            if forces.shape!=(4,) or not np.isfinite(forces).all():raise ValueError('Measured four-rotor forces are required for initial-force penalty')
            values=np.r_[values,forces]
        return values

    def seed(self,state,gates,normals,durations):
        m=self.c['cells_per_leg']; ts=np.r_[0,np.cumsum(durations)]
        speed=np.clip(np.linalg.norm(np.diff(np.vstack([state[:3],gates]),axis=0),axis=1)/durations,.8,2.5)
        spline=CubicHermiteSpline(ts,np.vstack([state[:3],gates]),np.vstack([state[3:6],normals*speed[:,None]]))
        grid=np.r_[0.,*[np.linspace(ts[j],ts[j+1],m+1)[1:] for j in range(len(durations))]]
        positions=torch.tensor(spline(grid)[None],dtype=torch.float64)
        cfg=dict(self.c,dt=float(sum(durations)/(2*m)),terminal_rest=False)
        x,u=seed_states(torch.tensor(state[None],dtype=torch.float64),positions,cfg)
        return np.r_[x[0].numpy().ravel(order='F'),u[0].numpy().ravel(order='F')]

    def solve(self,state,gates,normals,half,durations,warm=None,dual_start=None):
        durations=np.asarray(durations,dtype=float)
        if durations.shape!=(self.M,) or not np.isfinite(durations).all() or np.any(durations<=0):
            raise ValueError(f'{self.M} positive finite flight durations are required.')
        par=self.parameters(state,gates,normals,half,durations)
        start=self.seed(state,gates,normals,durations) if warm is None else warm
        if self.optimize_durations and len(start)==13*(self.n+1)+4*self.n:start=np.r_[start,durations]
        import time
        tick=time.perf_counter()
        trace=[]
        for attempt in range(self.c.get('phase_solve_retries',0)+1):
            dual={} if dual_start is None else dict(lam_x0=dual_start[0],lam_g0=dual_start[1])
            r=self.solver(x0=start,p=par,lbg=self.lbg,ubg=self.ubg,lbx=self.lbx,ubx=self.ubx,**dual)
            elapsed=time.perf_counter()-tick; stats=self.solver.stats()
            zz=np.asarray(r['x']).ravel();gg=np.asarray(r['g']).ravel()
            violation=max(np.maximum(self.lbg-gg,0).max(),np.maximum(gg-self.ubg,0).max(),
                np.maximum(self.lbx-zz,0).max(),np.maximum(zz-self.ubx,0).max())
            good=bool(stats['success'] and np.isfinite(zz).all() and violation<=2e-5)
            trace.append(dict(attempt=attempt,status=stats['return_status'],seconds=elapsed,iterations=int(stats['iter_count']),residual=float(violation)))
            trace[-1]['solver_timing']={key:float(value) for key,value in stats.items()
                if key.startswith('t_wall_') or key.startswith('t_proc_')}
            history=stats.get('iterations',{})
            trace[-1]['iteration_history']={key:list(value) for key,value in history.items()
                if key in ('inf_pr','inf_du','obj','alpha_pr','mu')}
            primal=np.asarray(history.get('inf_pr',[]),dtype=float)
            candidates=np.flatnonzero(primal<=2e-5)
            trace[-1]['first_primal_feasible_iteration']=int(candidates[0]) if len(candidates) else None
            self.last_trace=trace;self.last_iterate=zz.copy()
            if good:break
            if not np.isfinite(zz).all():break
            start=zz.copy()
        if not good:raise RuntimeError('Phase initialization failed: '+str(trace))
        ns=13*(self.n+1)
        if self.optimize_durations:durations=zz[-self.M:]
        result=dict(X=zz[:ns].reshape(13,self.n+1,order='F'),U=zz[ns:ns+4*self.n].reshape(4,self.n,order='F'),
            durations=durations.copy(),z=zz,value=float(r['f']),
            gradient=np.asarray(self.envelope(r['x'],par,r['lam_g'])).ravel(),
            seconds=elapsed,iterations=sum(v['iterations'] for v in trace),residual=float(violation),solve_trace=trace,
            gates=gates.copy(),normals=normals.copy(),half=half.copy(),
            lam_x=np.asarray(r['lam_x']).ravel(),lam_g=np.asarray(r['lam_g']).ravel())
        return result

    def audit(self,plan,state):
        c=self.c;m=c['cells_per_leg'];x=state.copy();xs=[x.copy()];times=[0.];t=0.
        cell_durations=plan.get('cell_durations',np.repeat(plan['durations']/m,m))
        event_nodes=list(plan.get('event_nodes',list(range(m,self.n+1,m))))
        node_error=0.; gate_records=[]
        batch=None
        if c.get('phase_batched_audit',False):
            batch=np.asarray(self.rollout_batch(state,np.repeat(plan['U'],8,axis=1),np.repeat(cell_durations/8,8)[None,:]))
        for k in range(self.n):
            h=cell_durations[k]
            for sub in range(8):
                x=(batch[:,8*k+sub] if batch is not None else np.asarray(self.step(x,plan['U'][:,k],h/8)).ravel()); t+=h/8
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
