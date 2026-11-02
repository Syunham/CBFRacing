"""Free-time CPC with explicit progress, decrement and distance slack.

Each physical gate owns one free point inside its aperture. These are the two
existing gate events, not additional waypoints. The original distance CPC is
mu*(||p-gate_point||^2-nu)=0 with bounded nu; plane/aperture/direction conditions
retain the physical gate interpretation. Network times initialize the solve.
"""
import phase_planner as legacy
from model import ca,np
import time,json


class OriginalCPC:
    def __init__(self,c,course):
        self.c=c;self.course=course;self.n=n=2*c['cells_per_leg'];self.ns=ns=13*(n+1)
        X=ca.MX.sym('X',13,n+1);U=ca.MX.sym('U',4,n)
        LI=ca.MX.sym('lambda_inner',2,n-1);M=ca.MX.sym('mu',2,n)
        V=ca.MX.sym('nu',2,n);P=ca.MX.sym('gate_offsets',2,2);T=ca.MX.sym('T')
        L=ca.horzcat(ca.DM.ones(2,1),LI,ca.DM.zeros(2,1))
        initial=ca.MX.sym('initial',13);G=ca.MX.sym('gates',3,2);N=ca.MX.sym('normals',3,2)
        H=ca.MX.sym('half',2,2);eps=ca.MX.sym('epsilon');p=ca.vertcat(initial,ca.vec(G),ca.vec(N),ca.vec(H),eps)
        z=ca.vertcat(ca.vec(X),ca.vec(U),ca.vec(LI),ca.vec(M),ca.vec(V),ca.vec(P),T)
        self.offsets=dict(lam=ns+4*n,mu=ns+4*n+2*(n-1),nu=ns+4*n+2*(n-1)+2*n,point=ns+4*n+2*(n-1)+4*n)
        g=[];lo=[];hi=[];labels=[];products=[];dt=T/n;world=np.array(course['world_box'])
        def add(a,l=0.,h=0.,name='constraint'):
            g.append(ca.vec(a));lo.extend([l]*a.numel());hi.extend([h]*a.numel());labels.extend([name]*a.numel())
        def product(a,name):products.append(ca.vec(a));add(a-eps,-np.inf,0.,name)
        def physical(s,start=False):
            pm=0. if start else c['physical_position_margin']
            add(ca.sumsqr(s[10:13]),-np.inf,(c['body_rate_limit']-10*pm)**2,'body_rate')
            add(s[2],c['height_bounds'][0]+pm,c['height_bounds'][1]-pm,'height')
            add(s[6],-1.,1.,'quaternion');add(s[7:10],-1.,1.,'quaternion')
            for d in range(3):add(s[d],float(world[d])+pm,float(world[d+3])-pm,'arena')
        add(X[:,0]-initial,name='initial');physical(X[:,0],True)
        for k in range(n):
            nxt,dense=legacy.integrate_symbolic(X[:,k],U[:,k],dt,c,c['integration_substeps'])
            add(X[:,k+1]-nxt,name='dynamics')
            for s in dense:physical(s)
        add(L[:,1:]-L[:,:-1]+M,name='progress_dynamics')
        add(L[0,1:n]-L[1,1:n],-np.inf,0.,'event_order')
        times=[]
        for j in range(2):
            tau=dt*ca.dot(ca.DM(np.arange(1,n+1)),M[j,:].T);times.append(tau)
            a=ca.cross(ca.DM([0,0,1]),N[:,j]);a/=ca.sqrt(ca.sumsqr(a));b=ca.cross(N[:,j],a)
            point=G[:,j]+a*P[0,j]+b*P[1,j]
            for hidx in range(2):add(P[hidx,j]-(H[hidx,j]-c['aperture_reserve']),-np.inf,0.,'gate_point');add(-P[hidx,j]-(H[hidx,j]-c['aperture_reserve']),-np.inf,0.,'gate_point')
            for k in range(1,n+1):
                d=X[:3,k]-G[:,j];mu=M[j,k-1];plane=ca.dot(N[:,j],d)
                phi=mu*(ca.sumsqr(X[:3,k]-point)-V[j,k-1])
                product(phi,'distance_cpc');product(-phi,'distance_cpc')
                product(mu*plane,'gate_plane');product(-mu*plane,'gate_plane')
                for axis,hidx in [(a,0),(b,1)]:
                    q=ca.dot(axis,d);bound=H[hidx,j]-c['aperture_reserve']
                    product(mu*(q-bound),'gate_aperture');product(mu*(-q-bound),'gate_aperture')
                product(mu*(c['gate_forward_speed_min']-ca.dot(N[:,j],X[3:6,k])),'gate_speed')
                product(mu*(ca.dot(N[:,j],X[:3,k-1]-G[:,j])+.005),'gate_before')
                if j==0 and k<n:product(mu*(-ca.dot(N[:,j],X[:3,k+1]-G[:,j])+.005),'gate_after')
                activation=L[j,k] if j==0 else (1-L[0,k])*L[1,k]
                product(activation*plane,'before_completion')
            if j==1:
                d=X[:3,-1]-G[:,j];add(ca.dot(N[:,j],d),name='terminal_gate')
                add(ca.dot(N[:,j],X[3:6,-1]),c['gate_forward_speed_min'],np.inf,'terminal_direction')
                for axis,hidx in [(a,0),(b,1)]:
                    q=ca.dot(axis,d);bound=H[hidx,j]-c['aperture_reserve']
                    add(q-bound,-np.inf,0.,'terminal_aperture');add(-q-bound,-np.inf,0.,'terminal_aperture')
        add(times[0],c['duration_min'],c['duration_max'],'time_bounds')
        add(T-times[0],c['duration_min'],c['duration_max'],'time_bounds')
        gg=ca.vertcat(*g);self.labels=labels;self.last_trace=[];self.last_diagnostics=None
        self.products=ca.Function('original_products',[z,p],[ca.vertcat(*products)])
        options=dict(print_time=False,error_on_fail=False,expand=True,ipopt=dict(print_level=0,sb='yes',tol=1e-8,acceptable_tol=1e-7,
            max_iter=c.get('cpc_max_iter',1400),max_wall_time=c.get('cpc_wall_time',30.),bound_relax_factor=0.,mu_strategy='adaptive',linear_solver='mumps'))
        self.solver=ca.nlpsol('original_cpc','ipopt',dict(x=z,p=p,f=T,g=gg),options)
        self.lbg=np.array(lo);self.ubg=np.array(hi)
        tol=c.get('cpc_waypoint_tolerance',.03)
        # mu>=0, lambda endpoints and progress dynamics already imply
        # lambda in[0,1] and mu<=1. Avoid duplicated active bound rows.
        self.lbx=np.r_[np.full(ns,-np.inf),np.full(4*n,c['vehicle']['thrust_min']),np.full(2*(n-1),-np.inf),np.zeros(4*n),np.full(4,-np.inf),2*c['duration_min']]
        self.ubx=np.r_[np.full(ns,np.inf),np.full(4*n,c['vehicle']['thrust_max']),np.full(2*(n-1)+2*n,np.inf),np.full(2*n,tol**2),np.full(4,np.inf),2*c['duration_max']]
        self.times=ca.Function('original_times',[z],[ca.vertcat(*times)])

    def seed(self,phase):
        m=self.c['cells_per_leg'];n=self.n;T=float(sum(phase['durations']))
        old=np.r_[np.linspace(0,phase['durations'][0],m+1),np.linspace(phase['durations'][0],T,m+1)[1:]]
        new=np.linspace(0,T,n+1);x=np.array([np.interp(new,old,row) for row in phase['X']]);x[6:10]/=np.linalg.norm(x[6:10],axis=0)
        cells=np.clip(np.searchsorted(old,(new[:-1]+new[1:])/2,side='right')-1,0,n-1);u=phase['U'][:,cells]
        mu=np.zeros((2,n));k=int(np.clip(round(n*phase['durations'][0]/T),1,n-1));mu[0,k-1]=1.;mu[1,-1]=1.
        lam=1-np.cumsum(mu,axis=1);pts=np.zeros((2,2));nu=np.zeros((2,n))
        for j in range(2):
            a,b=legacy.axes(phase['normals'][j]);node=[k,n][j];diff=x[:3,node]-phase['gates'][j]
            pts[:,j]=np.clip([diff@a,diff@b],-(phase['half'][j]-self.c['aperture_reserve']),phase['half'][j]-self.c['aperture_reserve'])
            point=phase['gates'][j]+a*pts[0,j]+b*pts[1,j]
            nu[j]=np.minimum(np.sum((x[:3,1:]-point[:,None])**2,axis=0),self.c.get('cpc_waypoint_tolerance',.03)**2)
        return np.r_[x.ravel(order='F'),u.ravel(order='F'),lam[:,:-1].ravel(order='F'),mu.ravel(order='F'),nu.ravel(order='F'),pts.ravel(order='F'),T]

    def solve(self,state,gates,normals,half,phase=None,initial_z=None):
        z0=self.seed(phase) if initial_z is None else np.array(initial_z).copy();z0[:13]=state
        tick=time.perf_counter();self.last_trace=[];self.last_diagnostics=None
        for eps in self.c.get('cpc_epsilon_schedule',[1e-3,1e-5,1e-7]):
            p=np.r_[state,gates.T.ravel(order='F'),normals.T.ravel(order='F'),half.T.ravel(order='F'),eps]
            result=self.solver(x0=z0,p=p,lbx=self.lbx,ubx=self.ubx,lbg=self.lbg,ubg=self.ubg)
            stats=self.solver.stats();zz=np.array(result['x']).ravel();g=np.array(result['g']).ravel()
            violation=np.maximum(np.maximum(self.lbg-g,0),np.maximum(g-self.ubg,0))
            primal=float(max(violation.max(),np.maximum(self.lbx-zz,0).max(),np.maximum(zz-self.ubx,0).max()))
            groups={name:float(violation[np.array(self.labels)==name].max()) for name in set(self.labels)}
            row=dict(epsilon=eps,status=stats['return_status'],iterations=int(stats['iter_count']),primal_residual=primal,horizon_time=float(zz[-1]),seconds=time.perf_counter()-tick)
            self.last_trace.append(row);self.last_diagnostics=dict(trace=self.last_trace,violation_by_constraint=groups)
            if hasattr(self,'stationarity'):
                row['stationarity_inf']=float(abs(np.array(self.stationarity(zz,p,result['lam_g'],result['lam_x']))).max())
            if not stats['success'] or not np.isfinite(zz).all() or primal>2e-6:
                raise RuntimeError('Original CPC solve failed: '+json.dumps(self.last_diagnostics))
            z0=zz
        n=self.n;off=self.offsets;mu=zz[off['mu']:off['nu']].reshape(2,n,order='F');lam=np.c_[np.ones(2),zz[off['lam']:off['mu']].reshape(2,n-1,order='F'),np.zeros(2)]
        tau=np.array(self.times(zz)).ravel();nodes=np.argmax(mu,axis=1)+1;T=float(zz[-1])
        return dict(X=zz[:self.ns].reshape(13,n+1,order='F'),U=zz[self.ns:off['lam']].reshape(4,n,order='F'),mu=mu,**{'lambda':lam},
            nu=zz[off['nu']:off['point']].reshape(2,n,order='F'),gate_offsets=zz[off['point']:-1].reshape(2,2,order='F'),
            durations=np.diff(np.r_[0,tau]),arrival_times=tau,horizon_time=T,cell_durations=np.full(n,T/n),event_nodes=nodes,
            z=zz,value=T,seconds=time.perf_counter()-tick,iterations=sum(v['iterations'] for v in self.last_trace),residual=primal,
            gates=gates.copy(),normals=normals.copy(),half=half.copy(),continuation=self.last_trace,diagnostics=self.last_diagnostics,
            cpc_residual=float(np.maximum(np.array(self.products(zz,p)).ravel(),0).max()))
