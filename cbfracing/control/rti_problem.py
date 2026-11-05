"""Full nonlinear MPCC with selectable direct learned safety constraints."""
from cbfracing.core.context import *
DT=.08
FEEDBACK=.02
import time

class RecoveryMPCC:
    def __init__(self,agent,n=20,barrier='soft',max_iter=180,wall_time=4.,restarts=0,penalty=100000.,execution_guard=True,cbf_checkpoint=None,cbf_mode='analytic',cbf_alpha=3.,gate_guard=False,tube_recovery=False,dense_physics=False,build_only=False):
        if barrier != 'soft':raise ValueError("Use barrier='soft' for the retained cost configuration; cbf_mode selects the safety constraints.")
        self.n=n;self.X=None;self.U=None;self.state=None;self.od=None
        self.restarts=restarts;self.barrier=barrier
        self.cbf_mode=cbf_mode;self.cbf_alpha=cbf_alpha;self.neural=None;self.h_rows=[];self.dh_rows=[]
        self.physical_gate_guard=os.environ.get('PHYSICAL_GATE_GUARD','0')=='1';self.guard_gate=None;self.guard_armed=False
        tracking_funnel=os.getenv('OPT_TRACKING_FUNNEL','1')=='1'
        if not tracking_funnel and not self.physical_gate_guard:raise ValueError('Physical crossing constraints are required without the tracking funnel')
        if cbf_checkpoint:
            from cbfracing.networks.barrier import load,symbolic
            self.neural=symbolic(load(cbf_checkpoint)[0])
        if cbf_mode not in ('analytic','none') and self.neural is None:raise ValueError('A learned CBF checkpoint is required')
        self.w=np.array([1,1,1,10000,1,1,10,.05,.05,.05,.05,2,1e5],float);self.mu=500.
        X=ca.MX.sym('X',19,n+1);U=ca.MX.sym('U',5,n);initial=ca.MX.sym('initial',19)
        OD=ca.MX.sym('OD',70,n+1);W=ca.MX.sym('W',13);mu=ca.MX.sym('mu')
        rhs,step=model_functions(2);expr=[];lb=[];ub=[];objective=0.;residuals=[];linear_objective=0.
        self.row_kinds=[];self.row_stages=[];stage=-1;gate_geometry_terms=[]
        def add(v,lo=0.,hi=0.,kind='physical'):
            expr.append(ca.vec(v));lb.extend([lo]*v.numel());ub.extend([hi]*v.numel());self.row_kinds.extend([kind]*v.numel());self.row_stages.extend([stage]*v.numel())
        def gate_crossing(before,after,od,next_gate=False):
            center=od[44:47] if next_gate else od[34:37]
            normal=od[47:50] if next_gate else od[37:40]
            a=before[:3]-center;b=after[:3]-center
            pa=ca.dot(a,normal);pb=ca.dot(b,normal)
            cross=(pb*a-pa*b)/ca.fmax(pb-pa,1e-6)
            lateral=-cross[0]*normal[1]+cross[1]*normal[0]+(0. if next_gate else od[53])
            active=ca.logic_and(pa<0.,pb>=0.)
            if next_gate:active=ca.logic_and(active,before[17]>=od[42]-.2)
            width=.70-float(os.environ.get('GATE_PREDICTION_MARGIN','0')) if next_gate else od[50]
            height=.70-float(os.environ.get('GATE_PREDICTION_MARGIN','0')) if next_gate else od[51]
            tag='next_gate_crossing' if next_gate else 'gate_crossing'
            if os.environ.get('GATE_CROSSING_FORM','rational')=='crossmultiply':
                # On the active forward crossing, pb-pa is strictly positive.
                # Multiplication therefore preserves the physical aperture.
                denominator=pb-pa; numerator=pb*a-pa*b
                lat=-numerator[0]*normal[1]+numerator[1]*normal[0]+(0. if next_gate else od[53])*denominator
                up=numerator[2]+(0. if next_gate else od[54])*denominator
                distance=ca.fmax(ca.fabs(lat)/ca.fmax(denominator,1e-12)-width,ca.fabs(up)/ca.fmax(denominator,1e-12)-height)
                gate_geometry_terms.append(ca.if_else(active,distance,-1.))
                for value,bound in [(lat,width),(up,height)]:
                    add(ca.if_else(active,value-bound*denominator,-1.),-np.inf,0.,kind=tag)
                    add(ca.if_else(active,-value-bound*denominator,-1.),-np.inf,0.,kind=tag)
            else:
                add(ca.if_else(active,lateral**2-width**2,-1.),-np.inf,0.,kind=tag)
                add(ca.if_else(active,(cross[2]+(0. if next_gate else od[54]))**2-height**2,-1.),-np.inf,0.,kind=tag)
        add(X[:,0]-initial,kind='initial')
        previous_h=None
        for k in range(n+1):
            stage=k
            x=X[:,k];u=U[:,min(k,n-1)];od=OD[:,k]
            d=x[17]-od[12];position=od[:3]+od[3:6]*d+od[6:9]*d*d+od[9:12]*d**3
            tangent=od[3:6]+2*od[6:9]*d+3*od[9:12]*d*d;tangent/=ca.sqrt(ca.sumsqr(tangent)+1e-10)
            error=x[:3]-position;lag=ca.dot(error,tangent);contour=error-tangent*lag
            floor=.5*(.5-x[2]+ca.sqrt((.5-x[2])**2+1e-6))
            objective+=od[13]*ca.sumsqr(contour)+W[3]*lag**2+ca.dot(W[4:7],x[10:13]**2)+W[12]*floor**2-mu*x[18]
            residuals.append(ca.vertcat(ca.sqrt(od[13])*contour,ca.sqrt(W[3])*lag,ca.sqrt(W[4:7])*x[10:13],ca.sqrt(W[12])*floor));linear_objective-=mu*x[18]
            velocity_weight=float(os.environ.get('VELOCITY_CONTOUR_WEIGHT','0'))
            if velocity_weight>0:
                transverse_velocity=x[3:6]-ca.dot(x[3:6],tangent)*tangent
                objective+=velocity_weight*ca.sumsqr(transverse_velocity)
                residuals.append(ca.sqrt(velocity_weight)*transverse_velocity)
            speed_weight=float(os.environ.get('REFERENCE_SPEED_WEIGHT','0'))
            if speed_weight>0:
                excess=ca.sqrt(ca.sumsqr(x[3:6])+1e-8)-(od[55]+od[56]*d)
                hinge=.5*(excess+ca.sqrt(excess**2+1e-6))
                objective+=speed_weight*hinge**2;residuals.append(ca.sqrt(speed_weight)*hinge)
            if k<n:
                objective+=ca.dot(W[7:11],u[:4]**2)+W[11]*u[4]**2
                residuals.append(ca.vertcat(ca.sqrt(W[7:11])*u[:4],ca.sqrt(W[11])*u[4]))
                add(X[:,k+1]-step(x,u,DT),kind='dynamics')
                if k==0 and execution_guard:
                    executed=step(x,u,FEEDBACK)
                    if self.physical_gate_guard:gate_crossing(x,executed,od)
                    add(ca.sumsqr(executed[10:13]),0.,(MODEL['omega_max']-.1)**2)
                    add(executed[13:17],-np.inf if os.environ.get('NODAL_THRUST_ONLY','0')=='1' else .00001,np.inf if os.environ.get('NODAL_THRUST_ONLY','0')=='1' else 6.99999)
                    add(executed[2],.05001,np.inf)
                    if dense_physics:
                        # Endpoint rate checks miss a short interior maximum.
                        # These constrain the actually applied feedback interval;
                        # the independent plant audit uses a denser time grid.
                        world=course()['world_box']
                        for fraction in (.25,.5,.75,1.):
                            interim=step(x,u,FEEDBACK*fraction)
                            add(ca.sumsqr(interim[10:13]),0.,(MODEL['omega_max']-.01)**2)
                            for dim in range(3):add(interim[dim],world[dim]+1e-5,world[dim+3]-1e-5)
            add(ca.sumsqr(x[10:13]),0.,MODEL['omega_max']**2);add(x[13:17],0.,7.0);add(x[2],.05,np.inf);add(x[18],0.,30.)
            add(x[17]-od[33],-np.inf,0.,kind='progress')
            world=course()['world_box']
            for dim in range(3):add(x[dim],world[dim],world[dim+3])
            gr=x[:3]-od[34:37];plane=ca.dot(gr,od[37:40]);lateral=-gr[0]*od[38]+gr[1]*od[37]
            width=od[40]+od[41]*ca.sqrt(plane**2+1e-6)+od[43]*ca.sqrt((x[17]-od[42])**2+1e-6)
            post_gap=x[17]-od[42]-.1
            post_release=float(os.getenv('POST_GATE_FUNNEL_SLOPE','0'))*.5*(post_gap+ca.sqrt(post_gap**2+1e-6))
            width+=post_release
            width_lat=width; width_up=width
            if self.physical_gate_guard and k>0:
                pieces=4 if k<=int(os.getenv('GATE_DENSE_PREFIX','0')) else 1
                previous=X[:,k-1]
                for part in range(1,pieces+1):
                    following=x if part==pieces else step(X[:,k-1],U[:,k-1],DT*part/pieces)
                    gate_crossing(previous,following,od);previous=following
                if os.environ.get('NEXT_GATE_GUARD','0')=='1':gate_crossing(X[:,k-1],x,od,True)
            if gate_guard:
                # A local progress window caps the physical opening. Release
                # both well before and after the event: the upper/lower Split-S
                # gates share a plane, so a new plan may start outside its gate.
                gap=x[17]-od[42];after=gap-.4;before=-gap-.4
                release=.5*(after+ca.sqrt(after*after+1e-6))+.5*(before+ca.sqrt(before*before+1e-6))
                for bound,dimension in [(od[50],0),(od[51],1)]:
                    cap=bound+od[41]*ca.sqrt(plane**2+1e-6)+od[43]*release+post_release
                    tight=.5*(width+cap-ca.sqrt((width-cap)**2+1e-8))
                    if dimension==0:width_lat=tight
                    else:width_up=tight
            if tracking_funnel and (k>0 or not tube_recovery):
                # A current tracking-tube error cannot be changed by this solve.
                # Keep all physical and learned-safety initial rows hard; require
                # the predicted future motion to return to the tracking funnel.
                add(lateral**2-width_lat**2,-np.inf,0.,kind='tracking');add(gr[2]**2-width_up**2,-np.inf,0.,kind='tracking')
            if cbf_mode!='none':
                distance=ca.sqrt(ca.sumsqr(x[:3]-od[16:19])+1e-10)
                inflation=0.
                geometric=distance-.55-inflation
                # Geometric safety is used only by the explicit analytic modes.
                if cbf_mode in ('analytic','hybrid'):add(geometric,0.,np.inf,kind='cbf')
                h=self.neural(x,od[16:22]) if self.neural is not None else distance-.55
                if k==0 and cbf_mode=='hard' and os.environ.get('RTI_DENSE_CBF','0')=='1':
                    for fraction in (.125,.25,.375,.5,.625,.75,.875,1.):
                        dt=FEEDBACK*fraction;executed=step(x,u,dt);tau=.35
                        otherp=od[16:19]+od[19:22]*dt+od[22:25]*tau*(dt-tau*(1-ca.exp(-dt/tau)))
                        otherv=od[19:22]+od[22:25]*tau*(1-ca.exp(-dt/tau))
                        he=self.neural(executed,ca.vertcat(otherp,otherv))
                        add(he,0.,np.inf,kind='applied_cbf')
                        add(he-ca.exp(-cbf_alpha*dt)*h,float(os.getenv('OPT_CBF_DECAY_MARGIN_RATE','0'))*dt,np.inf,kind='applied_decay')
                if self.neural is not None:
                    self.h_rows.append(len(lb));add(h,0.,np.inf,kind='cbf')
                if cbf_mode=='hard':
                    if previous_h is not None:
                        hard_steps=int(os.environ.get('CBF_HARD_STEPS','-1'))
                        decay=h-ca.exp(-cbf_alpha*DT)*previous_h
                        if hard_steps<0 or k<=hard_steps:
                            self.dh_rows.append(len(lb));add(decay,float(os.getenv('OPT_CBF_DECAY_MARGIN_RATE','0'))*DT,np.inf,kind='cbf_decay')
                        else:
                            # Forecast-tail decay is penalized; h>=0 stays hard.
                            # Applied-interval CBF audit remains unchanged.
                            negative=.5*(-decay+ca.sqrt(decay**2+1e-6))
                            weight=float(os.environ.get('CBF_TAIL_WEIGHT','2000'))
                            objective+=weight*negative**2;residuals.append(ca.sqrt(weight)*negative)
                else:
                    # Penalize negative distance barriers and discrete decay residuals.
                    # Geometric distance rows remain hard.
                    hinge=.5*(-h+ca.sqrt(h*h+1e-6))
                    objective+=penalty*hinge**2;residuals.append(ca.sqrt(penalty)*hinge)
                    if previous_h is not None:
                        residual=h-ca.exp(-cbf_alpha*DT)*previous_h
                        rh=.5*(-residual+ca.sqrt(residual*residual+1e-6))
                        objective+=penalty*.2*rh**2;residuals.append(ca.sqrt(penalty*.2)*rh)
                previous_h=h
        if os.environ.get('RTI_SHOOTING_GUARD','0')=='1':
            stage=0
            # Redundant at an exactly feasible multiple-shooting solution,
            # but constrains the first input directly in a finite-step RTI QP.
            actual_next=step(X[:,0],U[:,0],DT)
            add(ca.sumsqr(actual_next[10:13]),0.,MODEL['omega_max']**2,kind='shooting_guard')
            add(actual_next[13:17],-np.inf if os.environ.get('NODAL_THRUST_ONLY','0')=='1' else 0.,np.inf if os.environ.get('NODAL_THRUST_ONLY','0')=='1' else 7.,kind='shooting_guard')
            world=course()['world_box']
            for dim in range(3):add(actual_next[dim],world[dim],world[dim+3],kind='shooting_guard')
            if self.physical_gate_guard:gate_crossing(X[:,0],actual_next,OD[:,0])
            if cbf_mode=='hard':
                h0=self.neural(X[:,0],OD[16:22,0]);h1=self.neural(actual_next,OD[16:22,1])
                self.h_rows.append(len(lb));add(h1,0.,np.inf,kind='shooting_cbf')
                self.dh_rows.append(len(lb));add(h1-ca.exp(-cbf_alpha*DT)*h0,float(os.getenv('OPT_CBF_DECAY_MARGIN_RATE','0'))*DT,np.inf,kind='shooting_decay')
        z=ca.vertcat(ca.vec(X),ca.vec(U));p=ca.vertcat(initial,ca.vec(OD),W,mu);g=ca.vertcat(*expr)
        self.constraint_count=int(g.numel());self.barrier_constraint_count=len(self.h_rows)+len(self.dh_rows)+(n+1 if cbf_mode in ('analytic','hybrid') else 0)
        options=dict(print_time=False,error_on_fail=False,expand=True,ipopt=dict(print_level=0,sb='yes',tol=1e-6,acceptable_tol=2e-5,
            max_iter=max_iter,max_wall_time=wall_time,bound_relax_factor=0.,mu_strategy='adaptive',linear_solver='mumps'))
        self.z=z;self.p=p;self.g=g;self.objective=objective;self.residuals=ca.vertcat(*residuals);self.linear_objective=linear_objective
        self.gate_geometry=ca.Function('gate_geometry',[z,p],[ca.vertcat(*gate_geometry_terms)]).expand() if gate_geometry_terms else None
        self.solver=None if build_only else ca.nlpsol('nonlinear_mpcc','ipopt',dict(x=z,p=p,f=objective,g=g),options)
        self.lbg=np.array(lb);self.ubg=np.array(ub)
        self.lbx=np.r_[np.full(19*(n+1),-np.inf),np.tile([-200]*4+[-40],n)]
        self.ubx=np.r_[np.full(19*(n+1),np.inf),np.tile([200]*4+[40],n)]
    def seed(self,x,u):self.X=np.array(x).copy();self.U=np.array(u).copy()
    def current(self,x):self.state=np.array(x).copy()
    def data(self,od):
        from cbfracing.core.context import course
        track=course();centers=np.array(track['centers']);normals=np.array(track['normals'])
        gi=int(np.argmin(np.linalg.norm(centers-np.array(od)[0,34:37],axis=1)))
        context=np.r_[centers[(gi+1)%7],normals[(gi+1)%7]]
        self.od=np.c_[np.array(od),np.tile(context,(len(od),1)),np.zeros((len(od),20))]
        lat=np.cross([0.,0.,1.],normals[gi]);delta=np.asarray(od)[0,34:37]-centers[gi]
        margin=float(os.environ.get('GATE_PREDICTION_MARGIN','0'))
        self.od[:,50]=.70-abs(delta@lat)-margin;self.od[:,51]=.70-abs(delta[2])-margin
        if os.environ.get('EXACT_GATE_APERTURE','0')=='1':
            self.od[:,50:52]=.70-margin;self.od[:,53]=delta@lat;self.od[:,54]=delta[2]
            release=int(os.getenv('OPT_GATE_MARGIN_RELEASE_STEPS','0'))
            if release:self.od[:release,50:52]=.70
        if self.physical_gate_guard:
            if self.guard_gate!=gi:self.guard_gate=gi;self.guard_armed=False
            relative=self.state[:3]-self.od[0,34:37];plane=float(relative@normals[gi])
            if plane<-.1 and abs(relative@lat)<=self.od[0,50]+2*abs(plane)-.02 and abs(relative[2])<=self.od[0,51]+2*abs(plane)-.02:
                self.guard_armed=True
            self.od[:,52]=float(self.guard_armed)

    def weights(self,mu=500.,q_l=10000.,df=.05,at=2.,slack=1e6):
        self.mu=mu;self.w=np.array([1,1,1,q_l,1,1,10,df,df,df,df,at,1e5],float)
    def prediction(self):return self.X.copy(),self.U.copy()
    def step(self):
        tick=time.perf_counter();p=np.r_[self.state,self.od.ravel(),self.w,self.mu]
        z0=np.r_[self.X.ravel(),self.U.ravel()];trace=[]
        for retry in range(self.restarts+1):
            result=self.solver(x0=z0,p=p,lbx=self.lbx,ubx=self.ubx,lbg=self.lbg,ubg=self.ubg)
            z=np.asarray(result['x']).ravel();values=np.asarray(result['g']).ravel();stats=self.solver.stats()
            residual=float(max(np.maximum(self.lbg-values,0).max(),np.maximum(values-self.ubg,0).max(),np.maximum(self.lbx-z,0).max(),np.maximum(z-self.ubx,0).max()))
            good=bool(stats['success'] and np.isfinite(z).all() and residual<5e-5)
            trace.append(dict(retry=retry,status=stats['return_status'],primal_residual=residual,iterations=int(stats['iter_count']),elapsed=time.perf_counter()-tick))
            if good:break
            if retry<self.restarts:
                # Alternative initial guesses for exactly the same OCP. No
                # constraint relaxation and no alternative executed controller.
                cold=np.tile(self.state,(self.n+1,1));cold[:,17]+=np.arange(self.n+1)*DT*self.state[18]
                cold[:,17]=np.minimum(cold[:,17],self.od[:,33]);u0=np.zeros((self.n,5))
                if retry%2:
                    _,step=model_functions(2)
                    for k in range(self.n):cold[k+1]=np.asarray(step(cold[k],u0[k],DT)).ravel()
                z0=np.r_[cold.ravel(),u0.ravel()]
        self.X=z[:19*(self.n+1)].reshape(self.n+1,19);self.U=z[19*(self.n+1):].reshape(self.n,5)
        ds=np.linalg.norm(self.X[:,:3]-self.od[:,16:19],axis=1)
        diagnostic=dict(initial_geometric=float(ds[0]-.55),minimum_predicted_distance=float(ds.min()))
        if self.neural is not None:
            hs=np.array([float(self.neural(x,d[16:22])) for x,d in zip(self.X,self.od)])
            multiplier=np.asarray(result['lam_g']).ravel();hm=abs(multiplier[self.h_rows]);dm=abs(multiplier[self.dh_rows])
            diagnostic.update(initial_learned_h=float(hs[0]),minimum_predicted_h=float(hs.min()),minimum_predicted_discrete_residual=float((hs[1:]-np.exp(-self.cbf_alpha*DT)*hs[:-1]).min()),
                learned_active_rows=int(((hs<1e-3)&(hm>1e-4)).sum()),learned_dual_max=float(hm.max()),discrete_dual_max=float(dm.max()) if len(dm) else 0.,analytic_geometric_row=self.cbf_mode=='hybrid')
        return dict(status=0 if good else 1,message=stats['return_status'],seconds=time.perf_counter()-tick,barrier_diagnostics=diagnostic,
            preparation_seconds=0.,feedback_seconds=0.,kkt=float(residual),working_set=sum(r['iterations'] for r in trace),backend='full NLP diagnostic',solve_trace=trace)
