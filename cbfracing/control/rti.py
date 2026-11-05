"""One-QP SQP-RTI prototype using the exact retained nonlinear OCP.

QP backends are Clarabel or OSQP, not acados. GN changes curvature only;
the original objective gradient, all constraints, and nonlinear audits remain.
"""
from cbfracing.core.context import *
from cbfracing.control.rti_problem import RecoveryMPCC as Problem
from scipy import sparse
from scipy.sparse.linalg import splu
import clarabel, osqp


def csc(dm):
    pointers,rows=dm.sparsity().get_ccs()
    return sparse.csc_matrix((np.asarray(dm.nonzeros()),np.asarray(rows,dtype=np.int32),np.asarray(pointers,dtype=np.int32)),shape=dm.shape)


class RTIMPCC(Problem):
    def __init__(self,*args,qp_backend='clarabel',iterations=1,regularization=1e-5,trust_radius=2.,
                 native=False,accept_tolerance=5e-5,condense=False,globalize=False,feasibility_correction=False,
                 feasible_first=False,elastic_restore=False,acceptance='horizon',qp_tolerance=1e-8,**kwargs):
        super().__init__(*args,build_only=True,**kwargs)
        self.qp_backend=qp_backend;self.iterations=iterations;self.regularization=regularization
        self.trust_radius=trust_radius;self.accept_tolerance=accept_tolerance
        self.condense=condense
        self.globalize=globalize
        self.feasibility_correction=feasibility_correction
        self.feasible_first=feasible_first;self.elastic_restore=elastic_restore
        self.acceptance=acceptance;_,self.applied_step=model_functions(8)
        self.batched_audit=os.environ.get('RTI_BATCHED_AUDIT','0')=='1'
        if self.batched_audit:
            self.applied_map=self.applied_step.map(9)
            self.neural_map=self.neural.map(9) if self.neural is not None else None
            self.audit_world=np.array(course()['world_box'])
        self.qp_tolerance=qp_tolerance
        self.qp=None;self.qp_structure=None
        self.scale=np.r_[np.tile([5,5,3,10,10,10,1,1,1,1,5,5,5,5,5,5,5,10,10],self.n+1),np.tile([50,50,50,50,20],self.n)]
        J=ca.jacobian(self.residuals,self.z)
        GN=2*J.T@J
        self.prepare=ca.Function('rti_prepare',[self.z,self.p],[self.objective,self.g,ca.gradient(self.objective,self.z),ca.jacobian(self.g,self.z),GN]).expand()
        self.evaluate=ca.Function('rti_evaluate',[self.z,self.p],[self.objective,self.g]).expand()
        self.identity=ca.Function('objective_identity',[self.z,self.p],[self.objective-ca.sumsqr(self.residuals)-self.linear_objective]).expand()
        if native:self.load_native()
        self.fast_kernels=os.environ.get('FAST_KERNELS','0')!='0'
        self.kernel_loading_note='symbolic'
        if self.fast_kernels:
            stem=os.environ.get('FAST_KERNEL_STEM') or ('fast_mx' if os.environ.get('FAST_KERNELS')=='mx' else 'fast')
            try:
                meta=json.loads((HERE/'build'/(stem+'_compile.json')).read_text())
                if meta['key']!=self.native_key():raise ValueError('Native OCP hash mismatch')
                prepare=ca.external('rti_prepare',str(HERE/'build'/(stem+'.dll')))
                evaluate=ca.external('rti_evaluate',str(HERE/'build'/(stem+'.dll')))
                correction=ca.external('rti_correct',str(HERE/'build'/(stem+'.dll')))
            except (OSError,ValueError,RuntimeError) as error:
                if os.environ.get('FAST_KERNEL_POLICY')!='auto':raise
                self.fast_kernels=False
                self.kernel_loading_note=f'Symbolic fallback: {error}'
                print(self.kernel_loading_note,flush=True)
            else:
                self.prepare,self.evaluate,self.correct_prepare=prepare,evaluate,correction
                self.kernel_loading_note=stem
        self.cached_eye=sparse.eye(len(self.scale),format='csc')
        self.history=[]

    def load_native(self):
        # Model/network/constraint-specific function hash prevents stale DLL use.
        dll=HERE/'build'/f'rti_{self.native_key()}.dll'
        if not dll.is_file():raise FileNotFoundError('Compile this exact OCP with build_rti_native.py first: '+str(dll))
        self.prepare=ca.external('rti_prepare',str(dll))
        self.evaluate=ca.external('rti_evaluate',str(dll))

    def native_key(self):
        return hashlib.sha256((self.prepare.serialize()+self.evaluate.serialize()).encode()).hexdigest()[:20]

    def parameters(self):return np.r_[self.state,self.od.ravel(),self.w,self.mu]

    def violation(self,z,g):
        geometric=float(np.asarray(self.gate_geometry(z,self.parameters())).max(initial=0.)) if self.gate_geometry is not None else 0.
        return float(max(geometric,np.maximum(self.lbg-g,0.).max(),np.maximum(g-self.ubg,0.).max(),np.maximum(self.lbx-z,0.).max(),np.maximum(z-self.ubx,0.).max()))

    def applied_audit(self,z):
        """Predicted-neighbor audit of the next 20 ms; not a horizon certificate."""
        u=z[19*(self.n+1):19*(self.n+1)+5];times=np.linspace(0,.02,9)
        own=(np.asarray(self.applied_map(np.tile(self.state[:,None],(1,9)),np.tile(u[:,None],(1,9)),times[None,:])).T
            if self.batched_audit else np.array([np.asarray(self.applied_step(self.state,u,float(t))).ravel() for t in times]))
        obs=self.od[0];tau=.35;a=obs[22:25]
        pos=obs[16:19]+obs[19:22]*times[:,None]+a*tau*(times[:,None]-tau*(1-np.exp(-times[:,None]/tau)))
        vel=obs[19:22]+a*tau*(1-np.exp(-times[:,None]/tau))
        other=np.c_[pos,vel]
        hs=(np.asarray(self.neural_map(own.T,other.T)).ravel() if self.batched_audit and self.neural is not None else
            np.array([float(self.neural(x,o)) if self.neural is not None else np.linalg.norm(x[:3]-o[:3])-.55 for x,o in zip(own,other)]))
        dh=hs-np.exp(-self.cbf_alpha*times)*hs[0];world=self.audit_world if self.batched_audit else np.array(course()['world_box'])
        force_samples=own[[0,-1],13:17] if os.environ.get('NODAL_THRUST_ONLY','0')=='1' else own[:,13:17]
        values=[float(np.linalg.norm(own[:,10:13],axis=1).max()-(MODEL['omega_max']-.01)),
            float(-force_samples.min()),float(force_samples.max()-7.),float((world[:3]-own[:,:3]).max()),float((own[:,:3]-world[3:]).max()),
            float(np.max(abs(u[:4]))-200),float(abs(u[4])-40),float(-hs.min()),float(-dh.min()),float(-own[:,18].min()),float(own[:,18].max()-30)]
        gate_violation=0.
        if self.physical_gate_guard:
            n=obs[37:40];g=obs[34:37];signed=(own[:,:3]-g)@n
            for k in range(len(times)-1):
                if signed[k]<0<=signed[k+1]:
                    fraction=-signed[k]/(signed[k+1]-signed[k]);cross=(1-fraction)*own[k,:3]+fraction*own[k+1,:3]-g
                    reserve=float(os.environ.get('GATE_PREDICTION_MARGIN','0'))
                    physical_half=np.full(2,.70) if os.getenv('EXACT_GATE_APERTURE','0')=='1' else obs[50:52]+reserve
                    gate_violation=max(gate_violation,abs(-cross[0]*n[1]+cross[1]*n[0]+obs[53])-physical_half[0],abs(cross[2]+obs[54])-physical_half[1])
        violation=max(0.,gate_violation,*values)
        labels=['body_rate','thrust_lower','thrust_upper','world_lower','world_upper','thrust_rate','progress_acceleration','h','decay','progress_speed_lower','progress_speed_upper']
        return dict(passed=bool(np.isfinite(own).all() and np.isfinite(hs).all() and violation<1e-5),
            violation=violation,minimum_h=float(hs.min()),minimum_decay=float(dh.min()),samples=9,
            violations_by_kind={**dict(zip(labels,[max(0.,v) for v in values])),'gate':gate_violation},
            scope='Applied 20 ms only, causal neighbor forecast; original horizon residual separately logged')

    def qp_solve(self,B,gradient,A,lower,upper):
        alternative='' if getattr(self,'backend_dispatch_disabled',False) else os.environ.get('EXPERIMENT_QP','')
        # Elastic restoration adds a slack and changes the elimination system.
        # The fixed-dimension 399-state DAQP condensing applies only to nominal QPs.
        if alternative and B.shape[0]!=len(self.scale):alternative=''
        if alternative=='correction_daqp' and not (B.nnz==B.shape[0] and np.count_nonzero(gradient)==0):alternative=''
        if alternative:
            if not hasattr(self,'alternative_backend'):
                from cbfracing.control.qp_backends import QPBackend
                self.alternative_backend=QPBackend('daqp_direct' if alternative=='correction_daqp' else alternative,os.environ.get('EXPERIMENT_CONDENSE','0')=='1',self.qp_tolerance)
            delta,info=self.alternative_backend.solve(B,gradient,A,lower,upper)
            if not info['success'] and os.environ.get('OPT_QP_RETRY_CLARABEL','0')=='1':
                from cbfracing.control.qp_backends import QPBackend
                if not hasattr(self,'sparse_retry_backend'):self.sparse_retry_backend=QPBackend('clarabel',False,self.qp_tolerance)
                retry,second=self.sparse_retry_backend.solve(B,gradient,A,lower,upper)
                second['first_backend_failure']=info;second['same_qp_sparse_retry']=True
                second['total_seconds']+=info.get('total_seconds',0.)
                if second['success']:delta=retry
                info=second
            return delta,info
        eq=np.isfinite(lower)&np.isfinite(upper)&(abs(upper-lower)<1e-12)
        if self.condense:
            nx=19*(self.n+1);nu=5*self.n
            assert eq.sum()==nx
            E=A[eq,:nx].tocsc();F=A[eq,nx:].toarray();factor=splu(E)
            T=np.vstack([-factor.solve(F),np.eye(nu)]);offset=np.r_[factor.solve(upper[eq]),np.zeros(nu)]
            reduced_B=T.T@(B@T);reduced_B=(reduced_B+reduced_B.T)*.5
            reduced_gradient=T.T@(B@offset+gradient)
            reduced_A=sparse.csc_matrix(A[~eq]@T)
            shift=A[~eq]@offset;reduced_lower=lower[~eq]-shift;reduced_upper=upper[~eq]-shift
            self.condense=False
            try:delta,info=self.qp_solve(sparse.csc_matrix(reduced_B),reduced_gradient,reduced_A,reduced_lower,reduced_upper)
            finally:self.condense=True
            return T@delta+offset,info
        if self.qp_backend=='clarabel':
            up=np.isfinite(upper)&~eq;lo=np.isfinite(lower)&~eq
            matrix=sparse.vstack([A[eq],A[up],-A[lo]],format='csc')
            bound=np.r_[upper[eq],upper[up],-lower[lo]]
            P=sparse.triu(B,format='csc')
            signature=(P.shape,P.indptr.tobytes(),P.indices.tobytes(),matrix.shape,matrix.indptr.tobytes(),matrix.indices.tobytes())
            if self.qp is None or signature!=self.qp_structure:
                settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100
                settings.tol_gap_abs=self.qp_tolerance;settings.tol_gap_rel=self.qp_tolerance;settings.tol_feas=self.qp_tolerance
                settings.max_threads=1
                self.qp=clarabel.DefaultSolver(P,gradient,matrix,bound,[clarabel.ZeroConeT(int(eq.sum())),clarabel.NonnegativeConeT(int(up.sum()+lo.sum()))],settings)
                self.qp_structure=signature
            else:self.qp.update(P=P,q=gradient,A=matrix,b=bound)
            result=self.qp.solve();status=str(result.status)
            return np.asarray(result.x),dict(status=status,success=status in ['Solved','AlmostSolved'],iterations=result.iterations,backend_seconds=result.solve_time)
        else:
            P=sparse.triu(B,format='csc');signature=(P.shape,P.indptr.tobytes(),P.indices.tobytes(),A.shape,A.indptr.tobytes(),A.indices.tobytes())
            if self.qp is None or signature!=self.qp_structure:
                self.qp=osqp.OSQP();self.qp.setup(P=P,q=gradient,A=A,l=lower,u=upper,verbose=False,
                    eps_abs=1e-6,eps_rel=1e-6,max_iter=10000,polishing=True,adaptive_rho=True)
                self.qp_structure=signature
            else:self.qp.update(Px=P.data,q=gradient,Ax=A.data,l=lower,u=upper)
            result=self.qp.solve(raise_error=False);status=result.info.status
            return np.asarray(result.x),dict(status=status,success=result.info.status_val in [1,2],iterations=result.info.iter,backend_seconds=result.info.run_time)

    def step(self):
        start=time.perf_counter();p=self.parameters();z=np.r_[self.X.ravel(),self.U.ravel()]
        traces=[];preparation=0.;qp_seconds=0.;audit_seconds=0.;success=False;status='not_started';objective_done=False
        min_objective=int(os.environ.get('RTI_OBJECTIVE_UPDATES','1'));objective_updates=0;accepted=None
        certified=None;certificate_updates=None;used_certificate=False
        certify=os.getenv('OPT_RTI_CERTIFIED_BACKUP','0')=='1'
        initial_violation=None
        if certify:
            tick=time.perf_counter();_,g0=self.evaluate(z,p);g0=np.asarray(g0).ravel();initial_violation=self.violation(z,g0)
            if initial_violation<1e-5 and self.applied_audit(z)['passed']:
                certified=(z.copy(),g0.copy(),initial_violation);certificate_updates=0
            audit_seconds+=time.perf_counter()-tick
        for iteration in range(self.iterations):
            tick=time.perf_counter()
            known_correction=getattr(self,'reference_repair_only',False) or (self.feasibility_correction and objective_updates>=min_objective)
            if self.fast_kernels and known_correction:
                obj,g,grad,A=self.correct_prepare(z,p);B=None
            else:
                obj,g,grad,A,B=self.prepare(z,p);B=csc(B)
            g=np.asarray(g).ravel();grad=np.asarray(grad).ravel();A=csc(A)
            scaled_A=A.multiply(self.scale).tocsc()
            row_scale=1/np.maximum(np.sqrt(np.asarray(scaled_A.power(2).sum(axis=1)).ravel()),1.)
            scaled_A=scaled_A.multiply(row_scale[:,None]).tocsc()
            objective_scale=1e4
            gradient=grad*self.scale/objective_scale
            first_tolerance=min(self.accept_tolerance,1e-5) if certify else self.accept_tolerance
            correction=getattr(self,'reference_repair_only',False) or (self.feasibility_correction and objective_updates>=min_objective) or (self.feasible_first and self.violation(z,g)>=first_tolerance)
            if correction:
                # Project the first objective step onto the nonlinear feasible
                # set. These extra QPs are counted separately from strict RTI.
                B=self.cached_eye;gradient=np.zeros(len(z))
            else:
                B=B.multiply(self.scale[:,None]).multiply(self.scale[None,:]).tocsc()/objective_scale
                B=B+self.regularization*self.cached_eye
            lower=(self.lbg-g)*row_scale;upper=(self.ubg-g)*row_scale
            lo=np.maximum((self.lbx-z)/self.scale,-self.trust_radius)
            hi=np.minimum((self.ubx-z)/self.scale,self.trust_radius)
            finite=np.isfinite(lo)|np.isfinite(hi)
            Aqp=sparse.vstack([scaled_A,self.cached_eye[finite]],format='csc')
            lower=np.r_[lower,lo[finite]];upper=np.r_[upper,hi[finite]]
            preparation+=time.perf_counter()-tick
            tick=time.perf_counter();delta,qp=self.qp_solve(B,gradient,Aqp,lower,upper);qp_seconds+=time.perf_counter()-tick
            if np.isfinite(delta).all():
                linear_values=Aqp@delta
                qp['linear_primal_residual']=float(max(np.maximum(lower-linear_values,0.).max(),np.maximum(linear_values-upper,0.).max()))
                qp['success']=qp['success'] and qp['linear_primal_residual']<1e-4
            if (not qp['success'] or not np.isfinite(delta).all()) and self.elastic_restore:
                # Intermediate restoration only: one nonnegative shared slack
                # softens linearized rows. The executed solution must still pass
                # the configured original nonlinear acceptance check (applied
                # interval or horizon). Horizon residuals remain separately logged.
                tick=time.perf_counter();failed_qp=qp.copy();up=np.isfinite(upper);down=np.isfinite(lower)
                hard_prefix=int(os.environ.get('RTI_ELASTIC_PREFIX','-1'))
                soft=np.ones(len(lower))
                if hard_prefix>=0:
                    kinds=np.array(self.row_kinds);stages=np.array(self.row_stages)
                    soft[:len(g)]=((stages>hard_prefix)&~np.isin(kinds,['initial','dynamics'])).astype(float)
                    soft[len(g):]=0. # Input bounds and step trust bounds stay hard.
                hard=soft==0;up=up&~hard;down=down&~hard
                Ae=sparse.vstack([sparse.hstack([Aqp[hard],sparse.csc_matrix((hard.sum(),1))]),
                    sparse.hstack([Aqp[up],-soft[up,None]]),
                    sparse.hstack([-Aqp[down],-soft[down,None]]),
                    sparse.csc_matrix(([1.],([0],[len(z)])),shape=(1,len(z)+1))],format='csc')
                le=np.r_[lower[hard],np.full(up.sum()+down.sum(),-np.inf),0.]
                ue=np.r_[upper[hard],upper[up],-lower[down],np.inf]
                was_condensed=self.condense;self.condense=False
                try:
                    de,qe=self.qp_solve(sparse.diags(np.r_[np.ones(len(z)),1e-6],format='csc'),np.r_[np.zeros(len(z)),1000.],Ae,le,ue)
                finally:self.condense=was_condensed
                qp_seconds+=time.perf_counter()-tick
                if np.isfinite(de).all():
                    vals=Ae@de;qe['linear_primal_residual']=float(max(np.maximum(le-vals,0).max(),np.maximum(vals-ue,0).max()))
                    qe['success']=qe['success'] and qe['linear_primal_residual']<1e-4
                qe['initial_failed_qp']=failed_qp;qe['elastic_slack']=float(de[-1]);qe['qp_calls']=2;qe['hard_prefix_steps']=hard_prefix
                qe['backend_seconds']+=failed_qp['backend_seconds'];qe['iterations']+=failed_qp['iterations']
                delta=de[:-1];qp=qe;correction=True
            if not qp['success'] or not np.isfinite(delta).all():
                status='QP_'+qp['status'];traces.append(dict(iteration=iteration,qp=qp));break
            tick=time.perf_counter();alpha=1.;evaluations=0
            def scaled_l1(values,point):
                return float(np.sum((np.maximum(self.lbg-values,0)+np.maximum(values-self.ubg,0))*row_scale)+
                    np.sum(np.maximum((self.lbx-point)/self.scale,0)+np.maximum((point-self.ubx)/self.scale,0)))
            if self.globalize:
                penalty=100.;old_violation=scaled_l1(g,z)
                merit0=(0. if correction else float(obj)/objective_scale)+penalty*old_violation
                directional=(0. if correction else float(grad@(self.scale*delta))/objective_scale)-penalty*old_violation
                for trial in range(12):
                    candidate=z+alpha*self.scale*delta;f,gv=self.evaluate(candidate,p);gv=np.asarray(gv).ravel();evaluations+=1
                    merit=(0. if correction else float(f)/objective_scale)+penalty*scaled_l1(gv,candidate)
                    if merit<=merit0+1e-4*alpha*min(directional,-1e-10):break
                    alpha*=.5
                else:
                    status='merit_line_search_failed';traces.append(dict(iteration=iteration,qp=qp,merit=merit,merit0=merit0));break
                z=candidate
            else:
                z=z+self.scale*delta;f,gv=self.evaluate(z,p);gv=np.asarray(gv).ravel();evaluations=1
            violation=self.violation(z,gv)
            if not correction:objective_done=True;objective_updates+=1
            hs=gv[self.h_rows] if self.h_rows else np.array([np.nan])
            dh=gv[self.dh_rows] if self.dh_rows else np.array([np.nan])
            audit_seconds+=time.perf_counter()-tick
            traces.append(dict(iteration=iteration,feasibility_correction=correction,qp=qp,nonlinear_violation=violation,minimum_h=float(hs.min()),minimum_discrete_residual=float(dh.min()),objective=float(f),step_norm=float(np.linalg.norm(delta,np.inf)),alpha=alpha,evaluations=evaluations))
            applied=self.applied_audit(z)
            success=applied['passed'] and (self.acceptance=='applied' or violation<self.accept_tolerance) and (objective_done or not self.feasible_first)
            forecast_score=None
            if os.environ.get('RTI_FORECAST_FILTER','0')=='1':
                rv=np.maximum(np.maximum(self.lbg-gv,0),np.maximum(gv-self.ubg,0));kinds=np.array(self.row_kinds)
                limits=np.full(len(rv),.01)
                limits[np.isin(kinds,['gate_crossing','cbf','cbf_decay'])]=1e-4
                limits[np.isin(kinds,['initial','applied_cbf','applied_decay'])]=1e-5
                forecast_score=float((rv/limits).max(initial=0))
                if self.gate_geometry is not None:
                    gate_distance=float(np.asarray(self.gate_geometry(z,p)).max(initial=0.))
                    forecast_score=max(forecast_score,gate_distance/1e-4)
                    traces[-1]['gate_distance_violation_m']=gate_distance
                success=success and forecast_score<=1.
                traces[-1]['forecast_filter_score']=forecast_score
            if certify and violation<1e-5 and (forecast_score is None or forecast_score<=1.):
                check=applied if applied is not None else self.applied_audit(z)
                if check['passed']:
                    certified=(z.copy(),gv.copy(),violation);certificate_updates=objective_updates
            status=('RTI_applied_interval_accepted' if self.acceptance=='applied' else 'RTI_feasible') if success else ('RTI_applied_interval_violation' if self.acceptance=='applied' else 'RTI_nonlinear_residual')
            # One-iteration RTI always makes one update; additional requested
            # iterations are a separately identified residual-correction variant.
            if success:
                accepted=(z.copy(),gv.copy(),violation)
                if objective_updates>=min_objective or getattr(self,'reference_repair_only',False):break
        retained=False
        if certified is not None and (not success or status.startswith('QP_') or status=='merit_line_search_failed'):
            z,gv,violation=certified;success=True;retained=True;used_certificate=True;status='RTI_retained_full_feasible_iterate'
        elif accepted is not None and (not success or status.startswith('QP_') or status=='merit_line_search_failed'):
            z,gv,violation=accepted;success=True;retained=True;status='RTI_retained_accepted_iterate'
        self.X=z[:19*(self.n+1)].reshape(self.n+1,19);self.U=z[19*(self.n+1):].reshape(self.n,5)
        if retained:pass
        elif traces and 'nonlinear_violation'in traces[-1]:
            last=traces[-1];violation=last['nonlinear_violation']
        else:
            _,gv=self.evaluate(z,p);gv=np.asarray(gv).ravel();violation=self.violation(z,gv)
        result=dict(status=0 if success else 1,message=status,seconds=time.perf_counter()-start,preparation_seconds=preparation,
            feedback_seconds=qp_seconds,audit_seconds=audit_seconds,kkt=violation,primal_residual=violation,working_set=sum(t['qp']['iterations'] for t in traces),
            sqp_updates=len(traces),qp_backend=self.qp_backend,backend='Gauss-Newton SQP-RTI prototype',
            barrier_diagnostics=dict(minimum_predicted_h=float(gv[self.h_rows].min()) if self.h_rows else None,
                minimum_predicted_discrete_residual=float(gv[self.dh_rows].min()) if self.dh_rows else None),
            solve_trace=traces)
        result['objective_step_taken']=objective_done
        result['objective_updates']=objective_updates;result['retained_accepted_iterate']=retained
        result['retained_full_feasible_iterate']=used_certificate;result['starting_primal_violation']=initial_violation
        result['returned_iterate_objective_updates']=certificate_updates if used_certificate else objective_updates
        result['qp_calls']=sum(t['qp'].get('qp_calls',1) for t in traces)
        residual=np.maximum(np.maximum(self.lbg-gv,0),np.maximum(gv-self.ubg,0))
        equality=np.isfinite(self.lbg)&np.isfinite(self.ubg)&(self.lbg==self.ubg)
        result['residual_breakdown']=dict(dynamics_and_initial=float(residual[equality].max(initial=0)),
            nonlinear_inequality=float(residual[~equality].max(initial=0)),worst_constraint_row=int(np.argmax(residual)))
        result['stationarity_checked']=False
        result['acceptance']=self.acceptance;result['horizon_primal_feasible']=bool(violation<self.accept_tolerance)
        result['applied_audit']=self.applied_audit(z)
        self.history.append(result);return result
