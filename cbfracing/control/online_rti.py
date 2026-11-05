from cbfracing.core.context import *
from cbfracing.control.rti import RTIMPCC
from cbfracing.control.controller import RecoveryMPCC as IPOPT
class OnlineRTI(RTIMPCC):
    def __init__(self,*args,**kwargs):
        self.initialized=os.environ.get('OPT_RTI_COLD_START','0')=='1'
        self.initializer=None if self.initialized else IPOPT(*args,**kwargs)
        super().__init__(*args,qp_backend='clarabel',iterations=int(os.environ.get('RTI_UPDATES','1')),regularization=float(os.environ.get('RTI_LM','.001')),trust_radius=float(os.environ.get('RTI_TRUST','2')),globalize=os.environ.get('RTI_GLOBALIZE','0')=='1',feasibility_correction=os.environ.get('RTI_FEASIBILITY','0')=='1',feasible_first=os.environ.get('RTI_FIRST','0')=='1',elastic_restore=os.environ.get('RTI_ELASTIC','0')=='1',condense=os.environ.get('RTI_CONDENSE','0')=='1',native=os.environ.get('RTI_NATIVE','0')=='1',**kwargs)
    def step(self):
        if not self.initialized:
            s=self.initializer;s.state=self.state.copy();s.od=self.od.copy();s.w=self.w.copy();s.mu=self.mu;s.seed(self.X,self.U)
            result=s.step();self.X,self.U=s.prediction();self.initialized=result['status']==0
            result.update(initialization=True,backend='IPOPT initialization');return result
        self.acceptance=os.environ.get('RTI_ACCEPTANCE','horizon')
        self.qp_tolerance=float(os.environ.get('RTI_QP_TOL','1e-8'))
        result=super().step();result['initialization']=False
        if result['status']!=0 and os.environ.get('OPT_SQP_RECOVERY','0')=='1':
            original=result
            saved={k:getattr(self,k,None) for k in ['iterations','acceptance','elastic_restore','globalize','reference_repair_only','trust_radius']}
            previous_prefix=os.environ.get('RTI_ELASTIC_PREFIX')
            try:
                self.iterations=60;self.acceptance='horizon';self.elastic_restore=True
                self.globalize=False;self.reference_repair_only=True
                self.trust_radius=max(self.trust_radius,2.)
                os.environ['RTI_ELASTIC_PREFIX']=os.getenv('OPT_SQP_RECOVERY_PREFIX','0')
                restored=super().step()
            finally:
                for k,v in saved.items():setattr(self,k,v)
                if previous_prefix is None:os.environ.pop('RTI_ELASTIC_PREFIX',None)
                else:os.environ['RTI_ELASTIC_PREFIX']=previous_prefix
            check=self.applied_audit(np.r_[self.X.ravel(),self.U.ravel()])
            passed=restored['status']==0 and restored['horizon_primal_feasible'] and check['passed']
            result=dict(restored)
            result.update(status=0 if passed else 1,initialization=False,
                message='SQP_recovery_accepted' if passed else 'SQP_recovery_'+restored['message'],
                seconds=original['seconds']+restored['seconds'],sqp_recovery=True,
                sqp_recovery_seconds=restored['seconds'],original_rti_failure=original,
                applied_audit=check)
        budget=float(os.environ.get('RTI_NLP_RECOVERY_SECONDS','0'))
        if result['status']!=0 and budget>0:
            # Same experimental OCP, including dense applied CBF rows. A rare
            # nonlinear restoration is explicit, timed, and audited; no control
            # is accepted solely because the fallback solver reports success.
            begin=time.perf_counter();rti_result=result
            if not hasattr(self,'recovery_solver'):
                self.recovery_solver=ca.nlpsol('rti_recovery','ipopt',dict(x=self.z,p=self.p,f=self.objective,g=self.g),
                    dict(print_time=False,error_on_fail=False,expand=True,ipopt=dict(print_level=0,sb='yes',tol=1e-6,
                    acceptable_tol=2e-5,max_iter=300,max_wall_time=budget,bound_relax_factor=0.,mu_strategy='adaptive',linear_solver='mumps')))
            seed=np.r_[self.X.ravel(),self.U.ravel()]
            answer=self.recovery_solver(x0=seed,p=self.parameters(),lbx=self.lbx,ubx=self.ubx,lbg=self.lbg,ubg=self.ubg)
            z=np.asarray(answer['x']).ravel();gv=np.asarray(answer['g']).ravel();stats=self.recovery_solver.stats()
            residual=self.violation(z,gv);check=self.applied_audit(z)
            passed=bool(np.isfinite(z).all() and residual<self.accept_tolerance and check['passed'])
            if passed:
                self.X=z[:19*(self.n+1)].reshape(self.n+1,19);self.U=z[19*(self.n+1):].reshape(self.n,5)
            elapsed=time.perf_counter()-begin
            result=dict(rti_result)
            result.update(status=0 if passed else 1,message='NLP_recovery_accepted' if passed else 'NLP_recovery_'+stats['return_status'],
                seconds=rti_result['seconds']+elapsed,nlp_recovery=True,nlp_recovery_seconds=elapsed,
                nlp_recovery_status=stats['return_status'],nlp_recovery_iterations=int(stats['iter_count']),
                nlp_recovery_residual=residual,original_rti_failure=rti_result['message'],nlp_recovery_applied_audit=check)
            if passed:
                result.update(primal_residual=residual,kkt=residual,horizon_primal_feasible=True,applied_audit=check,
                    barrier_diagnostics=dict(minimum_predicted_h=float(gv[self.h_rows].min()),minimum_predicted_discrete_residual=float(gv[self.dh_rows].min())))
                rv=np.maximum(np.maximum(self.lbg-gv,0),np.maximum(gv-self.ubg,0))
                eq=np.isfinite(self.lbg)&np.isfinite(self.ubg)&(self.lbg==self.ubg)
                result['residual_breakdown']=dict(dynamics_and_initial=float(rv[eq].max(initial=0)),nonlinear_inequality=float(rv[~eq].max(initial=0)),worst_constraint_row=int(np.argmax(rv)))
        return result
