"""Experimental feasible SQP updates of the existing full nonlinear frontend."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import time
import numpy as np
import casadi as ca
from scipy import sparse
from scipy.sparse.linalg import splu
import daqp


class FeasibleFrontendSQP:
    def __init__(self,planner):
        self.planner=planner
        z,p=planner.nlp_definition['x'],planner.nlp_definition['p'];f=planner.nlp_definition['f']
        self.objective=ca.Function('frontend_sqp_objective',[z,p],[f,ca.gradient(f,z),ca.hessian(f,z)[0]])
        self.value=ca.Function('frontend_sqp_value',[z,p],[f])
        self.jacobian=planner.solver.get_function('nlp_jac_g')
        self.evaluate=planner.constraint_value if hasattr(planner,'constraint_value') else ca.external('nlp_g',planner.c['phase_native_library'])
        self.nx=13*(planner.n+1)
        self.scale=np.r_[np.tile(np.r_[np.full(3,5.),np.full(3,10.),np.ones(4),np.full(3,5.)],planner.n+1),np.full(4*planner.n,5.),np.ones(planner.M)]
        self.D=sparse.diags(self.scale,format='csc');self.I=sparse.eye(len(self.scale),format='csc')
        self.dynamic=np.r_[np.arange(13),*[np.arange(sl.start,sl.start+13) for sl in planner.cell_rows]]

    def solve(self,state,gates,normals,half,guess):
        begin=time.perf_counter();p=self.planner
        par=p.parameters(state,gates,normals,half)
        current=guess.copy();trace=[];passed=False;best_feasible=None;dual_cache=None
        objective_steps=0;accepted_objective_steps=0
        target_steps=int(p.c.get('phase_sqp_objective_steps',1))
        objective_update=target_steps>0
        trust=float(p.c.get('phase_sqp_trust',.5))
        for iteration in range(int(p.c.get('phase_sqp_max_iterations',8))):
            if time.perf_counter()-begin>float(p.c.get('phase_sqp_time_budget',1.)):break
            gg,J=self.jacobian(current,par);g=np.asarray(gg).ravel();A=J.sparse()@self.D
            correction=not objective_update
            if correction:
                fv=self.value(current,par);B=self.I;gradient=np.zeros_like(self.scale)
            else:
                fv,gradient,H=self.objective(current,par);gradient=np.asarray(gradient).ravel()*self.scale
                regularization=float(p.c.get('phase_sqp_polish_regularization',.1)) if best_feasible is not None else .1
                B=self.D@H.sparse()@self.D+regularization*self.I
                objective_steps+=1
            objective_update=False
            rs=1/np.maximum(np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel()),1.)
            A=sparse.diags(rs)@A;lower=(p.lbg-g)*rs;upper=(p.ubg-g)*rs
            lower=np.r_[lower,np.maximum((p.lbx-current)/self.scale,-trust)]
            upper=np.r_[upper,np.minimum((p.ubx-current)/self.scale,trust)]
            A=sparse.vstack([A,self.I],format='csc')
            warm={}
            if p.c.get('phase_sqp_backend','daqp')=='clarabel':
                if not hasattr(self,'sparse_backend'):
                    from scripts.planning.utils.sparse_frontend_qp import SparseFrontendQP
                    self.sparse_backend=SparseFrontendQP()
                delta,code,info,row_ids=self.sparse_backend.solve(B,gradient,A,lower,upper,trust)
                qp_rows=len(row_ids)
            else:
                if p.c.get('phase_sqp_chain_condensing',False):
                    from scripts.planning.utils.chain_condensing import condense
                    mapping,offset=condense(A,self.dynamic,self.nx,upper)
                else:
                    factor=splu(A[self.dynamic,:self.nx].tocsc())
                    mapping=np.vstack([-factor.solve(A[self.dynamic,self.nx:].toarray()),np.eye(len(self.scale)-self.nx)])
                    offset=np.r_[factor.solve(upper[self.dynamic]),np.zeros(len(self.scale)-self.nx)]
                reduced_H=mapping.T@(B@mapping);reduced_H=.5*(reduced_H+reduced_H.T)
                reduced_gradient=mapping.T@(B@offset+gradient)
                keep=np.ones(A.shape[0],dtype=bool);keep[self.dynamic]=False;remaining=np.flatnonzero(keep)
                reach=trust*np.asarray(abs(A[remaining]).sum(axis=1)).ravel()
                remaining=remaining[(lower[remaining]>=-reach-1e-10)|(upper[remaining]<=reach+1e-10)]
                reduced_A=A[remaining]@mapping;shift=A[remaining]@offset
                rlo=lower[remaining]-shift;rhi=upper[remaining]-shift
                active=np.linalg.norm(reduced_A,axis=1)>1e-12
                if np.any(rlo[~active]>1e-6) or np.any(rhi[~active]<-1e-6):
                    trace.append(dict(iteration=iteration,status='inconsistent_condensed_constant'));break
                reduced_A=reduced_A[active];rlo=rlo[active];rhi=rhi[active]
                sense=np.where(np.isfinite(rlo)&(abs(rlo-rhi)<1e-12),5,0).astype(np.int32)
                row_ids=remaining[active];warm={}
                if p.c.get('phase_sqp_dual_warm',False) and dual_cache is not None:
                    warm['dual_start']=dual_cache[row_ids].copy()
                qp_solver=daqp.solve
                if p.c.get('phase_sqp_constraint_generation',False):
                    from scripts.planning.utils.constraint_generation_qp import solve as qp_solver
                reduced,_,code,info=qp_solver(np.ascontiguousarray(reduced_H),reduced_gradient,np.ascontiguousarray(reduced_A),rhi,rlo,sense,primal_tol=1e-8,dual_tol=1e-8,iter_limit=1000,**warm)
                delta=mapping@reduced+offset
                qp_rows=len(rlo)
            linear=A@delta
            error=max(float(np.maximum(lower-linear,0.).max()),float(np.maximum(linear-upper,0.).max()))
            elastic=None
            if (code!=1 or not np.isfinite(delta).all() or error>1e-4) and p.c.get('phase_sqp_elastic',False):
                from scripts.planning.utils.frontend_restoration import restore
                first_info=info
                delta,elastic=restore(A,lower,upper,len(g),13)
                info=dict(elastic,lam=np.zeros(len(row_ids)))
                info['setup_time']+=first_info['setup_time'];info['solve_time']+=first_info['solve_time']
                info['iterations']+=first_info['iterations']
                code=1 if elastic['success'] else -99
                if not correction:objective_steps-=1
                error=elastic['elastic_qp_residual'];correction=True
            if code!=1 or not np.isfinite(delta).all() or error>1e-4:
                trace.append(dict(iteration=iteration,status=f'QP_{code}',native_status=info.get('native_status'),qp_backend=p.c.get('phase_sqp_backend','daqp'),dual_warm_start=bool(warm),linear_error=error,qp_setup_seconds=info['setup_time'],qp_solve_seconds=info['solve_time']));break
            dual_cache=np.zeros(A.shape[0]);dual_cache[row_ids]=info['lam']
            old_residual=np.maximum(np.maximum(p.lbg-g,0.),g-p.ubg)
            merit=(0. if correction else float(fv))+100*np.sum(old_residual*rs);chosen=None
            if elastic is not None:merit=100*float(np.max(old_residual*rs,initial=0.))
            filter_step=False
            for alpha in [1.,.5,.25,.125,.0625]:
                candidate=current+alpha*self.scale*delta;gv=np.asarray(self.evaluate(candidate,par)).ravel()
                residual=np.maximum(np.maximum(p.lbg-gv,0.),gv-p.ubg)
                val=float(self.value(candidate,par));new_merit=(0. if correction else val)+100*np.sum(residual*rs)
                if elastic is not None:new_merit=100*float(np.max(residual*rs,initial=0.))
                if new_merit<merit+1e-9:chosen=candidate;break
                # A feasible incumbent is retained while a small objective step
                # is restored. This candidate cannot become an output until all
                # nonlinear constraints and the independent audit pass.
                if (p.c.get('phase_sqp_polish_filter',False) and not correction
                    and best_feasible is not None and alpha<=.25
                    and val<float(fv)-1e-8 and float(residual.max())<.1):
                    chosen=candidate;filter_step=True;break
            violation=max(float(residual.max()),float(np.maximum(p.lbx-candidate,0.).max()),float(np.maximum(candidate-p.ubx,0.).max()))
            trace.append(dict(iteration=iteration,status='Solved',correction=correction,elastic_restoration=elastic,filter_step=filter_step,dual_warm_start=bool(warm),qp_iterations=info['iterations'],linear_error=error,residual=violation,alpha=alpha,objective=val,qp_rows=qp_rows,qp_backend=p.c.get('phase_sqp_backend','daqp'),qp_setup_seconds=info['setup_time'],qp_solve_seconds=info['solve_time']))
            if 'constraint_generation' in info:trace[-1]['constraint_generation']=info['constraint_generation']
            if chosen is None:break
            if not correction:accepted_objective_steps+=1
            current=chosen
            if violation<float(p.c.get('phase_sqp_feasibility_tol',2e-5)):
                if best_feasible is None or val<best_feasible[1]:best_feasible=(current.copy(),val,violation)
                if objective_steps>=target_steps:break
                objective_update=True
        if best_feasible is not None and objective_steps>=target_steps:
            current,_,violation=best_feasible;passed=True
        summary=dict(stage='feasible_sqp_update',status='feasible' if passed else 'not_feasible',iterations=len(trace),objective_steps=objective_steps,accepted_objective_steps=accepted_objective_steps,seconds=time.perf_counter()-begin,updates=trace)
        if not passed:return None,summary
        plan=dict(X=current[:self.nx].reshape(13,p.n+1,order='F'),U=current[self.nx:self.nx+4*p.n].reshape(4,p.n,order='F'),
                  durations=current[-p.M:].copy(),z=current.copy(),gates=gates.copy(),normals=normals.copy(),half=half.copy(),
                  lam_x=np.zeros_like(current),lam_g=np.zeros_like(p.lbg),value=float(self.value(current,par)),residual=violation,
                  solve_trace=[],stationarity_verified=False,gradient_available=False,frontend_update='feasibility_projection' if target_steps==0 else ('single_time_step_then_feasibility' if target_steps==1 else 'bounded_time_steps_then_feasibility'))
        audit_start=time.perf_counter()
        audit,tt,xx=p.audit(plan,state);audit_seconds=time.perf_counter()-audit_start
        summary.update(audit_passed=audit['passed'],audit_seconds=audit_seconds,seconds=time.perf_counter()-begin)
        if not audit['passed']:
            summary['status']='audit_failed';summary['failed_audit']=audit
            remaining=float(p.c.get('phase_sqp_time_budget',1.))-(time.perf_counter()-begin)
            if p.c.get('phase_sqp_audit_refine',False) and remaining>.02 and not getattr(self,'refining_audit',False):
                keys=['phase_sqp_objective_steps','phase_sqp_max_iterations','phase_sqp_time_budget','phase_sqp_feasibility_tol']
                saved={key:p.c.get(key) for key in keys}
                self.refining_audit=True
                try:
                    p.c.update(phase_sqp_objective_steps=0,phase_sqp_max_iterations=2,
                               phase_sqp_time_budget=remaining,phase_sqp_feasibility_tol=1e-7)
                    refined,refinement=self.solve(state,gates,normals,half,current)
                finally:
                    self.refining_audit=False
                    for key,value in saved.items():
                        if value is None:p.c.pop(key,None)
                        else:p.c[key]=value
                summary.update(audit_refinement=refinement,seconds=time.perf_counter()-begin,
                               iterations=summary['iterations']+refinement['iterations'])
                if refined is not None:
                    summary.update(status='feasible',audit_passed=True)
                    refined['frontend_update']='time_step_then_audit_refinement'
                    return refined,summary
            return None,summary
        plan.update(audit=audit,reference_time=tt,reference_states=xx,audited_initial_state=np.asarray(state).copy(),audit_seconds=audit_seconds)
        return plan,summary
