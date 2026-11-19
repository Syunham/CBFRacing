"""Fast-fly-style fixed-event time optimization with per-agent warm starts.

The physical/gate constraints and independent audit are shared with GatePlanner.
The formal objective is elapsed time. A bounded soft-dynamics warm-up supplies
the initial trajectory, and overlapping segments carry primal/dual guesses.
"""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import time

import casadi as ca
import numpy as np

from scripts.planning.cpc.phase_planner import GatePlanner
from scripts.planning.fast_fly.warm_seed import finite_seed


class FastFlyPlanner(GatePlanner):
    def __init__(self, config, course, optimize_durations=True):
        config = dict(config,cells_per_leg=config.get('phase_cells_per_leg',config['cells_per_leg']))
        super().__init__(dict(config, fast_fly_phase=True, phase_solve_retries=0), course,
                         optimize_durations)
        self.cache = {}
        self.past_lap_cache = {}
        self.last_objective_update = {}
        self.track_gate_count=len(course['centers'])
        self.solver_name = 'ipopt_fast_fly_style'
        self.active_agent = 0
        self.event_index = 0
        self.current_time = 0.
        if config.get('phase_online_sqp',False) and not config.get('phase_native_library'):
            raise ValueError('Experimental online SQP requires the verified native frontend library')
        s = self.symbols
        X, U, T = s['X'], s['U'], s['T']
        positions = ca.horzcat(*[X[:3, (j+1)*config['cells_per_leg']] for j in range(self.M)])
        objective = ca.sumsqr(s['dynamics'])+ca.sumsqr(positions-s['G'])+.01*ca.sumsqr(X[10:13, :])
        warm_parameters = ca.vertcat(s['p'], T) if optimize_durations else s['p']
        self.preheater = ca.nlpsol('phase_warmup', 'ipopt',
            dict(x=ca.vertcat(ca.vec(X),ca.vec(U)), p=warm_parameters,
                 f=objective, g=X[:,0]-s['x0']),
            dict(print_time=False,error_on_fail=False,expand=config.get('expand_solver_graph',False),
                 ipopt=dict(print_level=0,sb='yes',max_iter=200,max_wall_time=8.,tol=1e-5)))
        n = self.n
        lower = np.full((13,n+1),-np.inf)
        upper = np.full((13,n+1),np.inf)
        world = np.asarray(course['world_box'])
        lower[:3] = world[:3,None]
        upper[:3] = world[3:,None]
        lower[6:10], upper[6:10] = -1., 1.
        lower[10:13], upper[10:13] = -config['body_rate_limit'], config['body_rate_limit']
        vehicle = config['vehicle']
        self.warm_lower = np.r_[lower.ravel(order='F'),np.full(4*n,vehicle['thrust_min'])]
        self.warm_upper = np.r_[upper.ravel(order='F'),np.full(4*n,vehicle['thrust_max'])]
        if config.get('phase_optimize_tail',False):
            from scripts.planning.fast_fly.optimized_tail_seed import OptimizedTailSeed
            self.optimized_tail=OptimizedTailSeed(self.c,course)
        if config.get('phase_learned_initializer'):
            from scripts.planning.fast_fly.learned_warm_start import LearnedWarmStart
            self.learned_initializer=LearnedWarmStart(self,config['phase_learned_initializer'],config.get('phase_learned_budget',3.))
        if config.get('phase_reference_continuity',0)>0:
            from scripts.planning.fast_fly.reference_continuity import ReferenceContinuityNLP
            self.reference_backend=ReferenceContinuityNLP(self,config['phase_reference_continuity'])

    def seed(self, state, gates, normals, durations):
        # Analytic spline derivatives and the actual nonuniform time grid.
        return finite_seed(state,gates,normals,durations,self.c)

    def shifted_guess(self, previous, state, gates, normals, durations):
        self.tail_seed_trace=None
        m = self.c['cells_per_leg']
        nx = 13*(self.n+1)
        guess = self.seed(state,gates,normals,durations)
        X = guess[:nx].reshape(13,self.n+1,order='F')
        U = guess[nx:].reshape(4,self.n,order='F')
        old_index, old = previous
        advance = self.event_index-old_index
        if advance not in (0,1):
            return None
        shift = advance*m
        X[:,:self.n+1-shift] = old['X'][:,shift:]
        U[:,:self.n-shift] = old['U'][:,shift:]
        X[:,0] = state
        times = np.r_[old['durations'][advance:],durations[self.M-advance:]] if advance else old['durations'].copy()
        elapsed=self.current_time-old.get('start_time',self.current_time)
        if self.c.get('phase_time_shift',False) and elapsed>0:
            # Advance the old solution in physical time, then remesh only the
            # remaining first leg. This supplies an initial guess, never an
            # accepted trajectory without a fresh solve and independent audit.
            consumed=elapsed-float(np.sum(old['durations'][:advance]))
            remaining_lower=float(self.lbx[-self.M]) if self.optimize_durations else self.c.get('phase_remaining_min',.03)
            times[0]=max(remaining_lower,times[0]-consumed)
            new_grid=np.r_[0.,np.cumsum(np.repeat(times/m,m))]
            old_grid=np.r_[0.,np.cumsum(np.repeat(old['durations']/m,m))]
            query=elapsed+new_grid
            valid=query<=old_grid[-1]
            for j in range(13):
                X[j,valid]=np.interp(query[valid],old['reference_time'],old['reference_states'][:,j])
            X[6:10]/=np.linalg.norm(X[6:10],axis=0)
            old_cell=np.clip(np.searchsorted(old_grid[1:],query[:-1],side='right'),0,self.n-1)
            boundary_safe=self.c.get('phase_consistent_tail_boundary',False)
            seed_controls=U.copy() if boundary_safe else None
            boundary_tolerance=1e-10*max(1.,abs(old_grid[-1])) if boundary_safe else 0.
            control_valid=query[:-1]<old_grid[-1]-boundary_tolerance
            U[:,control_valid]=old['U'][:,old_cell[control_valid]]
            if self.c.get('phase_average_shifted_controls',False):
                # Preserve each remeshed cell's force/torque impulse instead
                # of holding the old control sampled at its left endpoint.
                cumulative=np.c_[np.zeros(4),np.cumsum(old['U']*np.diff(old_grid),axis=1)]
                contained=(query[:-1]>=0.)&(query[1:]<=old_grid[-1]+1e-10)
                for j in range(4):
                    integrated=np.interp(query,old_grid,cumulative[j])
                    U[j,contained]=(np.diff(integrated)/np.diff(new_grid))[contained]
                    if boundary_safe:
                        # A cell spanning the old horizon carries only the old
                        # impulse over its overlap; the rest uses its tail seed.
                        # Roundoff-sized overlaps belong entirely to the new tail.
                        partial=control_valid&~contained
                        width=np.diff(new_grid)
                        overlap=np.clip(old_grid[-1]-query[:-1],0.,width)
                        U[j,partial]=((np.diff(integrated)+(width-overlap)*seed_controls[j])/width)[partial]
            X[:,0]=state
        if advance and self.c.get('phase_tail_seed',False):
            seam=self.n-m
            tail=finite_seed(X[:,seam].copy(),gates[-1:],normals[-1:],times[-1:],
                             dict(self.c,seed_end_speed_max=15.,seed_preserve_heading=True))
            nt=13*(m+1)
            X[:,seam+1:]=tail[:nt].reshape(13,m+1,order='F')[:,1:]
            U[:,seam:]=tail[nt:].reshape(4,m,order='F')
            if self.c.get('phase_tail_pd',False):
                X[:,seam:],U[:,seam:]=self.tail_rollout(X[:,seam:].copy(),U[:,seam:].copy(),times[-1]/m)
        if advance and hasattr(self,'optimized_tail'):
            seam=self.n-m
            tail,self.tail_seed_trace=self.optimized_tail.solve(X[:,seam].copy(),gates[-1:],normals[-1:],
                self.active_half[-1:],times[-1:],U[:,seam-1].copy())
            if tail is not None:
                X[:,seam:]=tail['X'];U[:,seam:]=tail['U'];times[-1]=tail['durations'][0]
        result = np.r_[X.ravel(order='F'),U.ravel(order='F')]
        lam_x = np.zeros_like(old['lam_x'])
        lam_g = np.zeros_like(old['lam_g'])
        if not advance:
            lam_x[:] = old['lam_x']
            lam_g[:] = old['lam_g']
        else:
            lx = lam_x[:nx].reshape(13,self.n+1,order='F')
            lx[:,:-shift] = old['lam_x'][:nx].reshape(13,self.n+1,order='F')[:,shift:]
            lu = lam_x[nx:nx+4*self.n].reshape(4,self.n,order='F')
            lu[:,:-shift] = old['lam_x'][nx:nx+4*self.n].reshape(4,self.n,order='F')[:,shift:]
            for k in range(self.n-shift):
                lam_g[self.cell_rows[k]] = old['lam_g'][self.cell_rows[k+shift]]
            for j in range(self.M-1):
                new, prior = self.gate_rows[j], self.gate_rows[j+1]
                count = min(new.stop-new.start,prior.stop-prior.start)
                lam_g[new.start:new.start+count] = old['lam_g'][prior.start:prior.start+count]
            if self.optimize_durations:
                lam_x[-self.M:-1] = old['lam_x'][-self.M+1:]
        lam_x[:13] = 0.
        lam_g[:13] = 0.
        if self.c.get('phase_time_shift',False) and self.c.get('phase_reset_remeshed_duals',False):
            lam_x[:13*(m+1)]=0.
            lam_x[nx:nx+4*m]=0.
            for rows in self.cell_rows[:m]:lam_g[rows]=0.
            if self.optimize_durations:lam_x[-self.M]=0.
        if self.optimize_durations:
            result = np.r_[result,times]
        if self.c.get('phase_zero_shifted_dual',False):
            lam_x[:]=0.;lam_g[:]=0.
        if self.c.get('phase_projection_steps',0)>0:
            result=self.project_warm(result,state,gates,normals,self.active_half)
            lam_x[:]=0.;lam_g[:]=0.
        return result,(lam_x,lam_g)

    def tail_rollout(self,target,controls,dt):
        """Physically roll out a bounded geometric controller as a warm guess."""
        from scipy.spatial.transform import Rotation
        v=self.c['vehicle'];inertia=np.asarray(v['inertia']);arm=v['arm_length']/np.sqrt(2);tc=v['torque_coefficient']
        mixing=np.array([[1,1,1,1],[arm,arm,-arm,-arm],[-arm,arm,arm,-arm],[tc,-tc,tc,-tc]])
        inverse=np.linalg.inv(mixing);states=np.zeros_like(target);states[:,0]=target[:,0]
        for k in range(controls.shape[1]):
            x=states[:,k];R=Rotation.from_quat(x[[7,8,9,6]]).as_matrix()
            desired_acc=(target[3:6,k+1]-target[3:6,k])/dt+8.*(target[:3,k]-x[:3])+4.*(target[3:6,k]-x[3:6])
            desired_force=v['mass']*(desired_acc+[0.,0.,v['gravity']])+R@np.diag(v['linear_drag_kg_s'])@R.T@x[3:6]
            z=desired_force/max(np.linalg.norm(desired_force),1e-8)
            heading=R[:,0];side=np.cross(z,heading)
            if np.linalg.norm(side)<1e-6:side=np.cross(z,R[:,1])
            side/=np.linalg.norm(side);Rd=np.column_stack([np.cross(side,z),side,z])
            error=.5*(Rd.T@R-R.T@Rd);er=np.array([error[2,1],error[0,2],error[1,0]])
            acceleration=-64.*er+12.*(target[10:13,k]-x[10:13])
            predicted=x[10:13]+dt*acceleration;predicted*=min(1.,9./max(np.linalg.norm(predicted),1e-9))
            torque=inertia*((predicted-x[10:13])/dt)+np.cross(x[10:13],inertia*x[10:13])
            force=float(desired_force@R[:,2])
            controls[:,k]=np.clip(inverse@np.r_[force,torque],v['thrust_min'],v['thrust_max'])
            states[:,k+1]=np.asarray(self.step(x,controls[:,k],dt)).ravel()
        return states,controls

    def project_warm(self,guess,state,gates,normals,half):
        """Bounded sparse least-squares feasibility corrections of a guess."""
        from scipy import sparse
        from scipy.sparse.linalg import lsmr
        parameters=self.parameters(state,gates,normals,half)
        scales=np.r_[np.tile(np.r_[np.ones(3),np.full(3,5.),np.full(4,.3),np.full(3,4.)],self.n+1),
                     np.full(4*self.n,2.),np.full(self.M,.25)]
        derivative=self.solver.get_function('nlp_jac_g');evaluate=self.solver.get_function('nlp_g')
        current=guess.copy();records=[];start=time.perf_counter()
        for iteration in range(self.c['phase_projection_steps']):
            gv,jac=derivative(current,parameters);gv=np.asarray(gv).ravel();jac=jac.sparse()
            residual=np.clip(gv,self.lbg,self.ubg)-gv
            selected=((self.lbg==self.ubg)| (np.abs(residual)>1e-8))
            js=jac@sparse.diags(scales)
            row_scale=np.maximum(.1,1./np.maximum(np.sqrt(np.asarray(js.multiply(js).sum(axis=1)).ravel()),1.))
            system=sparse.diags(row_scale[selected])@js[selected]
            answer=lsmr(system,row_scale[selected]*residual[selected],damp=.001,atol=1e-5,btol=1e-5,maxiter=100)
            delta=answer[0];delta/=max(1.,float(np.abs(delta).max()))
            merit=float(np.linalg.norm(row_scale*residual));accepted=False
            for fraction in [1.,.5,.25,.1,.05,.01]:
                candidate=np.clip(current+fraction*scales*delta,self.lbx,self.ubx)
                candidate[:13]=state
                cg=np.asarray(evaluate(candidate,parameters)).ravel()
                cr=np.clip(cg,self.lbg,self.ubg)-cg
                if np.linalg.norm(row_scale*cr)<merit and np.abs(cr).max()<=np.abs(residual).max()+1e-8:
                    current=candidate;accepted=True;break
            records.append(dict(iteration=iteration,initial_violation=float(np.abs(residual).max()),
                final_violation=float(np.abs(cr).max()) if accepted else float(np.abs(residual).max()),
                accepted=accepted,lsmr_iterations=int(answer[2]),fraction=fraction if accepted else 0.))
            if not accepted:break
        self.projection_trace=dict(seconds=time.perf_counter()-start,iterations=len(records),records=records)
        return current

    def preheat(self, state, gates, normals, half, durations, seed):
        nxu = 13*(self.n+1)+4*self.n
        parameters = self.parameters(state,gates,normals,half,durations)
        if self.optimize_durations:parameters=np.r_[parameters,durations]
        start = time.perf_counter()
        result = self.preheater(x0=seed[:nxu],p=parameters,lbx=self.warm_lower,
                                ubx=self.warm_upper,lbg=np.zeros(13),ubg=np.zeros(13))
        stats = self.preheater.stats()
        guess = np.asarray(result['x']).ravel()
        self.warmup_trace = dict(stage='warmup',seconds=time.perf_counter()-start,
                                iterations=int(stats['iter_count']),status=stats['return_status'])
        if not np.isfinite(guess).all():
            raise RuntimeError('Warm-up produced a nonfinite initial trajectory')
        return guess

    def solve(self,state,gates,normals,half,durations,warm=None):
        start = time.perf_counter()
        traces = []
        self.active_half=half
        self.projection_trace=None
        previous = self.cache.get(self.active_agent)
        gate_changed=bool(previous is not None and self.event_index!=previous[0])
        if self.optimize_durations and self.c.get('phase_time_shift',False):
            self.lbx[-self.M]=self.c.get('phase_remaining_min',.03) if previous else self.c['duration_min']
            if previous and self.c.get('phase_adaptive_remaining_min',False):
                distance=max(0.,-float((state[:3]-gates[0])@normals[0]))
                speed=max(.8,float(state[3:6]@normals[0]))
                self.lbx[-self.M]=min(self.lbx[-self.M],max(1e-7,.25*distance/speed))
            if previous and self.c.get('phase_adaptive_first_buffer',False):
                distance=max(0.,-float((state[:3]-gates[0])@normals[0]))
                row=self.gate_rows[0].start+6+self.c['cells_per_leg']
                self.ubg[row]=-min(.005,max(1e-7,.5*distance/self.c['cells_per_leg']))
        saved_tail=self.c.get('phase_tail_seed',False)
        if self.c.get('phase_gate_tail_nlp',False):self.c['phase_tail_seed']=gate_changed
        try:shifted = self.shifted_guess(previous,state,gates,normals,durations) if previous else None
        finally:self.c['phase_tail_seed']=saved_tail
        if shifted and self.tail_seed_trace is not None:traces.append(self.tail_seed_trace)
        used_cache = shifted is not None
        if shifted:
            guess, dual = shifted
            if gate_changed and self.c.get('phase_gate_tail_nlp',False):dual=None
        else:
            guess = self.seed(state,gates,normals,durations) if warm is None else warm
            guess = self.preheat(state,gates,normals,half,durations,guess)
            traces.append(self.warmup_trace)
            dual = None
        plan=None
        if used_cache and self.c.get('phase_online_sqp',False) and not (gate_changed and self.c.get('phase_gate_tail_nlp',False)):
            self.c['phase_sqp_trust']=self.c.get('phase_sqp_gate_trust',.5) if self.event_index!=previous[0] else .5
            self.c['phase_sqp_objective_steps']=0 if self.c.get('phase_periodic_projection',False) and not gate_changed else 1
            objective_period=float(self.c.get('phase_objective_period',0.))
            if objective_period>0:
                last=self.last_objective_update.get(self.active_agent,-np.inf)
                due=gate_changed or self.current_time-last>=objective_period-1e-9
                self.c['phase_sqp_objective_steps']=int(due)
                traces.append(dict(stage='objective_cadence',iterations=0,seconds=0.,
                    time=self.current_time,last_objective_time=last,period=objective_period,
                    gate_changed=gate_changed,requested_objective_step=due))
            if not hasattr(self,'online_sqp'):
                from scripts.planning.fast_fly.fast_fly_sqp import FeasibleFrontendSQP
                self.online_sqp=FeasibleFrontendSQP(self)
            plan,sqp_trace=self.online_sqp.solve(state,gates,normals,half,guess)
            traces.append(sqp_trace)
            if plan is None and self.c.get('phase_past_lap_warm',False):
                candidates=[(event,old) for event,old in self.past_lap_cache.get((self.active_agent,self.event_index%self.track_gate_count),[])
                            if event<self.event_index]
                if candidates:
                    scales=np.r_[np.full(3,1/5),np.full(3,1/10)]
                    event,old=min(candidates,key=lambda pair:np.linalg.norm((pair[1]['X'][:6,0]-state[:6])*scales))
                    history_guess=np.r_[old['X'].ravel(order='F'),old['U'].ravel(order='F'),old['durations']]
                    history_guess[:13]=state
                    plan,history_trace=self.online_sqp.solve(state,gates,normals,half,history_guess)
                    history_trace.update(stage='history_sqp_update',template_event=event,template_time=old['start_time'],
                                         query_event=self.event_index,query_time=self.current_time,same_agent_only=True)
                    traces.append(history_trace)
        if plan is None and used_cache and hasattr(self,'learned_initializer'):
            plan,learned_trace=self.learned_initializer.solve(self,state,gates,normals,half,durations)
            traces.append(learned_trace)
        try:
            if plan is None:
                if used_cache and hasattr(self,'reference_backend'):
                    plan=self.reference_backend.solve(state,gates,normals,half,durations,guess,dual)
                else:plan = super().solve(state,gates,normals,half,durations,warm=guess,dual_start=dual)
        except RuntimeError:
            traces.extend(dict(row,stage='failed_hot_solve') for row in self.last_trace)
            if not used_cache:
                self.last_trace = traces
                raise
            guess = self.preheat(state,gates,normals,half,durations,guess)
            traces.append(self.warmup_trace)
            try:
                plan = super().solve(state,gates,normals,half,durations,warm=guess)
            except RuntimeError:
                self.last_trace = traces+[dict(row,stage='failed_recovery') for row in self.last_trace]
                raise
        traces.extend(dict(row,stage='time_optimal') for row in plan['solve_trace'])
        plan.update(seconds=time.perf_counter()-start,solve_trace=traces,
                    warm_start_used=used_cache,optimizer='fast_fly_style',
                    objective_kind='elapsed_time_plus_thrust_smooth' if self.c.get('phase_thrust_smooth',0)>0 else 'elapsed_time',iterations=sum(row['iterations'] for row in traces),
                    cells_per_leg=self.c['cells_per_leg'])
        plan['warm_projection']=self.projection_trace
        if self.initial_force_weight:
            plan.update(initial_forces=np.asarray(self.initial_forces).copy(),initial_force_weight=self.initial_force_weight,
                        initial_force_cost=self.initial_force_weight*float(np.sum((plan['U'][:,0]-self.initial_forces)**2)),
                        objective_kind='elapsed_time_plus_thrust_smooth_and_initial_force')
        if 'reference_continuity_cost' in plan:plan['objective_kind']+=' and causal reference continuity'
        audit_start=time.perf_counter()
        if 'audited_initial_state' in plan and np.array_equal(plan['audited_initial_state'],state):
            audit,reference_time,reference_states=plan['audit'],plan['reference_time'],plan['reference_states']
        else:audit, reference_time, reference_states = self.audit(plan,state)
        plan.update(audit=audit,reference_time=reference_time,reference_states=reference_states,
                    audited_initial_state=np.asarray(state).copy(),audit_seconds=plan.get('audit_seconds',0.)+time.perf_counter()-audit_start,
                    start_time=self.current_time)
        quantum=float(self.c.get('phase_audit_retry_quantum',0.))
        if not audit['passed'] and used_cache and quantum>0.:
            retry_start=time.perf_counter()
            retry_guess,_=self.shifted_guess(previous,state,gates,normals,durations)
            scale=np.r_[np.tile(np.r_[np.full(3,5.),np.full(3,10.),np.ones(4),np.full(3,5.)],self.n+1),np.full(4*self.n,5.),np.ones(self.M)]
            retry_guess=np.clip(np.round(retry_guess/scale/quantum)*quantum*scale,self.lbx,self.ubx)
            retry_guess[:13]=state
            retry=dict(stage='audit_retry',quantum=quantum,iterations=0,success=False,original_failed_audit=audit)
            try:
                candidate=GatePlanner.solve(self,state,gates,normals,half,durations,warm=retry_guess)
                check_start=time.perf_counter();check,tt,xx=self.audit(candidate,state);check_seconds=time.perf_counter()-check_start
                retry.update(iterations=candidate['iterations'],audit=check,nlp_trace=candidate['solve_trace'],success=check['passed'])
                if check['passed']:
                    plan.update(candidate)
                    plan.update(audit=check,reference_time=tt,reference_states=xx,audit_seconds=plan['audit_seconds']+check_seconds,
                                audited_initial_state=np.asarray(state).copy(),start_time=self.current_time)
                    audit=check
                    if self.initial_force_weight:
                        plan['initial_force_cost']=self.initial_force_weight*float(np.sum((plan['U'][:,0]-self.initial_forces)**2))
            except RuntimeError as error:
                retry.update(error=str(error),nlp_trace=self.last_trace,iterations=sum(r['iterations'] for r in self.last_trace))
            retry['seconds']=time.perf_counter()-retry_start;traces.append(retry)
            plan.update(solve_trace=traces,iterations=sum(r['iterations'] for r in traces),audit_retry=retry)
        plan['seconds']=time.perf_counter()-start
        self.last_trace = traces
        if audit['passed']:
            if (self.c.get('phase_objective_period',0.)>0
                and plan.get('frontend_update')!='feasibility_projection'):
                self.last_objective_update[self.active_agent]=self.current_time
            self.cache[self.active_agent] = (self.event_index,plan)
            if self.c.get('phase_past_lap_warm',False):
                key=(self.active_agent,self.event_index%self.track_gate_count)
                records=self.past_lap_cache.setdefault(key,[])
                # Retain this and the preceding lap only; templates are local
                # online history and always undergo a new solve and full audit.
                records[:]=[(event,old) for event,old in records if event>=self.event_index-self.track_gate_count]
                records.append((self.event_index,{k:plan[k].copy() for k in ['X','U','durations']}|{'start_time':self.current_time}))
        return plan
