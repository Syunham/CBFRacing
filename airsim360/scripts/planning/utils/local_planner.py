"""Rolling physical-gate preview with learned aperture allocations."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

from scripts.simulation.course import *
from scripts.planning.utils.allocation import observation,load_allocator
from scripts.planning.cpc.phase_planner import GatePlanner
from scripts.planning.utils.reference import reference_from_plan
from scripts.planning.cpc.original_cpc import OriginalCPC
from scripts.planning.utils.timing import ArrivalTimeNetwork

class LocalPlanner:
    def __init__(self,checkpoint,seconds=35.,phase_retries=0):
        self.c=configuration();self.c.update(cpc_wall_time=seconds,cpc_max_iter=1600,cpc_epsilon_schedule=[1e-3,1e-5,1e-6])
        phase_optimizer=os.environ.get('PHASE_OPTIMIZER','fast-fly')
        self.fixed_time_sqp=os.environ.get('CPC_FRONTEND')=='phase' and phase_optimizer=='fixed-time-sqp'
        self.c['phase_solve_retries']=phase_retries
        self.c['phase_thrust_smooth']=float(os.getenv('FASTFLY_THRUST_SMOOTH','0'))
        self.c['phase_initial_force_weight']=float(os.getenv('PHASE_INITIAL_FORCE_WEIGHT','0'))
        self.c['phase_online_sqp']=os.getenv('PHASE_ONLINE_SQP','0')=='1'
        self.c['phase_sqp_audit_refine']=os.getenv('PHASE_SQP_AUDIT_REFINE','0')=='1'
        self.c['phase_sqp_constraint_generation']=os.getenv('PHASE_SQP_CONSTRAINT_GENERATION','0')=='1'
        self.c['phase_reference_continuity']=float(os.getenv('PHASE_REFERENCE_CONTINUITY','0'))
        if not np.isfinite(self.c['phase_reference_continuity']) or self.c['phase_reference_continuity']<0:raise ValueError('Reference continuity weight must be finite and nonnegative')
        self.c['duration_min']=float(os.getenv('PHASE_DURATION_MIN',str(self.c['duration_min'])))
        if not 0.<self.c['duration_min']<self.c['duration_max']:raise ValueError('Duration minimum must be positive and below the maximum')
        self.c['integration_substeps']=int(os.getenv('PHASE_INTEGRATION_SUBSTEPS',str(self.c['integration_substeps'])))
        if self.c['integration_substeps']<1:raise ValueError('Integration substeps must be positive')
        warm_mode=os.getenv('FASTFLY_WARM_MODE','shift')
        if warm_mode in ('tail','tail-project'):
            self.c.update(phase_tail_seed=True,phase_zero_shifted_dual=True)
        if warm_mode=='tail-project':self.c['phase_projection_steps']=3
        if os.getenv('PHASE_SQP_TAIL_SEED','0')=='1':self.c['phase_tail_seed']=True
        self.c['phase_sqp_gate_trust']=float(os.getenv('PHASE_SQP_GATE_TRUST','.5'))
        self.c['phase_periodic_projection']=os.getenv('PHASE_PERIODIC_PROJECTION','0')=='1'
        self.c['phase_objective_period']=float(os.getenv('PHASE_OBJECTIVE_PERIOD','0'))
        if not np.isfinite(self.c['phase_objective_period']) or self.c['phase_objective_period']<0:
            raise ValueError('Objective update period must be finite and nonnegative')
        if self.c['phase_objective_period']>0 and self.c['phase_periodic_projection']:
            raise ValueError('Choose objective cadence or gate-only objective updates')
        self.c['phase_gate_tail_nlp']=os.getenv('PHASE_GATE_TAIL_NLP','0')=='1'
        self.c['phase_sqp_dual_warm']=os.getenv('PHASE_SQP_DUAL_WARM','0')=='1'
        self.c['phase_sqp_chain_condensing']=os.getenv('PHASE_SQP_CHAIN_CONDENSING','0')=='1'
        self.c['phase_consistent_tail_boundary']=os.getenv('PHASE_CONSISTENT_TAIL_BOUNDARY','0')=='1'
        self.c['phase_audit_retry_quantum']=float(os.getenv('PHASE_AUDIT_RETRY_QUANTUM','0'))
        if not np.isfinite(self.c['phase_audit_retry_quantum']) or self.c['phase_audit_retry_quantum']<0:
            raise ValueError('Audit retry quantum must be finite and nonnegative')
        self.c['phase_sqp_backend']=os.getenv('PHASE_SQP_BACKEND','daqp')
        self.c['phase_optimize_tail']=os.getenv('PHASE_OPTIMIZE_TAIL','0')=='1'
        self.c['phase_past_lap_warm']=os.getenv('PHASE_PAST_LAP_WARM','0')=='1'
        self.c['phase_learned_initializer']=os.getenv('PHASE_LEARNED_INITIALIZER','')
        self.c['phase_learned_budget']=float(os.getenv('PHASE_LEARNED_BUDGET','3'))
        self.c['phase_learned_sqp']=os.getenv('PHASE_LEARNED_SQP','0')=='1'
        self.c['phase_tail_solve_seconds']=float(os.getenv('PHASE_TAIL_SOLVE_SECONDS','.3'))
        if os.getenv('FASTFLY_PERIODIC','0')=='1':
            self.c.update(phase_time_shift=True,phase_remaining_min=float(os.getenv('PHASE_REMAINING_MIN','.03')),
                          phase_adaptive_first_buffer=os.getenv('PHASE_ADAPTIVE_FIRST_BUFFER','0')=='1')
            self.c['phase_average_shifted_controls']=os.getenv('PHASE_AVERAGE_SHIFTED_CONTROLS','0')=='1'
            self.c['phase_adaptive_remaining_min']=os.getenv('PHASE_ADAPTIVE_REMAINING_MIN','0')=='1'
        if not self.fixed_time_sqp and os.getenv('FASTFLY_NATIVE','0')=='1':
            self.c.update(phase_native_library=str(HERE/'build/fastfly_native_o0.dll'),
                          phase_cell_function=True,expand_solver_graph=False,phase_batched_audit=True)
            if os.getenv('FASTFLY_BLOCK_HESSIAN','0')=='1':
                self.c['phase_hessian_library']=str(HERE/'build/fastfly_block_hessian.dll')
            if os.getenv('FASTFLY_BLOCK_JACOBIAN','0')=='1':
                self.c['phase_jacobian_library']=str(HERE/'build/fastfly_block_jacobian.dll')
            if os.getenv('FASTFLY_NATIVE_DIR'):
                library=Path(os.environ['FASTFLY_NATIVE_DIR']).resolve()/'fastfly_native.dll'
                if not library.is_file():raise FileNotFoundError(library)
                self.c.update(phase_native_library=str(library),phase_hessian_library=str(library),
                              phase_jacobian_library=str(library))
        if not self.fixed_time_sqp and os.getenv('FASTFLY_COMPACT_DERIVATIVES','0')=='1':
            if os.getenv('FASTFLY_NATIVE','0')!='1' or os.getenv('FASTFLY_NATIVE_DIR'):
                raise ValueError('Compact derivatives require the original native five-gate mesh')
            directory=HERE/'build/compact_derivatives'
            for key,filename in [('phase_jacobian_library','compact_jacobian.dll'),('phase_hessian_library','compact_hessian.dll')]:
                path=directory/filename
                if not path.is_file():raise FileNotFoundError(path)
                self.c[key]=str(path)
            self.c['phase_compact_derivatives']=True
        self.course=course();self.net=load_allocator(checkpoint)
        self.event_net=None;self.event_checkpoint=os.environ.get('EVENT_CHECKPOINT','')
        if self.event_checkpoint and not self.fixed_time_sqp:
            from scripts.planning.utils.event_supervised import load_event
            self.event_net,self.event_metadata=load_event(self.event_checkpoint)
        self.minimum_snap=os.environ.get('MINIMUM_SNAP','0')=='1'
        self.scp_enabled=False
        self.rate_backend=''
        if self.minimum_snap:
            from scripts.planning.minsnap.minimum_snap import MinimumSnapPlanner
            self.phase=MinimumSnapPlanner(self.c,self.course)
        elif os.environ.get('CPC_FRONTEND')=='phase' and phase_optimizer in ('fast-fly','fixed-time-sqp'):
            from scripts.planning.fast_fly.fast_fly import FastFlyPlanner
            self.c['phase_cells_per_leg']=int(os.environ.get('PHASE_CELLS_PER_LEG','12'))
            phase_class = getattr(self, "phase_class", FastFlyPlanner)
            self.phase=phase_class(self.c.copy() if self.fixed_time_sqp else self.c,self.course,self.event_net is None)
            if self.fixed_time_sqp:
                from scripts.planning.fast_fly.fixed_time_frontend import FixedTimeFrontend
                self.launch_phase=self.phase
                self.phase=FixedTimeFrontend(self.c,self.course,os.getenv('FIXED_TIME_SQP_CHECKPOINT') or None,
                                             timing_recovery=os.getenv('FIXED_TIME_SQP_TIMING_RECOVERY')=='1')
                self.event_checkpoint=str(self.phase.network.checkpoint)
        else:
            self.phase=GatePlanner(self.c,self.course,self.event_net is None)
        self.full=None;self.previous_plans={};self.passage_memory={}
        self.timing=ArrivalTimeNetwork(torch.load(HERE/'models/time_initializer.pt',map_location='cpu',weights_only=False));self.timing.eval()
    def solve(self,own,other,index):
        self.last_failed_plan=None
        launching=self.fixed_time_sqp and index==0
        use_fixed_time=self.fixed_time_sqp and not launching
        phase_solver=self.launch_phase if launching else self.phase
        self.active_phase=phase_solver
        start=time.perf_counter();M=self.c.get('preview_gates',2);g,n,half,a=gate_pair(index,M);targets=g.copy();alloc=[];raw_alloc=[]
        memory_key=getattr(self,'active_agent',0);memory=self.passage_memory.get(memory_key,{})
        update_limit=float(os.getenv('OPT_PASSAGE_UPDATE_LIMIT','0'))
        for j in range(M):
            g2,n2,_,_=gate_pair(index+j+1)
            if other is not None:
                z=observation(self.allocation_own(memory_key) if getattr(self,'allocation_own',None) else own,other,g[j],n[j],a[j],g2[0],n2[0])
                with torch.no_grad():offset=float(self.net(torch.tensor(z[None]))) if self.net is not None else 0.
                offset*=float(os.environ.get('OPT_POINT_SCALE','1'))
            else:
                decision = self.visual_passage(memory_key, g[j], n[j], a[j])
                offset = decision
                if np.ndim(decision):
                    offset, vertical = decision
                    targets[j, 2] += vertical
            raw_alloc.append(offset)
            if update_limit>0 and index+j in memory:
                offset=memory[index+j]+float(np.clip(offset-memory[index+j],-update_limit,update_limit))
            targets[j]+=offset*a[j];alloc.append(offset)
        # Existing physical gate events are constrained around the predicted
        # passage point. The original aperture is retained for final scoring.
        local_half=np.full((M,2),float(os.environ.get('OPT_PASSAGE_HALF','.12'))) if self.net is not None else half
        if self.net is not None and os.getenv('OPT_PASSAGE_VERTICAL_HALF'):
            ids=[int(v) for v in os.environ['OPT_VERTICAL_GATE_IDS'].split(',')] if os.getenv('OPT_VERTICAL_GATE_IDS') else list(range(7))
            for j in range(M):
                if (index+j)%7 in ids:local_half[j,1]=float(os.environ['OPT_PASSAGE_VERTICAL_HALF'])
        if self.net is not None:
            # Retain the 5 cm physical frame reserve for every allowed point.
            local_half=np.minimum(local_half,.70-np.c_[np.abs(alloc),np.abs(targets[:,2]-g[:,2])])
        if np.any(local_half<=self.c['aperture_reserve']):raise ValueError('Passage box must retain positive aperture after reserve')
        if os.getenv('PHASE_FIRST_GATE_APERTURE','0')=='1':
            # Match the current physical opening available to MPCC. Only the
            # artificial learned-point box is removed for this imminent gate.
            targets[0]=g[0]
            local_half[0]=half[0]-.05
        network_start=time.perf_counter()
        if use_fixed_time:
            self.last_seed_diagnostics={}
            window=phase_solver.window(own,index,targets,local_half,getattr(self,'current_time',0.))
            targets,local_half,durations=window.gates,window.half,window.durations
            arrival=np.cumsum(durations)
        elif getattr(self,'snap_time_network',None) is not None:
            durations=self.snap_time_network.predict(np.asarray(own),targets,n,local_half)
            arrival=np.cumsum(durations)
        elif self.event_net is not None:
            from scripts.planning.utils.event_supervised import predict
            event_half=np.full_like(local_half,float(os.environ['OPT_EVENT_NOMINAL_HALF'])) if os.getenv('OPT_EVENT_NOMINAL_HALF') else local_half
            durations=predict(self.event_net,np.asarray(own),targets,n,event_half)
            arrival=np.cumsum(durations)
        else:
            with torch.no_grad():arrival=self.timing(*[torch.tensor(v,dtype=torch.float64) for v in [own[:13],targets[:2],n[:2],local_half[:2]]]).numpy()
            durations=np.clip(np.diff(np.r_[0,arrival]),.5,7.)
            if M>2:durations=np.r_[durations,np.clip(np.linalg.norm(np.diff(g[1:],axis=0),axis=1)/8.,.5,7.)]
        event_network_seconds=time.perf_counter()-network_start
        if self.minimum_snap:
            plan=self.phase.solve(np.asarray(own),targets,n,local_half,durations)
            if update_limit>0:self.passage_memory[memory_key]={index+j:alloc[j] for j in range(M)}
            plan['allocation_raw']=np.array(raw_alloc)
            plan.update(allocation=alloc,physical_gates=g,half=local_half,network_arrival_time_prior=arrival,
                wall_seconds=time.perf_counter()-start,extra_waypoints=0,gate_event_count=M,
                seed_diagnostics={'fallback_used':False,'polynomial_seed':not bool(plan.get('kinematics_only',False)),
                    'kinematic_reference_only':bool(plan.get('kinematics_only',False))},phase_solve_trace=plan['solve_trace'],
                event_network_seconds=event_network_seconds,event_policy='synthetic_time_network_fixed_qp' if getattr(self,'snap_time_network',None) is not None else ('network_fixed_polynomial' if self.event_net is not None else 'two_gate_time_prior_then_distance_over_8'),event_durations_input=durations.copy(),
                event_checkpoint=self.event_checkpoint,frontend_solver='clarabel_minimum_snap')
            if self.event_net is not None:plan['event_feature_half']=event_half.copy()
            return plan
        key=getattr(self,'active_agent',0)
        if use_fixed_time:
            phase=phase_solver.solve(window,key)
            self.last_seed_diagnostics.update(fallback_used=phase['frontend_update'] in ('ipopt_fallback','timing_recovery'),
                                               method=phase['frontend_update'])
            event_policy=phase_solver.event_policy
        else:
            warm=phase_solver.seed(np.asarray(own[:13]),targets,n,durations)
            self.last_seed_diagnostics=dict(original_nonfinite=int((~np.isfinite(warm)).sum()),fallback_used=False)
            if not np.isfinite(warm).all():
                from scripts.planning.fast_fly.warm_seed import finite_seed
                warm=finite_seed(np.asarray(own[:13]),targets,n,durations,phase_solver.c)
                self.last_seed_diagnostics['fallback_used']=True
            phase_solver.active_agent=key
            phase_solver.event_index=index
            phase_solver.current_time=getattr(self,'current_time',0.)
            phase_solver.initial_forces=np.asarray(own[13:17],dtype=float).copy()
            phase=phase_solver.solve(np.asarray(own[:13]),targets,n,local_half,durations,warm=warm)
            if launching:
                event_policy='free_time_launch'
            else:
                event_policy=('network_initialized_free' if phase_solver.optimize_durations else 'network_fixed') if self.event_net is not None else 'numerical_free'
        self.last_seed_diagnostics['phase_warm_start_used']=phase.get('warm_start_used',False)
        if os.environ.get('CPC_FRONTEND','original')=='phase':
            plan=phase
        else:
            if self.full is None:self.full=OriginalCPC(self.c,self.course)
            plan=self.full.solve(np.asarray(own[:13]),targets,n,local_half,phase=phase)
        if 'audited_initial_state' in plan and np.array_equal(plan['audited_initial_state'],np.asarray(own[:13])):
            audit,t,x=plan['audit'],plan['reference_time'],plan['reference_states']
        else:
            audit,t,x=phase_solver.audit(plan,np.asarray(own[:13]))
        if not audit['passed']:
            self.last_failed_plan=dict(plan,allocation=np.asarray(alloc),initial_actual=np.asarray(own).copy(),
                initial_other=np.asarray(other).copy(),event_durations_input=durations.copy())
            raise RuntimeError('CPC reference audit: '+json.dumps(audit))
        self.previous_plans[key]=(index,plan)
        plan.update(audit=audit,reference_time=t,reference_states=x,allocation=alloc,physical_gates=g,half=local_half,
            network_arrival_time_prior=arrival,wall_seconds=time.perf_counter()-start,extra_waypoints=0,gate_event_count=M,seed_diagnostics=self.last_seed_diagnostics,phase_solve_trace=phase.get('solve_trace',[]),
            event_network_seconds=event_network_seconds,event_policy=event_policy,event_durations_input=durations.copy(),event_checkpoint='' if launching else self.event_checkpoint,frontend_solver=getattr(phase_solver,'solver_name','ipopt'))
        if launching:
            self.phase.remember_launch(key,index,own,plan,phase_solver.current_time)
        return plan

def reference(plan,initial):
    ref=reference_from_plan(plan,initial)
    ref['physical_next_gate']=plan['physical_gates'][0]
    return ref

def save(path,plan):
    arrays={k:v for k,v in plan.items() if isinstance(v,np.ndarray)}
    arrays['allocation']=np.array(plan['allocation']);np.savez_compressed(path,**arrays)
