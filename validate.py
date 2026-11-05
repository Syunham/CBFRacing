"""Full two-lap closed-loop evaluation, causal neighbor prediction, no waiting."""
from cbfracing.core.context import *
from cbfracing.planning.planning import LocalPlanner,reference,save
from cbfracing.control.controller import RecoveryMPCC,DT,FEEDBACK
from cbfracing.planning.reference import online_data
from cbfracing.core.cli_profiles import resolve_safety
import argparse,shutil

def main(args):
    from cbfracing.core.runtime import configure
    configure(args)
    output=HERE/'outputs'/args.name;output.mkdir(parents=True,exist_ok=False);(output/'plans').mkdir();(output/'source').mkdir()
    for p in [*HERE.glob('*.py'), *(HERE/'cbfracing').rglob('*.py')]:
        target=output/'source'/p.relative_to(HERE)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    os.environ['GATE_PREDICTION_MARGIN']=str(args.gate_prediction_margin)
    os.environ['EXACT_GATE_APERTURE']='1' if args.exact_gate_aperture else '0'
    os.environ['NEXT_GATE_GUARD']='1' if args.next_gate_guard else '0'
    os.environ['VELOCITY_CONTOUR_WEIGHT']=str(args.velocity_contour_weight)
    os.environ['REFERENCE_SPEED_WEIGHT']=str(args.reference_speed_weight)
    os.environ['CPC_PREVIEW_GATES']=str(args.preview_gates);os.environ['CPC_FRONTEND']=args.frontend;os.environ['PREVIEW_SPLINE']=args.preview_spline;os.environ['PHYSICAL_GATE_GUARD']='1' if args.physical_gate_guard else '0';c=course();save_json(output/'course.json',c);cfg=dict(vars(args),gates=c['centers'],dynamics=MODEL,
        dt=DT,feedback=FEEDBACK,physical_speed_limit=None,manual_waiting=False,manual_priority=False,
        actuator_limits=f'CPC and MPCC0..{MODEL["thrust_max"]}N/rotor; MPC body-rate norm{MODEL["omega_max"]}rad/s',body_rate_norm_limit=MODEL['omega_max'],future_trajectory_exchange=False,
        target_events_per_agent=args.target_gates or 7*args.laps,cbf=True,clearance=.5,clearance_margin=.05,cbf_radius=np.inf,
        qwp=args.gate_weight,qnom=args.contour_weight,sigma=.6,cbf_rate=3.,gate_funnel=True,gate_slope=2.,gate_progress_slope=args.gate_progress_slope)
    cfg['experimental_acceleration']={k:os.environ.get(k,'') for k in ['EXPERIMENT_QP','EXPERIMENT_CONDENSE','FAST_KERNELS','FAST_DENSE_QP','CONDENSE_METHOD']}
    cfg['gate_crossing_form']=os.environ.get('GATE_CROSSING_FORM','rational')
    cfg['passage_halfwidth_requested']=float(os.environ.get('OPT_PASSAGE_HALF','.12'))
    cfg['passage_vertical_halfwidth_requested']=float(os.getenv('OPT_PASSAGE_VERTICAL_HALF',os.getenv('OPT_PASSAGE_HALF','.12')))
    cfg['event_nominal_halfwidth']=os.getenv('OPT_EVENT_NOMINAL_HALF')
    cfg['vertical_gate_ids_zero_based']=os.getenv('OPT_VERTICAL_GATE_IDS')
    cfg['native_kernel_stem']=os.environ.get('FAST_KERNEL_STEM','fast_mx')
    cfg['frontend_native']=os.environ.get('DIRECT_NATIVE','0')=='1'
    cfg['frontend_drop_redundant_quaternion']=os.environ.get('DIRECT_DROP_REDUNDANT_QUATERNION','0')=='1'
    cfg['frontend_integration_substeps']=int(os.environ.get('DIRECT_INTEGRATION_SUBSTEPS','4'))
    cfg['event_policy']=('network_initialized_free' if os.environ.get('SCP_FREE_TIME')=='1' else 'network_fixed') if os.environ.get('EVENT_CHECKPOINT') else 'numerical_free'
    cfg['event_checkpoint']=os.environ.get('EVENT_CHECKPOINT','')
    cfg['scp_settings']={k:v for k,v in os.environ.items() if k.startswith('SCP_') or k=='FRONTEND_SCP'}
    cfg['frontend_solver']='scp' if os.environ.get('FRONTEND_SCP')=='1' else 'ipopt'
    cfg['rti_cold_start']=os.environ.get('OPT_RTI_COLD_START','0')=='1'
    cfg['hover_initial_seed']=os.environ.get('OPT_HOVER_INITIAL_SEED','0')=='1'
    cfg['local_gate_window']=os.environ.get('OPT_LOCAL_GATE_WINDOW','0')=='1'
    cfg['gate_crossing_form']=os.getenv('GATE_CROSSING_FORM','rational')
    cfg['gate_margin_release_steps']=int(os.getenv('OPT_GATE_MARGIN_RELEASE_STEPS','0'))
    cfg['deferred_plan_artifacts']=os.getenv('OPT_DEFER_PLAN_WRITES','0')=='1'
    cfg['certified_feasible_backup']=os.getenv('OPT_RTI_CERTIFIED_BACKUP','0')=='1'
    cfg['fast_kernel_stem']=os.getenv('FAST_KERNEL_STEM','fast_mx' if os.getenv('FAST_KERNELS')=='mx' else 'fast')
    cfg['same_qp_sparse_retry']=os.environ.get('OPT_QP_RETRY_CLARABEL','0')=='1'
    cfg['full_horizon_sqp_recovery']=os.environ.get('OPT_SQP_RECOVERY','0')=='1'
    cfg['elastic_recovery_hard_prefix']=int(os.getenv('OPT_SQP_RECOVERY_PREFIX','0'))
    cfg['cbf_decay_margin_rate']=float(os.getenv('OPT_CBF_DECAY_MARGIN_RATE','0'))
    if os.environ.get('MINIMUM_SNAP')=='1':
        cfg.update(frontend_solver='clarabel_minimum_snap',event_policy='network_fixed_polynomial',polynomial_settings={k:v for k,v in os.environ.items() if k.startswith('SNAP_')})
    cfg['body_rate_frontend']=os.environ.get('BODY_RATE_FRONTEND','')
    cfg['rate_envelope']=os.environ.get('RATE_ENVELOPE','nodes')
    cfg['nodal_thrust_only']=os.environ.get('NODAL_THRUST_ONLY','0')=='1'
    cfg['start_mode']=c['start_mode']
    cfg['rate_omega_reserve']=float(os.environ.get('RATE_OMEGA_RESERVE','0'))
    cfg['direct_input_model']=os.environ.get('DIRECT_INPUT_MODEL','0')=='1'
    cfg['direct_rate_smooth']=float(os.environ.get('DIRECT_RATE_SMOOTH','0'))
    cfg['direct_eta_max']=float(os.environ.get('DIRECT_ETA_MAX','1'))
    cfg['direct_omega_max']=float(os.environ.get('DIRECT_OMEGA_MAX','10'))
    cfg['point_offset_scale']=float(os.environ.get('OPT_POINT_SCALE','1'))
    cfg['curvature_reward_cap']=float(os.environ.get('OPT_TURN_REWARD','0'))
    cfg['curvature_reward_decay']=float(os.environ.get('OPT_TURN_CURVATURE_DECAY','0'))
    cfg['curvature_preview_distance']=float(os.environ.get('OPT_TURN_PREVIEW_DISTANCE','8'))
    cfg['short_segment_time_scale']=float(os.getenv('SNAP_SHORT_TIME_SCALE','1'))
    cfg['snap_path_energy_weight']=float(os.getenv('SNAP_PATH_ENERGY_WEIGHT','0'))
    cfg['hard_tracking_funnel']=os.getenv('OPT_TRACKING_FUNNEL','1')=='1'
    cfg['gate_dense_prefix']=int(os.getenv('GATE_DENSE_PREFIX','0'))
    if cfg['curvature_preview_distance']<=0:raise ValueError('Curvature preview distance must be positive')
    cfg['carry_replan_warmstart']=os.environ.get('OPT_REPLAN_CARRY','0')=='1'
    cfg['project_replan_progress']=os.environ.get('OPT_PROJECT_REPLAN_PROGRESS','0')=='1'
    cfg['reference_repair_only']=os.environ.get('OPT_REFERENCE_REPAIR_ONLY','0')=='1'
    cfg['shifted_time_warmstart']=os.environ.get('OPT_SHIFTED_TIME_WARM','0')=='1'
    cfg['local_warm_extension']=os.environ.get('OPT_EXTEND_WARM','0')=='1'
    save_json(output/'config.json',cfg)
    save_json(output/'planner_config.json',configuration())
    cfg['cpc_mode']='original' if args.frontend=='original' else ('minimum_snap_network_durations' if args.frontend=='minimum-snap' else ('fixed_phase_events_network_durations' if os.environ.get('EVENT_CHECKPOINT') else 'fixed_phase_events_free_durations'))
    save_json(output/'config.json',cfg)
    cfg['initial_progress_slope']=float(os.getenv('OPT_INITIAL_PROGRESS_SLOPE','-1'))
    cfg['geometry_funnel_release']=os.getenv('OPT_GEOMETRY_RELEASE','0')=='1'
    cfg['post_gate_funnel_slope']=float(os.getenv('POST_GATE_FUNNEL_SLOPE','0'))
    cfg['passage_update_limit']=float(os.getenv('OPT_PASSAGE_UPDATE_LIMIT','0'))
    cfg['neighbor_forecast_model']=os.getenv('OPT_NEIGHBOR_FORECAST','ca')
    if cfg['neighbor_forecast_model'] not in ('ca','turn','turn_tail'):raise ValueError('Unknown neighbor forecast model')
    states=initial_states(args.separation,args.forward_offset);ts=[0.];history=[states.copy()];logs=[];front=[];events=[[],[]];counts=[0,0];refs=[None,None]
    build_tick=time.perf_counter();planner=LocalPlanner(args.allocator,args.cpc_seconds,args.phase_retries)
    cfg['planner_construction_seconds']=time.perf_counter()-build_tick;plan_artifacts=[]
    solver_class=RecoveryMPCC
    if args.mpcc_backend=='rti':
        from cbfracing.control.online_rti import OnlineRTI
        os.environ['RTI_UPDATES']=str(args.rti_iterations);os.environ['RTI_LM']=str(args.rti_regularization);os.environ['RTI_GLOBALIZE']='1' if args.rti_globalize else '0'
        os.environ['RTI_FEASIBILITY']='1' if args.rti_feasibility_correction else '0'
        os.environ['RTI_TRUST']=str(args.rti_trust_radius)
        os.environ['RTI_FIRST']='1' if args.rti_feasible_first else '0';os.environ['RTI_ELASTIC']='1' if args.rti_elastic_restore else '0';os.environ['RTI_CONDENSE']='1' if args.rti_condense else '0'
        os.environ['RTI_ACCEPTANCE']=args.rti_acceptance
        os.environ['RTI_NATIVE']='1' if args.rti_native else '0'
        os.environ['RTI_QP_TOL']=str(args.rti_qp_tolerance)
        os.environ['RTI_SHOOTING_GUARD']='1' if args.rti_shooting_guard else '0'
        os.environ['CBF_HARD_STEPS']=str(args.cbf_hard_steps);os.environ['CBF_TAIL_WEIGHT']=str(args.cbf_tail_weight)
        os.environ['RTI_DENSE_CBF']='1' if args.rti_dense_cbf else '0'
        os.environ['RTI_ELASTIC_PREFIX']=str(args.rti_elastic_prefix)
        os.environ['RTI_FORECAST_FILTER']='1' if args.rti_forecast_filter else '0'
        os.environ['RTI_OBJECTIVE_UPDATES']=str(args.rti_objective_updates)
        os.environ['RTI_NLP_RECOVERY_SECONDS']=str(args.rti_nlp_recovery_seconds)
        os.environ['RTI_BATCHED_AUDIT']='1' if args.rti_batched_audit else '0'
        solver_class=OnlineRTI
    build_tick=time.perf_counter()
    solvers=[solver_class(i,n=args.horizon,barrier=args.barrier,penalty=args.penalty,max_iter=args.iterations,wall_time=args.solver_seconds,
        execution_guard=True,cbf_checkpoint=args.h_checkpoint,cbf_mode=args.cbf_mode,cbf_alpha=args.cbf_alpha,
        tube_recovery=args.tube_recovery,dense_physics=args.dense_physics,gate_guard=cfg['local_gate_window']) for i in range(2)]
    cfg['controller_construction_seconds']=time.perf_counter()-build_tick
    cfg['loaded_compiled_kernels']=[getattr(s,'fast_kernels',False) for s in solvers]
    cfg['kernel_loading_notes']=[getattr(s,'kernel_loading_note','IPOPT') for s in solvers]
    if hasattr(solvers[0],'row_kinds'):
        from collections import Counter
        cfg['constraint_groups']=dict(Counter(solvers[0].row_kinds))
        if not cfg['hard_tracking_funnel']:
            assert cfg['constraint_groups'].get('tracking',0)==0
            assert cfg['constraint_groups'].get('gate_crossing',0)>0
            assert cfg['constraint_groups'].get('cbf',0)>0
    save_json(output/'config.json',cfg)
    _,step=model_functions(8);_,warm_step=model_functions(2);acc=np.zeros((2,3));previous_velocity=states[:,3:6].copy();reason=None;target=args.target_gates or 7*args.laps
    if len(history)>1:acc=np.clip((history[-1][:,3:6]-history[-2][:,3:6])/FEEDBACK,-35,35)
    def reward_now(i):
        fraction=1. if args.progress_start is None else float(np.clip((ts[-1]-args.progress_ramp_delay)/args.progress_ramp_seconds,0.,1.))
        requested=args.progress
        turn_cap=float(os.environ.get('OPT_TURN_REWARD','0'))
        if turn_cap>0 and refs[i] is not None:
            ref=refs[i];s=float(np.clip(states[i,17],0.,ref['length']))
            query=np.linspace(s,min(ref['length'],s+cfg['curvature_preview_distance']),25)
            d1=ref['spline'](query,1);d2=ref['spline'](query,2)
            curvature=np.linalg.norm(np.cross(d1,d2),axis=1)/np.maximum(np.linalg.norm(d1,axis=1)**3,1e-8)
            turn=float(np.clip((curvature.max()-.05)/.15,0.,1.))
            decay=float(os.environ.get('OPT_TURN_CURVATURE_DECAY','0'))
            turn_cap=max(args.reward_floor,turn_cap/(1+decay*max(float(curvature.max())-.2,0.)))
            requested=(1-turn)*args.progress+turn*min(args.progress,turn_cap)
        if args.reward_policy=='reference-speed' and refs[i] is not None:
            ref=refs[i];s=float(np.clip(states[i,17],0.,ref['length']))
            query=np.linspace(s,min(ref['length'],s+25.),101)
            vref=np.interp(query,ref['speed_arc'],ref['speed_values'])+args.reward_speed_allowance
            # Braking preview changes a cost coefficient, never a speed bound.
            desired=float(np.sqrt(np.min(vref**2+2*args.reward_braking*(query-s))))
            error=(desired-np.linalg.norm(states[i,3:6]))/args.reward_transition
            blend=float(1/(1+np.exp(-np.clip(error,-30.,30.))))
            requested=args.reward_floor+(requested-args.reward_floor)*blend
        return requested if args.progress_start is None else args.progress_start+(requested-args.progress_start)*fraction
    def install_data(solver,od,ref):
        solver.data(od)
        if args.reference_speed_weight>0:
            arc,unique=np.unique(ref['speed_arc'],return_index=True);speed=ref['speed_values'][unique]
            slope=np.gradient(speed,arc)
            solver.od[:,55]=np.interp(od[:,12],arc,speed)+args.reference_speed_extra
            solver.od[:,56]=np.interp(od[:,12],arc,slope)
    def prepare(i):
        begin=time.perf_counter();row=dict(agent=i,time=ts[-1],gate_index=counts[i])
        carry=solvers[i].prediction() if cfg['carry_replan_warmstart'] and counts[i]>0 else None
        try:
            planner.active_agent=i
            plan=planner.solve(states[i],states[1-i],counts[i]);refs[i]=reference(plan,states[i])
            if cfg['geometry_funnel_release']:
                delta=states[i,:3]-plan['gates'][0];normal=plan['normals'][0]
                plane=abs(float(delta@normal));side=np.cross([0.,0.,1.],normal)
                needed=max(abs(float(delta@side)),abs(float(delta[2])))+.15
                gate_s=refs[i]['first_gate_s'];plane_width=cfg['gate_slope']*np.sqrt(plane**2+1e-6)
                # Include the actual replan state in both tracking-funnel
                # expressions. Width contracts as physical and virtual gate
                # distances shrink; physical gate crossing remains unchanged.
                base_requirement=(needed-args.gate_halfwidth-plane_width)/np.sqrt(gate_s**2+1e-6)
                release=.5*(gate_s-.4+np.sqrt((gate_s-.4)**2+1e-6))+.5*(-gate_s-.4+np.sqrt((-gate_s-.4)**2+1e-6))
                cap_requirement=(needed-(.70-args.gate_prediction_margin)-plane_width)/max(release,1e-6)
                slope=max(args.gate_progress_slope,base_requirement,cap_requirement)
                refs[i]['geometry_progress_slope']=slope
                row['geometry_progress_slope']=slope
                row['geometry_release_active']=bool(slope>args.gate_progress_slope)
                row['geometry_initial_required_width']=needed
            path=output/'plans'/f'{len(front):03d}_agent{i}_event{counts[i]}.npz'
            if cfg['deferred_plan_artifacts']:
                snapshot={k:v.copy() for k,v in plan.items() if isinstance(v,np.ndarray)};snapshot['allocation']=np.array(plan['allocation'])
                plan_artifacts.append((path,snapshot));row['plan_artifact_save_seconds']=0.;row['plan_artifact_deferred']=True
            else:
                save_tick=time.perf_counter();save(path,plan);row['plan_artifact_save_seconds']=time.perf_counter()-save_tick
            grid=np.arange(args.horizon+1)*DT;tt=plan['reference_time'];xx=plan['reference_states'];X=np.tile(states[i],(args.horizon+1,1))
            kinematic_only=bool(plan.get('kinematics_only',False))
            if kinematic_only:
                assert carry is not None or (cfg['hover_initial_seed'] and ts[-1]==0.),'A physical warm start is required with kinematic-only references'
                row['kinematic_reference_only']=True
            columns=6 if kinematic_only else 13
            X[:,:columns]=np.array([np.interp(grid,tt,xx[:,j]) for j in range(columns)]).T
            extra=np.maximum(grid-tt[-1],0.);X[:,:3]+=extra[:,None]*xx[-1,3:6]
            X[:,6:10]/=np.linalg.norm(X[:,6:10],axis=1)[:,None]
            if not kinematic_only:
                cells=plan.get('cell_durations',np.repeat(plan['durations']/planner.c['cells_per_leg'],planner.c['cells_per_leg']))
                ii=np.clip(np.searchsorted(np.cumsum(cells),grid,side='right'),0,plan['U'].shape[1]-1);X[:,13:17]=plan['U'][:,ii].T
                if 'reference_forces' in plan:X[:,13:17]=np.array([np.interp(grid,tt,plan['reference_forces'][:,j]) for j in range(4)]).T
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xx[:,:3],axis=0),axis=1))];X[:,17]=np.interp(grid,tt,arc);X[:,18]=np.clip(np.linalg.norm(X[:,3:6],axis=1),0.,30.)
            refs[i]['speed_arc']=arc;refs[i]['speed_values']=np.linalg.norm(xx[:,3:6],axis=1)
            states[i,17]=0.;states[i,18]=np.linalg.norm(states[i,3:6]);X[0]=states[i]
            U=np.c_[np.clip(np.diff(X[:,13:17],axis=0)/DT,-200,200),np.clip(np.diff(X[:,18])/DT,-40,40)]
            if carry is not None:
                oldX,oldU=carry
                X[:,:17]=oldX[:,:17];X[0,:17]=states[i,:17]
                X[:,17]=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(X[:,:3],axis=0),axis=1))]
                X[:,18]=np.clip(np.linalg.norm(X[:,3:6],axis=1),0.,30.)
                if cfg['project_replan_progress']:
                    # Map the carried physical prediction to the new geometric
                    # reference; this changes only the numerical warm start.
                    sample_s=np.arange(0.,refs[i]['length']+.001,.025)
                    sample_p=refs[i]['spline'](sample_s);projected=np.zeros(len(X))
                    for k in range(1,len(X)):
                        reach=max(.5,np.linalg.norm(X[k,:3]-X[k-1,:3])+1.)
                        candidates=np.flatnonzero((sample_s>=projected[k-1])&(sample_s<=projected[k-1]+reach))
                        if not len(candidates):projected[k]=projected[k-1];continue
                        best=candidates[np.argmin(np.linalg.norm(sample_p[candidates]-X[k,:3],axis=1))]
                        projected[k]=sample_s[best]
                    X[:,17]=projected;X[:,18]=np.clip(np.gradient(projected,grid),0.,30.);X[0,18]=states[i,18]
                    row['projected_progress_warmstart']=True
                U[:,:4]=oldU[:,:4];U[:,4]=np.clip(np.diff(X[:,18])/DT,-40,40)
                row['physical_warmstart_carried']=True
            if cfg['hover_initial_seed'] and counts[i]==0 and ts[-1]==0.:
                # Feasible stationary guess only at the prescribed resting start.
                # It supplies no executed waiting policy and no later-state reset.
                assert np.linalg.norm(states[i,3:6])<1e-9 and np.linalg.norm(states[i,10:13])<1e-9
                assert np.linalg.norm(states[i,7:9])<1e-9
                assert np.max(abs(states[i,13:17]-MODEL['mass']*MODEL['gravity']/4))<1e-9
                X=np.tile(states[i],(args.horizon+1,1));X[:,17:19]=0.;U=np.zeros((args.horizon,5))
                row['hover_initial_seed']=True
            solvers[i].seed(X,U);solvers[i].current(states[i]);solvers[i].weights(mu=reward_now(i),q_l=args.lag_weight,df=args.thrust_rate_weight,at=args.progress_acceleration_weight)
            row.update(success=True,allocation=plan['allocation'],duration=plan['durations'],audit=plan['audit'],cpc_seconds=plan['wall_seconds'],seed_diagnostics=plan['seed_diagnostics'],phase_solve_trace=plan.get('phase_solve_trace',[]))
            row.update(event_network_seconds=plan.get('event_network_seconds',0.),event_policy=plan.get('event_policy'),event_durations_input=plan.get('event_durations_input'))
        except Exception as e:row.update(success=False,error=str(e),seed_diagnostics=getattr(planner,'last_seed_diagnostics',None),phase_solve_trace=getattr(planner.phase,'last_trace',None))
        row['seconds']=time.perf_counter()-begin
        row['compute_seconds_excluding_plan_save']=row['seconds']-row.get('plan_artifact_save_seconds',0.)
        front.append(row);save_json(output/'frontends.json',front);print('front',row,flush=True)
        return row['success']
    for i in range(2):
        if not prepare(i):reason='CPC reference failure';break
    for k in range(int((args.max_time-ts[-1])/FEEDBACK)):
        if reason or min(counts)>=target:break
        snap=states.copy();chosen=[];roundlog=[]
        if k:
            raw=(snap[:,3:6]-previous_velocity)/FEEDBACK;acc=.5*acc+.5*np.clip(raw,-35,35)
        previous_velocity=snap[:,3:6].copy()
        for i,solver in enumerate(solvers):
            tick=time.perf_counter();solver.current(snap[i]);X,U=solver.prediction();refs[i]['current_position']=snap[i,:3]
            od=online_data(refs[i],X[:,17],snap[1-i],cfg,acc[1-i] if args.predictor=='ca' else None)
            if cfg['geometry_funnel_release']:
                od[:,43]=refs[i]['geometry_progress_slope']
            if counts[i]==0 and cfg['initial_progress_slope']>=0:
                od[:,43]=cfg['initial_progress_slope']
            # Use the observer's .35s acceleration decay on the controller grid.
            # The reference helper also uses cfg['dt']; this overrides its
            # generic acceleration-decay default for this observer model.
            times=np.arange(args.horizon+1)*DT;a=acc[1-i] if args.predictor=='ca' else np.zeros(3);tau=.35
            od[:,16:19]=snap[1-i,:3]+snap[1-i,3:6]*times[:,None]+a[None,:]*tau*(times[:,None]-tau*(1-np.exp(-times[:,None]/tau)))
            od[:,19:22]=snap[1-i,3:6]+a[None,:]*tau*(1-np.exp(-times[:,None]/tau));od[:,22:25]=a[None,:]*np.exp(-times[:,None]/tau)
            if cfg['neighbor_forecast_model']!='ca':
                from cbfracing.control.neighbor_prediction import coordinated_turn,turn_tail
                forecast=coordinated_turn if cfg['neighbor_forecast_model']=='turn' else turn_tail
                od[:,16:19],od[:,19:22],od[:,22:25]=forecast(snap[1-i,:3],snap[1-i,3:6],a,times)
                # The immediate applied-interval safety audit keeps the same
                # current measured/estimated acceleration as the baseline.
                od[0,22:25]=a
            # Constrain the learned passage point tightly near its gate; it is
            # already strictly inside the actual physical aperture.
            od[:,40]=args.gate_halfwidth if planner.net is not None else .68
            install_data(solver,od,refs[i]);solver.weights(mu=reward_now(i),q_l=args.lag_weight,df=args.thrust_rate_weight,at=args.progress_acceleration_weight)
            attempts=[]
            for retry in range(args.solve_retries+1):
                result=solver.step();attempts.append(result)
                if result['status']==0:break
                # Continue the same constrained OCP from its last iterate. No
                # rejected iterate becomes a control and no constraints change.
                recoverable_status=('Maximum_Iterations_Exceeded','Maximum_WallTime_Exceeded','Error_In_Step_Computation','Search_Direction_Becomes_Too_Small','Restoration_Failed')
                if args.retry_infeasible:recoverable_status+=('Infeasible_Problem_Detected',)
                if result['message'] not in recoverable_status or not np.isfinite(solver.X).all() or not np.isfinite(solver.U).all():break
            X,U=solver.prediction();refinement=[]
            for outer in range(args.reference_refine):
                delta=X[:,17]-solver.od[:,12]
                local=solver.od[:,:3]+solver.od[:,3:6]*delta[:,None]+solver.od[:,6:9]*delta[:,None]**2+solver.od[:,9:12]*delta[:,None]**3
                exact=refs[i]['spline'](X[:,17]);error=float(np.linalg.norm(local-exact,axis=1).max())
                refinement.append(dict(iteration=outer,maximum_reference_error=error,success=result['status']==0))
                if result['status']==0 and error<.015:break
                if not np.isfinite(X).all():break
                progress=np.clip(X[:,17],0.,refs[i]['length'])
                for order in range(4):od[:,3*order:3*order+3]=refs[i]['spline'](progress,order)/[1,1,2,6][order]
                od[:,12]=progress
                curve=refs[i]['spline'](progress);d2=np.sum((curve[:,None,:]-np.array(cfg['gates'])[None,:,:])**2,axis=2)
                od[:,13]=cfg['qnom']+cfg['qwp']*np.exp(-.5*d2/cfg['sigma']**2).sum(axis=1)
                install_data(solver,od,refs[i])
                solver.reference_repair_only=bool(cfg['reference_repair_only'] and result['status']==0 and args.mpcc_backend=='rti')
                try:
                    result=solver.step();result['reference_repair_only']=solver.reference_repair_only
                finally:solver.reference_repair_only=False
                attempts.append(result);X,U=solver.prediction()
            row=dict(time=ts[-1],agent=i,controller_seconds=time.perf_counter()-tick,solve=result,solve_attempts=attempts,reference_refinement=refinement,control=U[0].copy(),progress_reward=reward_now(i))
            chosen.append((X,U));roundlog.append(row)
            if result['status']!=0:
                reason=f'Agent{i} MPCC failure: '+result['message'];np.savez_compressed(output/f'failed_ocp_{k}_{i}.npz',X=X,U=U,initial=snap[i],od=solver.od,weights=solver.w,mu=solver.mu,snapshot=snap)
        if reason:logs.extend(roundlog);break
        dense=np.array([[np.asarray(step(snap[i],chosen[i][1][0],FEEDBACK*j/8)).ravel() for j in range(1,9)] for i in range(2)])
        states=dense[:,-1].copy();distance=np.linalg.norm(dense[0,:,:3]-dense[1,:,:3],axis=1)
        if distance.min()<.5-1e-4:reason='Actual clearance violation'
        world=np.array(c['world_box'])
        force_samples=states[:,13:17] if cfg['nodal_thrust_only'] else dense[:,:,13:17]
        if (np.linalg.norm(dense[:,:,10:13],axis=-1).max()>MODEL['omega_max']+.001 or force_samples.min()<-.001 or force_samples.max()>7.001 or
            (dense[:,:,:3]<world[:3]-.001).any() or (dense[:,:,:3]>world[3:]+.001).any()):reason='Actual physical bound violation'
        for r in roundlog:r['min_actual_distance']=float(distance.min())
        logs.extend(roundlog);ts.append(ts[-1]+FEEDBACK);history.append(states.copy());pending=[]
        for i in range(2):
            gi=counts[i]%7;g=np.array(c['centers'][gi]);n=np.array(c['normals'][gi]);side=np.cross([0,0,1.],n)
            if (snap[i,:3]-g)@n<0<=(states[i,:3]-g)@n:
                lo=0.;hi=FEEDBACK
                for _ in range(25):
                    mid=(lo+hi)/2;y=np.asarray(step(snap[i],chosen[i][1][0],mid)).ravel()
                    if (y[:3]-g)@n>=0:hi=mid
                    else:lo=mid
                y=np.asarray(step(snap[i],chosen[i][1][0],hi)).ravel();delta=y[:3]-g;offset=np.array([delta@side,delta[2]])
                event=dict(gate=gi+1,sequence=counts[i]+1,time=ts[-2]+hi,position=y[:3],offset=offset,forward_speed=float(y[3:6]@n))
                if abs(offset).max()<=.75 and event['forward_speed']>0:events[i].append(event);counts[i]+=1;pending.append(i)
                else:reason=f'Agent{i} missed gate{gi+1}';roundlog[i]['missed_gate']=event
        for i,(X,U) in enumerate(chosen):
            fraction=FEEDBACK/DT;tail=np.asarray(step(X[-1],U[-1],FEEDBACK)).ravel()
            xn=np.vstack([(1-fraction)*X[:-1]+fraction*X[1:],tail]);un=np.vstack([(1-fraction)*U[:-1]+fraction*U[1:],U[-1]])
            xn[:,6:10]/=np.linalg.norm(xn[:,6:10],axis=1)[:,None];xn[0]=states[i];solvers[i].seed(xn,un)
            if args.mpcc_backend=='rti' and args.rti_warmstart=='rollout':
                # Restore multiple-shooting dynamics in the warm start only.
                # No predicted bound is accepted without the usual NLP audit.
                for j in range(args.horizon):xn[j+1]=np.asarray(warm_step(xn[j],un[j],DT)).ravel()
                solvers[i].seed(xn,un)
        if min(counts)<target and not reason:
            for i in pending:
                if not prepare(i):reason='Rolling CPC failure';break
        history[-1]=states.copy()
        if k%25==0 or pending or reason:
            print('flight',k,ts[-1],counts,'d',float(distance.min()),reason,flush=True)
            save_json(output/'planning_log.json',logs);save_json(output/'events.json',events);np.savez_compressed(output/'execution.partial.npz',time=ts,state=history)
        if reason:break
    if reason is None and min(counts)<target:reason='Flight time budget exhausted'
    x=np.array(history);durations=[r['controller_seconds']*1000 for r in logs]
    dense_min=[r['min_actual_distance'] for r in logs if 'min_actual_distance'in r]
    flush_tick=time.perf_counter()
    for path,plan in plan_artifacts:save(path,plan)
    plan_flush_seconds=time.perf_counter()-flush_tick
    summary=dict(success=reason is None and min(counts)>=target,gates_passed=counts,target_per_agent=target,duration=ts[-1],stop_reason=reason or 'completed',
        min_distance=min(dense_min) if dense_min else float(np.linalg.norm(x[0,0,:3]-x[0,1,:3])),
        controller_p50_ms=float(np.median(durations)) if durations else None,controller_p95_ms=float(np.percentile(durations,95)) if durations else None,
        frontend_seconds=sum(r['seconds'] for r in front),frontend_queries=len(front),max_speed=float(np.linalg.norm(x[:,:,3:6],axis=-1).max()),
        manual_waiting=False,expert_fallbacks=0,physical_latency_simulated=False,
        controller_construction_seconds=cfg['controller_construction_seconds'],planner_construction_seconds=cfg['planner_construction_seconds'],
        deferred_plan_artifact_seconds=plan_flush_seconds)
    save_json(output/'summary.json',summary);save_json(output/'planning_log.json',logs);save_json(output/'events.json',events);np.savez_compressed(output/'execution.npz',time=ts,state=x)
    print('SUMMARY',summary,flush=True)

def build_parser():
    defaults=configuration()['online_planning']
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--allocator',default=str(HERE/'models/allocation.pt'))
    p.add_argument('--mpcc-backend',choices=['ipopt','rti'],default='rti')
    p.add_argument('--event-checkpoint',help='Five-event time network checkpoint')
    p.add_argument('--start-mode',choices=['original','gate1_aligned'])
    p.add_argument('--rti-kernels',choices=['auto','native','symbolic'],default='auto')
    p.add_argument('--rti-cold-start',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--sqp-recovery',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--gate-dense-prefix',type=int,default=3)
    p.add_argument('--cbf-decay-margin',type=float,default=.04)
    p.add_argument('--passage-update-limit',type=float,default=.2)
    p.add_argument('--turn-reward',type=float)
    p.add_argument('--rti-iterations',type=int,default=1)
    p.add_argument('--rti-regularization',type=float,default=.001)
    p.add_argument('--rti-trust-radius',type=float,default=2.)
    p.add_argument('--rti-globalize',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--rti-feasibility-correction',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--rti-feasible-first',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--rti-elastic-restore',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--rti-condense',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--rti-warmstart',choices=['interpolate','rollout'],default='interpolate')
    p.add_argument('--rti-acceptance',choices=['horizon','applied'],default='horizon')
    p.add_argument('--rti-native',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--rti-qp-tolerance',type=float,default=1e-8)
    p.add_argument('--rti-shooting-guard',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--cbf-hard-steps',type=int,default=-1,help='Hard predicted CBF decay intervals; -1 keeps full horizon')
    p.add_argument('--cbf-tail-weight',type=float,default=2000.)
    p.add_argument('--rti-dense-cbf',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--rti-elastic-prefix',type=int,default=-1,help='Preserve hard near rows and all dynamics/input bounds in elastic restoration')
    p.add_argument('--rti-forecast-filter',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--rti-objective-updates',type=int,default=1)
    p.add_argument('--rti-nlp-recovery-seconds',type=float,default=0.,help='Explicit bounded same-OCP nonlinear restoration after RTI fails')
    p.add_argument('--rti-batched-audit',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--exact-gate-aperture',action=argparse.BooleanOptionalAction,default=False,help='Use true physical center with same 5 cm reserve instead of a symmetric inscribed allocation box')
    p.add_argument('--next-gate-guard',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--velocity-contour-weight',type=float,default=0.,help='Penalize velocity normal to the reference tangent')
    p.add_argument('--reference-speed-weight',type=float,default=0.,help='Soft penalty on exceeding the spatial reference speed profile; no speed bound')
    p.add_argument('--reference-speed-extra',type=float,default=1.)
    p.add_argument('--gate-prediction-margin',type=float,default=0.,help='Extra prediction-only gate reserve; applied aperture unchanged')
    p.set_defaults(cpc_mode='original',predictor='ca',barrier='soft',allocation_zero=False)
    p.add_argument('--laps',type=int,default=2);p.add_argument('--max-time',type=float,default=40.);p.add_argument('--horizon',type=int,default=defaults['horizon'])
    p.add_argument('--preview-spline',choices=['hermite','c2'],default=defaults['preview_spline']);p.add_argument('--reference-refine',type=int,default=defaults['reference_refine'])
    p.add_argument('--physical-gate-guard',action=argparse.BooleanOptionalAction,default=defaults['physical_gate_guard'])
    for name in ['lag_weight','gate_weight','contour_weight','thrust_rate_weight','progress_acceleration_weight']:
        p.add_argument('--'+name.replace('_','-'),type=float,default=defaults[name])
    p.add_argument('--frontend',choices=['original','phase','minimum-snap'],default='minimum-snap');p.add_argument('--preview-gates',type=int,default=defaults['preview_gates']);p.add_argument('--progress',type=float);p.add_argument('--penalty',type=float,default=100000.)
    p.add_argument('--progress-start',type=float);p.add_argument('--progress-ramp-delay',type=float,default=2.);p.add_argument('--progress-ramp-seconds',type=float,default=3.)
    p.add_argument('--reward-policy',choices=['constant','reference-speed'],default='constant')
    p.add_argument('--reward-floor',type=float,default=80.)
    p.add_argument('--reward-speed-allowance',type=float,default=2.)
    p.add_argument('--reward-braking',type=float,default=8.)
    p.add_argument('--reward-transition',type=float,default=.6)
    p.add_argument('--separation',type=float,default=1.3);p.add_argument('--forward-offset',type=float,default=0.)
    p.add_argument('--solver-seconds',type=float,help='Per MPCC attempt: analytic8s, learned35s');p.add_argument('--iterations',type=int,help='Per MPCC attempt: analytic900, learned4000');p.add_argument('--cpc-seconds',type=float,default=35.);p.add_argument('--solve-retries',type=int,default=2)
    p.add_argument('--gate-progress-slope',type=float,default=1.5);p.add_argument('--target-gates',type=int)
    p.add_argument('--cbf',dest='cbf_branch',choices=['analytic','learned'],default=defaults['cbf_branch'],help='Safety branch; learned selects the validated direct-output v7 profile')
    p.add_argument('--cbf-checkpoint',dest='h_checkpoint',help='Custom direct-h checkpoint for --cbf learned')
    p.add_argument('--cbf-alpha',type=float,help='Discrete CBF rate; default3')
    p.add_argument('--gate-halfwidth',type=float,help='Tracking funnel base halfwidth: analytic0.20m, learned0.16m')
    p.add_argument('--tube-recovery',action=argparse.BooleanOptionalAction,default=None,help='Omit current tracking-tube rows; learned default enabled')
    p.add_argument('--dense-physics',action=argparse.BooleanOptionalAction,default=None,help='Check physical bounds within each applied interval; learned default enabled')
    p.add_argument('--phase-retries',type=int,help='Same-problem phase initialization retries: analytic0, learned2')
    p.set_defaults(rti_iterations=16,rti_regularization=.1,rti_globalize=False,
                   rti_feasibility_correction=True,rti_acceptance='applied',
                   cbf_hard_steps=3,rti_dense_cbf=True,rti_forecast_filter=True,
                   rti_batched_audit=True,exact_gate_aperture=True,gate_prediction_margin=.05,
                   solve_retries=0,max_time=25.,thrust_rate_weight=.05,gate_weight=10000.,
                   gate_progress_slope=.6,reward_policy='reference-speed',reward_speed_allowance=4.)
    return p

def parse_args(argv=None):
    parser=build_parser();args=parser.parse_args(argv)
    try:
        if args.gate_dense_prefix<0 or args.gate_dense_prefix>args.horizon:
            raise ValueError('Dense gate prefix must fit the prediction horizon.')
        if args.cbf_decay_margin<0 or args.passage_update_limit<0:
            raise ValueError('CBF margin and passage update limit must be nonnegative.')
        if args.frontend=='minimum-snap' and not args.physical_gate_guard:
            raise ValueError('Minimum Snap requires --physical-gate-guard.')
        if args.event_checkpoint and args.preview_gates!=5:
            raise ValueError('The time network requires five gate events.')
        if args.preview_gates<2:raise ValueError('At least two physical gate events are required by the time initializer.')
        if args.rti_iterations<1 or args.rti_regularization<=0 or args.rti_trust_radius<=0:raise ValueError('RTI iteration limit, trust radius and regularization must be positive.')
        if not 1<=args.rti_objective_updates<=args.rti_iterations or args.rti_nlp_recovery_seconds<0:raise ValueError('Objective update count must fit the RTI budget and recovery budget must be nonnegative.')
        if args.reward_floor<0 or args.reward_speed_allowance<0 or args.reward_braking<=0 or args.reward_transition<=0:raise ValueError('Reward parameters must be nonnegative, with positive braking/transition scales.')
        if not 0<args.rti_qp_tolerance<=1e-4:raise ValueError('QP tolerance must be in (0,1e-4].')
        if args.progress_ramp_delay<0 or args.progress_ramp_seconds<=0:raise ValueError('Progress delay must be nonnegative and ramp duration positive.')
        if args.cbf_hard_steps==0 or args.cbf_hard_steps< -1 or args.cbf_tail_weight<0:raise ValueError('Use at least one hard CBF interval, or -1 for full horizon.')
        if not 0<=args.gate_prediction_margin<.2:raise ValueError('Prediction-only gate margin must lie in [0,.2).')
        if args.frontend=='original' and args.preview_gates!=2:raise ValueError('The retained original complementarity CPC supports two gates; use --preview-gates 2 or --frontend phase.')
        if args.reference_refine<0 or args.horizon<1:raise ValueError('Reference iterations must be nonnegative and horizon positive.')
        if min(args.lag_weight,args.gate_weight,args.contour_weight,args.thrust_rate_weight,args.progress_acceleration_weight)<0:raise ValueError('Cost weights must be nonnegative.')
        return resolve_safety(args)
    except ValueError as error:parser.error(str(error))

if __name__=='__main__':main(parse_args())
