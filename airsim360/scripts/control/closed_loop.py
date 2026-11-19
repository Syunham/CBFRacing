"""Full two-lap closed-loop evaluation, causal neighbor prediction, no waiting."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from scripts.simulation.course import *
from scripts.planning.utils.local_planner import LocalPlanner,reference,save
from scripts.control.rti_problem import RecoveryMPCC,DT,FEEDBACK
from scripts.planning.utils.reference import online_data
from scripts.control.cli_profiles import resolve_safety
import argparse,shutil,json

def main(args, vision, output, planner_class=LocalPlanner, *, vehicle_count=2, track_full_course=False):
    if vehicle_count not in (1, 2):
        raise ValueError('Expected one or two vehicles')
    single = vehicle_count == 1
    agents = range(vehicle_count)
    if track_full_course and (not single or args.phase_optimizer != 'fast-fly' or
                              args.preview_gates != (args.target_gates or 7*args.laps) or args.replan_period != 0.):
        raise ValueError('Full-course tracking requires one joint Fast-fly plan for all gate events')
    from scripts.control.runtime import configure
    if args.phase_optimizer == 'fixed-time-sqp':
        from scripts.simulation.dynamics import select_vehicle, vehicle_parameters
        select_vehicle(vehicle_parameters())
    configure(args)
    output.mkdir(parents=True,exist_ok=False);(output/'plans').mkdir()
    os.environ['GATE_PREDICTION_MARGIN']=str(args.gate_prediction_margin)
    os.environ['EXACT_GATE_APERTURE']='1' if args.exact_gate_aperture else '0'
    os.environ['NEXT_GATE_GUARD']='1' if args.next_gate_guard else '0'
    os.environ['VELOCITY_CONTOUR_WEIGHT']=str(args.velocity_contour_weight)
    os.environ['REFERENCE_SPEED_WEIGHT']=str(args.reference_speed_weight)
    os.environ['CPC_PREVIEW_GATES']=str(args.preview_gates);os.environ['CPC_FRONTEND']=args.frontend;os.environ['PREVIEW_SPLINE']=args.preview_spline;os.environ['PHYSICAL_GATE_GUARD']='1' if args.physical_gate_guard else '0';c=course();save_json(output/'course.json',c);cfg=dict(vars(args),gates=c['centers'],dynamics=MODEL,
        dt=DT,feedback=FEEDBACK,physical_speed_limit=None,manual_waiting=False,manual_priority=False,
        actuator_limits=f'CPC and MPCC0..{MODEL["thrust_max"]}N/rotor; MPC body-rate norm{MODEL["omega_max"]}rad/s',body_rate_norm_limit=MODEL['omega_max'],future_trajectory_exchange=False,
        target_events_per_agent=args.target_gates or 7*args.laps,cbf=os.getenv('ABLATION_MODE') in ('learned_cbf','analytic_cbf'),safety_mode=os.getenv('ABLATION_MODE'),threat_weight=float(os.getenv('THREAT_WEIGHT','0')),clearance=.5,clearance_margin=.05,cbf_radius=np.inf,
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
    if args.frontend=='phase' and args.phase_optimizer=='fast-fly':
        cfg['frontend_solver']='ipopt_fast_fly_style'
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
        if not args.event_checkpoint:cfg['event_policy']='two_gate_time_prior_then_distance_over_8'
        if getattr(args,'optimize_gate_times',False):cfg['event_policy']='optimized_gate_times'
        if getattr(args,'snap_time_network',None):cfg['event_policy']='synthetic_time_network_fixed_qp'
        if getattr(args,'guan_time_network',None):cfg.update(event_policy='guan_projected_network_times',frontend_solver='guan_minimum_jerk')
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
    cfg['passage_memory_active']=args.frontend=='minimum-snap'
    cfg['neighbor_forecast_model']=os.getenv('OPT_NEIGHBOR_FORECAST','ca')
    if cfg['neighbor_forecast_model'] not in ('ca','turn','turn_tail','spatial_turn','jerk'):raise ValueError('Unknown neighbor forecast model')
    cfg['vehicle_count']=vehicle_count
    if track_full_course:
        cfg.update(track_full_course=True, full_course_terminal_extension_s=.4)
    states=(initial_states(0.,0.)[:1].copy() if single else initial_states(args.separation,args.forward_offset))
    ts=[0.];history=[states.copy()];logs=[];front=[];events=[[] for _ in agents];counts=[0]*vehicle_count;refs=[None]*vehicle_count
    from collections import deque
    diagnostic_window=deque(maxlen=max(1,2*args.diagnostic_window_steps))
    build_tick=time.perf_counter();planner=planner_class(args.allocator,args.cpc_seconds,args.phase_retries)
    save_json(output/'planner_config.json',planner.phase.c)
    if args.frontend=='phase' and args.phase_optimizer=='fast-fly':
        cfg['frontend_native']=bool(planner.phase.c.get('phase_native_library'))
        cfg['frontend_integration_substeps']=planner.phase.c['integration_substeps']
        cfg['frontend_native_libraries']={k:v for k,v in planner.phase.c.items() if k.endswith('_library')}
    elif args.phase_optimizer=='fixed-time-sqp':
        cfg.update(frontend_native=False, frontend_solver=planner.phase.solver_name,
                   frontend_integration_substeps=planner.phase.c['integration_substeps'],
                   phase_cells_per_leg=planner.phase.c['cells_per_leg'],
                   event_checkpoint=planner.event_checkpoint, event_policy=planner.phase.event_policy,
                   launch_frontend=planner.launch_phase.solver_name, launch_gate_events=1,
                   cpc_mode='fixed_phase_events_network_durations')
    cfg['planner_construction_seconds']=time.perf_counter()-build_tick;plan_artifacts=[]
    solver_class=RecoveryMPCC
    if args.mpcc_backend=='rti':
        from scripts.control.online_rti import OnlineRTI
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
        tube_recovery=args.tube_recovery,dense_physics=args.dense_physics,gate_guard=cfg['local_gate_window']) for i in agents]
    cfg['controller_construction_seconds']=time.perf_counter()-build_tick
    cfg['loaded_compiled_kernels']=[getattr(s,'fast_kernels',False) for s in solvers]
    cfg['kernel_loading_notes']=[getattr(s,'kernel_loading_note','IPOPT') for s in solvers]
    if hasattr(solvers[0],'row_kinds'):
        from collections import Counter
        cfg['constraint_groups']=dict(Counter(solvers[0].row_kinds))
        if not cfg['hard_tracking_funnel']:
            assert cfg['constraint_groups'].get('tracking',0)==0
            assert cfg['constraint_groups'].get('gate_crossing',0)>0
            if os.getenv('ABLATION_MODE')!='threat':assert cfg['constraint_groups'].get('cbf',0)>0
    save_json(output/'config.json',cfg)
    _,step=model_functions(8);_,warm_step=model_functions(2);acc=np.zeros((vehicle_count,3));previous_velocity=states[:,3:6].copy();reason=None;target=args.target_gates or 7*args.laps
    if len(history)>1:acc=np.clip((history[-1][:,3:6]-history[-2][:,3:6])/FEEDBACK,-35,35)
    observed_jerk=np.zeros((vehicle_count,3))
    from scripts.control.progress_reward import ProgressReward
    adaptive_rewards=[ProgressReward(args) for _ in agents] if args.reward_policy=='adaptive' else None
    if args.flying_gate_schedule:
        if len(c['centers'])!=7:raise ValueError('Per-gate schedule requires the seven-gate Split-S course')
        save_json(output/'reward_schedule.json',dict(gains=adaptive_rewards[0].gate_gains,applies_after_completed_laps=1,learned=False))
    def reward_now(i):
        if adaptive_rewards is not None:
            return adaptive_rewards[i].update(ts[-1],states[i],refs[i],solvers[i],completed_laps=counts[i]//len(c['centers']),gate_index=counts[i])
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
        local = od.copy()
        if vision is not None:
            vision.install(solver, local)
        solver.reference_delta = local[:, :3]-od[:, :3]
        solver.data(local)
        if args.reference_speed_weight>0:
            arc,unique=np.unique(ref['speed_arc'],return_index=True);speed=ref['speed_values'][unique]
            slope=np.gradient(speed,arc)
            solver.od[:,55]=np.interp(od[:,12],arc,speed)+args.reference_speed_extra
            solver.od[:,56]=np.interp(od[:,12],arc,slope)
    last_replan_time=np.full(vehicle_count,-np.inf)
    previous_forecast=[None,None];previous_forecast_time=[None,None]
    def prepare(i):
        if hasattr(planner, "keep_approach") and planner.keep_approach(i, counts[i], states[i]):
            return True
        begin=time.perf_counter();row=dict(agent=i,time=ts[-1],gate_index=counts[i])
        carry=solvers[i].prediction() if cfg['carry_replan_warmstart'] and ts[-1]>0 else None
        try:
            planner.active_agent=i
            planner.current_time=ts[-1]
            peer=None
            if not single:
                peer=vision.allocation_peer(i,states[i]) if getattr(args,'visual_peer',False) else (states[1-i] if getattr(args,'original_allocation',False) else None)
            plan=planner.solve(states[i],peer,counts[i])
            tracking_plan=plan
            if track_full_course:
                # Give the tracker progress beyond the timing line. The audited
                # racing plan and its 14 event times end at the final gate.
                x=plan['reference_states'];t=plan['reference_time']
                extension=np.linspace(0.,cfg['full_course_terminal_extension_s'],21)[1:]
                tail=np.tile(x[-1],(len(extension),1))
                tail[:,:3]+=extension[:,None]*x[-1,3:6]
                tracking_plan=dict(plan,reference_time=np.r_[t,t[-1]+extension],reference_states=np.vstack((x,tail)))
            refs[i]=reference(tracking_plan,states[i])
            if track_full_course:
                xyz=plan['reference_states'][:,:3]
                arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
                refs[i].update(course_targets=plan['gates'],course_normals=plan['normals'],
                               course_physical_gates=plan['physical_gates'],
                               course_gate_progress=np.interp(np.cumsum(plan['durations']),plan['reference_time'],arc))
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
            U=np.c_[np.clip(np.diff(X[:,13:17],axis=0)/DT,-MODEL['thrust_rate_max'],MODEL['thrust_rate_max']),np.clip(np.diff(X[:,18])/DT,-40,40)]
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
            solvers[i].seed(X,U);solvers[i].current(states[i]);solvers[i].weights(mu=adaptive_rewards[i].weight if adaptive_rewards is not None else reward_now(i),q_l=args.lag_weight,df=args.thrust_rate_weight,at=args.progress_acceleration_weight)
            row.update(success=True,allocation=plan['allocation'],duration=plan['durations'],audit=plan['audit'],cpc_seconds=plan['wall_seconds'],seed_diagnostics=plan['seed_diagnostics'],phase_solve_trace=plan.get('phase_solve_trace',[]))
            if 'initial_force_cost' in plan:
                row.update(initial_force_weight=plan['initial_force_weight'],initial_force_cost=plan['initial_force_cost'],
                           initial_force_jump_max=float(np.max(np.abs(plan['U'][:,0]-plan['initial_forces']))))
            if 'reference_continuity_cost' in plan:
                row.update(reference_continuity_weight=plan['reference_continuity_weight'],
                    reference_continuity_cost=plan['reference_continuity_cost'],base_objective_value=plan['base_objective_value'])
            row.update(event_network_seconds=plan.get('event_network_seconds',0.),event_policy=plan.get('event_policy'),event_durations_input=plan.get('event_durations_input'))
            row.update(frontend_solver=plan.get('frontend_solver'),frontend_update=plan.get('frontend_update'))
        except Exception as e:
            phase=getattr(planner,'active_phase',planner.phase)
            row.update(success=False,error=str(e),seed_diagnostics=getattr(planner,'last_seed_diagnostics',None),
                       phase_solve_trace=getattr(phase,'last_trace',None),frontend_solver=getattr(phase,'solver_name',None))
            window=getattr(phase,'last_window',None)
            if window is not None:
                row.update(event_policy=phase.event_policy,event_durations_input=window.durations)
            failed=getattr(planner,'last_failed_plan',None)
            if failed is not None:
                path=output/f'failed_frontend_{len(front):03d}_agent{i}.npz'
                save(path,failed);row['failed_candidate_file']=path.name
        row['seconds']=time.perf_counter()-begin
        row['compute_seconds_excluding_plan_save']=row['seconds']-row.get('plan_artifact_save_seconds',0.)
        front.append(row);save_json(output/'frontends.json',front);print('front',row,flush=True)
        if row['success']:last_replan_time[i]=ts[-1]
        return row['success']
    reuse_last_mpcc = getattr(args, 'reuse_last_mpcc', False)
    last_nominal = [None]*vehicle_count
    last_success_time = [None]*vehicle_count
    if vision is not None:
        vision.observe(0., states)
    planner.visual_passage = (lambda *_: 0.) if single else getattr(vision, "passage", None)
    planner.allocation_own = getattr(vision,'allocation_own',None)
    for i in agents:
        if not prepare(i):reason='CPC reference failure';break
    benchmark_agent = getattr(vision, "benchmark_agent", None)
    for k in range(int((args.max_time-ts[-1])/FEEDBACK)):
        if reason or min(counts)>=target:break
        snap=states.copy();chosen=[];roundlog=[]
        observation_tick=time.perf_counter()
        if k and vision is not None: vision.observe(ts[-1], snap)
        observation_seconds=time.perf_counter()-observation_tick
        for i,solver in enumerate(solvers):
            tick=time.perf_counter()
            if benchmark_agent is not None:
                benchmark_agent(i, k)
            solver.current(snap[i]);X,U=solver.prediction();refs[i]['current_position']=snap[i,:3]
            od=online_data(refs[i], X[:,17], np.zeros(19), cfg)
            if cfg['geometry_funnel_release']:
                od[:,43]=refs[i]['geometry_progress_slope']
            if counts[i]==0 and cfg['initial_progress_slope']>=0:
                od[:,43]=cfg['initial_progress_slope']
            # Constrain the learned passage point tightly near its gate; it is
            # already strictly inside the actual physical aperture.
            od[:,40]=args.gate_halfwidth if planner.net is not None else .68
            install_data(solver,od,refs[i]);solver.weights(mu=reward_now(i),q_l=args.lag_weight,df=args.thrust_rate_weight,at=args.progress_acceleration_weight)
            diagnostic=None
            if args.diagnostic_window_steps:
                diagnostic=dict(time=np.array(ts[-1]),agent=np.array(i),warm_X=solver.X.copy(),warm_U=solver.U.copy(),
                    initial=snap[i].copy(),snapshot=snap.copy(),od_before=solver.od.copy(),weights=solver.w.copy(),mu=np.array(solver.mu))
            if reuse_last_mpcc:
                warm_X, warm_U = X.copy(), U.copy()
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
                exact=refs[i]['spline'](X[:,17])+solver.reference_delta;error=float(np.linalg.norm(local-exact,axis=1).max())
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
            row['observation_processing_seconds']=observation_seconds/vehicle_count
            row['controller_seconds']+=observation_seconds/vehicle_count
            if adaptive_rewards is not None:
                row['progress_adaptation']=dict(adaptive_rewards[i].diagnostics)
                adaptive_rewards[i].observe(result,ts[-1])
            chosen.append((X,U));roundlog.append(row)
            if diagnostic is not None:
                diagnostic.update(X=X.copy(),U=U.copy(),od=solver.od.copy(),status=np.array(result['status']))
                diagnostic_window.append(diagnostic)
            if reuse_last_mpcc and result['status'] == 0:
                last_nominal[i] = U[0].copy()
                last_success_time[i] = ts[-1]
            elif reuse_last_mpcc and last_nominal[i] is not None:
                warm_U[0] = last_nominal[i]
                chosen[i] = (warm_X, warm_U)
                row.update(control=last_nominal[i].copy(), reused_mpcc=True,
                           reused_from_time=last_success_time[i])
            elif result['status']!=0:
                reason=f'Agent{i} MPCC failure: '+result['message'];np.savez_compressed(output/f'failed_ocp_{k}_{i}.npz',X=X,U=U,initial=snap[i],od=solver.od,weights=solver.w,mu=solver.mu,snapshot=snap)
        if reason:
            logs.extend(roundlog)
            if args.diagnostic_window_steps:
                destination=output/'diagnostics';destination.mkdir(exist_ok=True)
                for di,d in enumerate(diagnostic_window):np.savez_compressed(destination/f'{di:03d}.npz',**d)
            break
        if not single and k==0 and getattr(args,'cbf_study',None)=='filtered':
            from scripts.control.pair_cbf import PairCBF
            pair_filter=PairCBF(continue_infeasible=getattr(args,'continue_infeasible_cbf',False))
        if (not single and getattr(args,'cbf_study',None)=='filtered'
                and (not getattr(args,'cbf_after_first_gate',False) or min(counts)>=1)):
            nominal=np.stack([U[0] for X,U in chosen])
            try:
                applied,audit=vision.visual_projection(pair_filter,snap,nominal,FEEDBACK) if getattr(args,'visual_peer',False) else pair_filter.project(snap,nominal,FEEDBACK)
            except ValueError as error:
                reason=str(error)
                for row in roundlog:row['cbf_infeasible']=reason
                logs.extend(roundlog)
                break
            for i,(X,U) in enumerate(chosen):
                U=U.copy()
                U[0]=applied[i]
                chosen[i]=(X,U)
                roundlog[i]['nominal_control']=nominal[i]
                roundlog[i]['control']=applied[i]
                roundlog[i]['pair_cbf']=audit
            # Gate crossing interpolation and the dense execution audit now use applied U.
        dense=np.array([[np.asarray(step(snap[i],chosen[i][1][0],FEEDBACK*j/8)).ravel() for j in range(1,9)] for i in agents])
        states=dense[:,-1].copy()
        min_distance=None if single else float(np.linalg.norm(dense[0,:,:3]-dense[1,:,:3],axis=1).min())
        if hasattr(vision, 'record_applied_control'):
            vision.record_applied_control(np.stack([U[0] for X,U in chosen]), FEEDBACK)
        if min_distance is not None and args.stop_separation is not None and min_distance<args.stop_separation:reason='Actual clearance violation'
        world=np.array(c['world_box'])
        force_samples=states[:,13:17] if cfg['nodal_thrust_only'] else dense[:,:,13:17]
        if (np.linalg.norm(dense[:,:,10:13],axis=-1).max()>MODEL['omega_max']+.001 or force_samples.min()<-.001 or force_samples.max()>MODEL['thrust_max']+.001 or
            (dense[:,:,:3]<world[:3]-.001).any() or (dense[:,:,:3]>world[3:]+.001).any()):reason='Actual physical bound violation'
        if min_distance is not None:
            for r in roundlog:r['min_actual_distance']=min_distance
        logs.extend(roundlog);ts.append(ts[-1]+FEEDBACK);history.append(states.copy());pending=[]
        for i in agents:
            gi=counts[i]%7;g=np.array(c['centers'][gi]);n=np.array(c['normals'][gi]);side=np.cross([0,0,1.],n)
            if (snap[i,:3]-g)@n<0<=(states[i,:3]-g)@n:
                lo=0.;hi=FEEDBACK
                for _ in range(25):
                    mid=(lo+hi)/2;y=np.asarray(step(snap[i],chosen[i][1][0],mid)).ravel()
                    if (y[:3]-g)@n>=0:hi=mid
                    else:lo=mid
                y=np.asarray(step(snap[i],chosen[i][1][0],hi)).ravel();delta=y[:3]-g;offset=np.array([delta@side,delta[2]])
                event=dict(gate=gi+1,sequence=counts[i]+1,time=ts[-2]+hi,position=y[:3],offset=offset,forward_speed=float(y[3:6]@n))
                if single:
                    event.update(velocity=y[3:6],speed=float(np.linalg.norm(y[3:6])))
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
        if track_full_course and not reason and pending and counts[0]<target:
            ref=refs[0];next_event=counts[0]
            ref.update(next_gate=ref['course_targets'][next_event],next_normal=ref['course_normals'][next_event],
                       physical_next_gate=ref['course_physical_gates'][next_event],
                       first_gate_s=ref['course_gate_progress'][next_event])
        elif not track_full_course and min(counts)<target and not reason:
            if vision is not None and (pending or (args.replan_period>0 and np.any(ts[-1]-last_replan_time>=args.replan_period-1e-8))):
                vision.observe(ts[-1],states)
            if args.replan_period>0:
                pending=sorted(set(pending)|{i for i in agents if counts[i]<target and ts[-1]-last_replan_time[i]>=args.replan_period-1e-8})
            for i in pending:
                if not prepare(i):reason='Rolling CPC failure';break
        history[-1]=states.copy()
        if k%25==0 or pending or reason:
            print('flight',k,ts[-1],counts,'d',min_distance,reason,flush=True)
            save_json(output/'planning_log.json',logs);save_json(output/'events.json',events);np.savez_compressed(output/'execution.partial.npz',time=ts,state=history)
        if reason:break
    if reason is None and min(counts)<target:reason='Flight time budget exhausted'
    x=np.array(history);durations=[r['controller_seconds']*1000 for r in logs]
    dense_min=[r['min_actual_distance'] for r in logs if 'min_actual_distance'in r]
    flush_tick=time.perf_counter()
    for path,plan in plan_artifacts:save(path,plan)
    plan_flush_seconds=time.perf_counter()-flush_tick
    summary=dict(success=reason is None and min(counts)>=target,gates_passed=counts,target_per_agent=target,duration=ts[-1],stop_reason=reason or 'completed',
        stop_separation_m=args.stop_separation,
        min_distance=None if single else (min(dense_min) if dense_min else float(np.linalg.norm(x[0,0,:3]-x[0,1,:3]))),
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
    p.add_argument('--phase-optimizer',choices=['fast-fly','fixed-time-sqp','legacy'],default='fast-fly')
    p.add_argument('--phase-cells-per-leg',type=int,default=12,help='Cells per segment in the Fast-fly-style phase optimizer; legacy retains config.json')
    p.add_argument('--phase-integration-substeps',type=int,default=4,help='Integration and physical constraint samples per frontend shooting cell')
    p.add_argument('--phase-native-dir',type=Path,help='Mesh-specific compiled frontend directory; requires FASTFLY_NATIVE=1')
    p.add_argument('--phase-compact-derivatives',action='store_true',help='Use verified one-pass sparse Jacobian/Hessian assembly for the original five-gate mesh')
    p.add_argument('--phase-initial-force-weight',type=float,default=0.,help='Optional penalty on first planned rotor forces minus measured forces (Fast-fly experiment)')
    p.add_argument('--reference-resampling',choices=['linear','hermite'],default='linear',help='Use physical velocity tangents when resampling the frontend path')
    p.add_argument('--phase-first-gate-aperture',action='store_true',help='Use the physical aperture for the imminent gate; later gates keep learned passage boxes')
    p.add_argument('--phase-average-shifted-controls',action='store_true',help='Preserve force and torque impulse in time-shifted warm-start cells')
    p.add_argument('--phase-duration-min',type=float,default=.4,help='Numerical lower bound for freely optimized frontend segment durations')
    p.add_argument('--diagnostic-window-steps',type=int,default=0,help='Save the last accepted and warm MPCC predictions when a solve fails')
    p.add_argument('--forecast-new-weight',type=float,default=1.,help='Weight of fresh neighbor forecast versus time-shifted anchored forecast (0..1)')
    p.add_argument('--forecast-anchor',choices=['decay','persistent'],default='decay',help='How observed position/velocity corrections propagate across the forecast')
    p.add_argument('--phase-online-sqp',action='store_true',help='Try audited feasible SQP frontend updates; include IPOPT fallback in timing')
    p.add_argument('--phase-audit-refine',action='store_true',help='After failed independent frontend audit, use remaining SQP budget for up to two pure feasibility corrections')
    p.add_argument('--phase-constraint-generation',action='store_true',help='Solve frontend QPs with constraint subsets and full-row verification; include complete fallback cost')
    p.add_argument('--phase-reference-continuity',type=float,default=0.,help='NLP fallback penalty on deviation from causally shifted first-two-leg reference positions')
    p.add_argument('--phase-sqp-backend',choices=['daqp','clarabel'],default='daqp',help='Frontend QP only: condensed DAQP or sparse uncondensed Clarabel')
    p.add_argument('--phase-optimize-tail',action='store_true',help='Optimize the new last gate segment only as a warm start; still validate the full five-gate solution')
    p.add_argument('--phase-past-lap-warm',action='store_true',help='After failed SQP, try a previous lap plan from the same agent; requires fresh full solve/audit')
    p.add_argument('--phase-learned-initializer',type=Path,help='Optional primal initialization checkpoint; same physical optimization and audit still required')
    p.add_argument('--phase-learned-budget',type=float,default=3.,help='Bounded learned-seed IPOPT attempt before original numerical fallback')
    p.add_argument('--phase-learned-sqp',action='store_true',help='Try original feasible SQP from the learned primal guess before bounded NLP')
    p.add_argument('--phase-tail-solve-seconds',type=float,default=.3,help='IPOPT budget for the optional new-tail warm start')
    p.add_argument('--prediction-rate-margin',type=float,default=0.,help='Interior prediction margin below the physical body-rate norm bound')
    p.add_argument('--phase-remaining-min',type=float,default=.03,help='Minimum remaining duration of the imminent event during periodic replanning')
    p.add_argument('--phase-adaptive-first-buffer',action='store_true',help='Scale the artificial pre-event plane buffer to the actual remaining distance')
    p.add_argument('--phase-adaptive-remaining-min',action='store_true',help='Reduce the remaining-time lower bound when an observed gate crossing is imminent')
    p.add_argument('--phase-sqp-tail-seed',action='store_true',help='Construct appended tail from the carried physical terminal state')
    p.add_argument('--phase-sqp-gate-trust',type=float,default=.5,help='SQP step trust radius specifically at gate changes')
    p.add_argument('--phase-periodic-projection',action='store_true',help='Between gate events, use a feasibility update without a new time-objective step')
    p.add_argument('--phase-objective-period',type=float,default=0.,help='Seconds between successful frontend time-objective updates; gate changes force an update, zero retains the original policy')
    p.add_argument('--phase-gate-tail-nlp',action='store_true',help='At gate changes, optimize the full NLP from a physical-tail warm seed')
    p.add_argument('--phase-sqp-dual-warm',action='store_true',help='Map QP multipliers by original row identity between SQP iterations')
    p.add_argument('--phase-sqp-chain-condensing',action='store_true',help='Use block forward substitution for the shooting-chain elimination; retain every QP constraint')
    p.add_argument('--phase-consistent-tail-boundary',action='store_true',help='Give the new tail ownership of roundoff-sized old-horizon overlap; average genuine partial control-cell overlap')
    p.add_argument('--phase-audit-retry-quantum',type=float,default=0.,help='One independently audited NLP retry from a quantized causal initial guess after frontend audit failure; zero disables')
    p.add_argument('--replan-period',type=float,default=0.,help='Periodic frontend replanning in simulated seconds; zero retains gate-triggered replans')
    p.add_argument('--phase-passage-half',type=float,help='Planner target-box halfwidth, still clipped inside the physical aperture')
    p.add_argument('--phase-vertical-half',type=float,help='Planner target-box vertical halfwidth at every gate')
    p.add_argument('--phase-thrust-smooth',type=float,default=0.,help='Optional thrust-difference integral regularizer in the Fast-fly time objective')
    p.add_argument('--phase-warm-start',choices=['shift','tail','tail-project'],default='shift')
    p.add_argument('--start-mode',choices=['original','gate1_aligned'])
    p.add_argument('--rti-kernels',choices=['auto','native','symbolic'],default='auto')
    p.add_argument('--rti-cold-start',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--sqp-recovery',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--gate-dense-prefix',type=int,default=3)
    p.add_argument('--cbf-decay-margin',type=float,default=.04)
    p.add_argument('--passage-update-limit',type=float,default=.2)
    p.add_argument('--turn-reward',type=float)
    p.add_argument('--rti-iterations',type=int,default=1)
    p.add_argument('--prediction-thrust-margin',type=float,default=0.,help='Prediction-only rotor-force margin; actual physical limits remain 0 to 7 N')
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
    p.add_argument('--reference-spacing',type=float,default=.15,help='Arc-length spacing for the CPC reference interpolant')
    p.add_argument('--physical-gate-guard',action=argparse.BooleanOptionalAction,default=defaults['physical_gate_guard'])
    for name in ['lag_weight','gate_weight','contour_weight','thrust_rate_weight','progress_acceleration_weight']:
        p.add_argument('--'+name.replace('_','-'),type=float,default=defaults[name])
    p.add_argument('--frontend',choices=['original','phase','minimum-snap'],default='minimum-snap');p.add_argument('--preview-gates',type=int,default=defaults['preview_gates']);p.add_argument('--progress',type=float);p.add_argument('--penalty',type=float,default=100000.)
    p.add_argument('--progress-start',type=float);p.add_argument('--progress-ramp-delay',type=float,default=2.);p.add_argument('--progress-ramp-seconds',type=float,default=3.)
    p.add_argument('--reward-policy',choices=['constant','reference-speed','adaptive'],default='constant')
    p.add_argument('--progress-backoff',action='store_true',help='Retry rejected MPCC queries from the original warm start with a smaller progress cost')
    p.add_argument('--adaptive-lateral-acceleration',type=float,default=24.)
    p.add_argument('--adaptive-turn-preview',type=float,default=8.,help='Distance used to reduce the progress cost before a turn; speed/braking preview remains unchanged')
    p.add_argument('--adaptive-flying-gain',type=float,default=1.,help='Progress ceiling multiplier after the first completed lap; existing safety factors and slew limits remain active')
    p.add_argument('--flying-gate-schedule',type=Path,help='Seven bounded per-gate progress gains used only after the first completed lap')
    p.add_argument('--adaptive-tracking-scale',type=float,default=.5)
    p.add_argument('--adaptive-safety-h',type=float,default=.15)
    p.add_argument('--adaptive-decay-reserve',type=float,default=0.,help='Positive discrete CBF headroom target for anticipatory progress cost scheduling only')
    p.add_argument('--forecast-decay-tau',type=float,default=.35,help='Future neighbor acceleration decay in seconds; applied-interval safety model remains unchanged')
    p.add_argument('--neighbor-forecast',choices=['ca','turn','turn_tail','spatial_turn','jerk'],default='ca')
    p.add_argument('--forecast-jerk-gain',type=float,default=.5,help='Blend current observed jerk into the decaying acceleration forecast; zero recovers the baseline forecast')
    p.add_argument('--forecast-turn-tau',type=float,default=.35,help='Angular rate decay for spatial_turn prediction')
    p.add_argument('--adaptive-rise-rate',type=float,default=800.)
    p.add_argument('--adaptive-fall-rate',type=float,default=3000.)
    p.add_argument('--adaptive-solve-tolerance',type=float,default=0.,help='Positive threshold enables proactive prediction-residual feedback')
    p.add_argument('--adaptive-residual-tail-weight',type=float,default=1.,help='Reward feedback uses max(near residual through stage 3, weight * full residual); physical acceptance is unchanged')
    p.add_argument('--adaptive-residual-straights-only',action='store_true',help='Restore full residual feedback smoothly in curved sections using the existing turn factor')
    p.add_argument('--progress-ramp-increase-only',action='store_true',help='Startup ramp caps reward increases while allowing immediate lower target requests; the configured fall-rate limit remains')
    p.add_argument('--reward-floor',type=float,default=80.)
    p.add_argument('--adaptive-residual-memory-seconds',type=float,default=0.,help='Exponential decay time of peak residual feedback; zero preserves the original scheduler')
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
        if args.replan_period<0:raise ValueError('Replan period must be nonnegative')
        if args.adaptive_decay_reserve<0:raise ValueError('Reward decay reserve must be nonnegative')
        if not .25<=args.adaptive_turn_preview<=25.:raise ValueError('Turn reward preview must lie in [.25,25] metres')
        if not np.isfinite(args.adaptive_flying_gain) or not .5<=args.adaptive_flying_gain<=2.:raise ValueError('Flying progress gain must be in [.5,2]')
        if args.flying_gate_schedule:
            if args.reward_policy!='adaptive':raise ValueError('Per-gate schedule requires the adaptive reward policy')
            args.flying_gate_schedule=args.flying_gate_schedule.resolve()
            if not args.flying_gate_schedule.is_file():raise ValueError('Flying-gate schedule file does not exist')
        if not .02<=args.forecast_decay_tau<=2.:raise ValueError('Forecast decay time must lie in [.02,2] seconds')
        if not .02<=args.forecast_turn_tau<=2.:raise ValueError('Turn decay time must lie in [.02,2] seconds')
        if not np.isfinite(args.forecast_jerk_gain) or not 0.<=args.forecast_jerk_gain<=1.:raise ValueError('Jerk forecast gain must lie in [0,1]')
        if args.neighbor_forecast=='jerk' and args.predictor!='ca':raise ValueError('Jerk forecast requires causal acceleration observations')
        if not 0<=args.prediction_thrust_margin<.1:raise ValueError('Prediction thrust margin must lie in [0,.1) N')
        if not .05<=args.reference_spacing<=1.:raise ValueError('Reference spacing must lie in [.05,1] m')
        if args.phase_thrust_smooth<0:raise ValueError('Thrust regularization must be nonnegative')
        if not 0<=args.forecast_new_weight<=1:raise ValueError('Forecast new weight must be in [0,1]')
        if not 0<=args.prediction_rate_margin<1.:raise ValueError('Prediction rate margin must be in [0,1) rad/s')
        if not 0<args.phase_remaining_min<=args.phase_duration_min:raise ValueError('Remaining duration minimum must be positive and no larger than the segment minimum')
        for width in [args.phase_passage_half,args.phase_vertical_half]:
            if width is not None and not .02<width<=.7:raise ValueError('Passage halfwidth must lie in (.02,.7]')
        os.environ['FASTFLY_PERIODIC']='1' if args.replan_period>0 else '0'
        if min(args.adaptive_lateral_acceleration,args.adaptive_tracking_scale,args.adaptive_safety_h,args.adaptive_rise_rate,args.adaptive_fall_rate)<=0:
            raise ValueError('Adaptive reward scales and rate limits must be positive.')
        if args.adaptive_solve_tolerance<0:
            raise ValueError('Adaptive solve tolerance must be nonnegative.')
        if args.phase_cells_per_leg<6:
            raise ValueError('Phase planning requires at least six cells per leg.')
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

if __name__=='__main__':
    from scripts.control.run_spatial_mpcc import main as run_comparison
    run_comparison()
