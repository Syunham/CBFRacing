"""Full two-lap closed-loop evaluation, causal neighbor prediction, no waiting."""
from context import *
from planning import LocalPlanner,reference,save
from controller import RecoveryMPCC,DT,FEEDBACK
from reference import online_data
from cli_profiles import resolve_safety
import argparse,shutil

def main(args):
    output=HERE/'outputs'/args.name;output.mkdir(parents=True,exist_ok=False);(output/'plans').mkdir();(output/'source').mkdir()
    for p in HERE.glob('*.py'):shutil.copy2(p,output/'source'/p.name)
    c=course();save_json(output/'course.json',c);cfg=dict(vars(args),gates=c['centers'],dynamics=MODEL,
        dt=DT,feedback=FEEDBACK,physical_speed_limit=None,manual_waiting=False,manual_priority=False,
        actuator_limits='CPC and MPCC0..7N/rotor; MPC body-rate norm9.4rad/s',future_trajectory_exchange=False,
        target_events_per_agent=args.target_gates or 7*args.laps,cbf=True,clearance=.5,clearance_margin=.05,cbf_radius=np.inf,
        qwp=10000.,qnom=10.,sigma=.6,cbf_rate=3.,gate_funnel=True,gate_slope=2.,gate_progress_slope=args.gate_progress_slope)
    save_json(output/'config.json',cfg)
    states=initial_states(args.separation,args.forward_offset);ts=[0.];history=[states.copy()];logs=[];front=[];events=[[],[]];counts=[0,0];refs=[None,None]
    planner=LocalPlanner(args.allocator,args.cpc_seconds,args.phase_retries)
    solvers=[RecoveryMPCC(i,n=args.horizon,barrier=args.barrier,penalty=args.penalty,max_iter=args.iterations,wall_time=args.solver_seconds,
        execution_guard=True,cbf_checkpoint=args.h_checkpoint,cbf_mode=args.cbf_mode,cbf_alpha=args.cbf_alpha,
        tube_recovery=args.tube_recovery,dense_physics=args.dense_physics) for i in range(2)]
    _,step=model_functions(8);acc=np.zeros((2,3));previous_velocity=states[:,3:6].copy();reason=None;target=args.target_gates or 7*args.laps
    if len(history)>1:acc=np.clip((history[-1][:,3:6]-history[-2][:,3:6])/FEEDBACK,-35,35)
    def prepare(i):
        begin=time.perf_counter();row=dict(agent=i,time=ts[-1],gate_index=counts[i])
        try:
            plan=planner.solve(states[i],states[1-i],counts[i]);refs[i]=reference(plan,states[i]);save(output/'plans'/f'{len(front):03d}_agent{i}_event{counts[i]}.npz',plan)
            grid=np.arange(args.horizon+1)*DT;tt=plan['reference_time'];xx=plan['reference_states'];X=np.tile(states[i],(args.horizon+1,1))
            X[:,:13]=np.array([np.interp(grid,tt,xx[:,j]) for j in range(13)]).T
            extra=np.maximum(grid-tt[-1],0.);X[:,:3]+=extra[:,None]*xx[-1,3:6]
            X[:,6:10]/=np.linalg.norm(X[:,6:10],axis=1)[:,None]
            cells=plan.get('cell_durations',np.repeat(plan['durations']/planner.c['cells_per_leg'],planner.c['cells_per_leg']))
            ii=np.clip(np.searchsorted(np.cumsum(cells),grid,side='right'),0,plan['U'].shape[1]-1);X[:,13:17]=plan['U'][:,ii].T
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xx[:,:3],axis=0),axis=1))];X[:,17]=np.interp(grid,tt,arc);X[:,18]=np.clip(np.linalg.norm(X[:,3:6],axis=1),0.,30.)
            states[i,17]=0.;states[i,18]=np.linalg.norm(states[i,3:6]);X[0]=states[i]
            U=np.c_[np.clip(np.diff(X[:,13:17],axis=0)/DT,-200,200),np.clip(np.diff(X[:,18])/DT,-40,40)]
            solvers[i].seed(X,U);solvers[i].current(states[i]);solvers[i].weights(mu=args.progress,df=.05,at=2.)
            row.update(success=True,allocation=plan['allocation'],duration=plan['durations'],audit=plan['audit'],cpc_seconds=plan['wall_seconds'],seed_diagnostics=plan['seed_diagnostics'],phase_solve_trace=plan.get('phase_solve_trace',[]))
        except Exception as e:row.update(success=False,error=str(e),seed_diagnostics=getattr(planner,'last_seed_diagnostics',None),phase_solve_trace=getattr(planner.phase,'last_trace',None))
        row['seconds']=time.perf_counter()-begin;front.append(row);save_json(output/'frontends.json',front);print('front',row,flush=True)
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
            # Native helper's time grid is0.06s; all observer forecasts are
            # overwritten on this controller's actual0.08s prediction grid.
            times=np.arange(args.horizon+1)*DT;a=acc[1-i] if args.predictor=='ca' else np.zeros(3);tau=.35
            od[:,16:19]=snap[1-i,:3]+snap[1-i,3:6]*times[:,None]+a[None,:]*tau*(times[:,None]-tau*(1-np.exp(-times[:,None]/tau)))
            od[:,19:22]=snap[1-i,3:6]+a[None,:]*tau*(1-np.exp(-times[:,None]/tau));od[:,22:25]=a[None,:]*np.exp(-times[:,None]/tau)
            # Constrain the learned passage point tightly near its gate; it is
            # already strictly inside the actual physical aperture.
            od[:,40]=args.gate_halfwidth if planner.net is not None else .68
            solver.data(od);solver.weights(mu=args.progress,df=.05,at=2.)
            attempts=[]
            for retry in range(args.solve_retries+1):
                result=solver.step();attempts.append(result)
                if result['status']==0:break
                # Continue the same constrained OCP from its last iterate. No
                # rejected iterate becomes a control and no constraints change.
                recoverable_status=('Maximum_Iterations_Exceeded','Maximum_WallTime_Exceeded','Error_In_Step_Computation','Search_Direction_Becomes_Too_Small','Restoration_Failed')
                if args.retry_infeasible:recoverable_status+=('Infeasible_Problem_Detected',)
                if result['message'] not in recoverable_status or not np.isfinite(solver.X).all() or not np.isfinite(solver.U).all():break
            X,U=solver.prediction();row=dict(time=ts[-1],agent=i,controller_seconds=time.perf_counter()-tick,solve=result,solve_attempts=attempts,control=U[0].copy())
            chosen.append((X,U));roundlog.append(row)
            if result['status']!=0:
                reason=f'Agent{i} MPCC failure: '+result['message'];np.savez_compressed(output/f'failed_ocp_{k}_{i}.npz',X=X,U=U,initial=snap[i],od=solver.od,weights=solver.w,mu=solver.mu,snapshot=snap)
        if reason:logs.extend(roundlog);break
        dense=np.array([[np.asarray(step(snap[i],chosen[i][1][0],FEEDBACK*j/8)).ravel() for j in range(1,9)] for i in range(2)])
        states=dense[:,-1].copy();distance=np.linalg.norm(dense[0,:,:3]-dense[1,:,:3],axis=1)
        if distance.min()<.5-1e-4:reason='Actual clearance violation'
        world=np.array(c['world_box'])
        if (np.linalg.norm(dense[:,:,10:13],axis=-1).max()>9.401 or dense[:,:,13:17].min()<-.001 or dense[:,:,13:17].max()>7.001 or
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
    summary=dict(success=reason is None and min(counts)>=target,gates_passed=counts,target_per_agent=target,duration=ts[-1],stop_reason=reason or 'completed',
        min_distance=min(dense_min) if dense_min else float(np.linalg.norm(x[0,0,:3]-x[0,1,:3])),
        controller_p50_ms=float(np.median(durations)) if durations else None,controller_p95_ms=float(np.percentile(durations,95)) if durations else None,
        frontend_seconds=sum(r['seconds'] for r in front),frontend_queries=len(front),max_speed=float(np.linalg.norm(x[:,:,3:6],axis=-1).max()),
        manual_waiting=False,expert_fallbacks=0,physical_latency_simulated=False)
    save_json(output/'summary.json',summary);save_json(output/'planning_log.json',logs);save_json(output/'events.json',events);np.savez_compressed(output/'execution.npz',time=ts,state=x)
    print('SUMMARY',summary,flush=True)

def build_parser():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--allocator',default=str(HERE/'models/allocation.pt'))
    p.set_defaults(cpc_mode='original',predictor='ca',barrier='soft',allocation_zero=False)
    p.add_argument('--laps',type=int,default=2);p.add_argument('--max-time',type=float,default=40.);p.add_argument('--horizon',type=int,default=20)
    p.add_argument('--progress',type=float,default=80.);p.add_argument('--penalty',type=float,default=100000.)
    p.add_argument('--separation',type=float,default=1.3);p.add_argument('--forward-offset',type=float,default=0.)
    p.add_argument('--solver-seconds',type=float,help='Per MPCC attempt: analytic8s, learned35s');p.add_argument('--iterations',type=int,help='Per MPCC attempt: analytic900, learned4000');p.add_argument('--cpc-seconds',type=float,default=35.);p.add_argument('--solve-retries',type=int,default=2)
    p.add_argument('--gate-progress-slope',type=float,default=1.5);p.add_argument('--target-gates',type=int)
    p.add_argument('--cbf',dest='cbf_branch',choices=['analytic','learned'],default='analytic',help='Safety branch; learned selects the validated direct-output v7 profile')
    p.add_argument('--cbf-checkpoint',dest='h_checkpoint',help='Custom direct-h checkpoint for --cbf learned')
    p.add_argument('--cbf-alpha',type=float,help='Discrete CBF rate; default3')
    p.add_argument('--gate-halfwidth',type=float,help='Tracking funnel base halfwidth: analytic0.20m, learned0.16m')
    p.add_argument('--tube-recovery',action=argparse.BooleanOptionalAction,default=None,help='Omit current tracking-tube rows; learned default enabled')
    p.add_argument('--dense-physics',action=argparse.BooleanOptionalAction,default=None,help='Check physical bounds within each applied interval; learned default enabled')
    p.add_argument('--phase-retries',type=int,help='Same-problem phase initialization retries: analytic0, learned2')
    return p

def parse_args(argv=None):
    parser=build_parser();args=parser.parse_args(argv)
    try:return resolve_safety(args)
    except ValueError as error:parser.error(str(error))

if __name__=='__main__':main(parse_args())
