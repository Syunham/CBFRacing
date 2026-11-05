"""Rolling physical-gate preview with learned aperture allocations."""
from cbfracing.core.context import *
from cbfracing.networks.allocation import observation,load_allocator
from cbfracing.planning.phase_planner import GatePlanner
from cbfracing.planning.reference import reference_from_plan
from cbfracing.planning.original_cpc import OriginalCPC
from cbfracing.networks.timing import ArrivalTimeNetwork

class LocalPlanner:
    def __init__(self,checkpoint,seconds=35.,phase_retries=0):
        self.c=configuration();self.c.update(cpc_wall_time=seconds,cpc_max_iter=1600,cpc_epsilon_schedule=[1e-3,1e-5,1e-6])
        self.c['phase_solve_retries']=phase_retries
        self.course=course();self.net=load_allocator(checkpoint)
        self.event_net=None;self.event_checkpoint=os.environ.get('EVENT_CHECKPOINT','')
        if self.event_checkpoint:
            from cbfracing.networks.event_supervised import load_event
            self.event_net,self.event_metadata=load_event(self.event_checkpoint)
        self.minimum_snap=os.environ.get('MINIMUM_SNAP','0')=='1'
        self.scp_enabled=False
        self.rate_backend=''
        if self.minimum_snap:
            from cbfracing.planning.minimum_snap import MinimumSnapPlanner
            self.phase=MinimumSnapPlanner(self.c,self.course)
        else:
            self.phase=GatePlanner(self.c,self.course,self.event_net is None)
        self.full=None;self.previous_plans={};self.passage_memory={}
        self.timing=ArrivalTimeNetwork(torch.load(HERE/'models/time_initializer.pt',map_location='cpu',weights_only=False));self.timing.eval()
    def solve(self,own,other,index):
        start=time.perf_counter();M=self.c.get('preview_gates',2);g,n,half,a=gate_pair(index,M);targets=g.copy();alloc=[];raw_alloc=[]
        memory_key=getattr(self,'active_agent',0);memory=self.passage_memory.get(memory_key,{})
        update_limit=float(os.getenv('OPT_PASSAGE_UPDATE_LIMIT','0'))
        for j in range(M):
            g2,n2,_,_=gate_pair(index+j+1)
            z=observation(own,other,g[j],n[j],a[j],g2[0],n2[0])
            with torch.no_grad():offset=float(self.net(torch.tensor(z[None]))) if self.net is not None else 0.
            offset*=float(os.environ.get('OPT_POINT_SCALE','1'))
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
            local_half=np.minimum(local_half,np.c_[.70-np.abs(alloc),np.full(M,.70)])
        if np.any(local_half<=self.c['aperture_reserve']):raise ValueError('Passage box must retain positive aperture after reserve')
        network_start=time.perf_counter()
        if self.event_net is not None:
            from cbfracing.networks.event_supervised import predict
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
                event_network_seconds=event_network_seconds,event_policy='network_fixed_polynomial',event_durations_input=durations.copy(),
                event_checkpoint=self.event_checkpoint,frontend_solver='clarabel_minimum_snap')
            if self.event_net is not None:plan['event_feature_half']=event_half.copy()
            return plan
        warm=self.phase.seed(np.asarray(own[:13]),targets,n,durations)
        self.last_seed_diagnostics=dict(original_nonfinite=int((~np.isfinite(warm)).sum()),fallback_used=False)
        if not np.isfinite(warm).all():
            from cbfracing.planning.warm_seed import finite_seed
            warm=finite_seed(np.asarray(own[:13]),targets,n,durations,self.c)
            self.last_seed_diagnostics['fallback_used']=True
        key=getattr(self,'active_agent',0)
        phase=self.phase.solve(np.asarray(own[:13]),targets,n,local_half,durations,warm=warm)
        if os.environ.get('CPC_FRONTEND','original')=='phase':
            plan=phase
        else:
            if self.full is None:self.full=OriginalCPC(self.c,self.course)
            plan=self.full.solve(np.asarray(own[:13]),targets,n,local_half,phase=phase)
        audit,t,x=self.phase.audit(plan,np.asarray(own[:13]))
        if not audit['passed']:raise RuntimeError('CPC reference audit: '+json.dumps(audit))
        self.previous_plans[key]=(index,plan)
        plan.update(audit=audit,reference_time=t,reference_states=x,allocation=alloc,physical_gates=g,half=local_half,
            network_arrival_time_prior=arrival,wall_seconds=time.perf_counter()-start,extra_waypoints=0,gate_event_count=M,seed_diagnostics=self.last_seed_diagnostics,phase_solve_trace=phase.get('solve_trace',[]),
            event_network_seconds=event_network_seconds,event_policy=('network_initialized_free' if self.phase.optimize_durations else 'network_fixed') if self.event_net is not None else 'numerical_free',event_durations_input=durations.copy(),event_checkpoint=self.event_checkpoint,frontend_solver='bodyrate_'+self.rate_backend if self.rate_backend else ('scp' if self.scp_enabled else 'ipopt'))
        return plan

def reference(plan,initial):
    ref=reference_from_plan(plan,initial)
    ref['physical_next_gate']=plan['physical_gates'][0]
    return ref

def save(path,plan):
    arrays={k:v for k,v in plan.items() if isinstance(v,np.ndarray)}
    arrays['allocation']=np.array(plan['allocation']);np.savez_compressed(path,**arrays)
