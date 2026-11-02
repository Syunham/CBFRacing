"""Online two-physical-gate CPC with learned gate aperture allocations."""
from context import *
from allocation import observation,load_allocator
from phase_planner import GatePlanner
from reference import reference_from_plan
from original_cpc import OriginalCPC
from timing import ArrivalTimeNetwork

class LocalPlanner:
    def __init__(self,checkpoint,seconds=35.,phase_retries=0):
        self.c=configuration();self.c.update(cpc_wall_time=seconds,cpc_max_iter=1600,cpc_epsilon_schedule=[1e-3,1e-5,1e-6])
        self.c['phase_solve_retries']=phase_retries
        self.course=course();self.net=load_allocator(checkpoint)
        self.phase=GatePlanner(self.c,self.course,True);self.full=None
        self.timing=ArrivalTimeNetwork(torch.load(HERE/'models/time_initializer.pt',map_location='cpu',weights_only=False));self.timing.eval()
    def solve(self,own,other,index):
        start=time.perf_counter();g,n,half,a=gate_pair(index);targets=g.copy();alloc=[]
        for j in range(2):
            g2,n2,_,_=gate_pair(index+j+1)
            z=observation(own,other,g[j],n[j],a[j],g2[0],n2[0])
            with torch.no_grad():offset=float(self.net(torch.tensor(z[None]))) if self.net is not None else 0.
            targets[j]+=offset*a[j];alloc.append(offset)
        # Existing physical gate events are constrained around the predicted
        # passage point. The original aperture is retained for final scoring.
        local_half=np.full((2,2),.12) if self.net is not None else half
        with torch.no_grad():arrival=self.timing(*[torch.tensor(v,dtype=torch.float64) for v in [own[:13],targets,n,local_half]]).numpy()
        durations=np.clip(np.diff(np.r_[0,arrival]),.5,7.)
        warm=self.phase.seed(np.asarray(own[:13]),targets,n,durations)
        self.last_seed_diagnostics=dict(original_nonfinite=int((~np.isfinite(warm)).sum()),fallback_used=False)
        if not np.isfinite(warm).all():
            from warm_seed import finite_seed
            warm=finite_seed(np.asarray(own[:13]),targets,n,durations,self.c)
            self.last_seed_diagnostics['fallback_used']=True
        phase=self.phase.solve(np.asarray(own[:13]),targets,n,local_half,durations,warm=warm)
        if self.full is None:self.full=OriginalCPC(self.c,self.course)
        plan=self.full.solve(np.asarray(own[:13]),targets,n,local_half,phase=phase)
        audit,t,x=self.phase.audit(plan,np.asarray(own[:13]))
        if not audit['passed']:raise RuntimeError('CPC reference audit: '+json.dumps(audit))
        plan.update(audit=audit,reference_time=t,reference_states=x,allocation=alloc,physical_gates=g,half=local_half,
            network_arrival_time_prior=arrival,wall_seconds=time.perf_counter()-start,extra_waypoints=0,gate_event_count=2,seed_diagnostics=self.last_seed_diagnostics,phase_solve_trace=phase.get('solve_trace',[]))
        return plan

def reference(plan,initial):
    ref=reference_from_plan(plan,initial)
    ref['physical_next_gate']=plan['physical_gates'][0]
    return ref

def save(path,plan):
    arrays={k:v for k,v in plan.items() if isinstance(v,np.ndarray)}
    arrays['allocation']=np.array(plan['allocation']);np.savez_compressed(path,**arrays)
