"""Bounded one-gate physical optimization used only to initialize a new tail."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import time
import numpy as np
from scripts.planning.cpc.phase_planner import GatePlanner
from scripts.planning.fast_fly.warm_seed import finite_seed

class OptimizedTailSeed:
    def __init__(self,config,course):
        c=dict(config,preview_gates=1,cells_per_leg=config['cells_per_leg'],fast_fly_phase=True,
               phase_solve_retries=0,phase_batched_audit=True,phase_online_sqp=False,
               phase_ipopt_options=dict(max_wall_time=config.get('phase_tail_solve_seconds',.3),max_iter=120),
               seed_end_speed_max=15.,seed_preserve_heading=True)
        for key in ['phase_native_library','phase_hessian_library','phase_jacobian_library']:c.pop(key,None)
        self.planner=GatePlanner(c,course,True)

    def solve(self,state,gates,normals,half,duration,forces):
        tick=time.perf_counter();p=self.planner;p.initial_forces=np.asarray(forces).copy()
        seed=finite_seed(state,gates,normals,duration,p.c)
        result=None;trace=dict(stage='tail_seed',success=False,iterations=0)
        try:
            candidate=p.solve(state,gates,normals,half,duration,warm=seed)
            audit,_,_=p.audit(candidate,state)
            trace.update(audit=audit,iterations=candidate['iterations'],trace=candidate['solve_trace'])
            if audit['passed']:result=candidate;trace['success']=True
        except RuntimeError as error:
            trace.update(error=str(error),trace=p.last_trace,iterations=sum(v['iterations'] for v in p.last_trace))
        trace['seconds']=time.perf_counter()-tick
        return result,trace
