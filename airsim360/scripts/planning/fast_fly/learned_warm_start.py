"""Bounded numerical solve initialized by a learned primal trajectory."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import time
import numpy as np
import torch
from scripts.planning.fast_fly.frontend_initializer import load,encode_features,decode_target
from scripts.planning.cpc.phase_planner import GatePlanner

class LearnedWarmStart:
    def __init__(self,outer,path,budget=3.):
        if outer.M!=5 or outer.n!=60 or not outer.optimize_durations:
            raise ValueError('Learned primal initializer requires five gates and twelve cells per gate with free durations')
        self.model,self.checkpoint=load(path)
        c=dict(outer.c,phase_solve_retries=0,phase_ipopt_options=dict(max_wall_time=budget,max_iter=150))
        self.planner=GatePlanner(c,outer.course,True)
        if c.get('phase_learned_sqp',False):
            from scripts.planning.fast_fly.fast_fly_sqp import FeasibleFrontendSQP
            self.sqp=FeasibleFrontendSQP(self.planner)

    def solve(self,outer,state,gates,normals,half,durations):
        start=time.perf_counter();p=self.planner;p.initial_forces=np.asarray(outer.initial_forces).copy()
        p.lbx=outer.lbx.copy();p.ubx=outer.ubx.copy();p.lbg=outer.lbg.copy();p.ubg=outer.ubg.copy()
        tick=time.perf_counter()
        features=encode_features(state,p.initial_forces,gates,normals,half,p.initial_force_weight,p.c.get('phase_thrust_smooth',0.))
        with torch.no_grad():target=self.model(torch.tensor(features,dtype=torch.float32)[None])[0].numpy()
        guess=decode_target(target,state,p.lbx[-5:],p.ubx[-5:])
        trace=dict(stage='learned_seed_attempt',iterations=0,inference_seconds=time.perf_counter()-tick,success=False)
        plan=None
        if hasattr(self,'sqp'):
            p.c['phase_sqp_trust']=outer.c.get('phase_sqp_trust',.5)
            p.c['phase_sqp_objective_steps']=outer.c.get('phase_sqp_objective_steps',1)
            trace['requested_objective_steps']=p.c['phase_sqp_objective_steps']
            plan,sqp_trace=self.sqp.solve(state,gates,normals,half,guess)
            trace.update(sqp_trace=sqp_trace,iterations=sqp_trace['iterations'])
            if plan is not None:
                trace.update(success=True,learned_update='sqp',nlp_iterations=0,audit_passed=True,seconds=time.perf_counter()-start)
                return plan,trace
        try:
            candidate=p.solve(state,gates,normals,half,durations,warm=guess)
            audit,tt,xx=p.audit(candidate,state)
            trace.update(nlp_iterations=candidate['iterations'],audit_passed=audit['passed'],learned_update='nlp')
            if audit['passed']:
                candidate.update(audit=audit,reference_time=tt,reference_states=xx,audited_initial_state=np.asarray(state).copy())
                plan=candidate;trace['success']=True
            else:trace.update(iterations=trace['iterations']+candidate['iterations'],failed_nlp_trace=candidate['solve_trace'])
        except RuntimeError as error:
            trace.update(error=str(error),failed_nlp_trace=p.last_trace,iterations=trace['iterations']+sum(r['iterations'] for r in p.last_trace))
        trace['seconds']=time.perf_counter()-start
        return plan,trace
