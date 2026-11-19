"""Optional NLP objective for continuity with a causally shifted reference."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import casadi as ca
import numpy as np
from scripts.planning.cpc.phase_planner import GatePlanner

class ReferenceContinuityNLP:
    def __init__(self,planner,weight):
        if not np.isfinite(weight) or weight<=0:raise ValueError('Positive continuity weight required')
        self.planner=planner;self.weight=float(weight)
        self.nodes=min(planner.n+1,2*planner.c['cells_per_leg']+1)
        z=planner.nlp_definition['x'];p=planner.nlp_definition['p'];X=planner.symbols['X']
        reference=ca.MX.sym('causal_reference',3,self.nodes)
        parameters=ca.vertcat(p,ca.vec(reference))
        penalty=self.weight/self.nodes*ca.sumsqr(X[:3,:self.nodes]-reference)
        f=planner.nlp_definition['f']+penalty;g=planner.nlp_definition['g']
        sigma=ca.MX.sym('sigma');lam=ca.MX.sym('lam',g.numel())
        opts=dict(planner.solver_options)
        original_j=planner.solver.get_function('nlp_jac_g');original_h=planner.solver.get_function('nlp_hess_l')
        opts.update(grad_f=ca.Function('continuous_grad',[z,parameters],[f,ca.gradient(f,z)]),
            jac_g=ca.Function('continuous_jac',[z,parameters],list(original_j(z,p))),
            hess_lag=ca.Function('continuous_hess',[z,parameters,sigma,lam],[original_h(z,p,sigma,lam)+sigma*ca.triu(ca.hessian(penalty,z)[0])]))
        self.solver=ca.nlpsol('continuous_reference','ipopt',dict(x=z,p=parameters,f=f,g=g),opts)
        self.reference=None

    def __call__(self,**kwargs):
        kwargs['p']=np.r_[kwargs['p'],self.reference.ravel(order='F')]
        return self.solver(**kwargs)

    def stats(self):return self.solver.stats()

    def solve(self,state,gates,normals,half,durations,guess,dual_start=None):
        p=self.planner;nx=13*(p.n+1)
        self.reference=guess[:nx].reshape(13,p.n+1,order='F')[:3,:self.nodes].copy()
        original=p.solver;p.solver=self
        try:result=GatePlanner.solve(p,state,gates,normals,half,durations,warm=guess,dual_start=dual_start)
        finally:p.solver=original
        cost=self.weight/self.nodes*float(np.sum((result['X'][:3,:self.nodes]-self.reference)**2))
        result.update(reference_continuity_weight=self.weight,reference_continuity_cost=cost,
            reference_continuity_positions=self.reference.copy(),base_objective_value=result['value']-cost)
        return result
