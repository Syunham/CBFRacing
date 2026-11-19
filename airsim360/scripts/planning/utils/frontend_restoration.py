"""Elastic intermediate QP; never a replacement for original trajectory checks."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import time
import numpy as np
from scipy import sparse
import clarabel

def restore(A,lower,upper,nphysical,ninitial,penalty=1000.):
    start=time.perf_counter();nz=A.shape[1]
    # Initial state and trust/variable bounds stay hard. All remaining original
    # rows get a shared temporary slack, driven to zero by the objective.
    hard=np.r_[np.ones(ninitial,dtype=bool),np.zeros(nphysical-ninitial,dtype=bool),np.ones(len(lower)-nphysical,dtype=bool)]
    eq=hard&np.isfinite(lower)&np.isfinite(upper)&(abs(upper-lower)<1e-12)
    up=~eq&np.isfinite(upper);dn=~eq&np.isfinite(lower);soft=(~hard).astype(float)
    matrix=sparse.vstack([sparse.hstack([A[eq],sparse.csc_matrix((eq.sum(),1))]),
        sparse.hstack([A[up],-soft[up,None]]),sparse.hstack([-A[dn],-soft[dn,None]]),
        sparse.csc_matrix(([-1.],([0],[nz])),shape=(1,nz+1))],format='csc')
    bound=np.r_[upper[eq],upper[up],-lower[dn],0.]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=100
    settings.tol_feas=1e-8;settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-8
    solver=clarabel.DefaultSolver(sparse.diags(np.r_[np.ones(nz),1e-6],format='csc'),np.r_[np.zeros(nz),penalty],matrix,bound,
        [clarabel.ZeroConeT(int(eq.sum())),clarabel.NonnegativeConeT(int(up.sum()+dn.sum()+1))],settings)
    setup=time.perf_counter()-start;start=time.perf_counter();result=solver.solve();seconds=time.perf_counter()-start
    z=np.asarray(result.x);v=matrix@z-bound;ne=int(eq.sum())
    residual=max(float(np.max(abs(v[:ne]),initial=0.)),float(np.max(v[ne:],initial=0.)))
    success=str(result.status) in ('Solved','AlmostSolved') and np.isfinite(z).all() and residual<1e-4
    info=dict(native_status=str(result.status),iterations=int(result.iterations),setup_time=setup,solve_time=seconds,
              elastic_slack=float(z[-1]),elastic_qp_residual=residual,success=success)
    return z[:-1],info
