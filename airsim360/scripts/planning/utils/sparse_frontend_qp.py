"""Sparse frontend QP; original-coordinate residuals are checked by the caller."""

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

class SparseFrontendQP:
    def __init__(self):
        self.solver=None;self.structure=None

    def solve(self,B,gradient,A,lower,upper,trust):
        tick=time.perf_counter()
        eq=np.isfinite(lower)&np.isfinite(upper)&(abs(upper-lower)<1e-12)
        reach=trust*np.asarray(abs(A).sum(axis=1)).ravel()
        keep=eq|(lower>=-reach-1e-10)|(upper<=reach+1e-10)
        up=keep&~eq&np.isfinite(upper);down=keep&~eq&np.isfinite(lower)
        matrix=sparse.vstack([A[eq],A[up],-A[down]],format='csc')
        bound=np.r_[upper[eq],upper[up],-lower[down]];P=sparse.triu(B,format='csc')
        signature=tuple((m.shape,m.indptr.tobytes(),m.indices.tobytes()) for m in [P,matrix])
        if self.solver is None or signature!=self.structure:
            settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=100
            settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-8;settings.tol_feas=1e-8
            self.solver=clarabel.DefaultSolver(P,gradient,matrix,bound,
                [clarabel.ZeroConeT(int(eq.sum())),clarabel.NonnegativeConeT(int(up.sum()+down.sum()))],settings)
            self.structure=signature
        else:self.solver.update(P=P,q=gradient,A=matrix,b=bound)
        setup=time.perf_counter()-tick;tick=time.perf_counter();result=self.solver.solve();elapsed=time.perf_counter()-tick
        ids=np.flatnonzero(keep);multipliers=np.zeros(A.shape[0]);dual=np.asarray(result.z)
        ne=int(eq.sum());nu=int(up.sum());multipliers[eq]=dual[:ne]
        multipliers[up]+=dual[ne:ne+nu];multipliers[down]-=dual[ne+nu:]
        info=dict(iterations=int(result.iterations),setup_time=setup,solve_time=elapsed,lam=multipliers[ids],
                  backend='sparse_clarabel',native_status=str(result.status))
        return np.asarray(result.x),1 if str(result.status) in ('Solved','AlmostSolved') else -99,info,ids
