"""Solve a working subset, adding violated rows and checking the full QP."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import time
import numpy as np
from daqp import solve as dense_solve

def solve(H,g,A,upper,lower,sense,initial_rows=512,max_rounds=4,**options):
    start=time.perf_counter();m=len(lower)
    norm=np.maximum(np.linalg.norm(A,axis=1),1e-12)
    margin=np.minimum(upper,-lower)/norm
    equal=(sense & 1)!=0
    selected=np.zeros(m,dtype=bool)
    selected[np.argsort(margin,kind='stable')[:min(initial_rows,m)]]=True
    selected[equal | (lower>0) | (upper<0)]=True
    setup=0.;qp_time=0.;iterations=0;rounds=[]
    for _ in range(max_rounds):
        ids=np.flatnonzero(selected);opts=dict(options)
        if 'dual_start' in opts:opts['dual_start']=opts['dual_start'][ids]
        x,obj,code,info=dense_solve(H,g,np.ascontiguousarray(A[ids]),upper[ids],lower[ids],sense[ids],**opts)
        setup+=info['setup_time'];qp_time+=info['solve_time'];iterations+=info['iterations']
        if code!=1 or not np.isfinite(x).all():
            rounds.append(dict(rows=len(ids),code=int(code)));break
        ax=A@x;violation=np.maximum(lower-ax,ax-upper)
        error=float(max(0.,violation.max()))
        rounds.append(dict(rows=len(ids),code=int(code),full_constraint_error=error))
        if error<=1e-7:
            lam=np.zeros(m);lam[ids]=info['lam']
            info=dict(info,lam=lam,setup_time=setup,solve_time=qp_time,iterations=iterations,
                      constraint_generation=dict(rounds=rounds,full_fallback=False,seconds=time.perf_counter()-start))
            return x,obj,code,info
        new=(violation>1e-7)&~selected
        if not np.any(new):break
        selected[new]=True
    x,obj,code,info=dense_solve(H,g,A,upper,lower,sense,**options)
    info=dict(info,setup_time=setup+info['setup_time'],solve_time=qp_time+info['solve_time'],iterations=iterations+info['iterations'],
              constraint_generation=dict(rounds=rounds,full_fallback=True,seconds=time.perf_counter()-start))
    return x,obj,code,info
