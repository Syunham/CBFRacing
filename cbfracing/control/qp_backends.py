"""Interchangeable QP solvers; all results are checked in the original QP coordinates."""
from cbfracing.core.context import *
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.linalg import solve_triangular
from cbfracing.control.rti import RTIMPCC


class QPBackend:
    def __init__(self,kind='clarabel',condensed=False,tolerance=1e-8):
        self.kind=kind;self.condensed=condensed;self.tolerance=tolerance;self.cache={}
        self.original=object.__new__(RTIMPCC)
        self.original.qp_backend=kind;self.original.condense=False;self.original.qp=None;self.original.qp_structure=None;self.original.qp_tolerance=tolerance
        self.original.backend_dispatch_disabled=True

    def solve(self,B,q,A,lo,hi):
        begin=time.perf_counter()
        equality=np.isfinite(lo)&np.isfinite(hi)&(abs(hi-lo)<1e-12)
        nx=int(equality.sum())
        if self.condensed:
            eq=np.isfinite(lo)&np.isfinite(hi)&(abs(hi-lo)<1e-12)
            assert eq.sum()==nx
            E=A[eq,:nx].tocsc();F=A[eq,nx:].toarray()
            if os.environ.get('CONDENSE_METHOD','lu')=='triangular':
                mat=E.toarray()
                if np.count_nonzero(np.triu(mat,1)):raise ValueError('Dynamics equality matrix is not lower triangular')
                solution=solve_triangular(mat,np.c_[F,hi[eq]],lower=True,check_finite=False)
                T=np.vstack([-solution[:,:-1],np.eye(B.shape[0]-nx)])
                offset=np.r_[solution[:,-1],np.zeros(B.shape[0]-nx)]
            else:
                factor=splu(E)
                T=np.vstack([-factor.solve(F),np.eye(B.shape[0]-nx)])
                offset=np.r_[factor.solve(hi[eq]),np.zeros(B.shape[0]-nx)]
            H=T.T@(B@T);H=(H+H.T)*.5
            gr=T.T@(B@offset+q);Ar=A[~eq]@T;shift=A[~eq]@offset
            condense_seconds=time.perf_counter()-begin
            if self.kind in ('daqp_direct','daqp_warm') and os.environ.get('FAST_DENSE_QP','0')=='1':
                delta,info=self._daqp_dense(H,gr,Ar,lo[~eq]-shift,hi[~eq]-shift)
            else:delta,info=self._solve(sparse.csc_matrix(H),gr,sparse.csc_matrix(Ar),lo[~eq]-shift,hi[~eq]-shift)
            info['condense_seconds']=condense_seconds
            x=T@delta+offset
        else:x,info=self._solve(B,q,A,lo,hi)
        info['total_seconds']=time.perf_counter()-begin
        if np.isfinite(x).all():
            v=A@x;info['original_residual']=float(max(np.maximum(lo-v,0).max(),np.maximum(v-hi,0).max()))
            info['objective']=float(.5*x@(B@x)+q@x)
            info['success']=bool(info['success'] and info['original_residual']<1e-4)
        else:info.update(success=False,original_residual=float('inf'),objective=float('inf'))
        return x,info

    def _solve(self,B,q,A,lo,hi):
        if self.kind=='piqp_direct':
            import piqp
            eq=np.isfinite(lo)&np.isfinite(hi)&(abs(hi-lo)<1e-12)
            E=A[eq].tocsc();G=A[~eq].tocsc();P=sparse.triu(B,format='csc')
            signature=tuple((m.shape,m.indptr.tobytes(),m.indices.tobytes()) for m in (P,E,G))
            data=dict(P=P,c=q,A=E,b=hi[eq],G=G,h_l=lo[~eq],h_u=hi[~eq])
            tick=time.perf_counter()
            if signature not in self.cache:
                solver=piqp.SparseSolver();solver.settings.eps_abs=self.tolerance;solver.settings.eps_rel=self.tolerance
                solver.settings.eps_duality_gap_abs=self.tolerance;solver.settings.eps_duality_gap_rel=self.tolerance
                solver.settings.max_iter=100;solver.settings.compute_timings=True;solver.setup(**data);self.cache[signature]=solver
            else:solver=self.cache[signature];solver.update(**data)
            status=solver.solve();info=solver.result.info
            return np.asarray(solver.result.x).copy(),dict(success=str(status)=='Status.PIQP_SOLVED',status=str(status),iterations=int(info.iter),backend_seconds=time.perf_counter()-tick)
        if self.kind in ('daqp_direct','daqp_warm'):
            return self._daqp_dense(B.toarray(),q,A.toarray(),lo,hi)
        if self.kind in ('clarabel','osqp'):
            return RTIMPCC.qp_solve(self.original,B,q,A,lo,hi)
        dense=self.kind in ('qpoases','daqp')
        H=ca.DM(B.toarray() if dense else B);Ac=ca.DM(A.toarray() if dense else A)
        signature=(H.sparsity().serialize(),Ac.sparsity().serialize())
        if signature not in self.cache:
            opts=dict(error_on_fail=False,print_time=False)
            if self.kind=='qpoases':opts.update(printLevel='none',nWSR=500,terminationTolerance=self.tolerance,enableEqualities=True)
            elif self.kind=='proxqp':opts['proxqp']=dict(eps_abs=self.tolerance,eps_rel=self.tolerance,max_iter=1000,verbose=False)
            elif self.kind=='piqp':opts['piqp']=dict(eps_abs=self.tolerance,eps_rel=self.tolerance,max_iter=100,verbose=False)
            elif self.kind=='daqp':opts['daqp']=dict(primal_tol=self.tolerance,dual_tol=self.tolerance,iter_limit=1000)
            self.cache[signature]=[ca.conic('backend_'+self.kind,self.kind,dict(h=H.sparsity(),a=Ac.sparsity()),opts),None]
        solver,last=self.cache[signature]
        kw=dict(h=H,g=q,a=Ac,lba=lo,uba=hi)
        if last is not None and self.kind in ('proxqp','qpoases','daqp'):
            kw.update(x0=last['x'],lam_a0=last['lam_a'],lam_x0=last['lam_x'])
        tick=time.perf_counter();answer=solver(**kw);elapsed=time.perf_counter()-tick
        st=solver.stats();self.cache[signature][1]=answer
        return np.asarray(answer['x']).ravel(),dict(success=bool(st.get('success',False)),status=str(st.get('return_status',st.get('unified_return_status','unknown'))),
            iterations=int(st.get('iter_count',0)),backend_seconds=elapsed)

    def _daqp_dense(self,H,q,A,lo,hi):
        import daqp
        eq=np.isfinite(lo)&np.isfinite(hi)&(abs(hi-lo)<1e-12)
        sense=np.where(eq,5,0).astype(np.int32)
        kw={};cache_key=(H.shape,A.shape)
        if self.kind=='daqp_warm' and cache_key in self.cache:kw['dual_start']=self.cache[cache_key]
        tick=time.perf_counter();x,obj,status,info=daqp.solve(np.ascontiguousarray(H),q,np.ascontiguousarray(A),hi,lo,sense,primal_tol=self.tolerance,dual_tol=self.tolerance,iter_limit=1000,**kw)
        self.cache[cache_key]=info['lam']
        return x,dict(success=status==1,status=str(status),iterations=int(info['iterations']),backend_seconds=time.perf_counter()-tick,daqp_setup_seconds=float(info['setup_time']),daqp_solve_seconds=float(info['solve_time']))


def read_sample(path):
    d=np.load(path)
    matrices={name:sparse.csc_matrix((d[name+'_data'],d[name+'_indices'],d[name+'_indptr']),shape=d[name+'_shape']) for name in ['A','B']}
    return d,matrices
