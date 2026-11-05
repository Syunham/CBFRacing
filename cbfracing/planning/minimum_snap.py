"""Fixed-time convex Bernstein planning; a kinematic reference, not a rotor certificate."""
import math, os, time
import numpy as np
import scipy.sparse as sp
from scipy.linalg import null_space,lstsq
from scipy.spatial.transform import Rotation
import clarabel
from cbfracing.core.model import MODEL, model_functions


def basis(t, derivative=0):
    n=7-derivative
    b=np.array([math.comb(n,k)*t**k*(1-t)**(n-k) for k in range(n+1)])
    return math.factorial(7)/math.factorial(n)*b@np.diff(np.eye(8),n=derivative,axis=0)


class MinimumSnapPlanner:
    def __init__(self,config,course):
        self.c=config;self.course=course;self.optimize_durations=False
        self.rhs,_=model_functions(substeps=1)
        self.basis_cache={};self.row_cache={}

    def solve(self,own,gates,normals,half,durations):
        start=time.perf_counter();event_T=np.array(durations)*float(os.getenv('SNAP_TIME_SCALE','1'))
        short_scale=float(os.getenv('SNAP_SHORT_TIME_SCALE','1'))
        if short_scale<=0:raise ValueError('Short-segment time scale must be positive')
        segment_distances=np.linalg.norm(np.diff(np.vstack([own[:3],gates]),axis=0),axis=1)
        event_T*=np.where(segment_distances<3.,short_scale,1.)
        pieces=int(os.getenv('SNAP_PIECES','1'))
        if pieces<1 or pieces>4:raise ValueError('SNAP_PIECES must be1..4')
        T=np.repeat(event_T/pieces,pieces)
        M=len(T);ncoeff=M*24;soft=os.getenv('SNAP_SOFT_ENVELOPE','0')=='1';nv=ncoeff+(M if soft else 0);eq=[];beq=[];ineq=[];bineq=[];soc=[];bsoc=[]
        Q=np.zeros((nv,nv));D=np.diff(np.eye(8),n=4,axis=0)
        G=np.array([[math.comb(3,k)*math.comb(3,l)/(7*math.comb(6,k+l)) for l in range(4)] for k in range(4)])
        def row(j,t,d=0):
            key=(nv,j,float(t),d)
            if key not in self.row_cache:
                out=np.zeros((3,nv));out[:,j*24:(j+1)*24]=np.kron(basis(t,d)[None,:],np.eye(3));self.row_cache[key]=out
            return self.row_cache[key]/T[j]**d
        def equal(A,b):eq.extend(np.atleast_2d(A));beq.extend(np.atleast_1d(b))
        def upper(A,b):ineq.extend(np.atleast_2d(A));bineq.extend(np.atleast_1d(b))
        for j in range(M):Q[j*24:(j+1)*24,j*24:(j+1)*24]=np.kron(840**2/T[j]**7*(D.T@G@D),np.eye(3))
        Q/=np.max(np.diag(Q));Q+=np.eye(nv)*float(os.getenv('SNAP_REGULARIZATION','1e-10'))
        path_weight=float(os.getenv('SNAP_PATH_ENERGY_WEIGHT','0'))
        if path_weight<0:raise ValueError('Path energy weight must be nonnegative')
        if path_weight:
            # Integral ||p_dot||^2 is a convex surrogate, not exact arc length.
            D1=np.diff(np.eye(8),axis=0)
            G6=np.array([[math.comb(6,k)*math.comb(6,l)/(13*math.comb(12,k+l)) for l in range(7)] for k in range(7)])
            Qv=np.zeros_like(Q)
            for j in range(M):Qv[j*24:(j+1)*24,j*24:(j+1)*24]=np.kron(49/T[j]*(D1.T@G6@D1),np.eye(3))
            Q+=path_weight*Qv/np.max(np.diag(Qv))
        if soft:Q[ncoeff:,ncoeff:]=np.eye(M)*float(os.getenv('SNAP_SLACK_WEIGHT','10'))
        dx=np.array(self.rhs(own,np.zeros(5))).ravel()
        for d,value in enumerate([own[:3],own[3:6],dx[3:6]]):equal(row(0,0,d),value)
        initial_R=Rotation.from_quat(np.r_[own[7:10],own[6]]).as_matrix()
        if os.getenv('SNAP_INITIAL_JERK','1')=='1':
            eps=1e-5
            jp=(np.array(self.rhs(own+eps*dx,np.zeros(5))).ravel()[3:6]-np.array(self.rhs(own-eps*dx,np.zeros(5))).ravel()[3:6])/(2*eps)
            equal(initial_R[:,:2].T@row(0,0,3),initial_R[:,:2].T@jp)
        box=np.array(self.course['world_box']);axes=[];eta=float(os.getenv('SNAP_ETA','.95'))
        thrust_soc=os.getenv('SNAP_THRUST_SOC','1')=='1';nodes=int(os.getenv('SNAP_NODES','17'))
        for j in range(M):
            if soft:
                slack_row=np.zeros(nv);slack_row[ncoeff+j]=-1;upper(slack_row,0.)
            if j:
                for d in range(4):equal(row(j,0,d)-row(j-1,1,d),np.zeros(3))
            if (j+1)%pieces==0:
                event=j//pieces;n=normals[event];axis=np.cross([0.,0.,1.],n);axis/=np.linalg.norm(axis);axes.append(axis)
                equal(n@row(j,1),n@gates[event]);upper(-n@row(j,1,1),-.8)
                for k,a in enumerate([axis,np.array([0.,0.,1.])]):
                    width=half[event,k]-self.c['aperture_reserve']
                    upper(a@row(j,1),a@gates[event]+width);upper(-a@row(j,1),-a@gates[event]+width)
            for tau in np.linspace(0,1,nodes):
                B=row(j,tau);V=row(j,tau,1);A=row(j,tau,2)
                upper(B,box[3:]);upper(-B,-box[:3])
                if thrust_soc and not (j==0 and tau==0):
                    # Frozen drag approximation gives a convex reference envelope.
                    # Exact body-frame drag is checked during reconstruction and actual execution.
                    drag_matrix=initial_R@np.diag(MODEL['drag'])@initial_R.T if os.getenv('SNAP_DRAG_FRAME','isotropic')=='initial' else float(os.getenv('SNAP_DRAG','.42'))*np.eye(3)
                    F=MODEL['mass']*A+drag_matrix@V
                    cone_top=np.zeros(nv)
                    if soft:cone_top[ncoeff+j]=-1.
                    soc.append(np.vstack([cone_top,-F]));bsoc.append(np.r_[eta*28.,0.,0.,MODEL['mass']*MODEL['gravity']])
        matrix=np.vstack([eq,ineq,*soc]);bounds=np.r_[beq,bineq,*bsoc]
        if os.getenv('SNAP_SCALE_ROWS','0')=='1':
            linear=len(beq)+len(bineq);scales=np.ones(len(bounds));scales[:linear]=np.maximum(np.linalg.norm(matrix[:linear],axis=1),1e-10)
            for k in range(len(soc)):
                sl=slice(linear+4*k,linear+4*k+4);scales[sl]=max(1.,np.linalg.norm(matrix[sl],axis=1).max())
            matrix=matrix/scales[:,None];bounds=bounds/scales
        cones=[clarabel.ZeroConeT(len(beq)),clarabel.NonnegativeConeT(len(bineq))]+[clarabel.SecondOrderConeT(4) for _ in soc]
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=150
        settings.tol_gap_abs=1e-8;settings.tol_feas=1e-8;settings.tol_gap_rel=1e-8
        settings.max_threads=1
        if os.getenv('SNAP_SCALE_ROWS','0')=='1':settings.tol_feas=1e-10
        if os.getenv('SNAP_LINEAR_SOLVER'):settings.direct_solve_method=os.environ['SNAP_LINEAR_SOLVER']
        assembly_seconds=time.perf_counter()-start;tick=time.perf_counter()
        eliminate=os.getenv('SNAP_ELIMINATE_EQUALITIES','0')=='1';null=None;part=None
        if eliminate:
            E=np.array(eq);scale=np.maximum(np.linalg.norm(E,axis=1),1e-12);E=E/scale[:,None];b=np.array(beq)/scale
            part=lstsq(E,b,lapack_driver='gelsy')[0];null=null_space(E)
            original_matrix=matrix[len(beq):];bounds=bounds[len(beq):]-original_matrix@part;matrix=original_matrix@null
            q=null.T@Q@part;P=null.T@Q@null;P=(P+P.T)*.5;cones=cones[1:]
        else:P=Q;q=np.zeros(nv)
        solver=clarabel.DefaultSolver(sp.triu(sp.csc_matrix(P),format='csc'),q,sp.csc_matrix(matrix),bounds,cones,settings)
        result=solver.solve();solve_seconds=time.perf_counter()-tick
        self.last_trace=[dict(status=str(result.status),iterations=result.iterations,solve_seconds=solve_seconds,
            primal_residual=float(result.r_prim),dual_residual=float(result.r_dual),objective=float(result.obj_val),dual_objective=float(result.obj_val_dual))]
        if str(result.status) not in ('Solved','AlmostSolved'):self.last_trace[0]['last_primal_iterate']=list(result.x)
        if str(result.status) not in ('Solved','AlmostSolved'):raise RuntimeError(f'Minimum snap {result.status}; times={T.tolist()}')
        z=np.array(result.x);z=part+null@z if eliminate else z
        coeff=z[:ncoeff].reshape(M,8,3);ts=[];ps=[];vs=[];acc=[];jerks=[]
        for j in range(M):
            tau=np.linspace(0,1,65)[0 if j==0 else 1:]
            ts.extend(sum(T[:j])+tau*T[j])
            for d,arr in enumerate([ps,vs,acc,jerks]):
                key=(j==0,d)
                if key not in self.basis_cache:self.basis_cache[key]=np.array([basis(t,d) for t in tau])
                arr.extend(self.basis_cache[key]@coeff[j]/T[j]**d)
        tt=np.array(ts);p=np.array(ps);v=np.array(vs);a=np.array(acc);jerk=np.array(jerks)
        kinematics_only=os.getenv('SNAP_KINEMATICS_ONLY','0')=='1'
        if kinematics_only:
            states=np.c_[p,v];states[0]=own[:6];forces=None;w=None;rotor_seed='not_generated'
        else:
            R=initial_R;qs=[];forces=[];drag=np.diag(MODEL['drag'])
            if os.getenv('SNAP_FRAME','fixed_heading')=='fixed_heading':
                R=np.tile(initial_R,(len(v),1,1))
                for _ in range(12):
                    force=MODEL['mass']*(a+[0,0,MODEL['gravity']])+np.einsum('nij,j,nkj,nk->ni',R,np.array(MODEL['drag']),R,v)
                    bz=force/np.linalg.norm(force,axis=1)[:,None]
                    bx=initial_R[:,0]-bz*(bz@initial_R[:,0])[:,None]
                    bad=np.linalg.norm(bx,axis=1)<1e-5
                    bx[bad]=initial_R[:,1]-bz[bad]*(bz[bad]@initial_R[:,1])[:,None]
                    bx/=np.linalg.norm(bx,axis=1)[:,None];R=np.stack([bx,np.cross(bz,bx),bz],axis=2)
                qr=Rotation.from_matrix(R).as_quat();q=np.c_[qr[:,3],qr[:,:3]]
                for k in range(1,len(q)):
                    if q[k]@q[k-1]<0:q[k]*=-1
                forces=np.repeat((np.linalg.norm(force,axis=1)/4)[:,None],4,axis=1)
            else:
                for vi,ai in zip(v,a):
                    for _ in range(8):
                        old_z=R[:,2].copy()
                        force=MODEL['mass']*(ai+[0,0,MODEL['gravity']])+R@drag@R.T@vi
                        bz=force/np.linalg.norm(force);bx=R[:,0]-bz*(R[:,0]@bz)
                        if np.linalg.norm(bx)<1e-7:bx=R[:,1]-bz*(R[:,1]@bz)
                        bx/=np.linalg.norm(bx);R=np.column_stack([bx,np.cross(bz,bx),bz])
                        if np.linalg.norm(bz-old_z)<1e-8:break
                    qr=Rotation.from_matrix(R).as_quat();q=np.r_[qr[3],qr[:3]]
                    if qs and q@qs[-1]<0:q=-q
                    qs.append(q);forces.append(np.full(4,np.linalg.norm(force)/4))
                q=np.array(qs)
            dq=np.gradient(q,tt,axis=0);w=2*(q[:,:1]*dq[:,1:]-dq[:,:1]*q[:,1:]-np.cross(q[:,1:],dq[:,1:]))
            rotor_seed=os.getenv('SNAP_ROTOR_SEED','uniform')
            if rotor_seed=='differential':
                # Initialization only: reconstruct torque from the polynomial attitude.
                # The QP has no individual-rotor feasibility certificate.
                inertia=np.array(MODEL['inertia']);dw=np.gradient(w,tt,axis=0)
                torque=dw*inertia+np.cross(w,w*inertia);arm=MODEL['arm']/np.sqrt(2);ct=MODEL['torque_coefficient']
                allocation=np.array([[1,1,1,1],[arm,arm,-arm,-arm],[-arm,arm,arm,-arm],[ct,-ct,ct,-ct]])
                forces=np.c_[np.sum(forces,axis=1),torque]@np.linalg.inv(allocation).T
                forces=np.clip(forces,MODEL['thrust_min'],MODEL['thrust_max'])
            states=np.c_[p,v,q,w];states[0]=own[:13];forces=np.array(forces);forces[0]=own[13:17]
        equality_error=float(np.max(np.abs(np.array(eq)@z-beq)))
        inequality_error=float(max(0,np.max(np.array(ineq)@z-bineq)))
        cone_error=max([max(0.,np.linalg.norm((b-A@z)[1:])-(b-A@z)[0]) for A,b in zip(soc,bsoc)],default=0.)
        audit=dict(scope='polynomial kinematics only; actual rigid-body feasibility checked separately',rotor_seed=rotor_seed,frame=os.getenv('SNAP_FRAME','fixed_heading'),passed=equality_error<1e-5 and inequality_error<1e-5 and cone_error<1e-5,
                   cone_residual=float(cone_error),
                   soft_surrogate_envelope=soft,surrogate_envelope_slacks=z[ncoeff:].tolist(),
                   equality_error=equality_error,linear_inequality_error=inequality_error,reference_peak_speed=float(np.linalg.norm(v,axis=1).max()),
                   reconstructed_peak_rotor_seed=None if forces is None else float(forces.max()),reconstructed_peak_body_rate=None if w is None else float(np.linalg.norm(w,axis=1).max()),
                   assembly_seconds=assembly_seconds,convex_solve_seconds=solve_seconds,iterations=result.iterations,status=str(result.status))
        if not audit['passed']:raise RuntimeError(str(audit))
        optimized_passages=coeff[pieces-1::pieces,-1,:].copy()
        tracking_gates=optimized_passages if os.getenv('SNAP_TRACK_OPTIMIZED_PASSAGE','0')=='1' else gates
        return dict(durations=event_T,coefficient_durations=T,pieces_per_event=pieces,gates=tracking_gates,learned_gates=np.array(gates),optimized_passages=optimized_passages,normals=normals,coefficients=coeff,reference_time=tt,reference_states=states,reference_forces=forces,
                    reference_acceleration=a,reference_jerk=jerk,U=None if forces is None else forces[:-1].T,cell_durations=np.diff(tt),audit=audit,kinematics_only=np.array(kinematics_only),
                    solve_trace=[audit],polynomial_thrust_soc=thrust_soc,time_scale=float(os.getenv('SNAP_TIME_SCALE','1')))
