"""Five-waypoint minimum-time, minimum-jerk teacher for Guan et al. (ICRA 2025)."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

from scripts.paths import FLIGHT_ASSETS
import argparse
import json
import math
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.optimize import minimize


M = 5
MASS, GRAVITY = .85, 9.81
DRAG = (.26, .28, .42)
F_MAX, W_MAX = 28., 10.
# Project choices where numerical settings are unspecified in the paper.
GRID_SIZE = AUDIT_SIZE = 13
SAMPLES = 10000
SEED = 20261113
BOUNDS = [(np.log(.025), np.log(20.))]*M


def hermite_basis():
    # Map normalized endpoint position, velocity and acceleration to powers of u=t/T.
    boundary = np.zeros((6,6))
    for side,u in enumerate((0.,1.)):
        for d in range(3):
            for i in range(d,6):
                boundary[3*side+d,i] = math.prod(range(i-d+1,i+1))*u**(i-d)
    power = np.linalg.inv(boundary)
    jerk = np.diag([6.,24.,60.]) @ power[3:]
    gram = np.fromfunction(lambda i,j: 1/(i+j+1), (3,3))
    return power, jerk.T@gram@jerk


def reconstruct():
    """Eliminate interior knot velocities/accelerations from the integrated jerk cost."""
    y = ca.SX.sym("input",21)
    times = ca.SX.sym("times",M)
    p = ca.vertcat(ca.SX.zeros(1,3),ca.reshape(y[6:],3,M).T)
    fixed = ca.SX.zeros(3*(M+1),3)
    fixed[::3,:] = p
    fixed[1,:], fixed[2,:] = y[:3].T,y[3:6].T
    free = [3*i+d for i in range(1,M) for d in (1,2)]
    power, gram = hermite_basis()
    hessian = ca.SX.zeros(3*(M+1),3*(M+1))
    scales = []
    for i in range(M):
        t = times[i]
        scale = ca.diag(ca.vertcat(1,t,t*t,1,t,t*t))
        scales.append(scale)
        ids = slice(3*i,3*i+6)
        hessian[ids,ids] += scale@ca.DM(gram)@scale/t**5
    # Small block-tridiagonal SPD system. LDL factorization keeps the symbolic
    # derivative graph banded and yields the unique minimizer for positive times.
    A = hessian[free,free]
    rhs = -(hessian@fixed)[free,:]
    size = len(free)
    L,D = ca.SX.eye(size),ca.SX.zeros(size)
    for i in range(size):
        for j in range(max(0,i-3),i):
            L[i,j] = (A[i,j]-sum(L[i,k]*D[k]*L[j,k] for k in range(j)))/D[j]
        D[i] = A[i,i]-sum(L[i,k]**2*D[k] for k in range(i))
        rhs[i,:] -= sum((L[i,j]*rhs[j,:] for j in range(i)),ca.SX.zeros(1,3))
    for i in reversed(range(size)):
        rhs[i,:] = rhs[i,:]/D[i]-sum((L[j,i]*rhs[j,:] for j in range(i+1,size)),ca.SX.zeros(1,3))
    fixed[free,:] = rhs
    pieces = [ca.DM(power)@scales[i]@fixed[3*i:3*i+6,:] for i in range(M)]
    return ca.Function("minjerk",[y,times],[ca.vertcat(*pieces)])


def flat_outputs():
    v,a,j = [ca.SX.sym(n,3) for n in ("v","a","j")]
    force = MASS*(a+ca.DM([0,0,GRAVITY]))
    def unit(z):
        return z/ca.sqrt(ca.dot(z,z)+1e-16)
    # Fixed yaw = 0, differential flatness with body-frame linear drag.
    bx = unit(ca.cross(ca.DM([0,1,0]),force+DRAG[0]*v))
    by = unit(ca.cross(force+DRAG[1]*v,bx))
    bz = ca.cross(bx,by)
    R = ca.horzcat(bx,by,bz)
    dR = ca.reshape(ca.jtimes(ca.vec(R),ca.vertcat(v,a),ca.vertcat(a,j)),3,3)
    skew = R.T@dR
    omega = ca.vertcat(skew[2,1],skew[0,2],skew[1,0])
    thrust = ca.dot(bz,force+DRAG[2]*v)
    return ca.Function("flat",[v,a,j],[ca.vertcat(thrust,ca.sqrt(ca.dot(omega,omega)+1e-15))])


def derivative_basis(u,d):
    return np.stack([np.zeros_like(u) if k<d else math.prod(range(k-d+1,k+1))*u**(k-d)
                     for k in range(6)],axis=1)


def teacher_functions():
    curve,flat = reconstruct(),flat_outputs()
    x,z = ca.MX.sym("x",21),ca.MX.sym("log_t",M)
    t = ca.exp(z)
    coeff = curve(x,t)
    grid = np.linspace(0.,1.,GRID_SIZE)
    margins=[]
    for i in range(M):
        c = coeff[6*i:6*i+6,:]
        derivatives = [(ca.DM(derivative_basis(grid,d))@c).T/t[i]**d for d in (1,2,3)]
        rates = flat.map(len(grid))(*derivatives)
        f,w = rates[0,:]/F_MAX,rates[1,:]/W_MAX
        margins.extend([f.T,1-f.T,1-w.T])
    g=ca.vertcat(*margins)
    objective=ca.sum1(t)
    fun = ca.Function("time_cost",[x,z],[objective,ca.gradient(objective,z),g,ca.jacobian(g,z)])
    return curve,flat,fun


def solve_label(x,fun):
    waypoints = x[6:].reshape(M,3)
    distances = np.linalg.norm(np.diff(np.vstack([np.zeros(3),waypoints]),axis=0),axis=1)
    seed = np.log(np.maximum(.25,np.sqrt(distances/6.)))
    last_z,values=None,None
    def evaluate(z):
        nonlocal last_z,values
        if last_z is None or not np.array_equal(z,last_z):
            val,grad,g,jac=fun(x,z)
            values=float(val),np.asarray(grad).ravel(),np.asarray(g).ravel(),np.asarray(jac)
            last_z=z.copy()
        return values
    # A small numerical buffer keeps accepted labels strictly inside physical bounds.
    con=dict(type="ineq",fun=lambda z:evaluate(z)[2]-1e-6,jac=lambda z:evaluate(z)[3])
    for scale in (1.,1.7):
        result=minimize(lambda z:evaluate(z)[:2],seed+np.log(scale),method="SLSQP",jac=True,
                        bounds=BOUNDS,constraints=[con],options=dict(maxiter=85,ftol=2e-7))
        if result.success and np.min(evaluate(result.x)[2])>=0:
            return np.exp(result.x),result.fun,result.nit
    return None


def repair_prediction(x,pred,fun):
    """Closest feasible log-duration vector from a network seed, on the same 13 nodes."""
    start=np.log(pred)
    def margins(z):
        return np.asarray(fun(x,z)[2]).ravel()
    if margins(start).min()>=0:
        return pred
    constraint=dict(type="ineq",fun=lambda z:margins(z)-1e-6,
                    jac=lambda z:np.asarray(fun(x,z)[3]))
    for seed in (start,np.minimum(start+.3,np.log(20.))):
        out=minimize(lambda z:(.5*np.sum((z-start)**2),z-start),seed,jac=True,
                     constraints=[constraint],bounds=BOUNDS,method="SLSQP",
                     options=dict(ftol=2e-7,maxiter=85))
        if out.success and margins(out.x).min()>=0:
            return np.exp(out.x)
    return None


def audit(x,t,c,flat_map):
    grid = np.linspace(0,1,AUDIT_SIZE)
    ends = np.array([[derivative_basis(np.array([0.,1.]),d)@piece/duration**d
                      for d in range(5)] for piece,duration in zip(c,t)])
    residual = max(abs(ends[:,0,1]-x[6:].reshape(M,3)).max(),
                   abs(ends[0,:3,0]-np.vstack([np.zeros(3),x[:3],x[3:6]])).max(),
                   abs(ends[-1,1:3,1]).max(),
                   abs(ends[1:,:,0]-ends[:-1,:,1]).max())
    # Continuity through fourth derivative independently checks jerk optimality.
    derivatives = [((derivative_basis(grid,d)@c)/t[:,None,None]**d).reshape(-1,3).T for d in (1,2,3)]
    dynamic = np.asarray(flat_map(*derivatives))
    return np.array([residual,dynamic[0].min(),dynamic[0].max(),dynamic[1].max()])


def flight_features(own, gates):
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(np.roll(own[6:10], -1)).as_matrix()
    velocity = own[3:6]
    drag = rot @ (np.asarray(DRAG) * (rot.T @ velocity))
    acceleration = (rot[:, 2] * own[13:17].sum() - drag) / MASS - [0, 0, GRAVITY]
    return np.r_[velocity, acceleration, (gates - own[:3]).ravel()]


def generate(folder):
    from scripts.training.synthetic_snap_teacher import generate_group, load_allocator
    from scripts.training.snap_transition_study import transition_problems
    folder.mkdir(parents=True,exist_ok=True)
    shards = folder/"shards"
    shards.mkdir(exist_ok=True)
    curve,flat,fun = teacher_functions()
    flat_map = flat.map(M*AUDIT_SIZE)
    allocator=load_allocator(FLIGHT_ASSETS/"models/allocation.pt")
    for batch in range(SAMPLES//100):
        target = shards/f"{batch:03d}.npz"
        if target.exists():
            continue
        records = []
        attempts = 0
        # 26 complete three-layout groups and 22 transition states per shard.
        # Disjoint seed ranges avoid overlap with the original held-out runs.
        regular_group=0
        def transitions():
            block=0
            while True:
                candidates=list(transition_problems(100000+5000*batch+block))
                order=np.random.default_rng(SEED+5000*batch+block).permutation(len(candidates))
                for i in order:
                    yield candidates[i]
                block+=1
        special=transitions()
        while len(records)<100:
            if len(records)<78:
                number=1000+1000*batch+regular_group
                rows=list(generate_group(number,(batch+regular_group)%7,allocator))
                regular_group+=1
            else:
                rows=[next(special)]
            group_samples=[]
            for r in rows:
                attempts+=1
                own=r["own"]
                x=flight_features(own,r["gates"])
                result=solve_label(x,fun)
                if result is None:
                    break
                t,obj,nit=result
                c=np.asarray(curve(x,t)).reshape(M,6,3)
                checks=audit(x,t,c,flat_map)
                if (not np.isfinite(checks).all() or checks[0]>2e-4 or checks[1]<0
                        or checks[2]>F_MAX or checks[3]>W_MAX):
                    break
                group_samples.append((x,t,c,obj,nit,checks,r["gate"],r["kind"],r["offsets"],r["gates"],own,
                                      batch*100+len(records)))
            if len(group_samples)==len(rows):
                records.extend(group_samples)
        fields=("inputs","times","coefficients","objective","iterations","audit","gate","kind","offsets","gates","own","group")
        arrays=dict(zip(fields,map(np.stack,zip(*records))))
        temporary = target.with_suffix(".partial.npz")
        np.savez_compressed(temporary,**arrays,attempts=attempts)
        temporary.replace(target)
        if batch%5==0 or batch==99:
            print("generated",100*(batch+1),"/",SAMPLES,flush=True)
    return curve,flat_map,fun


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    folder=parser.parse_args().folder
    functions=generate(folder)
    from scripts.training.train_guan_times import train
    train(folder,*functions)
