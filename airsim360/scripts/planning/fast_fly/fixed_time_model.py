"""Local five-gate Split-S model shared by SQP and the direct IPOPT baseline."""

from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation
from scripts.simulation.dynamics import BASE_CONFIG, rhs13, vehicle_parameters

LEGS = 5
CELLS = 30
STEPS = LEGS * CELLS
NX = 13 * (STEPS + 1)
NU = 4 * STEPS
VEHICLE = vehicle_parameters()
MASS = VEHICLE["mass"]
GRAVITY = VEHICLE["gravity"]
INERTIA = np.array(VEHICLE["inertia"])
THRUST_MAX = VEHICLE["thrust_max"]
ARM = VEHICLE["arm_length"] / np.sqrt(2)
YAW = VEHICLE["torque_coefficient"]
FIXED_BODY = dict(mass=MASS, gravity=GRAVITY, arm=VEHICLE["arm_length"],
                  inertia=INERTIA, torque_coefficient=YAW, drag=VEHICLE["linear_drag_kg_s"])
TORQUE = np.array([[ARM, ARM, -ARM, -ARM], [-ARM, ARM, ARM, -ARM], [YAW, -YAW, YAW, -YAW]])
MIX = np.vstack((np.ones(4), TORQUE))
WORLD_LOWER = np.array([-10., -8., .05])
WORLD_UPPER = np.array([14., 9., 6.])
RATE_LIMIT = BASE_CONFIG["body_rate_limit"]
APERTURE_RESERVE = .02
FORWARD_MIN = .8
FEASIBILITY_TOL = 2e-5

CENTERS = np.array([[-.6, -.86, 3.68], [9., 6.45, 1.05], [8.85, -3.8, 1.05],
                    [-4.3, -5.6, 3.4], [-4.3, -5.6, 1.42], [4.5, -.45, 1.05], [-1.95, 6.81, 1.05]])
GATE_YAW_QUAT = np.array([[.9848078, -.1736482], [1., 0.], [.4226183, -.9063078],
                          [0., -1.], [1., 0.], [.7660444, .6427876], [.258819, -.9659258]])
qw, qz = (GATE_YAW_QUAT / np.linalg.norm(GATE_YAW_QUAT, axis=1, keepdims=True)).T
NORMALS = np.column_stack((1. - 2. * qz**2, 2. * qw * qz, np.zeros(len(CENTERS))))


@dataclass(frozen=True)
class Window:
    own: np.ndarray
    event: int
    gates: np.ndarray
    normals: np.ndarray
    half: np.ndarray
    durations: np.ndarray
    time: float

    @property
    def ids(self):
        return (self.event + np.arange(LEGS)) % len(CENTERS)

    @property
    def mesh(self):
        return np.r_[0., np.cumsum(np.repeat(self.durations / CELLS, CELLS))]

    @property
    def parameters(self):
        return np.r_[self.own[:13], self.gates.ravel(), self.normals.ravel(),
                     self.half.ravel(), self.durations, self.own[13:17]]


def make_window(own, event, targets, time, network):
    own, targets = np.asarray(own, dtype=float), np.asarray(targets, dtype=float)
    if own.shape != (19,) or targets.shape != (LEGS, 3):
        raise ValueError("Expected a 19-value state and five world-frame targets")
    if not np.isfinite(np.r_[own, targets.ravel(), time]).all() or event < 0:
        raise ValueError("Invalid state, gate event, targets or time")
    ids = (event + np.arange(LEGS)) % len(CENTERS)
    normals = NORMALS[ids]
    offsets = targets - CENTERS[ids]
    plane_error = np.sum(offsets * normals, axis=1)
    if np.abs(plane_error).max() > 1e-6:
        raise ValueError("Targets must lie in the corresponding gate planes")
    gates = targets - plane_error[:, None] * normals
    side = np.cross([0., 0., 1.], normals)
    lateral = np.sum(offsets * side, axis=1)
    half = np.minimum(.75 - .05 - np.abs(np.column_stack((lateral, offsets[:, 2]))), [.075, .10])
    if np.any(half <= APERTURE_RESERVE):
        raise ValueError("Target leaves insufficient clearance")
    durations = network.predict(own, gates, normals, half)
    if durations.shape != (LEGS,) or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("Expected five positive finite predicted durations")
    return Window(own, int(event), gates, normals, half, durations, float(time))


def rigid_body(state, thrust):
    return rhs13(state, thrust, FIXED_BODY)


def rk4_map(name, substeps):
    x0, u, h = ca.SX.sym("x", 13), ca.SX.sym("u", 4), ca.SX.sym("h")
    rhs = ca.Function("fixed_body_rhs", [x0, u], [rigid_body(x0, u)])
    x, checkpoints = x0, []
    dt = h / substeps
    for _ in range(substeps):
        a = rhs(x, u)
        b = rhs(x + dt * a / 2, u)
        c = rhs(x + dt * b / 2, u)
        d = rhs(x + dt * c, u)
        x = x + dt * (a + 2 * b + 2 * c + d) / 6
        x = ca.vertcat(x[:6], x[6:10] / ca.norm_2(x[6:10]), x[10:])
        checkpoints.append(x)
    return ca.Function(name, [x0, u, h], [x, ca.horzcat(*checkpoints)])


def mesh_integrator():
    x, u, h = ca.SX.sym("x", 13), ca.SX.sym("u", 4), ca.SX.sym("h")
    dae = dict(x=x, p=ca.vertcat(u, h), ode=h * rigid_body(x, u))
    integrate = ca.integrator("mesh_audit_cvodes", "cvodes", dae, 0., 1.,
                              dict(linear_multistep_method="bdf", abstol=1e-10, reltol=1e-9))
    x, u, h = ca.MX.sym("x", 13), ca.MX.sym("u", 4), ca.MX.sym("h")
    end = integrate(x0=x, p=ca.vertcat(u, h))["xf"]
    return ca.Function("mesh_audit_step", [x, u, h], [end])


def split_decisions(z):
    return z[:NX].reshape(-1, 13), z[NX:].reshape(-1, 4)


def violation(values, lower, upper):
    return np.maximum(np.maximum(lower - values, values - upper), 0.)


def seed_window(window):
    state = window.own[:13]
    points = np.vstack((state[:3], window.gates))
    speeds = np.clip(np.linalg.norm(np.diff(points, axis=0), axis=1) / window.durations, .8, 2.5)
    times = window.mesh
    spline = CubicHermiteSpline(times[::CELLS], points, np.vstack((state[3:6], window.normals * speeds[:, None])))
    position, velocity = spline(times), spline(times, 1)
    accel = spline(times, 2) + [0., 0., GRAVITY]
    magnitude = np.linalg.norm(accel, axis=1)
    body_z = accel / np.maximum(magnitude[:, None], 1e-9)
    body_z[magnitude < 1e-9] = [0., 0., 1.]
    heading = np.tile([1., 0., 0.], (len(times), 1))
    heading[np.abs(body_z[:, 0]) > .97] = [0., 1., 0.]
    body_y = np.cross(body_z, heading)
    body_y /= np.linalg.norm(body_y, axis=1, keepdims=True)
    rotations = np.stack((np.cross(body_y, body_z), body_y, body_z), axis=2)
    quat = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
    quat[0] = state[6:10]
    for i in range(1, len(quat)):
        if quat[i - 1] @ quat[i] < 0:
            quat[i] *= -1
    dq = np.gradient(quat, times, axis=0)
    omega = 2 * (quat[:, :1] * dq[:, 1:] - dq[:, :1] * quat[:, 1:] -
                 np.cross(quat[:, 1:], dq[:, 1:]))
    omega *= np.minimum(1., 8. / np.maximum(np.linalg.norm(omega, axis=1), 1e-9))[:, None]
    X = np.column_stack((position, velocity, quat, omega))
    X[0] = state
    rate = (omega[:-1] + omega[1:]) / 2
    torque = np.diff(omega, axis=0) * INERTIA / np.diff(times)[:, None] + np.cross(rate, rate * INERTIA)
    total = MASS * (magnitude[:-1] + magnitude[1:]) / 2
    U = np.linalg.solve(MIX, np.column_stack((total, torque)).T).T
    return np.r_[X.ravel(), np.clip(U, 0., THRUST_MAX).ravel()]


class FixedTimeModel:
    def __init__(self):
        X, U = ca.MX.sym("X", 13, STEPS + 1), ca.MX.sym("U", 4, STEPS)
        par = ca.MX.sym("p", 13 + LEGS * 9 + 4)
        x0 = par[:13]
        gates = ca.reshape(par[13:28], 3, LEGS)
        normals = ca.reshape(par[28:43], 3, LEGS)
        half = ca.reshape(par[43:53], 2, LEGS)
        durations, forces = par[53:58], par[58:62]
        z = ca.vertcat(ca.vec(X), ca.vec(U))
        rows, lower, upper, defects, dynamics_rows = [], [], [], [], []

        def constrain(expr, lo=0., hi=0.):
            expr = ca.vec(expr)
            rows.append(expr)
            lower.extend(np.broadcast_to(lo, (expr.numel(),)))
            upper.extend(np.broadcast_to(hi, (expr.numel(),)))

        def physical(state, margin=0.):
            constrain(ca.sumsqr(state[10:]), -np.inf, (RATE_LIMIT - 10 * margin)**2)
            constrain(state[2], WORLD_LOWER[2] + margin, WORLD_UPPER[2] - margin)
            constrain(state[6:10], -1., 1.)
            constrain(state[:3], WORLD_LOWER + margin, WORLD_UPPER - margin)

        constrain(X[:, 0] - x0)
        physical(X[:, 0])
        interior_start = len(lower)
        cost = .03 * ca.sumsqr(U[:, 0] - forces)
        step = rk4_map("planning_rk4", 4)
        for k in range(STEPS):
            h = durations[k // CELLS] / CELLS
            end, dense = step(X[:, k], U[:, k], h)
            defect = X[:, k + 1] - end
            defects.append(defect)
            dynamics_rows.extend(range(len(lower), len(lower) + defect.numel()))
            constrain(defect)
            for sub in range(4):
                physical(dense[:, sub], .002)
            if k:
                cost += 1e-5 * ca.sumsqr(U[:, k] - U[:, k - 1]) / h

        self.interior_constraint_rows = slice(interior_start, len(lower))
        self.dynamics_rows = np.array(dynamics_rows, dtype=int)
        for leg in range(LEGS):
            k = (leg + 1) * CELLS
            gate, normal = gates[:, leg], normals[:, leg]
            side = ca.cross(ca.DM([0., 0., 1.]), normal)
            side /= ca.norm_2(side)
            vertical = ca.cross(normal, side)
            distance = X[:3, k] - gate
            constrain(ca.dot(normal, distance))
            for j, axis in enumerate((side, vertical)):
                across = ca.dot(axis, distance)
                gap = half[j, leg] - APERTURE_RESERVE
                constrain(ca.vertcat(across - gap, -across - gap), -np.inf, 0.)
            constrain(ca.dot(normal, X[3:6, k]), FORWARD_MIN, np.inf)
            for before in range(leg * CELLS, k):
                constrain(ca.dot(normal, X[:3, before] - gate), -np.inf, 1e-8)
            constrain(ca.dot(normal, X[:3, k - 1] - gate), -np.inf, -.005)
            if leg < LEGS - 1:
                constrain(ca.dot(normal, X[:3, k + 1] - gate), .005, np.inf)

        g = ca.vertcat(*rows)
        self.lower, self.upper = np.array(lower), np.array(upper)
        self.constraints = ca.Function("fixed_constraints", [z, par], [g])
        self.linearize = ca.Function("fixed_linearization", [z, par], [g, ca.jacobian(g, z)])
        self._leading_time_jacobian = None
        self.cost = ca.Function("fixed_cost", [z, par], [cost])
        self.quadratic = ca.Function("fixed_cost_derivatives", [z, par], [ca.gradient(cost, z), ca.hessian(cost, z)[0]])
        opts = {"print_time": False, "error_on_fail": False, "ipopt": {
            "print_level": 0, "sb": "yes", "tol": 2e-7, "acceptable_tol": 1e-6,
            "max_iter": 600, "max_wall_time": 180., "bound_relax_factor": 0.,
            "linear_solver": "mumps", "mu_strategy": "adaptive", "warm_start_init_point": "yes",
            "warm_start_bound_push": 1e-6, "warm_start_mult_bound_push": 1e-6,
            "warm_start_slack_bound_push": 1e-6,
        }}
        self.ipopt = ca.nlpsol("fixed_time_ipopt", "ipopt", dict(x=z, p=par, f=cost + ca.sum1(durations), g=g), opts)
        crossing_nodes = X[:3, [CELLS * (leg + 1) for leg in range(LEGS)]]
        preheat_cost = ca.sumsqr(ca.vertcat(*defects)) + ca.sumsqr(crossing_nodes - gates) + .01 * ca.sumsqr(X[10:, :])
        preheat_opts = {"print_time": False, "error_on_fail": False, "ipopt": {
            "print_level": 0, "sb": "yes", "max_iter": 200, "max_wall_time": 8., "tol": 1e-5,
        }}
        self.preheat = ca.nlpsol("fixed_time_preheat", "ipopt",
                                 dict(x=z, p=par, f=preheat_cost, g=X[:, 0] - x0), preheat_opts)
        self.integrate_cell = mesh_integrator()
        self.rollout = self.integrate_cell.mapaccum(STEPS)

    def leading_time_jacobian(self, current, parameters):
        """Differentiate constraints with respect to the first two leg times."""
        if self._leading_time_jacobian is None:
            z = ca.MX.sym("z", NX + NU)
            p = ca.MX.sym("p", len(parameters))
            durations = ca.MX.sym("durations", 2)
            g = self.constraints(z, ca.vertcat(p[:53], durations, p[55:]))
            self._leading_time_jacobian = ca.Function(
                "leading_time_jacobian", [z, p, durations], [ca.jacobian(g, durations)]
            )
        return self._leading_time_jacobian(current, parameters, parameters[53:55]).sparse()

    def bounds(self, window):
        lo, hi = np.r_[np.full(NX, -np.inf), np.zeros(NU)], np.r_[np.full(NX, np.inf), np.full(NU, THRUST_MAX)]
        for leg, (gate_id, normal) in enumerate(zip(window.ids, window.normals)):
            turn = gate_id in (3, 4)
            vmin, vmax, vertical = (5.5, 8., .65) if turn else (5., 14., 1.4)
            side = 1.2 * np.abs(np.cross([0., 0., 1.], normal)) + [0., 0., vertical]
            velocity = slice(13 * (leg + 1) * CELLS + 3, 13 * (leg + 1) * CELLS + 6)
            lo[velocity] = np.minimum(vmin * normal, vmax * normal) - side
            hi[velocity] = np.maximum(vmin * normal, vmax * normal) + side
            if gate_id == 4 and leg:
                hi[13 * (leg * CELLS + CELLS // 2)] = window.gates[leg, 0] - 1.2
        return lo, hi

    def audit(self, window, z):
        start = perf_counter()
        X, U = split_decisions(z)
        mesh = window.mesh
        states = self.rollout(window.own[:13], U.T, np.diff(mesh))
        states = np.vstack((window.own[:13], np.asarray(states).T))
        crossings = states[CELLS::CELLS]
        gap = crossings[:, :3] - window.gates
        side = np.cross([0., 0., 1.], window.normals)
        distance = np.sum(gap * window.normals, axis=1)
        transverse = np.column_stack((np.sum(gap * side, axis=1), gap[:, 2]))
        speed = np.sum(crossings[:, 3:6] * window.normals, axis=1)
        node_error = float(np.max(np.abs(states - X)))
        bounds_ok = (np.all(states[:, :3] >= WORLD_LOWER - 3e-5)
                     and np.all(states[:, :3] <= WORLD_UPPER + 3e-5)
                     and np.all(np.abs(states[:, 6:10]) <= 1. + 3e-5)
                     and np.all(np.linalg.norm(states[:, 10:], axis=1) <= RATE_LIMIT + 3e-5))
        passed = (bounds_ok and node_error < .002 and np.all(np.abs(distance) < .002)
                  and np.all(np.abs(transverse) <= window.half) and np.all(speed >= FORWARD_MIN - 3e-5))
        return dict(passed=bool(passed), node_error=node_error,
                    plane_error=float(np.max(np.abs(distance))), seconds=perf_counter() - start), mesh, states

    def solve_direct(self, window, guess=None):
        preheat_iterations, preheat_seconds = 0, 0.
        if guess is None:
            preheat_start = perf_counter()
            guess = seed_window(window)
            state_lo = np.r_[WORLD_LOWER, [-np.inf] * 3, [-1.] * 4, [-RATE_LIMIT] * 3]
            state_hi = np.r_[WORLD_UPPER, [np.inf] * 3, [1.] * 4, [RATE_LIMIT] * 3]
            initial = self.preheat(x0=guess, p=window.parameters, lbg=0., ubg=0.,
                                   lbx=np.r_[np.tile(state_lo, STEPS + 1), np.zeros(NU)],
                                   ubx=np.r_[np.tile(state_hi, STEPS + 1), np.full(NU, THRUST_MAX)])
            preheat_iterations = int(self.preheat.stats()["iter_count"])
            trial = np.asarray(initial["x"]).ravel()
            if np.isfinite(trial).all():
                guess = trial
            preheat_seconds = perf_counter() - preheat_start
        ipopt_start = perf_counter()
        lo, hi = self.bounds(window)
        result = self.ipopt(x0=guess, p=window.parameters, lbx=lo, ubx=hi, lbg=self.lower, ubg=self.upper)
        stats = self.ipopt.stats()
        z = np.asarray(result["x"]).ravel()
        residual = max(violation(np.asarray(result["g"]).ravel(), self.lower, self.upper).max(),
                       violation(z, lo, hi).max())
        info = dict(status=stats["return_status"], iterations=int(stats["iter_count"]),
                    preheat_iterations=preheat_iterations, preheat_seconds=preheat_seconds,
                    ipopt_seconds=perf_counter() - ipopt_start, residual=float(residual))
        if not stats["success"] or not np.isfinite(z).all() or not residual <= FEASIBILITY_TOL:
            return None, info
        audit, times, states = self.audit(window, z)
        info["audit"] = audit
        return (dict(z=z, time=times, states=states, window=window) if audit["passed"] else None), info
