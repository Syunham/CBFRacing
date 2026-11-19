"""Rolling fixed-time SQP with optional short-horizon timing recovery."""

from dataclasses import replace
import time

import clarabel
import numpy as np
from scipy import sparse

from .fixed_time_model import (
    CELLS, FEASIBILITY_TOL, LEGS, NU, STEPS, THRUST_MAX, FixedTimeModel, Window,
    seed_window, split_decisions, violation,
)

DEFAULT_TRUST_RADIUS = .75
QP_SUCCESS = ("Solved", "AlmostSolved")
TIME_EXTENSION_LIMIT = .100
TIMING_RECOVERY_ITERATIONS = 6


def shift_solution(window, previous, *, timing=None):
    old = previous["window"]
    advance = window.event - old.event
    elapsed = window.time - old.time
    if advance not in (0, 1) or elapsed <= 0.:
        return None
    if "control_time" in previous:
        old_mesh, Uold = previous["control_time"], previous["controls"]
    else:
        old_mesh = old.mesh
        _, Uold = split_decisions(previous["z"])
    old_events = np.r_[0., np.cumsum(old.durations)]
    start = max(elapsed, old_events[advance])
    if start >= old_events[advance + 1]:
        return None

    # Align the overlapping gate events while warping onto the new fixed grid.
    # Time warping only supplies the initial guess; all defects are corrected
    # against the new RK4 mesh.
    shared = (LEGS - advance) * CELLS
    new_mesh = window.mesh[:shared + 1]
    old_anchors = np.r_[start, old_events[advance + 1:]]
    query = np.interp(new_mesh, new_mesh[::CELLS], old_anchors)
    old_dt = np.diff(query)
    if np.any(old_dt <= 0.):
        return None
    X = np.column_stack([np.interp(query, previous["time"], v) for v in previous["states"].T])
    X[:, 6:10] /= np.linalg.norm(X[:, 6:10], axis=1, keepdims=True)
    if X[0, 6:10] @ window.own[6:10] < 0.:
        X[:, 6:10] *= -1.
    X[0] = window.own[:13]
    integral = np.vstack((np.zeros(4), np.cumsum(Uold * np.diff(old_mesh)[:, None], axis=0)))
    integrated = np.column_stack([np.interp(query, old_mesh, v) for v in integral.T])
    U = np.clip(np.diff(integrated, axis=0) / old_dt[:, None], 0., THRUST_MAX)
    if advance:
        tail_start = time.perf_counter()
        tail = Window(np.r_[X[-1], U[-1], 0., 0.], window.event + LEGS - 1,
                      window.gates[-1:], window.normals[-1:], window.half[-1:],
                      window.durations[-1:], window.time + new_mesh[-1])
        tail_seed = seed_window(tail)
        seam = 13 * (CELLS + 1)
        X = np.vstack((X, tail_seed[:seam].reshape(-1, 13)[1:]))
        U = np.vstack((U, tail_seed[seam:].reshape(-1, 4)))
        if timing is not None:
            timing["tail_seed"] = time.perf_counter() - tail_start
    return np.r_[X.ravel(), U.ravel()]


def rebuild_prefix(window, guess, integrate_cell, skip_seconds):
    """Replace close reference nodes and blend back to a later resampled node."""
    mesh = window.mesh
    skipped = int(np.searchsorted(mesh, skip_seconds, side="right") - 1)
    if skipped <= 0:
        return guess, 0

    # Rejoin at or before the next gate whenever there is room to blend there.
    # Retain an anchor even when the first gate falls inside the skipped prefix.
    skipped = min(skipped, STEPS - 1)
    blend_end = int(np.searchsorted(mesh, skip_seconds + .060))
    if skipped < CELLS:
        blend_end = min(blend_end, CELLS)
    blend_end = min(STEPS, max(skipped + 1, blend_end))
    seed = guess.copy()
    X, U = split_decisions(seed)
    propagated = X[0].copy()
    for k in range(1, blend_end):
        propagated = np.asarray(integrate_cell(propagated, U[k - 1], mesh[k] - mesh[k - 1])).ravel()
        if k <= skipped:
            X[k] = propagated
            continue
        weight = (mesh[k] - mesh[skipped]) / (mesh[blend_end] - mesh[skipped])
        weight = weight**2 * (3. - 2. * weight)
        if propagated[6:10] @ X[k, 6:10] < 0.:
            propagated[6:10] *= -1.
        X[k] = weight * X[k] + (1. - weight) * propagated
        X[k, 6:10] /= np.linalg.norm(X[k, 6:10])
    return seed, skipped


def reachable_sides(A, lower, upper, radius, equal):
    reach = radius * np.asarray(abs(A).sum(axis=1)).ravel()
    return (~equal & np.isfinite(upper) & (upper <= reach),
            ~equal & np.isfinite(lower) & (lower >= -reach))


def qp_settings():
    settings = clarabel.DefaultSettings()
    settings.verbose, settings.max_threads = False, 1
    settings.tol_feas, settings.tol_gap_abs, settings.tol_gap_rel = 1e-8, 1e-8, 1e-8
    return settings


def solve_qp(H, gradient, A, lower, upper, radius, *, screen_rows=None):
    start = time.perf_counter()
    equal = (lower == upper) & np.isfinite(lower)
    full_up, full_down = reachable_sides(A, lower, upper, radius, equal)
    settings = qp_settings()
    P = sparse.triu(H, format="csc")
    setup_seconds, solve_seconds, total_iterations, row_counts = 0., 0., 0, []

    def run(up, down):
        nonlocal start, setup_seconds, solve_seconds, total_iterations
        matrix = sparse.vstack((A[equal], A[up], -A[down]), format="csc")
        rhs = np.r_[upper[equal], upper[up], -lower[down]]
        cones = [clarabel.ZeroConeT(int(equal.sum())), clarabel.NonnegativeConeT(int(up.sum() + down.sum()))]
        solver = clarabel.DefaultSolver(P, gradient, matrix, rhs, cones, settings)
        solve_start = time.perf_counter()
        result = solver.solve()
        start_next = time.perf_counter()
        setup_seconds += solve_start - start
        solve_seconds += start_next - solve_start
        start = start_next
        total_iterations += int(result.iterations)
        row_counts.append(matrix.shape[0])
        return str(result.status), np.asarray(result.x)

    screen_passed = False
    if screen_rows is not None:
        outside = np.ones(len(lower), dtype=bool)
        outside[screen_rows] = False
        up = full_up & (outside | (upper <= .015))
        down = full_down & (outside | (lower >= -.015))
        for _ in range(3):
            status, step = run(up, down)
            if status not in QP_SUCCESS or not np.isfinite(step).all():
                break
            values = A @ step
            residual = float(violation(values, lower, upper).max())
            if residual <= 1e-7:
                screen_passed = True
                break
            up |= full_up & (values > upper + 1e-7)
            down |= full_down & (values < lower - 1e-7)
    if not screen_passed:
        status, step = run(full_up, full_down)
        residual = float(violation(A @ step, lower, upper).max())
    info = dict(qp_status=status, qp_iterations=total_iterations, linear_residual=residual,
                qp_setup_seconds=setup_seconds, qp_solve_seconds=solve_seconds, qp_rows=row_counts,
                screening_fallback=screen_rows is not None and not screen_passed)
    if status not in QP_SUCCESS or not np.isfinite(step).all() or not residual <= 1e-4:
        return None, info
    return step, info


def solve_elastic_qp(A, lower, upper, radius, dynamics, penalty):
    """Restore feasibility with one absolute-residual slack per RK4 equation."""
    start = time.perf_counter()
    equal = (lower == upper) & np.isfinite(lower)
    up, down = reachable_sides(A, lower, upper, radius, equal)
    equal[dynamics] = False
    nz, ns = A.shape[1], len(dynamics)
    # A slack bounds both directions of the same defect, implying s >= 0.
    hard = sparse.vstack((A[equal], A[up], -A[down]), format="csc")
    hard = sparse.hstack((hard, sparse.csc_matrix((hard.shape[0], ns))))
    eye = sparse.eye(ns, format="csc")
    soft = sparse.vstack((sparse.hstack((A[dynamics], -eye)),
                          sparse.hstack((-A[dynamics], -eye))))
    matrix = sparse.vstack((hard, soft), format="csc")
    rhs = np.r_[upper[equal], upper[up], -lower[down], upper[dynamics], -lower[dynamics]]
    neq = int(equal.sum())
    cones = [clarabel.ZeroConeT(neq), clarabel.NonnegativeConeT(matrix.shape[0] - neq)]
    H = sparse.diags(np.r_[np.ones(nz), np.full(ns, 1e-6)], format="csc")
    solver = clarabel.DefaultSolver(H, np.r_[np.zeros(nz), np.full(ns, penalty)], matrix, rhs, cones, qp_settings())
    solve_start = time.perf_counter()
    result = solver.solve()
    solve_end = time.perf_counter()
    info = dict(status=str(result.status), iterations=int(result.iterations), rows=matrix.shape[0],
                setup_seconds=solve_start - start, solve_seconds=solve_end - solve_start)
    solution = np.asarray(result.x)
    if info["status"] not in QP_SUCCESS or not np.isfinite(solution).all():
        return None, info
    step, slack = solution[:nz], solution[nz:]
    relaxed_lower, relaxed_upper = lower.copy(), upper.copy()
    relaxed_lower[dynamics] -= slack
    relaxed_upper[dynamics] += slack
    residual = float(violation(A @ step, relaxed_lower, relaxed_upper).max())
    info.update(residual=residual, max_slack=float(slack.max()), sum_slack=float(np.maximum(slack, 0.).sum()))
    return (step if residual <= 1e-4 else None), info


class FixedTimeSQP:
    def __init__(self, model=None, *, max_iterations=8, prefix_skip_seconds=.025, screen_qp=False,
                 trust_radius=DEFAULT_TRUST_RADIUS, elastic=True, timing_recovery=False):
        if not np.isfinite(prefix_skip_seconds) or prefix_skip_seconds < 0.:
            raise ValueError("The skipped reference time must be finite and nonnegative")
        if not np.isfinite(trust_radius) or trust_radius <= 0.:
            raise ValueError("The trust radius must be finite and positive")
        self.model = FixedTimeModel() if model is None else model
        self.max_iterations = max_iterations
        self.prefix_skip_seconds = prefix_skip_seconds
        self.screen_rows = self.model.interior_constraint_rows if screen_qp else None
        self.trust_radius = trust_radius
        self.elastic = elastic
        self.timing_recovery = timing_recovery
        state_scale = np.r_[[5.] * 3, [10.] * 3, [1.] * 4, [5.] * 3]
        self.scale = np.r_[np.tile(state_scale, STEPS + 1), np.full(NU, THRUST_MAX)]
        self.S = sparse.diags(self.scale, format="csc")
        self.I = sparse.eye(len(self.scale), format="csc")

    def solve(self, window, previous=None):
        """Return an audited plan and diagnostics, including any IPOPT recovery."""
        timing = dict(tail_seed=0.)
        start = time.perf_counter()
        guess = shift_solution(window, previous, timing=timing) if previous is not None else None
        shift_end = time.perf_counter()
        timing["resampling"] = shift_end - start - timing["tail_seed"]
        if guess is None:
            plan, direct = self.model.solve_direct(window)
            info = dict(method="ipopt_initialization", updates=[], ipopt=[direct], stage_seconds=timing)
            if plan is None and self.timing_recovery:
                plan = self.try_timing_recovery(window, seed_window(window), info)
            return plan, info
        guess, skipped = rebuild_prefix(window, guess, self.model.integrate_cell, self.prefix_skip_seconds)
        sqp_start = time.perf_counter()
        timing["prefix_rebuild"] = sqp_start - shift_end
        plan, current, updates = self.iterate(window, guess)
        timing["sqp_solver"] = time.perf_counter() - sqp_start - sum(
            update.get("audit", {}).get("seconds", 0.) for update in updates
        )
        if plan is not None:
            return plan, dict(method="sqp", updates=updates, ipopt=[], skipped_reference_nodes=skipped,
                              stage_seconds=timing)
        info = dict(method="ipopt_fallback", updates=updates, ipopt=[], skipped_reference_nodes=skipped,
                    stage_seconds=timing)
        if self.timing_recovery:
            plan = self.try_timing_recovery(window, current, info)
            if plan is not None:
                return plan, info
        plan, warm = self.model.solve_direct(window, current)
        info["ipopt"].append(warm)
        if plan is None:
            plan, cold = self.model.solve_direct(window)
            info["ipopt"].append(cold)
        return plan, info

    def try_timing_recovery(self, window, current, info):
        started = time.perf_counter()
        plan, history = self.restore_timing(window, current)
        audit_seconds = sum(update.get("audit", {}).get("seconds", 0.) for update in history)
        info["timing_recovery"] = dict(updates=history, seconds=time.perf_counter() - started - audit_seconds,
                                       audit_seconds=audit_seconds)
        if plan is not None:
            info["method"] = "timing_recovery"
            info["timing_recovery"]["extra_seconds"] = (plan["window"].durations - window.durations).tolist()
        return plan

    def restore_timing(self, original, current):
        """Allow the first two leg times to grow within a bounded SQP step."""
        model, radius = self.model, self.trust_radius
        lo, hi = model.bounds(original)
        time_scale = TIME_EXTENSION_LIMIT
        extension = np.zeros(2)
        lower_z, upper_z = np.r_[lo, 0., 0.], np.r_[hi, time_scale, time_scale]
        scale = np.r_[self.scale, time_scale, time_scale]
        identity = sparse.eye(len(scale), format="csc")
        hessian = sparse.diags(np.r_[np.ones(len(current)), 4., 4.], format="csc")
        history = []
        window = original
        for _ in range(TIMING_RECOVERY_ITERATIONS):
            p = window.parameters
            values, derivative = model.linearize(current, p)
            g = np.asarray(values).ravel()
            A = sparse.hstack((derivative.sparse() @ self.S, model.leading_time_jacobian(current, p) * time_scale),
                              format="csc")
            row_scale = 1. / np.maximum(1., np.sqrt(np.asarray(A.power(2).sum(axis=1)).ravel()))
            jacobian = sparse.vstack((sparse.diags(row_scale) @ A, identity), format="csc")
            z = np.r_[current, extension]
            lower = np.r_[(model.lower - g) * row_scale, np.maximum((lower_z - z) / scale, -radius)]
            upper = np.r_[(model.upper - g) * row_scale, np.minimum((upper_z - z) / scale, radius)]
            # After opening the time budget, prioritize removing nonlinear defects.
            gradient = np.zeros_like(z)
            step, update = solve_qp(hessian, gradient, jacobian, lower, upper, radius, screen_rows=self.screen_rows)
            if step is None and self.elastic:
                step, update["elastic_qp"] = solve_elastic_qp(jacobian, lower, upper, radius,
                                                              model.dynamics_rows, 100.)
            history.append(update)
            if step is None:
                break
            old_violation = row_scale @ violation(g, model.lower, model.upper)
            for alpha in (1., .5, .25, .125, .0625):
                proposal = z + alpha * scale * step
                trial_extension = np.clip(proposal[-2:], 0., time_scale)
                durations = original.durations.copy()
                durations[:2] += trial_extension
                trial_window = replace(original, durations=durations)
                trial = proposal[:-2]
                values = np.asarray(model.constraints(trial, trial_window.parameters)).ravel()
                defects = violation(values, model.lower, model.upper)
                if row_scale @ defects <= old_violation + 1e-9:
                    current, extension, window = trial, trial_extension, trial_window
                    break
            else:
                update["accepted"] = False
                break
            residual = float(max(defects.max(), violation(current, lo, hi).max()))
            update.update(accepted=True, alpha=alpha, residual=residual, extension=extension.tolist())
            if residual < FEASIBILITY_TOL:
                audit, times, states = model.audit(window, current)
                update["audit"] = audit
                if audit["passed"]:
                    return dict(z=current, time=times, states=states, window=window), history
        return None, history

    def iterate(self, window, current):
        p = window.parameters
        model = self.model
        lo, hi = model.bounds(window)
        radius, regularization, penalty = self.trust_radius, .1, 100.
        history = []
        for index in range(self.max_iterations):
            linearize_start = time.perf_counter()
            values, derivative = model.linearize(current, p)
            form_start = time.perf_counter()
            g, A = np.asarray(values).ravel(), derivative.sparse() @ self.S
            row_scale = 1. / np.maximum(1., np.sqrt(np.asarray(A.power(2).sum(axis=1)).ravel()))
            lower = np.r_[(model.lower - g) * row_scale, np.maximum((lo - current) / self.scale, -radius)]
            upper = np.r_[(model.upper - g) * row_scale, np.minimum((hi - current) / self.scale, radius)]
            jacobian = sparse.vstack((sparse.diags(row_scale) @ A, self.I), format="csc")
            correction = index > 0
            if correction:
                H, gradient = self.I, np.zeros_like(current)
            else:
                gradient, hessian = model.quadratic(current, p)
                gradient = np.asarray(gradient).ravel() * self.scale
                H = self.S @ hessian.sparse() @ self.S + regularization * self.I
            qp_start = time.perf_counter()
            step, info = solve_qp(H, gradient, jacobian, lower, upper, radius, screen_rows=self.screen_rows)
            if step is None and self.elastic:
                step, restore = solve_elastic_qp(jacobian, lower, upper, radius, model.dynamics_rows, penalty)
                info["elastic_qp"] = restore
                info["qp_iterations"] += restore["iterations"]
                info["qp_setup_seconds"] += restore["setup_seconds"]
                info["qp_solve_seconds"] += restore["solve_seconds"]
                correction = True
            search_start = time.perf_counter()
            info.update(linearize_seconds=form_start - linearize_start, qp_form_seconds=qp_start - form_start,
                        line_search_seconds=0.)
            history.append(info)
            info["correction"] = correction
            if step is None:
                break

            def merit(z, constraints):
                cost = 0. if correction else float(model.cost(z, p))
                return cost + penalty * (row_scale @ violation(constraints, model.lower, model.upper))

            old_merit = merit(current, g)
            for alpha in (1., .5, .25, .125, .0625):
                candidate = current + alpha * self.scale * step
                values = np.asarray(model.constraints(candidate, p)).ravel()
                if merit(candidate, values) < old_merit + 1e-9:
                    current = candidate
                    break
            else:
                info.update(accepted=False, line_search_seconds=time.perf_counter() - search_start)
                break

            residual = float(max(violation(values, model.lower, model.upper).max(), violation(current, lo, hi).max()))
            info.update(accepted=True, alpha=alpha, residual=residual,
                        line_search_seconds=time.perf_counter() - search_start)
            if residual < FEASIBILITY_TOL:
                audit, times, states = model.audit(window, current)
                info["audit"] = audit
                if audit["passed"]:
                    return dict(z=current, time=times, states=states, window=window), current, history
        return None, current, history
