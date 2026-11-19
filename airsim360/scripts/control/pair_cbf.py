"""Joint high-order CBF projection on the eight physical thrust-rate commands."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import casadi as ca
import numpy as np
from scipy.optimize import brentq
from scripts.simulation.dynamics import MODEL, model_functions


class PairCBF:
    radius = .55

    def __init__(self, gain=12., continue_infeasible=False):
        self.gain = gain
        self.continue_infeasible = continue_infeasible
        rhs, _ = model_functions(1)
        x = ca.SX.sym("pair", 38)
        u = ca.SX.sym("rates", 8)
        dx = ca.vertcat(*[rhs(x[19*i:19*(i+1)], ca.vertcat(u[4*i:4*i+4], 0)) for i in range(2)])
        h = ca.sumsqr(x[:3]-x[19:22])-self.radius**2
        psi1 = ca.jtimes(h, x, dx)+gain*h
        psi2 = ca.jtimes(psi1, x, dx)+gain*psi1
        residual = ca.jtimes(psi2, x, dx)+gain*psi2
        a = ca.jacobian(residual, u)
        if ca.depends_on(a,u) or ca.depends_on(psi2,u):
            raise ValueError("Expected relative degree three with affine thrust rates")
        self.barrier = ca.Function("pair_cbf", [x,u],[h,psi1,psi2,residual,a])

    def project(self, states, controls, dt):
        rates = np.asarray(controls)[:, :4].flatten()
        h, p1, p2, raw, jac = self.barrier(states.ravel(), rates)
        raw = float(raw)
        a = np.asarray(jac).ravel()
        record = dict(h=float(h), psi1=float(p1), psi2=float(p2), nominal=raw,
                      applied=raw, correction_norm=0., gain=self.gain)
        if raw >= 0.:
            return controls, record
        forces = states[:, 13:17].flatten()
        rate = MODEL.get("thrust_rate_max", 200.)
        lower = np.maximum(-rate, (MODEL["thrust_min"]-forces)/dt)
        upper = np.minimum(rate, (MODEL["thrust_max"]-forces)/dt)
        # KKT solution of min 1/2 ||v-u||^2 with one half-space and box bounds.
        def candidate(multiplier):
            return np.clip(rates+multiplier*a, lower, upper)
        def margin(multiplier):
            return raw + a @ (candidate(multiplier)-rates)
        best = float(raw+a@(np.where(a>=0, upper, lower)-rates))
        if np.any(lower>upper) or best<0:
            if self.continue_infeasible:
                record.update(infeasible=True, best=best, policy="nominal")
                return controls, record
            raise ValueError(f"No actuator-feasible CBF correction (nominal {raw:.6g})")
        multiplier = 0.
        if margin(0.) < 0:
            high = 1.
            while margin(high) < 0:
                high *= 2.
            multiplier = brentq(margin, 0., high, xtol=1e-14)
        corrected = candidate(multiplier)
        result = np.array(controls, copy=True)
        result[:, :4] = corrected.reshape(2,4)
        record.update(applied=float(self.barrier(states.ravel(), corrected)[3]),
                      correction_norm=float(np.linalg.norm(corrected-rates)))
        if record["applied"] < -1e-7:
            raise ArithmeticError("QP residual outside tolerance")
        return result, record
