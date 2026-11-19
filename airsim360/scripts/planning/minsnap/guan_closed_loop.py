"""Track the same projected quintics evaluated by train_guan_times.py."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
import torch

from scripts.planning.utils.local_planner import LocalPlanner
from scripts.planning.minsnap.guan_minjerk import (AUDIT_SIZE, F_MAX, M, W_MAX, audit, derivative_basis,
                         flight_features, repair_prediction, teacher_functions)


class ProjectedQuintics:
    def __init__(self, checkpoint):
        self.c = dict(frontend="guan_minimum_jerk", event_policy="guan_projected_network_times",
                      checkpoint=str(checkpoint), preview_gates=M, discrete_nodes_per_leg=AUDIT_SIZE,
                      thrust_limit_n=F_MAX, body_rate_limit_rad_s=W_MAX)
        self.network = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        self.curve, flat, self.constraints = teacher_functions()
        self.flat_map = flat.map(M * AUDIT_SIZE)
        self.grid = np.linspace(0, 1, 65)
        self.basis = [derivative_basis(self.grid, d) for d in range(4)]

    @torch.inference_mode()
    def predict(self, own, gates, normals, half):
        features = flight_features(own, gates)
        return self.network(torch.tensor(features[None], dtype=torch.float32))[0].numpy().astype(float)

    def solve(self, own, gates, normals, half, durations):
        x = flight_features(own, gates)
        raw_margin = float(np.min(np.asarray(self.constraints(x, np.log(durations))[2])))
        t = repair_prediction(x, durations, self.constraints)
        self.last_trace = [dict(raw_times=durations.tolist(), raw_min_margin=raw_margin)]
        if t is None:
            self.last_trace[0]["input"] = x.tolist()
            raise RuntimeError("Guan discrete time projection found no feasible durations")
        c = np.asarray(self.curve(x, t)).reshape(M, 6, 3)
        checks = audit(x, t, c, self.flat_map)
        report = dict(projection_applied=not np.array_equal(t, durations), raw_min_margin=raw_margin,
                      boundary_residual=float(checks[0]), min_thrust_n=float(checks[1]),
                      max_thrust_n=float(checks[2]), max_body_rate_rad_s=float(checks[3]))
        self.last_trace.append(report)
        if (not np.isfinite(checks).all() or checks[0] > 2e-4 or checks[1] < 0
                or checks[2] > F_MAX or checks[3] > W_MAX):
            raise RuntimeError(f"Guan quintic failed discrete audit: {report}")
        # Keep shared gate knots exactly once. Only positions need world translation.
        def join_legs(values):
            return np.concatenate([values[0], *[leg[1:] for leg in values[1:]]])

        edges = np.r_[0, np.cumsum(t)]
        tt = join_legs(edges[:-1, None] + t[:, None] * self.grid)
        p, v, a, j = [join_legs((basis @ c) / t[:, None, None]**d)
                      for d, basis in enumerate(self.basis)]
        return dict(durations=t, gates=gates, normals=normals, optimized_passages=gates,
                    reference_time=tt, reference_states=np.c_[p + own[:3], v],
                    reference_acceleration=a, reference_jerk=j, cell_durations=np.diff(tt),
                    guan_inputs=x, relative_quintic_coefficients=c, kinematics_only=np.array(True),
                    audit=report, solve_trace=self.last_trace)


class GuanPlanner(LocalPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = ProjectedQuintics(self.checkpoint)
        self.snap_time_network = self.phase

    def solve(self, own, other, index):
        plan = super().solve(own, other, index)
        plan.update(event_policy="guan_projected_network_times", frontend_solver="guan_minimum_jerk")
        return plan
