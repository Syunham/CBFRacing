"""Gate passage geometry for the rolling, five-gate Fast-fly optimizer."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
from scripts.planning.utils.local_planner import LocalPlanner
from scripts.planning.fast_fly.fast_fly import FastFlyPlanner


class PassagePhase(FastFlyPlanner):
    def solve(self, state, gates, normals, half, durations, warm=None):
        cells = self.c["cells_per_leg"]
        if not hasattr(self, "base_position_upper"):
            self.base_position_upper = self.ubx.copy()
            self.base_lower = self.lbx.copy()
        # Put crossing-velocity bounds into the existing native NLP and SQP;
        # both solvers consume these variable bounds on every solve.
        for j, n in enumerate(normals):
            physical = (self.event_index+j) % self.track_gate_count
            slow_turn = physical in (3, 4)
            low, high = (5.5, 8.) if slow_turn else (5., 14.)
            side = np.abs(np.cross([0., 0., 1.], n))
            a, b = low*n, high*n
            k = (j+1)*cells*13+3
            vertical = .65 if slow_turn else 1.4
            self.lbx[k:k+3] = np.minimum(a, b) - 1.2*side - [0., 0., vertical]
            self.ubx[k:k+3] = np.maximum(a, b) + 1.2*side + [0., 0., vertical]
            # The Split-S stacked pair opens onto the same negative-X side.
            # Make space behind its shared crossbars before reversing direction.
            mid_x = (j*cells+cells//2)*13
            self.ubx[mid_x] = self.base_position_upper[mid_x]
            if physical == 4 and j > 0:
                self.ubx[mid_x] = min(self.ubx[mid_x], gates[j, 0]-1.2)
        try:
            return super().solve(state, gates, normals, half, durations, warm)
        except RuntimeError as error:
            if not getattr(self,"recover_frontend",False) or "Phase initialization failed" not in str(error):
                raise
            failure = str(error).split("'solver_timing'")[0][:350]
        # A strongly deflected approach may not satisfy the nominal first-gate
        # velocity band. Re-solve the same course from the actual state with a
        # fresh seed. Gate aperture, forward crossing and plant limits still apply.
        self.cache.pop(self.active_agent,None)
        first_velocity = cells*13+3
        first = slice(first_velocity,first_velocity+3)
        self.lbx[first] = self.base_lower[first]
        self.ubx[first] = self.base_position_upper[first]
        old_min = self.c["duration_min"]
        remaining = max(1e-4, float((gates[0]-state[:3]) @ normals[0]))
        approach_speed = max(.8,float(state[3:6] @ normals[0]))
        self.c["duration_min"] = min(old_min,.25*remaining/approach_speed)
        retry_times = durations.copy()
        retry_times[0] = np.clip(remaining/approach_speed,self.c["duration_min"],durations[0])
        try:
            plan = super().solve(state,gates,normals,half,retry_times,warm=None)
        finally:
            self.c["duration_min"] = old_min
        plan["solve_trace"].append(dict(stage="fresh_first_gate_velocity_solve", prior_error=failure, iterations=0))
        return plan


class GuardedFastFly(LocalPlanner):
    phase_class = PassagePhase

    def keep_approach(self, agent, index, own):
        saved = self.previous_plans.get(agent)
        if saved is None or saved[0] != index:
            return False
        c = self.course
        j = index % len(c["centers"])
        n, center = np.array(c["normals"][j]), np.array(c["centers"][j])
        remaining = (center-own[:3]) @ n
        return 0. < remaining < .23*max(own[3:6] @ n, 0.)
