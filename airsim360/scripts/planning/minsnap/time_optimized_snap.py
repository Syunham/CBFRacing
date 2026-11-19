"""Optimize the five segment times around the local fixed-time polynomial QP."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import math

import numpy as np
from scipy.optimize import minimize

from scripts.planning.minsnap.minimum_snap import MinimumSnapPlanner
from scripts.planning.utils.local_planner import LocalPlanner


class VariableTimeSnap(MinimumSnapPlanner):
    time_weight = 100_000.
    gram = np.array([[math.comb(3, i)*math.comb(3, j)/(7*math.comb(6, i+j))
                      for j in range(4)] for i in range(4)])

    def solve(self, own, gates, normals, half, durations):
        # Reuse any remaining time to the *same* gate on periodic updates.
        # All five entries remain independent decision variables thereafter.
        seed = durations.copy()
        previous = self.previous.get(self.agent)
        if previous is not None:
            old_index, old_time, old_durations = previous
            advance = self.index - old_index
            if 0 <= advance < len(seed):
                remaining = old_durations[advance:].copy()
                remaining[0] -= self.stamp - old_time - old_durations[:advance].sum()
                seed[:len(remaining)] = remaining
        lower = np.full(len(seed), self.c['duration_min'])
        lower[0] = .02
        upper = np.full(len(seed), self.c['duration_max'])
        best, score = None, np.inf
        evaluated = {}

        def objective(times):
            nonlocal best, score
            key = tuple(times)
            if key in evaluated:
                return evaluated[key]
            try:
                plan = super(VariableTimeSnap, self).solve(own, gates, normals, half, times)
            except RuntimeError:
                evaluated[key] = 1e9
                return evaluated[key]
            derivative = 840*np.diff(plan['coefficients'], n=4, axis=1)
            energy = np.einsum('eic,ij,ejc->e', derivative, self.gram, derivative)
            energy = np.sum(energy/plan['coefficient_durations']**7)
            value = float(plan['durations'].sum() + energy/self.time_weight)
            if value < score:
                best, score = plan, value
            evaluated[key] = value
            return value

        seed = np.clip(seed, lower, upper)
        initial = objective(seed)
        if best is None:
            seed = np.clip(durations, lower, upper)
            initial = objective(seed)
        result = minimize(objective, seed, method='COBYQA', bounds=list(zip(lower, upper)),
                          options={'initial_tr_radius': .20, 'final_tr_radius': .015, 'maxfev': 40})
        if best is None:
            raise RuntimeError(f'No feasible minimum-snap trajectory in {len(evaluated)} time allocations')
        best['solve_trace'].append(dict(stage='optimize_gate_times', evaluations=len(evaluated),
            status=result.message, seed=seed.tolist(), selected=best['durations'].tolist(),
            seed_cost=initial, cost=score, time_weight=self.time_weight))
        self.previous[self.agent] = (self.index, self.stamp, best['durations'].copy())
        return best


class TimeOptimizedPlanner(LocalPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = VariableTimeSnap(self.c, self.course)
        self.phase.previous = {}

    def solve(self, own, other, index):
        self.phase.agent = self.active_agent
        self.phase.stamp = self.current_time
        self.phase.index = index
        plan = super().solve(own, other, index)
        plan['event_policy'] = 'optimized_gate_times'
        return plan
