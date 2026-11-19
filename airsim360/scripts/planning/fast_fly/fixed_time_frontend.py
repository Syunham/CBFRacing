"""Adapt the fixed-time SQP plan to the AirSim rolling frontend interface."""

from dataclasses import replace

import numpy as np

from .fixed_time_model import (
    APERTURE_RESERVE, CELLS, CENTERS, LEGS, NORMALS, Window, make_window, split_decisions,
)
from .fixed_time_sqp import FixedTimeSQP
from .time_network import SegmentTimeNetwork


class FixedTimeFrontend:
    solver_name = "fixed_time_sqp"
    event_policy = "network_fixed_sqp"
    optimize_durations = False

    def __init__(self, config, course, checkpoint=None, *, timing_recovery=False):
        if not np.allclose(course["centers"], CENTERS) or not np.allclose(course["normals"], NORMALS):
            raise ValueError("The bundled fixed-time network requires the Split-S gate geometry")
        self.network = SegmentTimeNetwork() if checkpoint is None else SegmentTimeNetwork(checkpoint)
        self.sqp = FixedTimeSQP(screen_qp=True, timing_recovery=timing_recovery)
        self.c = config
        self.c.update(preview_gates=LEGS, cells_per_leg=CELLS, phase_cells_per_leg=CELLS,
                      integration_substeps=4, fixed_time_sqp_trust_radius=self.sqp.trust_radius,
                      fixed_time_sqp_elastic=self.sqp.elastic, fixed_time_sqp_screening=True,
                      fixed_time_sqp_timing_recovery=self.sqp.timing_recovery,
                      fixed_time_sqp_prefix_skip_seconds=self.sqp.prefix_skip_seconds)
        self.cache = {}
        self.last_trace = []
        self.last_window = None

    def window(self, own, event, targets, half, now):
        self.last_trace = []
        self.last_window = None
        window = make_window(own, event, targets, now, self.network)
        half = np.minimum(window.half, half)
        if not np.isfinite(half).all() or np.any(half <= APERTURE_RESERVE):
            raise ValueError("The fixed-time targets must retain positive aperture after reserve")
        return replace(window, half=half)

    def remember_launch(self, agent, event, own, plan, now):
        """Use the audited free-time launch plan only as the next SQP initial guess."""
        cells = plan["cells_per_leg"]
        durations = plan["durations"].copy()
        control_durations = plan.get("cell_durations", np.repeat(durations / cells, cells))
        window = Window(np.asarray(own).copy(), event, plan["gates"].copy(), plan["normals"].copy(),
                        plan["half"].copy(), durations, now)
        self.cache[agent] = dict(window=window, time=plan["reference_time"], states=plan["reference_states"],
                                 control_time=np.r_[0., np.cumsum(control_durations)], controls=plan["U"].T)

    def solve(self, window, agent):
        self.last_window = window
        candidate, info = self.sqp.solve(window, self.cache.get(agent))
        recovery = info.get("timing_recovery")
        self.last_trace = [
            dict(stage=self.solver_name, method=info["method"], iterations=len(info["updates"]),
                 updates=info["updates"], stage_seconds=info["stage_seconds"],
                 skipped_reference_nodes=info.get("skipped_reference_nodes", 0)),
        ]
        if recovery is not None:
            self.last_trace.append(dict(stage="timing_recovery", iterations=len(recovery["updates"]), **recovery))
        self.last_trace.extend(dict(stage="fixed_time_ipopt", **attempt) for attempt in info["ipopt"])
        if candidate is None:
            reason = info["ipopt"][-1]
            status = "audit_failed" if "audit" in reason else reason["status"]
            raise RuntimeError(f"Fixed-time Fast-fly failed at gate event {window.event} ({status})")
        self.cache[agent] = candidate
        window = candidate["window"]
        X, U = split_decisions(candidate["z"])
        if info["method"] == "timing_recovery":
            audit = recovery["updates"][-1]["audit"]
        elif info["method"] == "sqp":
            audit = info["updates"][-1]["audit"]
        else:
            audit = info["ipopt"][-1]["audit"]
        return dict(X=X.T.copy(), U=U.T.copy(), audit=audit,
                    durations=window.durations.copy(), cell_durations=np.repeat(window.durations / CELLS, CELLS),
                    gates=window.gates.copy(), normals=window.normals.copy(), half=window.half.copy(),
                    gate_ids=window.ids, cells_per_leg=CELLS,
                    reference_time=candidate["time"], reference_states=candidate["states"],
                    audited_initial_state=window.own[:13].copy(), solve_trace=self.last_trace,
                    warm_start_used=bool(info["updates"]), frontend_update=info["method"])
