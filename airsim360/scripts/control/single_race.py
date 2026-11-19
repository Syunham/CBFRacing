"""Run the Split-S MPCC on one simulated Crazyflie."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.control.adapter import ROOT, controller
from scripts.simulation.dynamics import vehicle_parameters
from scripts.control.profile import baseline_options, configure_racing_tracker
from scripts.control.runtime import resolve_fixed_time_checkpoint
from scripts.paths import FLIGHT_ASSETS
from scripts.planning.fast_fly.guarded_fast_fly import GuardedFastFly, PassagePhase


class FullCoursePhase(PassagePhase):
    def __init__(self, config, course, optimize_durations=True):
        config = dict(config, phase_ipopt_options={"max_wall_time": 180.},
                      phase_cell_function=True, phase_batched_audit=True)
        super().__init__(config, course, optimize_durations)


class FullCourseFastFly(GuardedFastFly):
    phase_class = FullCoursePhase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--native-fast-fly", action="store_true",
                        help="Jointly optimize the 14 leg durations and states, then track that full plan")
    parser.add_argument("--gate-triggered-only", action="store_true",
                        help="Replan after each gate crossing and disable periodic replans")
    parser.add_argument("--fixed-time-checkpoint", type=Path,
                        default=FLIGHT_ASSETS / "models/segment_times_085.pt")
    parser.add_argument("--fixed-time-recovery", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow bounded timing recovery after a failed fixed-time solve")
    args = parser.parse_args()
    if not args.name.isidentifier():
        parser.error("Use a run identifier without path separators")
    if not args.native_fast_fly:
        try:
            checkpoint = resolve_fixed_time_checkpoint(args.fixed_time_checkpoint)
        except ValueError as error:
            parser.error(str(error))

    folder = ROOT / "reports/spatial_potential" / args.name
    folder.mkdir(exist_ok=False)
    os.environ.update(ABLATION_MODE="threat", THREAT_WEIGHT="0", RISK_TEMPORAL_DECAY="0",
                      FASTFLY_NATIVE="0", FASTFLY_BLOCK_HESSIAN="0", FASTFLY_BLOCK_JACOBIAN="0")
    vehicle = vehicle_parameters()
    validate = controller(folder / "adapter", vehicle=vehicle)
    options = baseline_options(validate, args.name)
    configure_racing_tracker(options)
    if args.gate_triggered_only or args.native_fast_fly:
        options.replan_period = 0.
        os.environ["FASTFLY_PERIODIC"] = "0"
    options.phase_optimizer = "fast-fly" if args.native_fast_fly else "fixed-time-sqp"
    options.phase_compact_derivatives = options.phase_online_sqp = False
    options.phase_native_dir = None
    options.allocator = str(FLIGHT_ASSETS / "models/allocation.pt")
    options.h_checkpoint = options.cbf_study = options.stop_separation = None
    options.cbf_mode, options.rti_kernels = "none", "symbolic"
    options.rti_cold_start = options.reuse_last_mpcc = True
    options.diagnostic_window_steps = 5
    options.laps, options.target_gates, options.max_time = 2, 14, 45.
    GuardedFastFly.phase_class.recover_frontend = True
    if args.native_fast_fly:
        options.preview_gates = options.target_gates
        options.phase_cells_per_leg = 30
    else:
        options.fixed_time_checkpoint = checkpoint
        options.fixed_time_recovery = args.fixed_time_recovery

    metadata = dict(vehicle_count=1, vehicle=vehicle, target_gates=options.target_gates,
                    passage_targets="gate centres", start="centred on the nominal start",
                    simulation="project rigid-body plant integrated with applied MPCC controls",
                    replan_period_s=options.replan_period, phase_optimizer=options.phase_optimizer,
                    track_full_course=args.native_fast_fly)
    if not args.native_fast_fly:
        metadata.update(fixed_time_recovery=options.fixed_time_recovery, fixed_time_checkpoint=str(checkpoint.relative_to(ROOT)),
                        fixed_time_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (folder / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    planner = FullCourseFastFly if args.native_fast_fly else GuardedFastFly
    validate.main(options, None, folder / "flight", planner, vehicle_count=1,
                  track_full_course=args.native_fast_fly)


if __name__ == "__main__":
    main()
