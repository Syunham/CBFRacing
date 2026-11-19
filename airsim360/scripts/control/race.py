"""Two-agent Split-S flight with Fast-fly, depth odometry and joint CBF-QP."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from scripts.paths import FLIGHT_ASSETS
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from scripts.control.adapter import ROOT, controller
from scripts.simulation.dynamics import vehicle_parameters
from scripts.control.profile import baseline_options, configure_racing_tracker
from scripts.control.runtime import resolve_fixed_time_checkpoint
from scripts.perception.depth_sensor import DepthBarrierVision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--phase-optimizer", choices=("fast-fly", "fixed-time-sqp"), default="fast-fly",
                        help="Select the rolling race planner")
    parser.add_argument("--fixed-time-checkpoint", type=Path,
                        help="Select network weights from assets/flight/models")
    parser.add_argument("--fixed-time-recovery", action="store_true",
                        help="Allow bounded timing recovery after a fixed-time SQP failure")
    parser.add_argument("--replan-period", type=float,
                        help="Periodic replanning in simulated seconds. Zero replans only on gate crossings.")
    args = parser.parse_args()
    if not args.name.isidentifier():
        parser.error("Use a run identifier without path separators")
    if args.fixed_time_recovery and args.phase_optimizer != "fixed-time-sqp":
        parser.error("--fixed-time-recovery requires --phase-optimizer fixed-time-sqp")
    if args.replan_period is not None and (not math.isfinite(args.replan_period) or args.replan_period < 0.):
        parser.error("--replan-period must be finite and nonnegative")
    checkpoint = None
    if args.fixed_time_checkpoint is not None or args.phase_optimizer == "fixed-time-sqp":
        if args.phase_optimizer != "fixed-time-sqp":
            parser.error("--fixed-time-checkpoint requires --phase-optimizer fixed-time-sqp")
        try:
            checkpoint = resolve_fixed_time_checkpoint(args.fixed_time_checkpoint or FLIGHT_ASSETS/"models/segment_times_085.pt")
        except ValueError as error:
            parser.error(str(error))
    folder = ROOT/"reports/spatial_potential"/args.name
    folder.mkdir(exist_ok=False)
    os.environ.update(ABLATION_MODE="threat", THREAT_WEIGHT="0", RISK_TEMPORAL_DECAY="0",
                      FASTFLY_NATIVE="0", FASTFLY_BLOCK_HESSIAN="0", FASTFLY_BLOCK_JACOBIAN="0",
                      RGBD_IMPLICIT="0", RGBD_LOCAL_FLOW="0", SINGLE_FLIGHT_CAMERA="1")
    for key in ("RGBD_TIME_CHECKPOINT", "RGBD_DIRECTIONAL", "RGBD_MIXTURE"):
        os.environ.pop(key, None)
    vehicle = vehicle_parameters()
    validate = controller(folder/"adapter", vehicle=vehicle)
    from scripts.planning.fast_fly.guarded_fast_fly import GuardedFastFly
    options = baseline_options(validate, args.name, replan_period=args.replan_period)
    configure_racing_tracker(options)
    options.phase_optimizer = args.phase_optimizer
    options.fixed_time_checkpoint = checkpoint
    options.fixed_time_recovery = args.fixed_time_recovery
    GuardedFastFly.phase_class.recover_frontend = True
    options.phase_compact_derivatives = options.phase_online_sqp = False
    options.phase_native_dir = None
    options.name = args.name
    options.separation = 1.2
    options.stop_separation = None
    options.cbf_study = "filtered"
    options.cbf_after_first_gate = options.continue_infeasible_cbf = options.reuse_last_mpcc = True
    options.original_allocation = options.visual_peer = options.rti_cold_start = True
    options.allocator = str(FLIGHT_ASSETS/"models/allocation.pt")
    options.h_checkpoint, options.cbf_mode, options.rti_kernels = None, "none", "symbolic"
    options.diagnostic_window_steps, options.max_time = 5, 45.
    metadata = dict(name=args.name, vehicle=vehicle, separation=options.separation,
                    stop_separation=None, cbf_study=options.cbf_study, depth_pair=True,
                    phase_optimizer=options.phase_optimizer, replan_period=options.replan_period,
                    guarded_fast_fly=True, original_allocation=True,
                    recover_frontend=GuardedFastFly.phase_class.recover_frontend,
                    rti_cold_start=True, cbf_after_first_gate=True, continue_infeasible_cbf=True,
                    reuse_last_mpcc=True, weight=0., resolution=[512, 256])
    sources = ("control/race.py", "control/profile.py", "control/pair_cbf.py",
               "planning/fast_fly/guarded_fast_fly.py", "perception/depth_pair.py",
               "perception/depth_sensor.py", "perception/panorama.py")
    if options.phase_optimizer == "fixed-time-sqp":
        metadata["launch_frontend"] = "fast-fly through first G1"
        metadata["fixed_time_checkpoint"] = str(checkpoint.relative_to(ROOT))
        metadata["fixed_time_checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        metadata["fixed_time_recovery"] = options.fixed_time_recovery
        sources += tuple(f"planning/fast_fly/{name}.py" for name in
                         ("fixed_time_frontend", "fixed_time_model", "fixed_time_sqp", "time_network"))
    (folder/"experiment.json").write_text(json.dumps(metadata, indent=2))
    for source in sources:
        shutil.copy2(ROOT/"scripts"/source, folder/"adapter"/Path(source).name)
    geometry = json.loads((ROOT/"captures/crazyflie21_split_s/geometry.json").read_text())
    sensor = DepthBarrierVision(geometry, metadata["resolution"], folder, options.separation)
    try:
        validate.main(options, sensor, folder/"flight", GuardedFastFly)
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
