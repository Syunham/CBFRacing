"""Run the original rigid-body Split-S test with live RGB-D spatial potentials."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from scripts.paths import FLIGHT_ASSETS
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from scripts.control.profile import baseline_options, configure_racing_tracker
from scripts.control.adapter import ROOT, controller
from scripts.perception.depth_sensor import DepthBarrierVision
from scripts.perception.live_vision import LiveVision, ZeroPotential


def main(argv=None, *, vision_class=LiveVision, load_controller=controller):
    p = argparse.ArgumentParser()
    models = ROOT/"reports/spatial_potential"
    current = "crazyflie21" if (models/"crazyflie21/training.json").is_file() else "sensor12hz"
    p.add_argument("--checkpoint", type=Path, default=models/current/"best.pt")
    p.add_argument("--time-checkpoint",type=Path,help="Frozen RGBDField used as the time-conditioned MPCC cost")
    p.add_argument("--future-checkpoint",type=Path,help="Learned local RGB-D future field for MPCC")
    p.add_argument("--implicit-checkpoint",type=Path,help="Continuous peer occupancy plus existing spatial support")
    p.add_argument("--flow-prior",type=Path,help="RGB-D occupancy transport fitted on training flights")
    p.add_argument("--local-flow",type=Path,help="Fuse the temporal network with a measured RGB-D relative-motion field")
    p.add_argument("--flow-guidance",action="store_true",help="Use local visual repulsion to adapt the MPCC racing reference")
    p.add_argument("--recover-frontend",action="store_true",help="Retry an infeasible guarded approach from the current state")
    p.add_argument("--rti-cold-start",action="store_true",help="Start RTI from its feasible hover seed")
    p.add_argument("--name", required=True)
    p.add_argument("--cbf-study", choices=("nominal","filtered"))
    p.add_argument("--continue-infeasible-cbf", action="store_true")
    p.add_argument("--reuse-last-mpcc", action="store_true",
                   help="On a failed solve reuse this aircraft's last successful nominal command before CBF")
    p.add_argument("--cbf-after-first-gate", action="store_true",
                   help="Enable the pair filter once both aircraft have passed the first gate")
    p.add_argument("--optimize-gate-times", action="store_true")
    p.add_argument("--snap-time-network",type=Path,help="Use learned segment times directly in the fixed-time Minimum Snap QP")
    p.add_argument("--guan-time-network",type=Path,help="Use projected network times with the Guan quintic reference")
    p.add_argument("--original-allocation", action="store_true", help="Use local allocation.pt with original joint state input")
    p.add_argument("--visual-peer", action="store_true", help="Supply RGB-D peer observations to allocation.pt and the pair CBF")
    p.add_argument("--depth-pair", action="store_true", help="Cooperative depth-only pair state for allocation and CBF")
    p.add_argument("--separation", type=float, required=True)
    p.add_argument("--stop-separation", type=float, help="Optional centre-to-centre stop distance in metres")
    p.add_argument("--weight", type=float, default=350.)
    p.add_argument("--passage-gain",type=float,default=1.,
                   help="Calibration of the visual gate offset, before the original aperture checks")
    p.add_argument("--passage-head",type=Path,help="Fitted visual gate-plane output weights")
    p.add_argument("--passage-weight", type=float, default=9.,
                   help="Weight on deviation from the original visual passing side")
    p.add_argument("--max-time", type=float, default=45.)
    p.add_argument("--visual-aperture", action="store_true",
                   help="Constrain the imminent Fast-fly passage around the RGB-D selected point")
    reference = p.add_mutually_exclusive_group()
    reference.add_argument("--guarded-fast-fly", action="store_true",
                           help="Reproduce the guarded Fast-fly reference")
    reference.add_argument("--minimum-snap", action="store_true",
                           help="Use the local five-gate minimum-snap planner")
    reference.add_argument("--corridor", dest="corridor", action="store_true",
                           help="Use the normal-aligned spatial gate reference (default)")
    reference.add_argument("--fast-fly", dest="corridor", action="store_false",
                           help="Use the original racing frontend for comparison")
    p.set_defaults(corridor=None)
    p.add_argument("--field-off", action="store_true")
    p.add_argument("--gate-offset", nargs=3, type=float, metavar=("GATE", "SIDE", "UP"),
                   help="Perturb rendered gate only; the controller retains the original map")
    p.add_argument("--visual-refine", action="store_true",
                   help="Use spatial queries to update the reference near observed gate frame ridges")
    p.add_argument("--replay-passages", type=Path,
                   help="For cost ablation, retain passage offsets chosen in a completed run")
    args = p.parse_args(argv)
    if args.depth_pair and (not args.original_allocation or args.cbf_study != "filtered" or args.weight != 0 or
                            any((args.flow_prior, args.time_checkpoint, args.field_off, args.visual_peer,
                                 args.implicit_checkpoint, args.future_checkpoint, args.local_flow))):
        p.error("--depth-pair requires original allocation, filtered CBF and zero risk weight")
    if args.visual_peer and (not args.flow_prior or not args.original_allocation):
        p.error("--visual-peer needs --flow-prior and --original-allocation")
    if args.implicit_checkpoint and (args.future_checkpoint or args.time_checkpoint or args.flow_prior or
                                     args.local_flow or args.field_off):
        p.error("Select one dynamic MPCC risk field")
    os.environ["RGBD_IMPLICIT"] = "1" if args.implicit_checkpoint else "0"
    if args.implicit_checkpoint:
        torch.set_num_threads(1)
    if args.future_checkpoint and (args.time_checkpoint or args.flow_prior or args.local_flow or args.field_off):
        p.error("Select one MPCC risk field")
    if not np.isfinite(args.passage_gain) or args.passage_gain <= 0:
        p.error("The gate output gain must be positive and finite")
    if args.passage_head and (args.passage_gain != 1 or args.field_off):
        p.error("The fitted gate head needs active vision and unit passage gain")
    if args.flow_prior and (args.field_off or args.time_checkpoint or args.visual_refine):
        p.error("--flow-prior requires the spatial support adapter")
    if args.flow_guidance and not args.flow_prior:
        p.error("--flow-guidance needs --flow-prior")
    if args.stop_separation is not None and not 0 < args.stop_separation <= args.separation:
        p.error("--stop-separation must be positive and at most --separation")
    if args.time_checkpoint and (args.field_off or args.visual_refine):
        p.error("The time-query comparison requires active risk and the original gate references")
    if args.time_checkpoint:
        os.environ["RGBD_TIME_CHECKPOINT"] = str(args.time_checkpoint.resolve())
        torch.set_num_threads(1)
    else:
        os.environ.pop("RGBD_TIME_CHECKPOINT",None)
    if args.local_flow and (not args.time_checkpoint or args.flow_prior):
        p.error("--local-flow requires a time checkpoint and a single visual field adapter")
    os.environ["RGBD_LOCAL_FLOW"] = "1" if args.local_flow else "0"
    if args.corridor is None:
        args.minimum_snap |= args.depth_pair and not args.guarded_fast_fly
        args.corridor = not (args.guarded_fast_fly or args.minimum_snap)
    if args.replay_passages and not (args.field_off and args.corridor):
        p.error("Passage replay requires --field-off --corridor")
    if args.optimize_gate_times and not args.minimum_snap:
        p.error("--optimize-gate-times requires minimum snap")
    if args.snap_time_network and (not args.minimum_snap or args.optimize_gate_times):
        p.error("--snap-time-network requires fixed-time minimum snap")
    if args.guan_time_network and (not args.minimum_snap or args.snap_time_network or args.optimize_gate_times):
        p.error("--guan-time-network requires the standalone five-gate polynomial frontend")
    folder = ROOT/"reports/spatial_potential"/args.name
    folder.mkdir(exist_ok=False)
    os.environ.update(ABLATION_MODE="threat", THREAT_WEIGHT=str(0 if args.field_off else args.weight),
                      RISK_TEMPORAL_DECAY="0", FASTFLY_NATIVE=str(int(not args.minimum_snap)),
                      FASTFLY_BLOCK_HESSIAN=str(int(not args.minimum_snap)),
                      FASTFLY_BLOCK_JACOBIAN=str(int(not args.minimum_snap)))
    validate = load_controller(folder/"adapter")
    planner_class = validate.LocalPlanner
    options = baseline_options(validate, args.name, args.corridor)
    options.name, options.separation = args.name, args.separation
    options.stop_separation = args.stop_separation
    options.cbf_study = args.cbf_study
    options.cbf_after_first_gate = args.cbf_after_first_gate
    options.continue_infeasible_cbf = args.continue_infeasible_cbf
    options.reuse_last_mpcc = args.reuse_last_mpcc
    options.optimize_gate_times = args.optimize_gate_times
    options.guan_time_network = args.guan_time_network
    options.snap_time_network = str(args.snap_time_network.resolve()) if args.snap_time_network else None
    options.original_allocation = args.original_allocation
    options.visual_peer = args.visual_peer or args.depth_pair
    if args.cbf_study:
        shutil.copyfile(ROOT/"scripts/control/pair_cbf.py", folder/"adapter/pair_cbf.py")
    options.max_time = args.max_time
    if args.rti_cold_start:
        options.rti_cold_start = True
    options.allocator = str(FLIGHT_ASSETS/"models/allocation.pt")
    options.h_checkpoint, options.cbf_mode, options.rti_kernels = None, "none", "symbolic"
    options.diagnostic_window_steps = 5
    if args.visual_aperture:
        options.phase_first_gate_aperture = False
        options.gate_prediction_margin = .20
    if args.guarded_fast_fly:
        from scripts.planning.fast_fly.guarded_fast_fly import GuardedFastFly
        GuardedFastFly.phase_class.recover_frontend = args.recover_frontend
        planner_class = GuardedFastFly
    if args.minimum_snap:
        if args.optimize_gate_times:
            from scripts.planning.minsnap.time_optimized_snap import TimeOptimizedPlanner
            planner_class = TimeOptimizedPlanner
            shutil.copyfile(ROOT/"scripts/planning/minsnap/time_optimized_snap.py", folder/"adapter/time_optimized_snap.py")
        options.frontend = "minimum-snap"
        options.phase_optimizer = "minimum-snap"
        options.phase_initial_force_weight = 0.
        options.phase_compact_derivatives = False
        snap_checkpoint = FLIGHT_ASSETS/"models/snap_short_residual_v2/events.pt"
        options.event_checkpoint = str(snap_checkpoint.resolve()) if snap_checkpoint.is_file() else ""
        if args.snap_time_network:
            from scripts.planning.minsnap.snap_time_network import FixedTimePredictor
            options.event_checkpoint = ""
            class NetworkTimePlanner(planner_class):
                snap_time_network = FixedTimePredictor(args.snap_time_network)
            planner_class = NetworkTimePlanner
            shutil.copyfile(ROOT/"scripts/planning/minsnap/snap_time_network.py",folder/"adapter/snap_time_network.py")
        if args.guan_time_network:
            from scripts.planning.minsnap.guan_closed_loop import GuanPlanner
            options.event_checkpoint = ""
            GuanPlanner.checkpoint = args.guan_time_network.resolve()
            planner_class = GuanPlanner
            for source in (ROOT/"scripts/planning/minsnap/guan_closed_loop.py", ROOT/"scripts/planning/minsnap/guan_minjerk.py"):
                shutil.copyfile(source, folder/"adapter"/source.name)
    if args.guarded_fast_fly or args.minimum_snap:
        configure_racing_tracker(options)
    if args.corridor:
        from scripts.planning.utils.spatial_gate_reference import GateReference
        planner_class = GateReference
        options.phase_optimizer = "spatial-corridor"
        options.phase_initial_force_weight = 0.
        options.replan_period = 0.
        options.reward_policy = "fixed"
        options.progress_start = options.progress = 110.
        options.turn_reward = 0.
        options.contour_weight, options.lag_weight = 1500., 2500.
        options.reference_speed_weight, options.reference_speed_extra = 350., 0.
        options.velocity_contour_weight = 25.
        options.gate_prediction_margin = .20
    metadata = dict(vars(args), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    if args.guan_time_network:
        metadata["guan_time_network_sha256"] = hashlib.sha256(args.guan_time_network.read_bytes()).hexdigest()
    if args.implicit_checkpoint:
        metadata["implicit_sha256"] = hashlib.sha256(args.implicit_checkpoint.read_bytes()).hexdigest()
        metadata["implicit_query"] = "world logit tangent at RTI nodes; maximum with spatial support"
        for source in (ROOT/"scripts/perception/implicit_rgbd.py", ROOT/"scripts/training/train_rgbd_field.py"):
            shutil.copyfile(source, folder/"adapter"/source.name)
    os.environ.pop("RGBD_DIRECTIONAL",None)
    os.environ.pop("RGBD_MIXTURE",None)
    if args.future_checkpoint:
        from scripts.perception.local_future_field import LocalFutureField, VisualMotionPrior, FlowPrior
        torch.set_num_threads(1)
        saved = torch.load(args.future_checkpoint,map_location="cpu",weights_only=True)
        future_model = LocalFutureField(saved.get("correction_speed",3.)).eval()
        future_model.load_state_dict(saved["model"])
        if "directional" in saved:
            from scripts.perception.directional_motion_risk import DirectionalRisk
            future_model.directional=DirectionalRisk().eval()
            future_model.directional.load_state_dict(saved["directional"])
            os.environ["RGBD_DIRECTIONAL"]="1"
            shutil.copyfile(ROOT/"scripts/perception/directional_motion_risk.py",folder/"adapter/directional_motion_risk.py")
        prior_path = ROOT/Path(saved["prior"])
        if "motion" in saved:
            from scripts.perception.context_motion import ContextMotionPrior
            future_prior = ContextMotionPrior(prior_path,saved)
            shutil.copyfile(ROOT/"scripts/perception/context_motion.py",folder/"adapter/context_motion.py")
            if 'mixture' in saved:
                os.environ["RGBD_MIXTURE"]="1"
                shutil.copyfile(ROOT/"scripts/training/train_motion_mixture.py",folder/"adapter/train_motion_mixture.py")
        else:
            future_prior = (VisualMotionPrior if saved.get("visual_only") else FlowPrior)(prior_path)
        metadata.update(future_sha256=hashlib.sha256(args.future_checkpoint.read_bytes()).hexdigest(),
                        future_prior=str(prior_path),future_prior_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                        encoder_history_period_s=.02,max_lead_s=1.6)
        for source in (ROOT/"scripts/perception/local_future_field.py", ROOT/"scripts/perception/occupancy_flow.py"):
            shutil.copyfile(source, folder/"adapter"/source.name)
    if args.passage_head:
        with np.load(args.passage_head) as head_file:
            head = head_file["weight"]
            geometry_weight = head_file.get("geometry_weight")
        if head.shape != (2,) or not np.isfinite(head).all():
            raise ValueError("Passage head must have two finite gate-plane weights")
        if geometry_weight is not None and (geometry_weight.shape != (2,2) or not np.isfinite(geometry_weight).all()):
            raise ValueError("Gate geometry calibration needs four finite weights")
        metadata["passage_head_sha256"] = hashlib.sha256(args.passage_head.read_bytes()).hexdigest()
    if args.local_flow:
        metadata["local_flow_sha256"] = hashlib.sha256(args.local_flow.read_bytes()).hexdigest()
        shutil.copyfile(ROOT/"scripts/perception/occupancy_flow.py",folder/"adapter/occupancy_flow.py")
    if args.flow_prior:
        metadata["flow_sha256"] = hashlib.sha256(args.flow_prior.read_bytes()).hexdigest()
        shutil.copyfile(ROOT/"scripts/perception/occupancy_flow.py",folder/"adapter/occupancy_flow.py")
    if args.time_checkpoint:
        time_spec = torch.load(args.time_checkpoint,map_location="cpu",weights_only=True)
        metadata.update(time_checkpoint_sha256=hashlib.sha256(args.time_checkpoint.read_bytes()).hexdigest(),
                        encoder_history_period_s=.02,encoder_resolution=[512,256] if time_spec.get("wide") else [256,128],max_lead_s=1.6)
        shutil.copyfile(ROOT/"scripts/perception/time_query_field.py",folder/"adapter/time_query_field.py")
    lane_limit = .24 if args.guarded_fast_fly or args.minimum_snap else .42
    if args.guarded_fast_fly or args.minimum_snap:
        metadata["lane_limit_m"] = lane_limit
        planner_source = (ROOT/"scripts/planning/minsnap/minimum_snap.py" if args.minimum_snap
                          else ROOT/"scripts/planning/fast_fly/guarded_fast_fly.py")
        for source in (Path(__file__), ROOT/"scripts/perception/live_vision.py", planner_source):
            shutil.copyfile(source, folder/"adapter"/source.name)
    (folder/"experiment.json").write_text(json.dumps(metadata, default=str, indent=2))
    if args.replay_passages:
        geometry = args.replay_passages.parent/"rendered_course.json"
        if geometry.exists():
            shutil.copyfile(geometry, folder/"rendered_course.json")
    if args.depth_pair:
        shutil.copyfile(ROOT/"scripts/perception/depth_pair.py", folder/"adapter/depth_pair.py")
        sensor = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        vision = DepthBarrierVision(sensor["vehicle_geometry"], sensor["resolution"], folder, args.separation)
    elif args.field_off:
        vision = ZeroPotential(args.replay_passages)
    else:
        vision = vision_class(args.checkpoint, folder, args.gate_offset, args.visual_refine,
                            args.passage_weight, lane_limit, args.time_checkpoint, args.flow_prior,
                            args.flow_guidance and args.weight>0, args.implicit_checkpoint)
    vision.passage_gain = args.passage_gain
    if args.visual_peer:
        from scripts.perception.visual_peer import VisualPeer
        shutil.copyfile(ROOT/"scripts/perception/visual_peer.py", folder/"adapter/visual_peer.py")
        vision.peer_tracks = [VisualPeer(), VisualPeer()]
        vision.peer_log = (folder/"visual_peer.jsonl").open("w", buffering=1)
    if args.future_checkpoint:
        vision.future_model, vision.future_prior = future_model, future_prior
        vision.future_history = deque(maxlen=4)
    vision.passage_head = head if args.passage_head else None
    vision.passage_geometry = geometry_weight if args.passage_head else None
    if vision.passage_geometry is not None:
        from scripts.planning.utils.passage_calibration import bend_features
        centers = np.asarray(vision.nominal["centers"])
        axes = np.asarray(vision.nominal["coordinate_axes"])
        vision.gate_bends = dict(zip(map(tuple,centers),bend_features(centers,axes)))
        shutil.copyfile(ROOT/"scripts/planning/utils/passage_calibration.py",folder/"adapter/passage_calibration.py")
    if args.local_flow:
        from scripts.perception.occupancy_flow import FlowPrior, OccupancyFlow
        local_prior = FlowPrior(args.local_flow)
        vision.local_flow = [OccupancyFlow(local_prior) for _ in range(2)]
    try:
        validate.main(options, vision, folder/"flight", planner_class)
    finally:
        vision.close()
        if args.visual_peer:
            vision.peer_log.close()


if __name__ == "__main__":
    main()
