"""Configure public planning and RTI profiles before constructing either solver."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import os
import math
from pathlib import Path

from scripts.paths import FLIGHT_ASSETS as ROOT


def resolve_fixed_time_checkpoint(checkpoint):
    path = Path(checkpoint).resolve()
    if not path.is_relative_to((ROOT / "models").resolve()) or not path.is_file():
        raise ValueError("Fixed-time SQP weights must be a file under assets/flight/models")
    return path


def configure(args):
    # Old saved experiment configs predate this optional objective parameter.
    if not hasattr(args,'phase_initial_force_weight'):args.phase_initial_force_weight=0.
    if not hasattr(args,'phase_native_dir'):args.phase_native_dir=None
    if not hasattr(args,'phase_sqp_backend'):args.phase_sqp_backend='daqp'
    if not hasattr(args,'phase_optimize_tail'):args.phase_optimize_tail=False
    if not hasattr(args,'phase_past_lap_warm'):args.phase_past_lap_warm=False
    if not hasattr(args,'phase_learned_initializer'):args.phase_learned_initializer=None
    if not hasattr(args,'phase_learned_budget'):args.phase_learned_budget=3.
    if not hasattr(args,'phase_learned_sqp'):args.phase_learned_sqp=False
    if args.phase_learned_initializer:
        if args.frontend!='phase' or args.phase_optimizer!='fast-fly':raise ValueError('Learned initializer requires the Fast-fly phase frontend')
        args.phase_learned_initializer=str(Path(args.phase_learned_initializer).resolve())
        if not Path(args.phase_learned_initializer).is_file():raise ValueError('Learned initializer checkpoint does not exist')
    if not math.isfinite(args.phase_learned_budget) or args.phase_learned_budget<=0:raise ValueError('Learned initializer solve budget must be positive')
    if not hasattr(args,'phase_tail_solve_seconds'):args.phase_tail_solve_seconds=.3
    if not math.isfinite(args.phase_tail_solve_seconds) or args.phase_tail_solve_seconds<=0:
        raise ValueError('Tail solve budget must be finite and positive')
    if not math.isfinite(args.phase_initial_force_weight) or args.phase_initial_force_weight<0:
        raise ValueError('Initial-force penalty weight must be finite and nonnegative')
    if args.phase_initial_force_weight and (args.frontend!='phase' or args.phase_optimizer not in ('fast-fly','fixed-time-sqp')):
        raise ValueError('Initial-force penalty requires the Fast-fly phase frontend')
    if args.phase_optimizer == 'fixed-time-sqp':
        if args.frontend != 'phase' or args.preview_gates != 5:
            raise ValueError('The fixed-time SQP frontend requires five phase events')
        if args.event_checkpoint:
            raise ValueError('Use the fixed-time SQP checkpoint option for the five-leg timing network')
    fixed_checkpoint = getattr(args, 'fixed_time_checkpoint', None)
    if fixed_checkpoint is not None:
        if args.phase_optimizer != 'fixed-time-sqp':
            raise ValueError('The fixed-time checkpoint requires the fixed-time-sqp frontend')
        fixed_checkpoint = resolve_fixed_time_checkpoint(fixed_checkpoint)
    os.environ['FIXED_TIME_SQP_CHECKPOINT'] = str(fixed_checkpoint) if fixed_checkpoint is not None else ''
    timing_recovery = getattr(args, 'fixed_time_recovery', False)
    if timing_recovery and args.phase_optimizer != 'fixed-time-sqp':
        raise ValueError('Timing recovery requires the fixed-time-sqp frontend')
    os.environ['FIXED_TIME_SQP_TIMING_RECOVERY'] = '1' if timing_recovery else '0'
    snap = args.frontend == 'minimum-snap'
    if args.rti_cold_start is None:
        args.rti_cold_start = snap
    if snap and args.preview_gates != 5:
        raise ValueError('The learned Minimum Snap checkpoint requires five gate events.')
    checkpoint = args.event_checkpoint
    if snap and checkpoint is None:
        checkpoint = ROOT / 'models/snap_short_residual_v2/events.pt'
    if checkpoint:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise ValueError(f'Time checkpoint does not exist: {checkpoint}')
    args.event_checkpoint = str(checkpoint) if checkpoint else None
    if args.start_mode is None:
        args.start_mode = 'gate1_aligned' if snap else 'original'
    if args.progress is None:
        args.progress = 4200. if snap or args.reward_policy=='adaptive' else 220.
    if args.progress_start is None and (snap or args.reward_policy=='adaptive'):
        args.progress_start = 220.
    if args.turn_reward is None:
        args.turn_reward = 900. if snap or args.reward_policy=='adaptive' else 0.
    values = {
        'MINIMUM_SNAP': int(snap), 'EVENT_CHECKPOINT': checkpoint or '',
        'PHASE_OPTIMIZER': args.phase_optimizer,
        'PHASE_CELLS_PER_LEG': args.phase_cells_per_leg,
        'PHASE_INTEGRATION_SUBSTEPS': args.phase_integration_substeps,
        'PHASE_INITIAL_FORCE_WEIGHT': args.phase_initial_force_weight,
        'FASTFLY_NATIVE_DIR': args.phase_native_dir or '',
        'FASTFLY_COMPACT_DERIVATIVES': int(getattr(args,'phase_compact_derivatives',False)),
        'REFERENCE_RESAMPLING': args.reference_resampling,
        'PHASE_FIRST_GATE_APERTURE': int(args.phase_first_gate_aperture),
        'PHASE_AVERAGE_SHIFTED_CONTROLS': int(args.phase_average_shifted_controls),
        'PHASE_DURATION_MIN': args.phase_duration_min,
        'PHASE_ONLINE_SQP': int(args.phase_online_sqp),
        'PHASE_SQP_AUDIT_REFINE': int(getattr(args,'phase_audit_refine',False)),
        'PHASE_SQP_CONSTRAINT_GENERATION': int(getattr(args,'phase_constraint_generation',False)),
        'PHASE_REFERENCE_CONTINUITY': float(getattr(args,'phase_reference_continuity',0.)),
        'PHASE_SQP_BACKEND': args.phase_sqp_backend,
        'PHASE_OPTIMIZE_TAIL': int(args.phase_optimize_tail),
        'PHASE_PAST_LAP_WARM': int(args.phase_past_lap_warm),
        'PHASE_LEARNED_INITIALIZER': args.phase_learned_initializer or '',
        'PHASE_LEARNED_BUDGET': args.phase_learned_budget,
        'PHASE_LEARNED_SQP': int(args.phase_learned_sqp),
        'PHASE_TAIL_SOLVE_SECONDS': args.phase_tail_solve_seconds,
        'PHASE_REMAINING_MIN': args.phase_remaining_min,
        'PHASE_ADAPTIVE_FIRST_BUFFER': int(args.phase_adaptive_first_buffer),
        'PHASE_ADAPTIVE_REMAINING_MIN': int(args.phase_adaptive_remaining_min),
        'PHASE_SQP_TAIL_SEED': int(args.phase_sqp_tail_seed),
        'PHASE_SQP_GATE_TRUST': args.phase_sqp_gate_trust,
        'PHASE_PERIODIC_PROJECTION': int(args.phase_periodic_projection),
        'PHASE_OBJECTIVE_PERIOD': float(getattr(args,'phase_objective_period',0.)),
        'PHASE_GATE_TAIL_NLP': int(args.phase_gate_tail_nlp),
        'PHASE_SQP_DUAL_WARM': int(args.phase_sqp_dual_warm),
        'PHASE_SQP_CHAIN_CONDENSING': int(getattr(args,'phase_sqp_chain_condensing',False)),
        'PHASE_CONSISTENT_TAIL_BOUNDARY': int(getattr(args,'phase_consistent_tail_boundary',False)),
        'PHASE_AUDIT_RETRY_QUANTUM': float(getattr(args,'phase_audit_retry_quantum',0.)),
        'START_MODE': args.start_mode, 'FRONTEND_SCP': 0, 'BODY_RATE_FRONTEND': '',
        'DIRECT_INPUT_MODEL': 0, 'OPT_REPLAN_CARRY': 1, 'OPT_HOVER_INITIAL_SEED': 1,
        'OPT_RTI_COLD_START': int(args.rti_cold_start), 'OPT_LOCAL_GATE_WINDOW': 1,
        'GATE_CROSSING_FORM': 'crossmultiply', 'OPT_GATE_MARGIN_RELEASE_STEPS': 2,
        'GATE_DENSE_PREFIX': args.gate_dense_prefix, 'OPT_TRACKING_FUNNEL': int(not args.physical_gate_guard),
        'POST_GATE_FUNNEL_SLOPE': 10, 'OPT_GEOMETRY_RELEASE': 1,
        'OPT_DEFER_PLAN_WRITES': 1, 'OPT_SQP_RECOVERY': int(args.sqp_recovery),
        'OPT_SQP_RECOVERY_PREFIX': 0, 'OPT_CBF_DECAY_MARGIN_RATE': args.cbf_decay_margin,
        'OPT_PASSAGE_HALF': .12, 'OPT_PASSAGE_VERTICAL_HALF': .12,
        'OPT_EVENT_NOMINAL_HALF': .12, 'OPT_VERTICAL_GATE_IDS': '3,4',
        'OPT_POINT_SCALE': 1, 'OPT_PASSAGE_UPDATE_LIMIT': args.passage_update_limit,
        'OPT_TURN_REWARD': args.turn_reward, 'OPT_TURN_PREVIEW_DISTANCE': 8,
        'OPT_NEIGHBOR_FORECAST': args.neighbor_forecast, 'OPT_QP_RETRY_CLARABEL': 0,
        'OPT_RTI_CERTIFIED_BACKUP': 0, 'OPT_PROJECT_REPLAN_PROGRESS': 0,
        'OPT_REFERENCE_REPAIR_ONLY': 0, 'OPT_SHIFTED_TIME_WARM': 0, 'OPT_EXTEND_WARM': 0,
        'SNAP_KINEMATICS_ONLY': 1, 'SNAP_THRUST_SOC': 0, 'SNAP_SOFT_ENVELOPE': 0,
        'SNAP_PIECES': 1, 'SNAP_INITIAL_JERK': 1, 'SNAP_NODES': 17,
        'SNAP_REGULARIZATION': '1e-10', 'SNAP_TIME_SCALE': 1,
        'SNAP_SHORT_TIME_SCALE': 1, 'SNAP_PATH_ENERGY_WEIGHT': 0,
        'SNAP_SCALE_ROWS': 0, 'SNAP_ELIMINATE_EQUALITIES': 0,
        'EXPERIMENT_QP': 'daqp_direct', 'EXPERIMENT_CONDENSE': 1,
        'FAST_DENSE_QP': 1, 'CONDENSE_METHOD': 'lu',
        'FAST_KERNELS': 0 if args.rti_kernels == 'symbolic' else 'mx',
        'FAST_KERNEL_POLICY': args.rti_kernels,
        'FAST_KERNEL_STEM': 'fast_physical_cross_dense',
    }
    os.environ.update({key: str(value) for key, value in values.items()})
    os.environ['PROGRESS_BACKOFF']='1' if args.progress_backoff else '0'
    os.environ['FASTFLY_THRUST_SMOOTH']=str(args.phase_thrust_smooth)
    os.environ['FASTFLY_WARM_MODE']=args.phase_warm_start
    os.environ['REFERENCE_SPACING']=str(args.reference_spacing)
    os.environ['PREDICTION_THRUST_MARGIN']=str(args.prediction_thrust_margin)
    os.environ['PREDICTION_RATE_MARGIN']=str(args.prediction_rate_margin)
    if args.phase_passage_half is not None:
        os.environ['OPT_PASSAGE_HALF']=str(args.phase_passage_half)
    if args.phase_vertical_half is not None:
        os.environ['OPT_PASSAGE_VERTICAL_HALF']=str(args.phase_vertical_half)
        os.environ['OPT_VERTICAL_GATE_IDS']='0,1,2,3,4,5,6'
    return args
