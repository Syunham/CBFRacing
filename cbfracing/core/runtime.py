"""Configure public planning and RTI profiles before constructing either solver."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def configure(args):
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
        args.progress = 4200. if snap else 220.
    if args.progress_start is None and snap:
        args.progress_start = 220.
    if args.turn_reward is None:
        args.turn_reward = 900. if snap else 0.
    values = {
        'MINIMUM_SNAP': int(snap), 'EVENT_CHECKPOINT': checkpoint or '',
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
        'OPT_NEIGHBOR_FORECAST': 'ca', 'OPT_QP_RETRY_CLARABEL': 0,
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
    return args
