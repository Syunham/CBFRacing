"""Resolve the original analytic defaults or the frozen direct-CBF v7 profile."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
ANALYTIC_DEFAULTS=dict(iterations=900,solver_seconds=8.,gate_halfwidth=.20,
    tube_recovery=False,dense_physics=False,phase_retries=0,cbf_alpha=3.)

def resolve_safety(args):
    learned=args.cbf_branch=='learned';defaults=ANALYTIC_DEFAULTS.copy()
    profile=json.loads((HERE/'learned_cbf_profile.json').read_text()) if learned else None
    if learned:
        for key in defaults:
            defaults[key]=profile['alpha' if key=='cbf_alpha' else key]
    for key,value in defaults.items():
        if getattr(args,key,None) is None:setattr(args,key,value)
    if learned:
        checkpoint=Path(args.h_checkpoint) if args.h_checkpoint else HERE/profile['checkpoint']
        if not checkpoint.is_file():raise ValueError('Learned CBF checkpoint does not exist: '+str(checkpoint))
        args.h_checkpoint=str(checkpoint.resolve())
    elif args.h_checkpoint:
        raise ValueError('--cbf-checkpoint requires --cbf learned')
    args.cbf_mode='hard' if learned else 'analytic'
    args.retry_infeasible=learned
    if args.cbf_alpha<=0 or args.phase_retries<0 or args.gate_halfwidth<=0:
        raise ValueError('CBF rate/gate width must be positive and phase retries nonnegative')
    return args
