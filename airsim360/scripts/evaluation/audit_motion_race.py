"""Measure synchrony and motion error on the same newly flown two-lap recording."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import sys
import numpy as np
import torch
from scripts.evaluation.audit_spatial_flight import audit
from scripts.perception.context_motion import ContextMotionPrior
from scripts.training.train_context_motion import ROOT, PRIOR, LocalFutureField, FlowPrior, extract, retrieval, metrics


@torch.inference_mode()
def report(name):
    torch.set_num_threads(1)
    folder=ROOT/name
    cfg=json.loads((folder/'experiment.json').read_text())
    old=LocalFutureField().eval()
    baseline=torch.load(ROOT/'cf21_local_future01/best.pt',map_location='cpu',weights_only=True)
    old.load_state_dict(baseline['model'])
    saved=torch.load(cfg['future_checkpoint'],map_location='cpu',weights_only=True)
    prior=ContextMotionPrior(PRIOR,saved)
    _,obs,_=extract('race',old,FlowPrior(PRIOR),[name],history=prior.use_history)
    b,=retrieval(obs,[(prior.context_keys,prior.shifts)])
    learned=metrics(prior.motion(*b[:3]),b)
    occupancy=None
    if hasattr(prior,'mixture'):
        from scripts.training.train_motion_mixture import group_fragments,data_for,scores,metrics as occupancy_metrics
        shifts,mass=group_fragments(prior.motion,b[:3])
        count,h=shifts.shape[:2]
        key=b[0][:,None].expand(-1,h,-1).flatten(0,1)
        m,_,_=prior.mixture(key,shifts.flatten(0,1),mass[:,None].expand(-1,h,-1).flatten(0,1),torch.arange(h).repeat(count)*.1)
        at88=.2*shifts[:,8]+.8*shifts[:,9]
        weights88,_,_=prior.mixture(b[0],at88,mass,torch.full((count,),.88))
        learned=metrics((m.reshape(count,h,3,1)*shifts).sum(2),b,(weights88[:,:,None]*at88).sum(1))
        d=data_for(b,prior.motion,FlowPrior(PRIOR),old,8411)
        occupancy=dict(single=occupancy_metrics(d[-1],d[-2]),mixture=occupancy_metrics(scores(prior.mixture,d),d[-2]))
    weights=(.03+b[1].norm(dim=-1)).pow(-2)
    bare=(weights[:,:,None,None]*b[2]).sum(1)/weights.sum(1)[:,None,None]
    no_learning=metrics(bare,b)
    flight=audit(folder/'flight',.5)
    (folder/'flight/spatial_audit.json').write_text(json.dumps(flight,indent=2))
    passes=flight['ordered_crossing_times_s']
    count=min(map(len,passes))
    delta=1000*(np.asarray(passes[0][:count])-passes[1][:count])
    summary=dict(flight=flight,gate_gaps_ms=delta.tolist(),mean_abs_ms=float(np.abs(delta).mean()) if count else None,
                 max_abs_ms=float(np.abs(delta).max()) if count else None,finish_gap_ms=float(delta[-1]) if count==14 else None,
                 model_errors_m=learned,retrieval_errors_m=no_learning,occupancy_scores=occupancy,
                 definition='Delta is Drone0 minus Drone1. Error pairs are previous model and evaluated variant on matched visible returns.')
    (folder/'race_evaluation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    report(sys.argv[1])
