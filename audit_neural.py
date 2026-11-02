"""Audit physical flight and direct learned-CBF use, without inference-time margins."""
from context import *
from barrier import load,features
from audit import audit

def verify(folder):
    folder=Path(folder);physical=audit(folder);cfg=json.loads((folder/'config.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    data=np.load(folder/'execution.npz');t=data['time'];x=data['state'];logs=json.loads((folder/'planning_log.json').read_text())
    if not cfg.get('h_checkpoint'):
        result=dict(direct_output=False,mode=cfg['cbf_mode'],full_goal_case_passed=False,scope='Baseline without a learned safety checkpoint')
        save_json(folder/'neural_audit.json',result);return result
    net,ck=load(cfg['h_checkpoint'])
    assert ck['direct_output'] is True
    with torch.no_grad():hs=np.column_stack([net(torch.tensor(features(x[:,i],x[:,1-i]))).numpy() for i in range(2)])
    residual=hs[1:]-np.exp(-cfg['cbf_alpha']*np.diff(t))[:,None]*hs[:-1]
    peaks=np.linalg.norm(x[:,:,3:6],axis=-1).max(axis=0)
    rows=[r['solve']['barrier_diagnostics'] for r in logs if r['solve']['status']==0]
    result=dict(direct_output=True,checkpoint_sha256=hashlib.sha256(Path(cfg['h_checkpoint']).read_bytes()).hexdigest(),mode=cfg['cbf_mode'],
        analytic_distance_constraint_present=cfg['cbf_mode']=='hybrid',learned_hard_constraint=cfg['cbf_mode'] in ('hard','soft_derivative','hybrid'),hard_discrete_condition=cfg['cbf_mode']=='hard',
        peaks_per_agent=peaks,peak_speed=float(peaks.max()),minimum_actual_h=hs.min(axis=0),actual_discrete_residual_min=residual.min(axis=0) if len(residual) else [],
        actual_discrete_violation_count=(residual < -1e-5).sum(axis=0) if len(residual) else [],learned_active_prediction_rows=sum(r.get('learned_active_rows',0) for r in rows),
        learned_dual_max=max((r.get('learned_dual_max',0) for r in rows),default=0),discrete_dual_max=max((r.get('discrete_dual_max',0) for r in rows),default=0),
        accepted_query_count=len(rows),physical_full_pass=physical['full_goal_case_passed'],speed_target_met=bool(peaks.max()>=12),both_agents_peak_target=bool(peaks.min()>=12),
        full_goal_case_passed=bool(physical['full_goal_case_passed'] and peaks.max()>=12 and cfg['cbf_mode'] in ('hard','soft_derivative') and len(rows)>0),
        scope='Actual trajectories under both simultaneous controls; predicted-neighbor model differs. Numerical checks, no invariant-set theorem.')
    save_json(folder/'neural_audit.json',result);np.savez_compressed(folder/'actual_barrier.npz',time=t,h=hs,residual=residual)
    print(folder.name,result,flush=True);return result

if __name__=='__main__':
    for folder in sys.argv[1:]:verify(folder)
