"""Bounded fixed-state counterfactual OCPs; not a closed-loop benchmark."""
import sys
import argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'swarm_cpc_reproduction/.deps'))
from context import *
from controller import RecoveryMPCC,DT
from reference import reference_from_plan,online_data
from scipy.spatial.distance import cdist
out=HERE/'diagnosis_speed_20261102';out.mkdir(exist_ok=True)
ap=argparse.ArgumentParser();ap.add_argument('--times',nargs='+',type=float,default=[2.8,5.6]);ap.add_argument('--suffix',default='');args=ap.parse_args()
folder=HERE/'outputs/integrated_learned_20261030';cfg=json.loads((folder/'config.json').read_text())
z=np.load(folder/'execution.npz');times=z['time'];states=z['state'];front=json.loads((folder/'frontends.json').read_text());logs=json.loads((folder/'planning_log.json').read_text())
rows=[];cache={};coverage=[]
for row in logs:
 i=row['agent'];ti=row['time'];k=int(round(ti/.02));candidates=[(j,f) for j,f in enumerate(front) if f['agent']==i and f['time']<=ti+1e-8 and f['success']]
 j,f=candidates[-1]
 if j not in cache:
  file=list((folder/'plans').glob(f'{j:03d}_*.npz'))[0];p=dict(np.load(file));ref=reference_from_plan(p,states[int(round(f['time']/.02)),i]);cache[j]=(p,ref)
 p,ref=cache[j];remaining=ref['length']-states[k,i,17];speed=np.linalg.norm(states[k,i,3:6]);coverage.append(dict(time=ti,agent=i,remaining=float(remaining),speed=float(speed),ratio=float(remaining/max(speed*DT*20,1e-9))))
save_json(out/'reference_coverage.json',dict(fraction_shorter_than_constant_speed_horizon=float(np.mean([r['ratio']<1 for r in coverage])),rows=coverage))
# Two representative states, one on an early leg and one near the Split-S turn.
for ti in args.times:
 i=0;k=int(round(ti/.02));own=states[k,i].copy();other=states[k,1-i].copy()
 j,f=[(j,f) for j,f in enumerate(front) if f['agent']==i and f['time']<=ti+1e-8 and f['success']][-1];p,ref=cache[j];ref=dict(ref,current_position=own[:3])
 od=online_data(ref,np.linspace(own[17],min(ref['length'],own[17]+20),21),other,cfg)
 a=np.clip((states[k,1-i,3:6]-states[k-1,1-i,3:6])/.02,-35,35);tt=np.arange(21)*DT;tau=.35
 od[:,16:19]=other[:3]+other[3:6]*tt[:,None]+a*tau*(tt[:,None]-tau*(1-np.exp(-tt[:,None]/tau)))
 od[:,19:22]=other[3:6]+a*tau*(1-np.exp(-tt[:,None]/tau));od[:,40]=.16
 # Archived future states are used ONLY as an offline numerical seed.
 grid=ti+tt;X=np.array([[np.interp(q,times,states[:,i,d]) for d in range(19)] for q in grid])
 X[:,17]=ref['arc'][np.argmin(cdist(X[:,:3],ref['position']),axis=1)];X[:,18]=np.linalg.norm(X[:,3:6],axis=1);X[0]=own
 U=np.c_[np.clip(np.diff(X[:,13:17],axis=0)/DT,-200,200),np.clip(np.diff(X[:,18])/DT,-40,40)]
 # Hold identical local reference coefficients across all counterfactuals.
 for q in range(4):od[:,3*q:3*q+3]=ref['spline'](X[:,17],q)/[1,1,2,6][q]
 od[:,12]=X[:,17]
 base=None
 variants=[('baseline',80.,'hard',.16,.05,2.,False),('progress500',500.,'hard',.16,.05,2.,False),
  ('no_cbf',80.,'none',.16,.05,2.,False),('wide_funnel',80.,'hard',.68,.05,2.,False),
  ('lower_control_cost',80.,'hard',.16,.005,.2,False),('long_progress_bound',80.,'hard',.16,.05,2.,True)]
 for label,mu,mode,width,df,at,extend in variants:
  solver=RecoveryMPCC(i,n=20,cbf_mode=mode,cbf_checkpoint=HERE/'models/direct_cbf_v2/h.pt' if mode=='hard' else None,
   max_iter=700,wall_time=6.,tube_recovery=True,dense_physics=True)
  xx,uu=(X,U) if base is None else base
  solver.seed(xx,uu);solver.current(own);oo=od.copy();oo[:,40]=width
  if extend:oo[:,33]+=15. # diagnostic spline extrapolation only; not valid track extension
  # Iterate local spline Taylor data, as a single online solve holds coefficients fixed.
  solver.data(oo);solver.weights(mu=mu,df=df,at=at);result=solver.step();px,pu=solver.prediction()
  if label=='baseline' and result['status']==0:base=(px.copy(),pu.copy())
  r=dict(time=ti,variant=label,success=result['status']==0,status=result['message'],seconds=result['seconds'],
   current_speed=float(np.linalg.norm(own[3:6])),predicted_peak_speed=float(np.linalg.norm(px[:,3:6],axis=1).max()),
   progress_gain=float(px[-1,17]-px[0,17]),terminal_progress_margin=float(oo[-1,33]-px[-1,17]),
   predicted_speed_0_4s=float(np.linalg.norm(px[5,3:6])),predicted_progress_0_4s=float(px[5,17]-px[0,17]),
   first_progress_acceleration=float(pu[0,4]),predicted_end_speed=float(np.linalg.norm(px[-1,3:6])),barrier=result['barrier_diagnostics'])
  rows.append(r);save_json(out/('fixed_state_probes'+args.suffix+'.json'),rows);print(json.dumps(r),flush=True)
save_json(out/('probe_scope'+args.suffix+'.json'),dict(scope='Fixed observed-state OCP sensitivity, not full-flight speed improvement',
 seed='Archived future states as offline initialization only; no deployable policy claim',
 progress_extension='Diagnostic extrapolation only; unsafe for operational use without a longer valid reference',max_seconds_per_solve=6,states=args.times))
