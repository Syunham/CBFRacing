"""Read-only analysis of archived flight and numerical CPC plans."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'swarm_cpc_reproduction/.deps'))
import numpy as np,json
OUT=ROOT/'diagnosis_speed_20261102';OUT.mkdir(exist_ok=True)
run=ROOT/'outputs/integrated_learned_20261030'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
z=np.load(run/'execution.npz');t=z['time'];x=z['state'];v=np.linalg.norm(x[:,:,3:6],axis=-1)
logs=load(run/'planning_log.json');events=load(run/'events.json');front=load(run/'frontends.json')
print('EVENT STRUCTURE',str(events)[:150])
stats=lambda a:dict(min=float(np.min(a)),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)))
queries=[]
for row in logs:
 d=row['solve']['barrier_diagnostics'];i=row['agent'];k=int(round(row['time']/.02));u=np.array(row['control']);vv=v[k,i]
 queries.append(dict(time=row['time'],agent=i,speed=float(vv),h=d['initial_learned_h'],active=d['learned_active_rows'],
  dual=d['learned_dual_max'],decay_dual=d['discrete_dual_max'],pred_h=d['minimum_predicted_h'],
  decay=d['minimum_predicted_discrete_residual'],force_rate=float(abs(u[:4]).max()),progress_acceleration=float(u[4])))
active=np.array([q['active']>0 for q in queries]);decay=np.array([q['decay_dual']>1e-3 and q['decay']<1e-4 for q in queries]);qs=np.array([q['speed'] for q in queries])
result=dict(run=str(run),summary=load(run/'summary.json'),speed=stats(v),per_agent=[],
 cbf=dict(active_prediction_fraction=float(active.mean()),active_decay_fraction=float(decay.mean()),
  mean_speed_active=float(qs[active].mean()) if active.any() else None,mean_speed_inactive=float(qs[~active].mean()) if (~active).any() else None,
  current_h=stats([q['h'] for q in queries]),slowest_queries=sorted(queries,key=lambda q:q['speed'])[20:32]),
 forces=stats(x[:,:,13:17]),rotor_above_6_9_fraction=float((x[:,:,13:17]>6.9).mean()),
 body_rate_norm=stats(np.linalg.norm(x[:,:,10:13],axis=-1)),body_rate_above_9_fraction=float((np.linalg.norm(x[:,:,10:13],axis=-1)>9).mean()),
 progress_speed=stats(x[:,:,18]),speed_below_3_fraction=float((v<3).mean()),
 force_rate=stats([q['force_rate'] for q in queries]),progress_acceleration=stats([q['progress_acceleration'] for q in queries]),plans=[])
for i in range(2):
 ee=events[i];last=0.;segments=[]
 for j,e in enumerate(ee):
  end=e['time'];mask=(t>=last)&(t<=end);path=float(np.trapezoid(v[mask,i],t[mask]));segments.append(dict(event=j+1,gate=j%7+1,start=last,end=end,duration=end-last,path=path,mean_speed=path/(end-last),min_speed=float(v[mask,i].min()),max_speed=float(v[mask,i].max())))
  last=end
 result['per_agent'].append(dict(agent=i,path=float(np.trapezoid(v[:,i],t)),mean_speed=float(np.trapezoid(v[:,i],t)/t[-1]),segments=segments))
for path in sorted((run/'plans').glob('*.npz')):
 p=np.load(path);r=p['reference_states'];rt=p['reference_time'];vv=np.linalg.norm(r[:,3:6],axis=1)
 fr=front[int(path.name.split('_')[0])];duration=p['durations'][0];mask=rt<=duration
 result['plans'].append(dict(file=path.name,agent=fr['agent'],time=fr['time'],event=fr['gate_index'],
  durations=p['durations'].tolist(),first_leg_mean_speed=float(np.trapezoid(vv[mask],rt[mask])/duration),
  max_speed=float(vv.max()),terminal_speed=float(vv[-1]),first_gate_speed=float(np.interp(duration,rt,vv)),
  allocation=p['allocation'].tolist(),prior=p['network_arrival_time_prior'].tolist(),
  residual=fr['audit'],actual_next_gate_duration=next((e['duration'] for e in result['per_agent'][fr['agent']]['segments'] if e['event']==fr['gate_index']+1),None)))
(OUT/'archive_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:val for k,val in result.items() if k not in ['plans','per_agent']},ensure_ascii=False,indent=2))
print('PLAN SAMPLE',json.dumps(result['plans'][:3],ensure_ascii=False))
print('SEGMENTS',json.dumps(result['per_agent'][0],ensure_ascii=False))
