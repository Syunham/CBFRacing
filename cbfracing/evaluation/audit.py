"""Independent dense physical replay and ordered gate-event verification."""
from cbfracing.core.context import *
import argparse
import re

def audit(folder):
    folder=Path(folder);cfg=json.loads((folder/'config.json').read_text());track=json.loads((folder/'course.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    z=np.load(folder/'execution.npz');t=z['time'];x=z['state'];logs=json.loads((folder/'planning_log.json').read_text());events=json.loads((folder/'events.json').read_text())
    assert np.allclose(track['centers'],POSITIONS,rtol=0,atol=1e-12) and np.allclose(track['rotations_wxyz'],QUATERNIONS,rtol=0,atol=1e-12)
    parameters=cfg['dynamics']
    historical_limit=re.search(r'body-rate norm([0-9.]+)',cfg.get('actuator_limits',''))
    rate_limit=cfg.get('body_rate_norm_limit',float(historical_limit[1]) if historical_limit else parameters['omega_max'])
    rhs,step=model_functions(16,parameters=parameters);_,old_step=model_functions(8,parameters=parameters);world=np.array(track['world_box']);current=x[0].copy();errors=[];dense_min=float(np.linalg.norm(current[0,:3]-current[1,:3]));max_rate=float(np.linalg.norm(current[:,10:13],axis=-1).max());max_force=float(current[:,13:17].max());min_force=float(current[:,13:17].min());max_qerror=float(abs(np.linalg.norm(current[:,6:10],axis=-1)-1).max());world_violation=0.;max_speed=float(np.linalg.norm(current[:,3:6],axis=-1).max());unsafe_executed=[];accepted_queries=0
    for k,dt in enumerate(np.diff(t)):
        pair=logs[2*k:2*k+2];assert len(pair)==2
        for row in pair:
            accepted_queries+=int(row['solve']['status']==0)
            if row['solve']['status']!=0:unsafe_executed.append(k)
        d=[]
        for i in range(2):
            u=np.array(pair[i]['control']);y=np.asarray(old_step(current[i],u,dt)).ravel();errors.append(float(abs(y[:17]-x[k+1,i,:17]).max()));current[i]=y
            d.append(np.array([np.asarray(step(x[k,i],u,dt*j/16)).ravel() for j in range(1,17)]))
        d=np.transpose(np.array(d),(1,0,2));dense_min=min(dense_min,float(np.linalg.norm(d[:,0,:3]-d[:,1,:3],axis=1).min()))
        max_rate=max(max_rate,float(np.linalg.norm(d[:,:,10:13],axis=-1).max()));max_force=max(max_force,float(d[:,:,13:17].max()));min_force=min(min_force,float(d[:,:,13:17].min()))
        max_qerror=max(max_qerror,float(abs(np.linalg.norm(d[:,:,6:10],axis=-1)-1).max()));max_speed=max(max_speed,float(np.linalg.norm(d[:,:,3:6],axis=-1).max()))
        world_violation=max(world_violation,float(np.maximum(world[:3]-d[:,:,:3],0).max()),float(np.maximum(d[:,:,:3]-world[3:],0).max()))
        current[:,17:19]=x[k+1,:,17:19]
    gate_checks=[]
    for i,ee in enumerate(events):
        for j,e in enumerate(ee):
            gi=j%7;assert e['gate']==gi+1 and e['sequence']==j+1
            k=min(int(np.searchsorted(t,e['time'],side='right')-1),len(t)-2);u=np.array(logs[2*k+i]['control']);y=np.asarray(step(x[k,i],u,e['time']-t[k])).ravel()
            g=POSITIONS[gi];n=np.array(track['normals'][gi]);side=np.cross([0.,0.,1.],n);delta=y[:3]-g
            gate_checks.append(dict(agent=i,sequence=j+1,gate=gi+1,time=e['time'],plane_error=float(abs(delta@n)),aperture_coordinate=float(max(abs(delta@side),abs(delta[2]))),forward_speed=float(y[3:6]@n)))
    checks=dict(exact_supplied_track=True,physical_replay=max(errors,default=0)<1e-4,all_executed_controls_from_successful_solves=not unsafe_executed,
        dense_clearance=dense_min>=.5-1e-4,angular_rate_norm=max_rate<=rate_limit+.001,rotor_thrust=min_force>=parameters['thrust_min']-.001 and max_force<=parameters['thrust_max']+.001,
        world_bounds=world_violation<=.001,quaternion=max_qerror<=1e-5,
        ordered_gates=all(e['plane_error']<1e-5 and e['aperture_coordinate']<=.7501 and e['forward_speed']>0 for e in gate_checks),
        two_laps_each=all(len(e)>=14 for e in events),no_explicit_waiting=not cfg['manual_waiting'] and not cfg['manual_priority'],
        fresh_from_initial_rest=not cfg.get('debug_resume_source') and np.max(abs(x[0,:,3:6]))<1e-12 and np.max(abs(x[0,:,10:13]))<1e-12)
    result=dict(full_goal_case_passed=all(checks.values()) and summary['success'],checks=checks,passed_gate_counts=[len(e) for e in events],gates=gate_checks,
        actual_dense_min_distance=dense_min,max_rate_norm=max_rate,min_force=min_force,max_force=max_force,world_violation=world_violation,
        max_quaternion_error=max_qerror,max_physical_replay_error=max(errors,default=0),max_speed=max_speed,executed_accepted_queries=accepted_queries,
        physical_integration_samples=1+16*max(len(t)-1,0),continuous_time_scope='Dense numerical replay, not a formal interval safety certificate',
        computation_latency_in_physics=False,replay_dynamics=parameters,body_rate_norm_limit=rate_limit)
    save_json(folder/'audit.json',result);print(folder.name,result['full_goal_case_passed'],checks,'distance',dense_min,flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='*');a=p.parse_args()
    for f in a.folders:audit(f)
