"""Isolate the dynamic MPCC cost with recorded camera frames, poses and Fast-fly plans."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from scripts.control.adapter import ROOT, controller
from scripts.perception.live_vision import LiveVision
from scripts.control.run_spatial_mpcc import main as run_experiment


def trial(folder, name, mode, candidate=None):
    cfg=json.loads((folder/'experiment.json').read_text())
    with np.load(folder/'flight/execution.npz') as f:
        times, states = f['time'], f['state']
    rows=json.loads((folder/'flight/frontends.json').read_text())
    files={(r['agent'],round(r['time'],2)):folder/'flight/plans'/f"{j:03d}_agent{r['agent']}_event{r['gate_index']}.npz"
           for j,r in enumerate(rows) if r['success']}
    diagnostics=[]
    class Replay(LiveVision):
        def start_renderer(self,**kwargs):
            pass
        def close(self):
            self.log.close()
        def capture(self,stamp,unused):
            return tuple(np.load(folder/d/f'{round(stamp*1000):06d}.npy') for d in ('rgb','depth'))
        def observe(self,stamp,unused):
            super().observe(stamp,states[round(stamp/.02)])
        def install(self,solver,od):
            super().install(solver,od)
            ego=solver.agent
            _,tree=self.future_views[ego]
            pts=solver.X[:,:3]
            dist,idx=tree.query(pts)
            static_centers=tree.data[idx]
            dyn=np.linalg.norm(od[:,16:19]-static_centers,axis=1)>1e-5
            logits=(od[:,19]-np.linalg.norm(od[:,16:19]-pts,axis=1))/od[:,20]
            j=int(np.argmax(logits))
            future=np.array([np.interp(self.control_stamp+.08*j,times,x) for x in states[:,1-ego,:3].T])
            diagnostics.append(dict(t=self.control_stamp,ego=ego,node=j,dynamic=bool(dyn[j]),risk=float(.5+.5*np.tanh(logits[j]/2)),
                                    predicted_center_error_m=float(np.linalg.norm(od[j,16:19]-future)),
                                    node_distance_to_recorded_future_m=float(np.linalg.norm(pts[j]-future))))
            if mode=='static':
                od[:,16:19]=static_centers
                od[:,19:21]=[.20,.055]
            elif mode=='oracle':
                # Replace only future motion of RGB-D returns on the recorded peer.
                # Keep the sensor coverage, other returns, radius, width and static field.
                from scripts.perception.local_future_field import supports
                flow,_=self.future_views[ego]
                observation=round(self.stamp/.02)
                for node,query in enumerate(pts):
                    lead=min(1.6,self.control_stamp-self.stamp+.08*node)
                    feat,pred,valid,ids=supports(flow,self.poses[ego],lead)
                    with torch.inference_mode():
                        c,r,w=(a.numpy() for a in self.future_model.kernels(torch.from_numpy(feat),torch.from_numpy(pred)))
                    delta=np.array([np.interp(self.stamp+lead,times,v) for v in states[:,1-ego,:3].T])-states[observation,1-ego,:3]
                    visible=np.flatnonzero(np.linalg.norm(flow.xyz[ids]-states[observation,1-ego,:3],axis=1)<.12)
                    c[visible]=flow.xyz[ids[visible]]+delta
                    logit=(r-np.linalg.norm(query-c,axis=1))/w
                    logit[~valid]=-np.inf
                    j=int(np.argmax(logit))
                    if logit[j]>(.20-dist[node])/.055:
                        od[node,16:19],od[node,19],od[node,20]=c[j],r[j],w[j]
                    else:
                        od[node,16:19],od[node,19:21]=static_centers[node],[.20,.055]
    def recorded_controller(adapter):
        mod=controller(adapter)
        from scripts.planning.fast_fly.guarded_fast_fly import GuardedFastFly
        solve=GuardedFastFly.solve
        def fixed(planner,own,other,index):
            result=solve(planner,own,other,index)
            with np.load(files[planner.active_agent,round(planner.current_time,2)]) as recorded:
                result.update({k:recorded[k] for k in recorded})
            return result
        GuardedFastFly.solve=fixed
        return mod
    torch.set_num_threads(1)
    options = ['--name',name,'--separation','1.2','--weight',str(0 if mode=='off' else cfg['weight']),
              '--checkpoint',cfg['checkpoint'],'--future-checkpoint',candidate or cfg['future_checkpoint'],'--passage-head',cfg['passage_head'],
              '--guarded-fast-fly','--recover-frontend','--rti-cold-start','--max-time','.599']
    run_experiment(options, vision_class=Replay, load_controller=recorded_controller)
    (folder.parent/name/'launch_peaks.json').write_text(json.dumps(diagnostics))


if __name__=='__main__':
    group=sys.argv[1]
    source=ROOT/'reports/spatial_potential'/sys.argv[2]
    if len(sys.argv)==3 or sys.argv[3]=='--compare-motion':
        modes=('full','oracle','static') if len(sys.argv)>3 else ('full','static','off')
        for mode in modes:
            subprocess.run([sys.executable,'-u','-m','scripts.replay.replay_local_future_launch',group,str(source),mode],check=True)
    else:
        trial(source,group+'_'+sys.argv[3],sys.argv[3],sys.argv[4] if len(sys.argv)>4 else None)
