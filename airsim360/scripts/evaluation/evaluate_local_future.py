"""Compare frozen fields on identical recorded inputs, event pairs and metric 3D probes."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.evaluation.evaluate_flight_count import make_probes
from scripts.simulation.gate_geometry import gate_boxes
from scripts.perception.local_future_field import LocalFutureField, VisualMotionPrior, observe, scores, static_tree
from scripts.perception.occupancy_flow import FlowPrior
from scripts.perception.spatial_potential import camera_points, rotation
from scripts.perception.time_query_field import encode_views, load_rgbd
from scripts.training.train_local_future import MANIFEST, ROOT
from scripts.training.train_rgbd_field import GateFrames, Windows, threat_labels

MODELS = ('global_rgbd','surface_prior','learned_local')
angle = np.arange(24)*np.pi/12
RING = np.c_[np.cos(angle),np.sin(angle),np.zeros(len(angle))]
r, f = np.meshgrid(np.linspace(-6,6,49),np.linspace(-2,6,33))


class Replay:
    def __init__(self, times, states, reader, gates, networks):
        self.t, self.x, self.reader, self.gates = times, states, reader, gates
        self.old, self.local, self.prior = networks
        self.k = None

    def frame(self,k):
        if k == self.k:
            return
        idx = np.maximum(0,np.arange(k-3,k+1))
        rgb, depth = self.reader(idx)
        self.encoded = encode_views(self.old,rgb,depth)
        self.flows, self.trees = [], []
        for ego in (0,1):
            poses = self.x[idx,ego]
            flow = observe(self.prior,self.t[idx],poses,rgb[:,ego],depth[:,ego])
            self.flows.append(flow)
            self.trees.append(static_tree(depth[-1,ego],poses[-1],flow.xyz))
        self.k = k

    def query(self,ego,lead,q):
        pose = self.x[self.k,ego]
        cam = torch.as_tensor(camera_points(q,pose).astype('f'))[None]
        old = self.old.query_logits(self.encoded[ego:ego+1],cam,torch.full((1,len(q),1),lead)).sigmoid()
        local, dyn, base = scores(self.local,self.flows[ego],self.trees[ego],pose,lead,q)
        return (old.numpy().ravel(),base,local), dyn


def local_metrics(replay, stride, out, name):
    rows = []
    for k in range(0,len(replay.x)-20,stride):
        replay.frame(k)
        for ego in (0,1):
            pose = replay.x[k,ego]
            peer = replay.x[k+20,1-ego,:3]
            front = rotation(pose).apply([1.,0.,0.])[:2]
            front /= np.linalg.norm(front)
            right = np.array([front[1],-front[0]])
            grid = np.broadcast_to(pose[:3],(*r.shape,3)).copy()
            grid[:,:,:2] += r[:,:,None]*right+f[:,:,None]*front
            q = np.vstack((peer,peer+RING,grid.reshape(-1,3)))
            labels = threat_labels(q,peer,replay.gates,.25)
            safe_ring, safe_area = labels[1:25]<.05, labels[25:]<.05
            values, dyn = replay.query(ego,.4,q)
            near = np.linalg.norm(replay.flows[ego].xyz-replay.x[k,1-ego,:3],axis=1)
            row = dict(ego=ego,k=k,peer_surface=bool(np.any(near<.12)),dynamic_center=float(dyn[0]))
            for m,v in zip(MODELS,values):
                ring = v[1:25][safe_ring]
                row[m] = [float(v[0]), float(v[0]-ring.mean()) if len(ring) else None,
                          bool(v[0]>ring.max()) if len(ring) else None, float(np.mean(v[25:][safe_area]>=.5))]
            if name=='run1' and k==288 and ego==0:
                xy = np.linspace(-2,2,101)
                xx,yy = np.meshgrid(xy,xy)
                plane = peer + np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
                maps, _ = replay.query(ego,.4,plane)
                np.savez_compressed(out/'example.npz',xy=xy,truth=threat_labels(plane,peer,replay.gates,.25).reshape(xx.shape),
                    maps=np.stack(maps).reshape(3,*xx.shape),peer=peer,now=replay.x[k,1-ego,:3],ego=pose[:3])
            rows.append(row)
    (out/(name+'_local.json')).write_text(json.dumps(rows))
    return rows


def summarize(rows):
    result = dict(n=len(rows),peer_surface=sum(r['peer_surface'] for r in rows))
    for m in MODELS:
        arr = np.array([r[m] for r in rows],float)
        result[m] = dict(median_center=float(np.median(arr[:,0])),median_contrast=float(np.nanmedian(arr[:,1])),
                         above_all=float(np.nanmean(arr[:,2])),median_safe_area_above_half=float(np.nanmedian(arr[:,3])))
    result['dynamic_center_median'] = float(np.median([r['dynamic_center'] for r in rows]))
    return result


def events(replay, episode, records):
    openings, warnings = make_probes(episode,records)
    queries = []
    rows = []
    for horizon,gate,stamp,pos,neg in warnings:
        if horizon!=.4:
            continue
        i=len(rows)
        rows.append(dict(gate=gate,time=stamp,ego=pos[1]))
        for key,item in (('pos',pos),('empty',neg)):
            k,e,t,q=item
            queries.append((k,e,t,q,i,key))
    bars={m:[] for m in MODELS}
    for k,e,t,q in openings:
        if t==.4 and k%8==3:
            queries.append((k,e,t,q,None,'bars'))
    for k,e,t,q,i,key in sorted(queries,key=lambda item:item[0]):
        replay.frame(k)
        vals,_=replay.query(e,t,q)
        for m,v in zip(MODELS,vals):
            if key=='bars':
                bars[m].append([float(v[0]),*v[1:].tolist()])
            else:
                rows[i].setdefault(m,{})[key]=float(v.max())
    summary={}
    for m in MODELS:
        a=np.array(bars[m])
        summary[m]=dict(bar_center_pairs=len(a)*4,bar_higher=float((a[:,1:]>a[:,:1]).mean()),
                        bar_above_half=float((a[:,1:]>=.5).mean()),center_above_half=float((a[:,0]>=.5).mean()))
    return rows,summary


def operating(rows,m,threshold):
    pos,neg = np.array([[r[m]['pos'],r[m]['empty']] for r in rows]).T
    return dict(n=len(rows),hits=int((pos>=threshold).sum()),false=int((neg>=threshold).sum()),threshold=threshold)


def budget(rows,m,n):
    return float(np.nextafter(sorted((r[m]['empty'] for r in rows),reverse=True)[n],np.inf))


@torch.inference_mode()
def main(out):
    torch.set_num_threads(1)
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    head=LocalFutureField(saved.get('correction_speed',3.)).eval()
    head.load_state_dict(saved['model'])
    if 'directional' in saved:
        from scripts.perception.directional_motion_risk import DirectionalRisk
        head.directional=DirectionalRisk().eval()
        head.directional.load_state_dict(saved['directional'])
    if 'motion' in saved:
        from scripts.perception.context_motion import ContextMotionPrior
        prior=ContextMotionPrior(saved['prior'],saved)
    else:
        prior=(VisualMotionPrior if saved.get('visual_only') else FlowPrior)(saved['prior'])
    nets=(load_rgbd(ROOT/'cf21_time8_rgbd_50hz_resume02/best.pt'),head,prior)
    summary, event_rows = {},{}
    entries=json.loads(MANIFEST.read_text())['episodes']
    for split in ('validation','test'):
        ep,=Windows(MANIFEST,split,1,.25,lazy_windows=True).episodes
        entry,=[e for e in entries if e['split']==split]
        rec=Replay(ep[0],ep[1],lambda idx:(ep[2][idx],ep[3][idx]),ep[5],nets)
        ev=json.loads((MANIFEST.parent / entry['trajectory']).with_name('events.json').read_text())
        event_rows[split],bars=events(rec,ep,ev)
        loc=local_metrics(rec,8,out,split)
        summary[split]=dict(episode=entry['episode_id'],local=summarize(loc),gates=bars)
        print(split,summary[split],flush=True)
    for split,rows in event_rows.items():
        summary[split]['events']={m:{'half':operating(rows,m,.5),'validation_budget_2':operating(rows,m,budget(event_rows['validation'],m,2)),
             'matched_2':operating(rows,m,budget(rows,m,2)), 'matched_7':operating(rows,m,budget(rows,m,7))} for m in MODELS}
    (out/'event_rows.json').write_text(json.dumps(event_rows,indent=2))
    all_rows=[]
    for i in (1,2):
        fold=ROOT/f'cf21_bendhead03_run{i}'
        with np.load(fold/'flight/execution.npz') as file:
            t,x=file['time'],file['state']
        def read(idx):
            return tuple(np.stack([np.load(fold/d/f'{k*20:06d}.npy') for k in idx]) for d in ('rgb','depth'))
        gates=GateFrames(gate_boxes(json.loads((fold/'rendered_course.json').read_text())))
        rows=local_metrics(Replay(t,x,read,gates,nets),4,out,f'run{i}')
        all_rows.extend(rows)
        summary[f'run{i}']=summarize(rows)
        print('rollout',i,summary[f'run{i}'],flush=True)
    summary['both_runs']=summarize(all_rows)
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    main(ROOT/sys.argv[1])
