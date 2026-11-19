"""Train the Guan five-interval MLP and evaluate reconstructed quintic references."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.planning.minsnap.guan_minjerk import AUDIT_SIZE, BOUNDS, DRAG, F_MAX, GRID_SIZE, MASS, M, SEED, W_MAX, audit, repair_prediction, solve_label


class TimeAllocator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("input_scale",torch.tensor([22.]*3+[40.]*3+[15.]*15))
        self.net=nn.Sequential(nn.Linear(21,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,5))

    def forward(self,x):
        return torch.nn.functional.softplus(self.net(x/self.input_scale))+1e-5


def metrics(y,p):
    absolute=abs(y-p)
    relative=absolute/y
    return dict(n=len(y),mae_s=float(absolute.mean()),mape_pct=float(100*relative.mean()),
                first_mae_s=float(absolute[:,0].mean()),first_p95_s=float(np.percentile(absolute[:,0],95)),
                first_mape_pct=float(100*relative[:,0].mean()),
                segment_mape_pct=(relative.mean(0)*100).tolist())


def evaluate(x,y,p,curve,flat,fun):
    result=metrics(y,p)
    dyn={}
    for name,alloc in [("teacher",y),("network",p)]:
        checks=[]
        objectives=[]
        for row,t in zip(x,alloc):
            c=np.asarray(curve(row,t)).reshape(M,6,3)
            checks.append(audit(row,t,c,flat))
            objectives.append(float(fun(row,np.log(t))[0]))
        a=np.stack(checks)
        assert np.isfinite(a).all()
        dyn[name]=np.asarray(objectives)
        result[name]=dict(reconstructed=int((a[:,0]<2e-4).sum()),max_boundary_and_continuity_residual=float(a[:,0].max()),
                          thrust_max_p50_N=float(np.median(a[:,2])),thrust_max_p95_N=float(np.percentile(a[:,2],95)),
                          rate_max_p50_rad_s=float(np.median(a[:,3])),rate_max_p95_rad_s=float(np.percentile(a[:,3],95)),
                          within_sampled_limits=int(((a[:,1]>=0)&(a[:,2]<=F_MAX)&(a[:,3]<=W_MAX)).sum()))
    result["median_duration_difference_pct"]=float(np.median(100*(dyn["network"]/dyn["teacher"]-1)))
    corrected=[]
    target=[]
    for row,t,truth in zip(x,p,y):
        out=repair_prediction(row,t,fun)
        if out is not None:
            c=np.asarray(curve(row,out)).reshape(M,6,3)
            err,lo,hi,w=audit(row,out,c,flat)
            if err<2e-4 and lo>=0 and hi<=F_MAX and w<=W_MAX:
                corrected.append(out)
                target.append(truth)
    result["network_plus_node_projection"]=dict(valid=len(corrected),attempted=len(y),
        metrics=metrics(np.stack(target),np.stack(corrected)) if corrected else None)
    return result


def track_inputs(folder):
    # Held-out course inputs only. Recompute all targets using the new teacher.
    from scipy.spatial.transform import Rotation
    x=[]
    for source in ("synthetic_snap_times01","synthetic_snap_transition01"):
        for path in sorted((folder.parent/source/"labels").glob("*.npz")):
            with np.load(path) as r:
                if str(r["split"])!="test":
                    continue
                own=r["own"]
                R=Rotation.from_quat(np.roll(own[6:10],-1)).as_matrix()
                a=(R[:,2]*own[13:17].sum()-R@(np.asarray(DRAG)*(R.T@own[3:6])))/MASS-[0,0,9.81]
                x.append(np.r_[own[3:6],a,(r["gates"]-own[:3]).ravel()])
    return np.stack(x)


def train(folder,curve,flat,fun):
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    files=sorted((folder/"shards").glob("[0-9][0-9][0-9].npz"))
    columns={}
    attempts=0
    for path in files:
        with np.load(path) as shard:
            attempts+=int(shard["attempts"])
            for k in ("inputs","times","coefficients","objective","audit","iterations","gate","kind","offsets","gates","own","group"):
                columns.setdefault(k,[]).append(shard[k])
    arrays={k:np.concatenate(v) for k,v in columns.items()}
    x,y=arrays["inputs"],arrays["times"]
    unique,inverse,counts=np.unique(x,axis=0,return_inverse=True,return_counts=True)
    assert len(x)==10000
    # The original allocator can legitimately select the same corner as the
    # edge policy for one state. Keep paired policy examples in the same group.
    assert all(np.unique(arrays["group"][inverse==j]).size==1 for j in np.flatnonzero(counts>1))
    shards=np.random.default_rng(SEED).permutation(100)
    train_ids,val_ids,test_ids=[(s[:,None]*100+np.arange(100)).ravel() for s in np.split(shards,[80,90])]
    assert len(set(arrays["group"][train_ids]) & set(arrays["group"][val_ids]))==0
    assert len(set(arrays["group"][train_ids]) & set(arrays["group"][test_ids]))==0
    assert len(set(arrays["group"][val_ids]) & set(arrays["group"][test_ids]))==0
    np.savez_compressed(folder/"dataset.npz",**arrays,train_ids=train_ids,val_ids=val_ids,test_ids=test_ids)
    manifest=dict(valid=len(x),unique_inputs=len(unique),attempts=attempts,split=[8000,1000,1000],seed=SEED,
                  inputs="v_xyz, a_xyz, five future waypoint positions relative to current position; xyz in each",
                  target="five locally optimized durations in seconds",sampling="original full-size Split-S; 2600 each allocator, stratified, edge/center; 2200 stacked-gate transitions; grouped splits",
                  input_normalization=[22,40,15],coefficient_basis="ascending powers of u=(t-t_start)/T, shape (5,6,3)",
                  yaw_rad=0,terminal_velocity_acceleration=0,collective_thrust_N=[0,F_MAX],bodyrate_limit_rad_s=W_MAX,
                  drag_body_kg_s=DRAG,mass_kg=MASS,objective="minimize sum(T), hard thrust and bodyrate constraints at uniformly spaced nodes including endpoints",
                  optimization_points_per_leg=GRID_SIZE,audit_points_per_leg=AUDIT_SIZE,
                  numerical_duration_bounds_seconds=np.exp(BOUNDS[0]).tolist(),solver="Hermite minimum jerk with free interior derivatives, SLSQP over log times",
                  paper="Guan et al., Learning Time-Optimal Online Replanning for Distributed Model Predictive Contouring Control of Quadrotors, ICRA 2025")
    (folder/"manifest.json").write_text(json.dumps(manifest,indent=2))
    xt,yt=[torch.tensor(a,dtype=torch.float32) for a in (x,y)]
    model=TimeAllocator()
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    history=[]
    best=float("inf")
    best_epoch=0
    lr_wait=0
    for epoch in range(1,351):
        model.train()
        loss_sum=0.
        for idx in torch.as_tensor(train_ids)[torch.randperm(len(train_ids))].split(192):
            loss=((model(xt[idx])-yt[idx]).abs()/yt[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum+=float(loss.detach())*len(idx)
        model.eval()
        with torch.no_grad():
            val=float(((model(xt[val_ids])-yt[val_ids]).abs()/yt[val_ids]).mean())
        history.append([epoch,100*loss_sum/len(train_ids),100*val,opt.param_groups[0]["lr"]])
        if val<best:
            best,best_epoch,state=val,epoch,copy.deepcopy(model.state_dict())
            lr_wait=0
        else:
            lr_wait+=1
        if lr_wait>=12:
            for group in opt.param_groups:
                group["lr"]=max(.00003,group["lr"]*.5)
            lr_wait=0
        if epoch%25==0:
            print("epoch",epoch,"validation MAPE",100*val,flush=True)
        if epoch-best_epoch>=45:
            break
    model.load_state_dict(state)
    torch.save(dict(state=state,architecture=[21,512,512,5],activation="ReLU",output="softplus",epoch=best_epoch),folder/"best.pt")
    # A portable inference file includes input scaling and positive output transform.
    model.eval()
    torch.jit.trace(model,xt[test_ids[:1]]).save(str(folder/"time_allocator.ts"))
    deployed=torch.jit.load(str(folder/"time_allocator.ts"))
    def predict(data):
        with torch.no_grad():
            return deployed(torch.tensor(data,dtype=torch.float32)).numpy().astype(float)
    pred=predict(x[test_ids])
    with torch.no_grad():
        np.testing.assert_allclose(pred,model(xt[test_ids]).numpy(),rtol=1e-5,atol=1e-6)
    report=dict(best_epoch=best_epoch,epochs=epoch,val_mape_pct=100*best,
                baseline_train_median=metrics(y[test_ids],np.broadcast_to(np.median(y[train_ids],axis=0),pred.shape)),
                test=evaluate(x[test_ids],y[test_ids],pred,curve,flat,fun))
    std_x=track_inputs(folder)
    course_x,course_y=[],[]
    for row in std_x:
        sol=solve_label(row,fun)
        if sol is not None:
            course_x.append(row)
            course_y.append(sol[0])
    course_x,course_y=map(np.stack,(course_x,course_y))
    course_pred=predict(course_x)
    report["splits_course_attempts"]=len(std_x)
    report["splits_course_external"]=evaluate(course_x,course_y,course_pred,curve,flat,fun)
    np.savez_compressed(folder/"evaluation.npz",test_ids=test_ids,predicted=pred,course_inputs=course_x,
                        course_times=course_y,course_predicted=course_pred)
    (folder/"training_history.json").write_text(json.dumps(history))
    (folder/"results.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
