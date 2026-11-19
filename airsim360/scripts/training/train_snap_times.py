"""Select a duration regressor on grouped validation problems; test the frozen model in Clarabel."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scripts.planning.minsnap.snap_time_network import TimeNet, FixedTimePredictor, features
from scripts.training.synthetic_snap_teacher import fixed_solver, cost


def load_rows(folder):
    rows=[]
    for path in sorted((folder/"labels").glob("*.npz")):
        if path.name.endswith(".partial.npz"):
            continue
        with np.load(path) as f:
            rows.append({k:f[k] for k in ("own","gates","normals","half","times","teacher_cost",
                                           "split","group","kind","mode","gate","offsets")})
    return rows


def inputs(row):
    return [row[k] for k in ("own","gates","normals","half")]


def fit(rows, train, valid, widths, geometry, path):
    torch.manual_seed(541)
    x=torch.from_numpy(np.stack([features(*inputs(r),geometry) for r in rows]))
    y=torch.tensor(np.stack([r["times"] for r in rows]),dtype=torch.float32)
    if geometry=="ballistic":
        from scripts.simulation.dynamics import MODEL
        from scipy.spatial.transform import Rotation
        states=np.stack([r["own"] for r in rows])
        rot=Rotation.from_quat(np.roll(states[:,6:10],-1,axis=1)).as_matrix()
        bz=rot[:,:,2]
        velocity=states[:,3:6]
        drag=np.einsum("nij,nj->ni",rot,np.asarray(MODEL["drag"])*np.einsum("nji,nj->ni",rot,velocity))
        accel=(bz*states[:,13:17].sum(1)[:,None]-drag)/MODEL["mass"]-[0,0,MODEL["gravity"]]
        target=np.stack([r["gates"][0]-r["own"][:3] for r in rows])
        delta,v,a,axis=[torch.tensor(z,dtype=torch.float32) for z in (target,velocity,accel,bz)]
        def near_cost(ids,t):
            t=t.clamp(.035,8.)[:,None]
            err=delta[ids]-v[ids]*t-.5*a[ids]*t.square()
            lateral=err-(err*axis[ids]).sum(1,keepdim=True)*axis[ids]
            # Free terminal derivatives and free initial jerk along the thrust axis.
            mismatch=(lateral.norm(dim=1)-.08).clamp_min(0)
            return torch.log1p(.00252*mismatch.square()/t[:,0].pow(7))
        base_cost=near_cost(train,y[train,0])
        reference=torch.zeros(len(rows))
        reference[train]=base_cost
    model=TimeNet(x.shape[1],widths)
    with torch.no_grad():
        model.center.copy_(x[train].mean(0))
        model.scale.copy_(x[train].std(0).clamp_min(.05))
        model.log_center.copy_(y[train].log().mean(0))
        model.log_scale.copy_(y[train].log().std(0).clamp_min(.12))
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.004)
    best_score=np.inf
    best_epoch=0
    for epoch in range(1400):
        epoch_start=time.perf_counter()
        for ids in torch.from_numpy(train)[torch.randperm(len(train))].split(96):
            pred=model(x[ids])
            losses=(pred-y[ids].log()).square().mean(0)
            loss=losses.mean()+.08*losses[0]
            if geometry=="ballistic":
                extra=(near_cost(ids,pred[:,0].exp())-reference[ids]).clamp_min(0.)
                loss=loss+.08*extra.square().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
        if os.getenv("SNAP_LOW_DUTY")=="1":
            time.sleep(2*(time.perf_counter()-epoch_start))
        if epoch in (250,550,900):
            for group in opt.param_groups:
                group["lr"]*=.4
        if epoch%10==0:
            with torch.no_grad():
                e=(model.durations(x[valid])/y[valid]).log().square()
                score=float(e.mean()+.08*e[:,0].mean())
            if score<best_score:
                best_score,best_epoch,best_state=score,epoch,copy.deepcopy(model.state_dict())
            if epoch-best_epoch>=240:
                break
    torch.save(dict(geometry=geometry,dimension=x.shape[1],widths=widths,state=best_state),path)
    return dict(widths=widths,geometry_features=geometry,best_epoch=best_epoch,validation_loss=best_score)


def summary(records):
    valid=[r for r in records if r["solved"]]
    errs=np.abs(np.array([r["prediction"] for r in records])-np.array([r["teacher"] for r in records]))
    gaps=np.array([r["gap_pct"] for r in valid])
    return dict(count=len(records),solved=len(valid),failures=len(records)-len(valid),
                mae_s=float(errs.mean()),p95_abs_s=float(np.percentile(errs,95)),
                segment_mae_s=errs.mean(0).tolist(),
                median_cost_gap_pct=float(np.median(gaps)) if len(gaps) else None,
                p95_cost_gap_pct=float(np.percentile(gaps,95)) if len(gaps) else None,
                max_eq=max((r["eq"] for r in valid),default=None),
                max_ineq=max((r["inequality"] for r in valid),default=None))


def evaluate(rows,ids,path,solver,save_examples=None):
    predictor=FixedTimePredictor(path)
    records=[]
    times=[]
    qp_ms=[]
    for i in ids:
        r=rows[i]
        args=inputs(r)
        tick=time.perf_counter()
        prediction=predictor.predict(*args)
        times.append((time.perf_counter()-tick)*1000)
        rec=dict(group=int(r["group"]),gate=int(r["gate"]),kind=str(r["kind"]),mode=int(r["mode"]),
                 prediction=prediction.tolist(),teacher=r["times"].tolist(),teacher_cost=float(r["teacher_cost"]))
        try:
            tick=time.perf_counter()
            plan=solver.solve(*args,prediction)
            qp_ms.append((time.perf_counter()-tick)*1000)
            np.testing.assert_allclose(plan["durations"],prediction,rtol=0,atol=1e-10)
            j=cost(plan)
            rec.update(solved=True,cost=j,gap_pct=100*(j/r["teacher_cost"]-1),
                       eq=plan["audit"]["equality_error"],inequality=plan["audit"]["linear_inequality_error"])
            if save_examples:
                np.savez_compressed(save_examples/f"{rec['group']:04d}_{rec['kind']}.npz",
                    **{k:r[k] for k in ("own","gates","normals","half","offsets")},
                    predicted_times=prediction,teacher_times=r["times"],coefficients=plan["coefficients"],
                    reference_time=plan["reference_time"],reference_states=plan["reference_states"],
                    optimized_passages=plan["optimized_passages"])
        except RuntimeError as exc:
            rec.update(solved=False,error=str(exc)[:400])
        records.append(rec)
    metrics=summary(records)
    metrics["inference_median_ms"]=float(np.median(times))
    metrics["inference_p95_ms"]=float(np.percentile(times,95))
    metrics["qp_median_ms"]=float(np.median(qp_ms)) if qp_ms else None
    return metrics,records


def plot_example(rows,test,records,folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import comb
    successful=[i for i,r in enumerate(records) if r["solved"]]
    if not successful:
        return None
    successful.sort(key=lambda i:records[i]["gap_pct"])
    i=successful[len(successful)//2]
    row=rows[test[i]]
    name=f"{int(row['group']):04d}_{str(row['kind'])}.npz"
    with np.load(folder/"labels"/name) as teacher, np.load(folder/"test_trajectories"/name) as predicted:
        tau=np.linspace(0,1,65)
        basis=np.column_stack([comb(7,k)*tau**k*(1-tau)**(7-k) for k in range(8)])
        xyz=(basis@teacher["coefficients"]).reshape(-1,3)
        ref=predicted["reference_states"]
        fig,axes=plt.subplots(1,2,figsize=(10.8,4.3),layout="constrained")
        for ax,y,caption in zip(axes,[1,2],["Top view","Side view"]):
            ax.plot(xyz[:,0],xyz[:,y],"--",color="#c75a27",lw=2.2,label="Offline teacher")
            ax.plot(ref[:,0],ref[:,y],color="#076b9a",lw=1.65,label="Learned times + QP")
            ax.scatter(row["gates"][:,0],row["gates"][:,y],s=28,facecolors="white",edgecolors="#315449",label="Gate targets",zorder=3)
            ax.scatter(row["own"][0],row["own"][y],s=45,c="#222222",marker="x",label="Start")
            ax.set(xlabel="x (m)",ylabel=("y (m)" if y==1 else "z (m)"),title=caption)
            ax.set_aspect("equal",adjustable="datalim")
            ax.grid(alpha=.18)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Held-out five-gate trajectory   |   cost gap {records[i]['gap_pct']:.2f}%")
        fig.savefig(folder/"test_trajectory.png",dpi=165)
        plt.close(fig)
    return name


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    folder=parser.parse_args().folder
    rows=load_rows(folder)
    splits={split:np.array([i for i,r in enumerate(rows) if str(r["split"])==split]) for split in ("train","val","test")}
    groups={s:{int(rows[i]["group"]) for i in ids} for s,ids in splits.items()}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])
    qp=fixed_solver()
    results={}
    previous={}
    if os.getenv("SNAP_LOW_DUTY")=="1" and (folder/"results.json").exists():
        previous=json.loads((folder/"results.json").read_text())["candidates"]
    for name,widths,geometry in [("baseline",[48,48],False),("geometry",[128,128,96,64],True),
                                ("ballistic",[160,160,128,96], "ballistic")]:
        path=folder/f"{name}.pt"
        if name in previous and path.is_file():
            results[name]=previous[name]
            continue
        info=fit(rows,splits["train"],splits["val"],widths,geometry,path)
        info["validation"],_=evaluate(rows,splits["val"],path,qp)
        print(name,json.dumps(info),flush=True)
        results[name]=info

    def rank(name):
        val=results[name]["validation"]
        return val["failures"],val["p95_cost_gap_pct"],val["mae_s"]

    winner=min(results,key=rank)
    from shutil import copyfile
    copyfile(folder/f"{winner}.pt",folder/"best.pt")
    out=folder/"test_trajectories"
    out.mkdir(exist_ok=True)
    metrics,records=evaluate(rows,splits["test"],folder/"best.pt",qp,out)
    subsets={}
    for key in ("kind","mode","gate"):
        subsets[key]={str(v):summary([r for r in records if r[key]==v]) for v in sorted({r[key] for r in records})}
    report=dict(counts={s:len(ids) for s,ids in splits.items()},group_counts={s:len(g) for s,g in groups.items()},
                candidates=results,selected=winner,test=metrics,test_subsets=subsets)
    report["illustrated_test_case"]=plot_example(rows,splits["test"],records,folder)
    external=np.array([i for i,r in enumerate(rows) if str(r["split"])=="external"])
    if len(external):
        assert not ({int(rows[i]["group"]) for i in external} & set.union(*groups.values()))
        metrics2,records2=evaluate(rows,external,folder/"best.pt",qp,out)
        report["external_test"]=metrics2
        (folder/"external_test_records.json").write_text(json.dumps(records2,indent=2))
    (folder/"test_records.json").write_text(json.dumps(records,indent=2))
    (folder/"results.json").write_text(json.dumps(report,indent=2))
    print("test",json.dumps(metrics),flush=True)


if __name__=="__main__":
    main()
