"""Compare a causal RGB-D field to executed neighbor positions 400 ms later."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit

from scripts.perception.occupancy_flow import FlowPrior, OccupancyFlow


def sample(folder, temp):
    cfg = json.loads((folder/"experiment.json").read_text())
    audit = json.loads((folder/"visual_comparison.json").read_text())
    with np.load(folder/"flight/execution.npz") as data:
        t, states = data["time"],data["state"]
    k = int(np.argmin(abs(t+.4-audit["closest_time_s"])))
    j = int(np.argmin(abs(t-t[k]-.4)))
    assert k>=4 and abs(t[j]-t[k]-.4)<1e-8
    shape = (2,256,512)
    scale=json.loads((folder/"vehicle_geometry.json").read_text())["scale_xyz"]
    worker=[".venv/python.exe","-u","-m","scripts.simulation.spatial_render_worker",str(temp/"sensor.jsonl"),
            "--resolution","512","256","--scale",*map(str,scale)]
    prior=FlowPrior(cfg["flow_prior"])
    fields=[OccupancyFlow(prior),OccupancyFlow(prior)]
    with subprocess.Popen(worker,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                          env=dict(os.environ,SAVE_FLIGHT_VIEWS="0",SINGLE_FLIGHT_CAMERA="1")) as proc:
        try:
            if proc.stdout.readline().strip()!=b"READY":raise RuntimeError("Renderer startup failed")
            for tick in range(k-4,k+1):
                proc.stdin.write((json.dumps([t[tick],states[tick].tolist()])+"\n").encode())
                proc.stdin.flush()
                rgb=np.frombuffer(proc.stdout.read(3*np.prod(shape)),np.uint8).reshape(*shape,3)
                depth=np.frombuffer(proc.stdout.read(4*np.prod(shape)),"<f4").reshape(shape)
                for i,field in enumerate(fields):
                    field.observe(t[tick],rgb[i],depth[i],states[tick,i])
        finally:proc.stdin.close()
        if proc.wait():raise RuntimeError("Renderer failed")
    panels=[]
    for i,field in enumerate(fields):
        centers = field.xyz+field.displacements[:,4] if len(field.xyz) else field.xyz
        path=states[k:j+1,1-i,:3]
        error = None if not len(centers) else np.median(centers,axis=0)-path[-1]
        score=float(field.score(path[-1:],.4)[0])
        panels.append(dict(observer=i,own=states[k,i,:3],path=path,centers=centers,score=score,
                           err_xy=None if error is None else np.linalg.norm(error[:2]),
                           err_3d=None if error is None else np.linalg.norm(error)))
    return dict(weight=cfg["weight"],time=t[k],target=t[j],panels=panels)


def plot(runs, output):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})
    cmap=LinearSegmentedColormap.from_list("risk",["#ffe99c","#ffc44e","#f16a3a","#a61e31"])
    fig,axes=plt.subplots(len(runs),2,figsize=(12,10),squeeze=False)
    fig.subplots_adjust(left=.07,right=.91,bottom=.095,top=.865,hspace=.37,wspace=.17)
    fig.text(.075,.96,"最近接前 0.4 秒的 OccFlow 预测与实飞位置",fontsize=20,weight="bold")
    fig.text(.075,.92,"世界坐标俯视 · 色面为沿高度取最大值的风险分数（≥ 0.5） · 黑色虚线为随后 0.4 秒的真实轨迹",fontsize=11)
    markers=[("o","#1984a2","本机当前位置"),("s","#536679","邻机当前位置"),
             ("+","#122b52","预测点簇中位数"),("*","#702e9b","邻机 0.4 秒后真值")]
    for row,run in enumerate(runs):
        xy=np.vstack([v[:,:2] for panel in run["panels"] for v in
                      (panel["own"][None],panel["path"],panel["centers"])])
        low,high=xy.min(0)-1,xy.max(0)+1
        center=(low+high)/2
        half=max(np.max(high-low)/2,2.8)
        x=np.linspace(center[0]-half,center[0]+half,400)
        y=np.linspace(center[1]-half,center[1]+half,400)
        xx,yy=np.meshgrid(x,y)
        for col,panel in enumerate(run["panels"]):
            ax=axes[row,col]
            pred=panel["centers"]
            if len(pred):
                distance=cKDTree(pred[:,:2]).query(np.c_[xx.ravel(),yy.ravel()])[0].reshape(xx.shape)
                risk=expit((.70-distance)/.13)
                ax.pcolormesh(x,y,np.ma.masked_less(risk,.5),cmap=cmap,vmin=.5,vmax=1,shading="auto",rasterized=True)
            path=panel["path"]
            ax.plot(*path[:,:2].T,"--",c="#323c48",lw=1.8)
            positions=[panel["own"],path[0],np.median(pred,axis=0) if len(pred) else None,path[-1]]
            for (m,c,_),pos,size in zip(markers,positions,[7,6,13,16]):
                if pos is not None:
                    ax.plot(*pos[:2],marker=m,color=c,ms=size,mew=2 if m=="+" else 1,
                            markeredgecolor=c if m=="+" else "white",linestyle="none")
            if len(pred):
                ax.plot([positions[2][0],path[-1,0]],[positions[2][1],path[-1,1]],":",c="#702e9b",lw=1.3)
            err="无预测点" if panel["err_xy"] is None else f"平面偏差 {panel['err_xy']:.2f} m"
            ax.set_title(f"权重 {run['weight']:.0f}   ·   {col} 号机预测 {1-col} 号机\n"
                         f"{run['time']:.2f} → {run['target']:.2f} s    |    {err}",loc="left",fontsize=11)
            ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),aspect="equal",xlabel="世界 X（m）",ylabel="世界 Y（m）")
            ax.grid(alpha=.17);ax.set_axisbelow(True)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(.5,1),cmap=cmap),cax=fig.add_axes([.925,.29,.014,.40]),label="动态风险分数")
    fig.legend(handles=[Line2D([],[],marker=m,color=c,lw=0,ms=10,label=label) for m,c,label in markers],
               loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(.49,.018))
    fig.savefig(output,dpi=170)
    plt.close(fig)


if __name__=="__main__":
    folders=list(map(Path,sys.argv[1:]))
    runs=[]
    for folder in folders:
        with TemporaryDirectory() as temp:
            runs.append(sample(folder,Path(temp)))
    output=folders[-1]/"occflow_xy_040_comparison.png"
    plot(runs,output)
    output.with_suffix(".json").write_text(json.dumps(runs,default=lambda a:a.tolist(),indent=2))
    print(output,flush=True)
