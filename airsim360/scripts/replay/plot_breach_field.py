"""Reconstruct a common metric section of the two fields at the last MPCC step."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.simulation.gate_geometry import gate_boxes
from scripts.perception.spatial_potential import PotentialNet, SpatialField, camera_points


def unit(vector):
    return vector / np.linalg.norm(vector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()
    run, out = args.run, args.replay
    experiment = json.loads((run / "experiment.json").read_text())
    checkpoint = Path(experiment["checkpoint"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == experiment["checkpoint_sha256"]
    assert (run / "rendered_course.json").read_bytes() == (out / "rendered_course.json").read_bytes()
    course = json.loads((run / "rendered_course.json").read_text())
    net = PotentialNet().eval()
    torch.set_num_threads(1)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    replay = np.load(out / "replay_depth.npz")
    rgb = np.stack([np.load(run / "rgb" / f"{ms:06d}.npy") for ms in replay["ms"]])
    traj = np.load(run / "flight/execution.npz")
    time, states = traj["time"], traj["state"]
    sample_time = replay["ms"][-1] / 1000
    sample_index = abs(time - sample_time).argmin()
    assert np.isclose(time[sample_index], sample_time)
    pose = states[sample_index]
    current, end = states[-2, :, :3], states[-1, :, :3]
    center = current.mean(axis=0)
    gate = np.asarray(course["centers"][0])
    forward = unit(gate-center)
    separation = current[1] - current[0]
    lateral = unit(separation - (separation @ forward)*forward)
    basis = np.stack([forward, lateral])
    project = lambda points: (points-center) @ basis.T
    x = np.linspace(-1.25, np.linalg.norm(gate-center)+.65, 360)
    y = np.linspace(-1.35, 1.35, 180)
    xx, yy = np.meshgrid(x, y)
    world = center + xx[..., None]*forward + yy[..., None]*lateral
    boxes = gate_boxes(course)[:4]
    distance_to_bars = []
    for box in boxes:
        c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
        local = (world - box["center"]) @ np.array([[c,-s,0],[s,c,0],[0,0,1]])
        distance_to_bars.append((abs(local) - np.asarray(box["size"])/2).max(axis=-1))
    peaks = [json.loads(line) for line in (run/"vision.jsonl").read_text().splitlines()][-2:]
    solver = json.loads((run/"flight/planning_log.json").read_text())[-2:]
    assert all(p["ego"] == i and np.isclose(p["time"],sample_time) and
               np.isclose(solver[i]["time"],time[-2]) for i,p in enumerate(peaks))
    info = dict(sensor_times_s=(replay["ms"]/1000).tolist(), decision_time_s=time[-2],
                stop_time_s=time[-1], original_warm_start_peaks=[p["peak"] for p in peaks],
                rgb_replay_mae_0_255=replay["rgb_mae"].tolist(), reconstructed=[])
    plt.rcParams.update({"font.family":"Microsoft YaHei", "axes.unicode_minus":False,
                         "font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig = plt.figure(figsize=(13,7.9), layout="constrained", facecolor="white")
    gs = fig.add_gridspec(2,2,height_ratios=[1.15,1])
    axes = [fig.add_subplot(gs[0,i]) for i in range(2)]
    profile = fig.add_subplot(gs[1,0])
    note = fig.add_subplot(gs[1,1])
    field_values, profiles = [], []
    length = np.linalg.norm(separation)
    line = np.linspace(0, length, 251)
    colors = ["#007272","#b64b15"]
    for i, ax in enumerate(axes):
        peer = 1-i
        field = SpatialField(net, rgb[:,i], replay["depth"][:,i])
        def risk(points):
            return field.query(camera_points(points, pose[i]))
        values = risk(world.reshape(-1,3)).reshape(xx.shape)
        field_values.append(values)
        curve = risk(current[0] + line[:,None]*unit(separation))
        profiles.append(curve)
        points = np.stack([pose[peer,:3], current[i], current[peer], end[i], end[peer], gate])
        metrics = dict(zip(["peer_at_capture","ego_at_decision","peer_at_decision",
                            "ego_at_stop","peer_at_stop","gate1_center"], risk(points).tolist()))
        peer_center = camera_points(points[:1], pose[i])[0]
        point_distances = np.linalg.norm(field.observation.points - peer_center, axis=1)
        metrics["surface_points_within_0p15m_of_peer_at_capture"] = int((point_distances < .15).sum())
        metrics["closest_surface_point_to_peer_center_at_capture_m"] = float(point_distances.min())
        metrics["ego_displacement_from_capture_to_decision_m"] = float(np.linalg.norm(current[i]-pose[i,:3]))
        info["reconstructed"].append(metrics)
        im = ax.pcolormesh(x,y,values,cmap="magma",vmin=0,vmax=1,shading="auto",rasterized=True)
        ax.contour(xx,yy,np.min(distance_to_bars,axis=0),levels=[0],colors=["#e4f4ff"],linewidths=1.1)
        for j, marker in enumerate(["o","^"]):
            xy = project(states[sample_index:,j,:3])
            ax.plot(*xy.T, color="#a6e9ff", lw=1.4)
            ax.scatter(*xy[-2],marker=marker,s=62,color="white",edgecolor="#16303b",lw=.9,zorder=5)
        ax.scatter(*project(pose[peer,:3]),marker="D",s=70,facecolors="none",edgecolors="#48e6fc",lw=1.5)
        ax.plot(*project(gate),marker="+",color="white",ms=10)
        ax.set(aspect="equal",title=f"{i} 号机看到的风险场",xlabel="朝向第一道门的距离（m）",ylabel="横向距离（m）")
        profile.plot(line,curve,color=colors[i],label=f"{i} 号机的风险场",lw=2.2)
        profile.scatter([0,length],curve[[0,-1]],color=colors[i],s=26,zorder=3)
    fig.colorbar(im,ax=axes,shrink=.72,label="风险值 U")
    profile.set(xlabel="从 0 号机沿两机连线的距离（m）",ylabel="重建风险值 U",ylim=(-.025,1.025),
                title=f"决策时刻两机之间的剖面（间距 {length:.3f} m）",xlim=(-.01,length+.01))
    profile.grid(alpha=.2)
    profile.legend(frameon=False)
    note.axis("off")
    note.set_title("如何读图",loc="left")
    note.text(0,.87,"○  0 号机     △  1 号机     ◇  采图时的邻机\n"
              f"白色标记为决策位置（{time[-2]:.2f} s），蓝线为实飞轨迹在切面上的投影\n"
              f"（{sample_time:.2f} 至 {time[-1]:.2f} s）。十字为门心，白色轮廓为门框截面。\n\n"
              "两个切面均穿过决策时的两机中心及第一道门心。\n"
              "下图沿两机中心连线查询，每个位置分别读取两架机的风险值。\n\n"
              "原始记录 · 最后一次求解前，预测节点上的最大风险值\n"
              f"0 号机  {peaks[0]['peak']:.3f}                 1 号机  {peaks[1]['peak']:.3f}",
              transform=note.transAxes,va="top",linespacing=1.65)
    fig.suptitle(f"越界前最后一次 MPCC 决策  |  权重 {experiment['weight']:,.0f}  ·  原始 RGB + 回放深度重建",
                 fontsize=17,fontweight="bold")
    fig.savefig(out/"breach_field.png",dpi=150)
    np.savez_compressed(out/"breach_field.npz",x=x,y=y,U=field_values,origin=center,basis=basis,
                        line_distance=line,line_U=profiles)
    (out/"breach_values.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info,indent=2))


if __name__ == "__main__":
    main()
