"""Render the same held-out gate at a crossing and its matched empty observation."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import torch

from scripts.perception.spatial_potential import PotentialNet, SpatialField, camera_points


def main():
    root = Path("reports/spatial_potential")
    manifest = json.loads(Path("reports/flight_count_comparison/manifest_8.json").read_text(encoding="utf-8-sig"))
    entry = next(e for e in manifest["episodes"] if e["split"] == "test")
    event = json.loads(Path("reports/projected_rgbd_8/events_0p4.json").read_text())["projected_8"][0]
    ego, g = event["observer"], event["gate"]-1
    bars = np.array([b["center"] for b in entry["gate_boxes"]]).reshape(-1, 4, 3)[g]
    center = bars.mean(0)
    side = bars[1]-bars[0]
    side /= np.linalg.norm(side)
    grid = np.linspace(-1.15, 1.15, 151)
    y, z = np.meshgrid(grid, grid)
    points = center + y[..., None]*side + z[..., None]*[0, 0, 1]
    with np.load(Path("reports/flight_count_comparison") / entry["trajectory"]) as f:
        t, states = f["time"], f["state"]
    with np.load(Path("reports/flight_count_comparison") / entry["observations"]) as f:
        rgb, depth = f["rgb"], f["depth"]
    ck = torch.load(root/"sensor12hz/best.pt", map_location="cpu", weights_only=True)
    net = PotentialNet().eval()
    net.load_state_dict(ck["model"])
    torch.set_num_threads(1)
    plt.rcParams["font.family"] = ["Microsoft YaHei"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.1), layout="constrained")
    for ax, label, time in zip(axes, ("空门", "邻机正在穿门"), (event["empty_observation_s"], event["event_s"])):
        k = int(abs(t-time).argmin())
        h = slice(k-3*ck["stride"], k+1, ck["stride"])
        field = SpatialField(net, rgb[h, ego], depth[h, ego])
        risk = field.query(camera_points(points.reshape(-1, 3), states[k, ego])).reshape(y.shape)
        im = ax.pcolormesh(y, z, risk, vmin=0, vmax=1, cmap="magma", shading="auto")
        ax.add_patch(Rectangle((-.75, -.75), 1.5, 1.5, fill=False, edgecolor="white",
                               linewidth=1.2, linestyle="--"))
        delta = states[k, 1-ego, :3]-center
        if abs(delta@np.cross(side, [0, 0, 1])) < .25:
            ax.add_patch(Circle((delta@side, delta[2]), .1, fill=False, edgecolor="#54ffd3", linewidth=2))
        ax.set_aspect("equal")
        ax.set_xlabel("门面横向位置（米）")
        ax.set_title(f"{label}   {t[k]:.2f} 秒")
    axes[0].set_ylabel("相对门中心高度（米）")
    fig.colorbar(im, ax=axes, shrink=.78, label="视觉空间风险值")
    fig.suptitle(f"留出航次 · 第 {g+1} 道门    虚线为门洞边界，圆圈为邻机仿真真值", fontsize=12)
    fig.savefig(root/"heldout_gate.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
