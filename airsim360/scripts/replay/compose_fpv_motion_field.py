"""Project the history motion model's spherical kernels onto its recorded RGB panoramas."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from scipy.special import expit
import torch

from scripts.perception.context_motion import ContextMotionPrior
from scripts.perception.local_future_field import LocalFutureField, kernel_geometry, observe
from scripts.perception.spatial_potential import camera_points
from scripts.perception.panorama import spherical_rays as rays


RACE = Path("reports/spatial_potential/cf21_motion_history01_race2")
W, H = 512, 256
RAYS = rays(H, W)
LUT = (255 * colormaps["inferno"](np.linspace(0, 1, 256))[:, :3]).astype("uint8")
BG, FG, DIM = "#101923", "#f2f5f7", "#b6c5d0"
FONT = {size: ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size) for size in (16, 19, 24)}


def projected_field(model, flow, pose):
    c, r, w, valid, _, coeff = kernel_geometry(model, flow, pose, [.4])
    assert np.all(coeff == 0), "Projection requires spherical kernels"
    logits = np.full((H, W), -60., dtype=np.float32)
    for center, radius, width in zip(c[0, valid[0]], r[0, valid[0]], w[0, valid[0]]):
        p = camera_points(center, pose)
        dot = RAYS @ p.astype("f")
        depth = dot.clip(.08, 12.)
        distance = np.sqrt(np.maximum(0., p @ p + depth * (depth - 2 * dot)))
        np.maximum(logits, (radius - distance) / width, out=logits)
    return expit(logits)


def overlay(image, phi, cutoff=None):
    gray = np.asarray(image.convert("L"), dtype="f")[..., None] * .48
    alpha = np.sqrt(phi)[..., None]
    pixels = np.uint8(gray * (1 - alpha) + LUT[np.rint(phi * 255).astype("uint8")] * alpha)
    if cutoff is not None:
        pixels[phi < cutoff] = 0
    return Image.fromarray(pixels)


def label(draw, x, y, string, size=19, fill=FG):
    draw.text((x, y), string, font=FONT[size], fill=fill)


def motion_network():
    cfg = json.loads((RACE / "experiment.json").read_text())
    checkpoint, appearance = map(Path, (cfg["future_checkpoint"], cfg["future_prior"]))
    for path, key in ((checkpoint, "future_sha256"), (appearance, "future_prior_sha256")):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == cfg[key]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prior = ContextMotionPrior(appearance, saved)
    model = LocalFutureField(saved.get("correction_speed", 3.)).eval()
    model.load_state_dict(saved["model"])
    assert prior.use_history and not hasattr(prior, "mixture") and "directional" not in saved
    return prior, model


def recorded_future(times, positions, k, lead):
    target = times[k] + lead
    if target > times[-1] + 1e-8:
        return None
    return np.array([np.interp(target, times, axis) for axis in positions.T])


class LegacyProjection:
    def __init__(self, source=RACE.parent / "cf21_bendhead03_run1", lead=.1):
        from scripts.perception.time_query_field import load_rgbd
        self.lead = lead
        cfg = json.loads((source / "experiment.json").read_text())
        weights = Path(cfg["time_checkpoint"])
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == cfg["time_checkpoint_sha256"]
        self.model = load_rgbd(weights)
        self.shape = (64, 128)
        directions = rays(*self.shape)
        depths = np.r_[.08, np.arange(.5, 12.01, .5)].astype("f")
        self.points = torch.from_numpy((directions[:, :, None] * depths[:, None]).reshape(-1, 3))

    def query(self, context, points):
        q = points.reshape(1, -1, 3)
        return self.model.query_logits(context, q, torch.full_like(q[..., :1], self.lead)).sigmoid()

    def project(self, context):
        values = [self.query(context, q).reshape(-1) for q in self.points.split(8192)]
        grid = torch.cat(values).reshape(*self.shape, -1).amax(-1).numpy()
        return np.asarray(Image.fromarray(grid).resize((W, H), Image.Resampling.BILINEAR)), float(grid.max())


@torch.inference_mode()
def render(ffmpeg, legacy=False, race=RACE, cutoff=None, output_folder=None, lead=.1):
    torch.set_num_threads(1)
    if legacy:
        from scripts.perception.time_query_field import encode_views
        config = json.loads((race / "experiment.json").read_text())
        projection = LegacyProjection(race,lead) if config["time_checkpoint"] else LegacyProjection(lead=lead)
    else:
        prior, model = motion_network()
    with np.load(race / "flight/execution.npz") as f:
        times, poses = f["time"], f["state"]
    log = {}
    for line in (race / "vision.jsonl").read_text().splitlines():
        row = json.loads(line)
        if abs(row["time"] - row["decision_s"]) < 1e-6:
            log[round(row["time"] * 1000), row["ego"]] = row
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1", "-filter_threads", "1"]
    output = RACE / "two_laps_fpv_dynamic_risk_field.mp4"
    if legacy:
        output = race / f"two_laps_fpv_cnn_gru_{round(1000*lead)}ms.mp4"
    if output_folder is not None:
        output = output_folder / output.name
    samples = []
    reader = [*args, "-i", str(race / "chase.mp4"), "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "-"]
    writer = [*args, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1520x644", "-r", "12.5",
              "-i", "-", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "17",
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    gradient = np.tile(np.linspace(0., 1., 401), (15, 1))
    bar = overlay(Image.new("RGB", (401, 15), "#777777"), gradient, cutoff)
    with subprocess.Popen(reader, stdout=subprocess.PIPE) as chase, subprocess.Popen(writer, stdin=subprocess.PIPE) as enc:
        try:
            for frame, k in enumerate(range(0, len(times), 4)):
                stamp = round(times[k] * 1000)
                ids = np.maximum(0, np.arange(k - 3, k + 1))
                buffers = {}
                for folder in ("rgb", "depth"):
                    buffers[folder] = np.stack([np.load(race / folder / f"{round(times[i]*1000):06d}.npy") for i in ids])
                data = chase.stdout.read(640 * 360 * 3)
                assert len(data) == 640 * 360 * 3, (frame, "missing chase view")
                canvas = Image.new("RGB", (1520, 644), BG)
                draw = ImageDraw.Draw(canvas)
                label(draw, 16, 9, "双机竞速  /  第三人称", 24)
                canvas.paste(Image.frombytes("RGB", (640, 360), data).resize((464, 261), Image.Resampling.LANCZOS), (12, 52))
                label(draw, 16, 333, f"飞行时间   {stamp/1000:.2f} s", 24)
                label(draw, 16, 389, f"旧 CNN–GRU  /  提前 {lead:.2f} 秒的预测风险" if legacy else "颜色表示 0.40 秒后的动态风险值", 19)
                canvas.paste(bar, (16, 429))
                for x, value in ((16, "0"), (208, "0.5"), (405, "1")):
                    label(draw, x, 451, value, 16, DIM)
                label(draw, 16, 496, "每个方向取 0.08–12 m 的采样峰值。" if legacy else "每个方向取 0.08–12 m 内的最高风险值。", 19)
                label(draw, 16, 533, f"邻机点 = 在 {lead:.2f} 秒后实飞位置查询的风险" if legacy else "两机共用色标，灰色底图为当前帧。", 19)
                label(draw, 16, 570, "画面中央朝前，两端在正后方相接。", 19)
                if cutoff is not None:
                    label(draw, 16, 606, f"黑色区域：预测风险值 < {cutoff:g}", 19)
                if legacy:
                    contexts = encode_views(projection.model, buffers["rgb"], buffers["depth"])
                for ego, x in ((0, 484), (1, 1004)):
                    rgb, depth = (buffers[key][:, ego] for key in ("rgb", "depth"))
                    image = Image.fromarray(rgb[-1])
                    assert image.size == (W, H)
                    if legacy:
                        context = contexts[ego:ego+1]
                        phi, peak = projection.project(context)
                        peer = None
                        future = recorded_future(times, poses[:,1-ego,:3], k, lead)
                        if future is not None:
                            q = camera_points(future, poses[k, ego])
                            peer = projection.query(context, torch.tensor(q, dtype=torch.float32)).item()
                        samples.append((stamp, ego, peak, peer))
                        label(draw, x, 325, f"{ego} 号机  /  风险", 24)
                        value = "—" if peer is None else f"{peer:.3f}"
                        label(draw, x+192, 332, f"峰值  {peak:.3f}      邻机点  {value}", 19, DIM)
                    else:
                        flow = observe(prior, times[ids], poses[ids, ego], rgb, depth)
                        assert len(flow.xyz) == log[stamp, ego]["flow_pixels"], (stamp, ego)
                        phi = projected_field(model, flow, poses[k, ego])
                        label(draw, x, 325, f"{ego} 号机  /  第一人称动态风险场", 24)
                        label(draw, x + 365, 332, f"峰值  {phi.max():.2f}", 19, DIM)
                    assert np.isfinite(phi).all() and 0 <= phi.min() <= phi.max() <= 1
                    label(draw, x, 9, f"{ego} 号机  /  第一人称 RGB", 24)
                    canvas.paste(image, (x, 52))
                    canvas.paste(overlay(image, phi, cutoff), (x, 370))
                enc.stdin.write(canvas.tobytes())
            assert not chase.stdout.read(1), "Extra chase frames"
            enc.stdin.close()
            assert chase.wait() == 0 and enc.wait() == 0
        finally:
            chase.stdout.close()
            if chase.poll() is None:
                chase.terminate()
    check = subprocess.run([*args, "-i", str(output), "-progress", "pipe:1", "-f", "null", "-"],
                           capture_output=True, text=True, check=True)
    counts = [int(line[6:]) for line in check.stdout.splitlines() if line.startswith("frame=")]
    assert counts[-1] == frame + 1
    if legacy:
        with output.with_suffix(".csv").open("w", newline="") as stream:
            table = csv.writer(stream)
            ms=round(1000*lead)
            table.writerow(("frame_time_ms", "ego", f"ray_grid_peak_{ms}ms", f"risk_at_recorded_peer_{ms}ms"))
            table.writerows(samples)
    print(f"{output}: verified {frame + 1} frames, last observation {stamp} ms")


if __name__ == "__main__":
    render(sys.argv[1], "--legacy-100ms" in sys.argv[2:])
