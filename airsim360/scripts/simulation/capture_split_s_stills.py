"""Render the Split-S course and metric depth stills at the recorded starting grid."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import argparse

import airsim
import cv2
import numpy as np

from scripts.simulation.airsim360_replay import create_renderer
from scripts.simulation.preview_split_s import place_course, set_camera, world_to_ned


def save(path, image):
    if not cv2.imwrite(str(path), image):
        raise IOError(path)


def perspective(image, direction, size, fov, interpolation):
    width, height = size
    forward = np.asarray(direction, np.float32)
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0, 0, 1], np.float32), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    focal = width / (2 * np.tan(np.deg2rad(fov / 2)))
    x = ((np.arange(width, dtype=np.float32) + .5 - width / 2) / focal)[None, :, None]
    result = np.empty((height, width) + image.shape[2:], image.dtype)
    for row in range(0, height, 96):
        end = min(height, row + 96)
        y = ((np.arange(row, end, dtype=np.float32) + .5 - height / 2) / focal)[:, None, None]
        rx, ry, rz = np.moveaxis(forward + x * right + y * down, -1, 0)
        u = (.5 + np.arctan2(ry, rx) / (2 * np.pi)) * image.shape[1] - .5
        v = (.5 + np.arctan2(rz, np.hypot(rx, ry)) / np.pi) * image.shape[0] - .5
        result[row:end] = cv2.remap(image, u, v, interpolation, borderMode=cv2.BORDER_WRAP)
    return result


def depth_figure(depth, low, high):
    def colors(values):
        norm = np.clip((np.nan_to_num(values, nan=high) - low) / (high-low), 0, 1)
        return cv2.applyColorMap(np.uint8((1-norm)*255), cv2.COLORMAP_VIRIDIS)
    field = colors(depth)
    field[~np.isfinite(depth)] = 0
    width = field.shape[1]
    bar = np.full((54, width, 3), 22, np.uint8)
    ramp = np.linspace(low, high, width-48, dtype=np.float32)[None]
    bar[8:18, 24:-24] = colors(ramp)
    for frac in (0, .25, .5, .75, 1):
        value = low + frac * (high-low)
        label = f'{value:.2f} m' if high < 1 else f'{value:g} m'
        x = max(2, min(width - 65, int(24 + frac*(width-48) - 18)))
        cv2.putText(bar, label, (x, 42), cv2.FONT_HERSHEY_SIMPLEX, .5, (230,230,230), 1, cv2.LINE_AA)
    return np.vstack((field, bar))


def context_depth_figure(depth):
    # Give the nearby airframe and distant scenery each part of one continuous scale.
    meters = [.20, .30, .40, .50, 2., 8., 32., 64.]
    stops = [0., .14, .28, .42, .54, .69, .86, 1.]
    norm = np.interp(np.nan_to_num(depth, nan=64.), meters, stops)
    image = cv2.applyColorMap(np.uint8((1-norm)*255), cv2.COLORMAP_VIRIDIS)
    image[~np.isfinite(depth)] = 0
    width = image.shape[1]
    legend = np.full((54, width, 3), 22, np.uint8)
    ramp = np.linspace(255, 0, width-64, dtype=np.uint8)[None]
    legend[8:18, 32:-32] = cv2.applyColorMap(ramp, cv2.COLORMAP_VIRIDIS)
    for value, stop in zip(meters, stops):
        x = round(32+stop*(width-65))
        label = f'{value:g} m'
        (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
        cv2.putText(legend, label, (x-w//2,40), cv2.FONT_HERSHEY_SIMPLEX, .5, (230,230,230), 1, cv2.LINE_AA)
    return np.vstack((image, legend))


def main(out, revised=False):
    cv2.setNumThreads(1)
    run = Path("reports/spatial_potential/cf21_depth_snap_joint01")
    placement_file = Path("reports/split_s_open_yard/placement.json")
    placement = json.loads(placement_file.read_text())
    course = json.loads((run / "rendered_course.json").read_text())
    scale = json.loads((run / "vehicle_geometry.json").read_text())["scale_xyz"]
    with np.load(run / "flight/execution.npz") as record:
        states = record["state"][0]
    out.mkdir(exist_ok=False)
    os.environ["RISK_REPLAY_LOG"] = str(out / "sensor.jsonl")
    with ExitStack() as stack:
        r = stack.enter_context(create_renderer(True, scale, (1536, 768)))
        client, vehicle = r.client, r.vehicles[0]
        r.translation = np.array(placement["translation_ned_m"])
        r.heading = np.deg2rad(placement["heading_ned_deg"])
        def ned(world):
            return world_to_ned(world, r.translation, r.heading)
        client.simRunConsoleCommand("r.MotionBlurQuality 0")
        actors = set(client.simListSceneObjects())
        for name in placement["removed_yard_fixtures"]:
            if name in actors and not client.simDestroyObject(name):
                raise RuntimeError(name)
        bars = place_course(client, course, stack, placement_file.with_name("gate_texture.png"), r.translation, r.heading)
        if len(bars) != 28 or not all(bar["textured"] for bar in bars):
            raise RuntimeError("Incomplete gate installation")
        if not revised:
            pair = r.capture(0., states)
            depth = pair["depth"][0].copy()
            save(out/"drone0_rgb.png", pair["rgb"][0, :, :, ::-1])
            save(out/"drone0_depth.png", depth_figure(depth, 0, 35))
            np.save(out/"drone0_range_m.npy", depth)
            print("Starting panoramas saved", flush=True)

        # Swap the two proxy poses left by capture: show only the peer at its recorded pose.
        from scripts.simulation.airsim360_replay import pose
        target = ned(states[1, :3])
        qw, qx, qy, qz = states[1, 6:10] * [1,1,-1,-1]
        c, s = np.cos(r.heading/2), np.sin(r.heading/2)
        peer_pose = pose(target, [c*qw-s*qz, c*qx-s*qy, c*qy+s*qx, c*qz+s*qw])
        client.simSetObjectPose(r.meshes[0], r.far, True)
        client.simSetObjectPose(r.meshes[1], peer_pose, True)
        offset = [-.29, .13, .055] if revised else [-.19, -.19, .19]
        eye = ned(states[1, :3] + offset)
        size = (960, 720) if revised else (480, 360)
        def closeup():
            set_camera(client, vehicle, eye)
            r.read(vehicle)
            ranges = r.read(vehicle)[1]
            return perspective(ranges, target-eye, size, 46 if revised else 31, cv2.INTER_NEAREST)
        seen = closeup()
        client.simSetObjectPose(r.meshes[1], r.far, True)
        empty = closeup()
        distance = float(np.linalg.norm(eye-target))
        mask = ((np.abs(seen-distance) < .10) & (~np.isfinite(empty) | (empty-seen > .12)))
        ys, xs = np.where(mask)
        if len(xs) < 100 or xs.min() == 0 or xs.max() == size[0]-1 or ys.min() == 0 or ys.max() == size[1]-1:
            raise RuntimeError(f"Peer silhouette missing or clipped: {len(xs)} pixels")
        shift = np.rint([(size[0]-1-xs.min()-xs.max())/2, (size[1]-1-ys.min()-ys.max())/2]).astype(int)
        isolated = np.roll(np.where(mask, seen, np.nan), (shift[1], shift[0]), axis=(0,1))
        if revised:
            # Recenter by cropping the real scene around the peer, keeping its background.
            left, top = (np.array(size) - [832, 576])//2 - shift
            if not (0 <= left <= size[0]-832 and 0 <= top <= size[1]-576):
                raise RuntimeError("Centered crop would extend beyond the camera view")
            context = seen[top:top+576, left:left+832].copy()
            np.save(out/"peer_range_m.npy", context)
            save(out/"peer_depth.png", context_depth_figure(context))
        else:
            np.save(out/"peer_range_m.npy", isolated)
            low = np.floor(np.nanmin(isolated)*100)/100
            high = np.ceil(np.nanmax(isolated)*100)/100
            save(out/"peer_depth.png", depth_figure(isolated, low, high))
        print("Centered peer saved", flush=True)

        eye_world, aim_world = [-11., -18., 14.], [1.2, .5, 1.9]
        if revised:
            eye_world, aim_world = [-18., -9.5, 4.1], [1.6, 1., 2.0]
        eye, aim = ned(eye_world), ned(aim_world)
        set_camera(client, vehicle, eye)
        camera = "panorama_original"
        client.client.call("simSetPanoramaResolution", camera, 4096, 2048, vehicle)
        for _ in range(3):
            client.client.call("simTriggerPanoramaCapture", camera, vehicle)
            frame, = client.simGetImages([airsim.ImageRequest(camera, 0, False, False)], vehicle_name=vehicle)
        pano = np.frombuffer(frame.image_data_uint8, np.uint8).reshape(frame.height, frame.width, 3)
        overview = perspective(pano, aim-eye, (1920, 1080), 47 if revised else 66, cv2.INTER_LINEAR)
        save(out/"split_s_course.png", overview)
        meta = dict(source=str(run), gate_count=len(bars)//4, start_positions_m=states[:, :3].tolist(),
                    peer_id=1, peer_camera_range_m=distance, peer_surface_range_m=[float(np.nanmin(isolated)),float(np.nanmax(isolated))],
                    peer_foreground_pixels=int(mask.sum()), overview_eye_world_m=eye_world, overview_aim_world_m=aim_world)
        (out/"capture.json").write_text(json.dumps(meta, indent=2))
        print(meta, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--revised", action="store_true")
    args = parser.parse_args()
    main(args.output, args.revised)
