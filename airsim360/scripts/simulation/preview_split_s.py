"""Place the original Split-S gate frames and render fixed views in the factory."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
import time

import airsim
import cv2
import numpy as np

from scripts.simulation.airsim360_replay import AXES, pose, xyz
from scripts.simulation.gate_geometry import gate_boxes

# Lower the original replay alignment by 4.4 m to clear the overhead beams.
TRANSLATION = np.array([8., 1., -.6])


def place_course(client, course, stack, texture, translation, heading):
    parts = []
    for box in gate_boxes(course):
        name, scale = box["name"], box["size"]
        target = world_to_ned(box["center"], translation, heading)
        yaw = heading - box["yaw"]
        quat = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
        name = client.simSpawnObject(name, "Cube", pose(target, quat),
                                     airsim.Vector3r(*scale), False, False)
        if not name:
            raise RuntimeError(f"Failed to spawn {box['name']}")
        stack.callback(client.simDestroyObject, name)
        actual = client.simGetObjectPose(name)
        if (not np.allclose(xyz(actual.position), target, atol=.005)
                or not np.allclose(xyz(client.simGetObjectScale(name)), scale, atol=.005)):
            raise RuntimeError(f"Incorrect transform for {name}")
        orientation = actual.orientation
        observed = [orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val]
        if abs(np.dot(observed, quat)) < .99999:
            raise RuntimeError(f"Incorrect rotation for {name}")
        textured = client.simSetObjectMaterialFromTexture(name, str(texture.resolve()))
        parts.append(dict(name=name, position=target.tolist(), scale=scale, textured=textured))
    return parts


def perspective(panorama, direction, fov_deg):
    forward = direction / np.linalg.norm(direction)
    right = np.cross([0., 0., 1.], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    width, height = 1100, 700
    y, x = np.indices((height, width), dtype=np.float32)
    focal = width / (2*np.tan(np.deg2rad(fov_deg/2)))
    rays = (forward + (x[...,None]+.5-width/2)*right/focal
            + (y[...,None]+.5-height/2)*down/focal)
    lon = np.arctan2(rays[:,:,1], rays[:,:,0])
    lat = np.arctan2(rays[:,:,2], np.hypot(rays[:,:,0], rays[:,:,1]))
    h, w = panorama.shape[:2]
    u = ((lon/(2*np.pi)+.5)*w-.5).astype("float32")
    v = ((lat/np.pi+.5)*h-.5).astype("float32")
    return cv2.remap(panorama, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def world_to_ned(point, translation, heading):
    x, y, z = np.asarray(point)*AXES
    c, s = np.cos(heading), np.sin(heading)
    return np.asarray(translation) + [c*x-s*y, s*x+c*y, z]


def set_camera(client, vehicle, eye):
    target = pose(eye)
    state = airsim.KinematicsState()
    state.position, state.orientation = target.position, target.orientation
    for _ in range(12):
        client.simSetKinematics(state, True, vehicle)
        client.simSetVehiclePose(target, True, vehicle)
        client.simContinueForFrames(1)
        time.sleep(.18)
        actual = client.simGetVehiclePose(vehicle)
        if (client.simIsPause() and np.linalg.norm(xyz(actual.position)-eye)<.03
                and abs(actual.orientation.w_val)>.99995):
            break
    else:
        raise RuntimeError(f"Preview camera pose did not settle at {eye}")


def render_view(client, vehicle, eye, aim, fov, path):
    set_camera(client, vehicle, eye)
    camera = "panorama_original"
    for _ in range(4):
        client.client.call("simTriggerPanoramaCapture", camera, vehicle)
        response, = client.simGetImages([airsim.ImageRequest(camera, 0, False, False)], vehicle_name=vehicle)
    pano = np.frombuffer(response.image_data_uint8, np.uint8).reshape(response.height,response.width,3)
    if not cv2.imwrite(str(path), perspective(pano, aim-eye, fov)):
        raise IOError(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translation", nargs=3, type=float, default=TRANSLATION)
    parser.add_argument("--heading-deg", type=float, default=0.)
    parser.add_argument("--clear-yard-posts", action="store_true",
                        help="Remove perimeter posts, fences and bollards in this disposable instance")
    args = parser.parse_args()
    course = json.loads(args.course.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    texture = out / "gate_texture.png"
    y, x = np.indices((128,128))
    stripes = np.where((((x+y)//24)%4 == 0)[...,None], [28,35,39], [12,117,250])
    cv2.imwrite(str(texture), stripes.astype(np.uint8))
    with ExitStack() as stack:
        client = airsim.MultirotorClient(timeout_value=30)
        stack.callback(client.client.close)
        if client.simListSceneObjects("SplitS_G.*"):
            raise RuntimeError("Existing preview gate objects found on server")
        vehicle, = client.listVehicles()
        stack.callback(client.simPause, client.simIsPause())
        client.simPause(True)
        client.simRunConsoleCommand("r.MotionBlurQuality 0")
        removed = []
        if args.clear_yard_posts:
            for pattern in ("SM_Pole_.*", "SM_Fence_.*", "SM_Cone_.*"):
                for actor in client.simListSceneObjects(pattern):
                    if not client.simDestroyObject(actor):
                        raise RuntimeError(f"Could not clear yard fixture {actor}")
                    removed.append(actor)
        heading = np.deg2rad(args.heading_deg)
        parts = place_course(client, course, stack, texture, args.translation, heading)
        client.client.call("simSetPanoramaResolution", "panorama_original", 4096, 2048, vehicle)
        centers = np.array(course["centers"])
        views = [
            ("start", course["start_position_center"], centers[0], 65),
            ("stacked", [-9.,-6.8,2.7], (centers[3]+centers[4])/2, 78),
            ("overview", [-9.,-5.,6.3], [1.,.5,2.], 101),
        ]
        for i, (center, normal) in enumerate(zip(centers, course["normals"]), 1):
            views.append((f"gate_{i}", center-3*np.asarray(normal), center, 76))
        for name, eye, aim, fov in views:
            render_view(client, vehicle, world_to_ned(eye, args.translation, heading),
                        world_to_ned(aim, args.translation, heading), fov, out/f"{name}.jpg")
            print(name, flush=True)
        (out/"placement.json").write_text(json.dumps(dict(course=str(args.course), bars=parts,
                            planner_to_ned_signs=AXES.tolist(), translation_ned_m=list(args.translation),
                            heading_ned_deg=args.heading_deg,
                            removed_yard_fixtures=removed,
                            views=[dict(name=n, camera_world=list(e), look_at_world=list(a), fov=f)
                                   for n,e,a,f in views]), indent=2))


if __name__ == "__main__":
    main()
