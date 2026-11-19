"""Paired ERP observations on a dedicated FactoryExteriorInterior server."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import time

import airsim
import numpy as np


WIDTH, HEIGHT = 256, 128
CAMERAS = ("panorama_original", "panorama_depth")
# Planner world/body are z-up. The simulator uses NED/FRD.
AXES = np.array([1., -1., -1.])
TRANSLATION = np.array([8., 1., -5.])
SECOND_ORIGIN = np.array([2., 0., 0.])


def pose(position, quaternion=(1., 0., 0., 0.)):
    w, x, y, z = map(float, quaternion)
    return airsim.Pose(airsim.Vector3r(*map(float, position)), airsim.Quaternionr(x, y, z, w))


def xyz(v):
    return np.array([v.x_val, v.y_val, v.z_val])


class Renderer:
    def __init__(self, client, vehicles, meshes, log, resolution=(WIDTH, HEIGHT), refresh_on_switch=True):
        self.width, self.height = resolution
        self.client = client
        self.vehicles = vehicles
        self.meshes = meshes
        self.log = log
        self.previous_stamp = 0
        self.refresh_on_switch = refresh_on_switch
        self.translation = TRANSLATION
        self.heading = 0.
        self.far = pose([900, 900, -900])

    def read(self, vehicle):
        for camera in CAMERAS:
            self.client.client.call("simTriggerPanoramaCapture", camera, vehicle)
        rgb, depth = self.client.simGetImages([
            airsim.ImageRequest(CAMERAS[0], 0, False, False),
            airsim.ImageRequest(CAMERAS[1], 0, True, False)], vehicle_name=vehicle)
        if any((r.width, r.height) != (self.width, self.height) for r in (rgb, depth)):
            raise RuntimeError("Unexpected panorama dimensions")
        bgr = np.frombuffer(rgb.image_data_uint8, np.uint8).reshape(self.height, self.width, 3)
        values = (np.asarray(depth.image_data_float, np.float32) if depth.image_data_float
                  else np.frombuffer(depth.image_data_uint8, "<f4"))
        distance = values.reshape(self.height, self.width).copy()
        distance[(distance <= 0) | (distance >= 32256)] = np.nan
        return bgr[:, :, ::-1].copy(), distance, [rgb.time_stamp, depth.time_stamp]

    def position_pair(self, local, quats):
        for name, position, quat in zip(self.vehicles, local, quats):
            state = airsim.KinematicsState()
            target = pose(position, quat)
            state.position, state.orientation = target.position, target.orientation
            self.client.simSetKinematics(state, True, name)
            self.client.simSetVehiclePose(target, True, name)
        self.client.simContinueForFrames(1)
        # The RPC can return before the render thread has dispatched this step.
        time.sleep(.08)
        deadline = time.monotonic() + 3
        while not self.client.simIsPause():
            if time.monotonic() > deadline:
                raise TimeoutError("Renderer did not pause after advancing a frame")
            time.sleep(.005)
        errors, angles = [], []
        for name, position, quat in zip(self.vehicles, local, quats):
            actual = self.client.simGetVehiclePose(name)
            errors.append(float(np.linalg.norm(xyz(actual.position) - position)))
            q = actual.orientation
            dot = abs(np.dot([q.w_val, q.x_val, q.y_val, q.z_val], quat))
            angles.append(float(np.degrees(2*np.arccos(np.clip(dot, 0, 1)))))
        return errors, angles

    def capture(self, time_s, snapshot):
        c, s = np.cos(self.heading), np.sin(self.heading)
        rot = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        positions = (snapshot[:, :3] * AXES) @ rot.T + self.translation
        quats = snapshot[:, 6:10] * [1., 1., -1., -1.]
        w,x,y,z = quats.T
        c,s = np.cos(self.heading/2), np.sin(self.heading/2)
        quats = np.stack((c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w),axis=1)
        if len(self.vehicles) == 1:
            views = []
            discarded_stamps = []
            for i in (0, 1):
                for mesh, target in zip(self.meshes, [self.far if j == i else pose(positions[j], quats[j]) for j in (0,1)]):
                    if not self.client.simSetObjectPose(mesh, target, True):
                        actual = self.client.simGetObjectPose(mesh)
                        def quat(p):
                            q = p.orientation
                            return [q.w_val,q.x_val,q.y_val,q.z_val]
                        if not (np.linalg.norm(xyz(actual.position)-xyz(target.position)) < .005 and
                                abs(np.dot(quat(actual),quat(target))) > .99995):
                            raise RuntimeError(f"Cannot position visual proxy {mesh}")
                for attempt in range(16):
                    errors, angles = self.position_pair(positions[i:i+1], quats[i:i+1])
                    if max(errors) <= .03 and max(angles) <= 1.5 and self.client.simIsPause():
                        if self.refresh_on_switch:
                            # A panorama triggered immediately after a view switch can omit the peer.
                            first_stamps = self.read(self.vehicles[0])[2]
                        view = self.read(self.vehicles[0])
                        if view[2][0] == view[2][1] and view[2][0] > self.previous_stamp:
                            break
                else:
                    raise RuntimeError(f"Camera pose/frame at t={time_s} agent={i}, error={errors}, angle={angles}, previous={self.previous_stamp}, state={self.client.getMultirotorState(self.vehicles[0]).timestamp}")
                self.previous_stamp = view[2][0]
                if self.refresh_on_switch:
                    discarded_stamps.append(first_stamps)
                views.append(view)
            self.log.write(json.dumps(dict(time=time_s, sensor_ns=[v[2] for v in views],
                                           discarded_sensor_ns=discarded_stamps))+"\n")
            self.log.flush()
            return dict(time=time_s, rgb=np.stack([v[0] for v in views]), depth=np.stack([v[1] for v in views]), valid=np.ones(2, bool))
        local = positions - np.array([[0., 0., 0.], SECOND_ORIGIN])
        for attempt in range(8):
            errors, angles = self.position_pair(local, quats)
            if (max(errors) <= .03 and max(angles) <= 1.5 and
                    self.client.getMultirotorState(self.vehicles[0]).timestamp > self.previous_stamp):
                break
        else:
            raise RuntimeError(f"Replay pose residuals {errors} m, {angles} degrees")
        views = []
        for i, name in enumerate(self.vehicles):
            other = 1-i
            for mesh, target in ((self.meshes[i], self.far),
                                 (self.meshes[other], pose(positions[other], quats[other]))):
                if not self.client.simSetObjectPose(mesh, target, True):
                    raise RuntimeError(f"Could not position {mesh}")
            views.append(self.read(name))
        stamps = [v[2] for v in views]
        stamp = stamps[0][0]
        if (len(set(np.ravel(stamps))) != 1 or stamp <= self.previous_stamp
                or not self.client.simIsPause()):
            raise RuntimeError(f"Image pair is not from one frozen timestamp: {stamps}")
        self.previous_stamp = stamp
        self.log.write(json.dumps(dict(time=time_s, sensor_ns=stamps, pose_attempts=attempt+1, position_error_m=errors,
                                       attitude_error_deg=angles)) + "\n")
        self.log.flush()
        return dict(time=time_s, rgb=np.stack([v[0] for v in views]),
                    depth=np.stack([v[1] for v in views]), valid=np.ones(2, bool))


@contextmanager
def create_renderer(single=False, scale=(1., 1., 1.), resolution=(WIDTH, HEIGHT), refresh_on_switch=True):
    """Use a disposable simulator instance; caller owns its shutdown."""
    with ExitStack() as stack:
        log = stack.enter_context(Path(os.environ["RISK_REPLAY_LOG"]).open("x", encoding="utf-8"))
        client = airsim.MultirotorClient(timeout_value=20)
        stack.callback(client.client.close)
        vehicles = client.listVehicles()
        if not vehicles or any(n != "ReplayB" for n in vehicles[1:]):
            raise RuntimeError(f"Expected dedicated replay server, got {vehicles}")
        paused = client.simIsPause()
        stack.callback(client.simPause, paused)
        client.simPause(True)
        if not single and "ReplayB" not in vehicles:
            if not client.simAddVehicle("ReplayB", "SimpleFlight", pose([2, 0, -4]), "DefaultQuadrotor"):
                raise RuntimeError("Could not add second vehicle")
        names = [vehicles[0]] if single else [vehicles[0], "ReplayB"]
        meshes = []
        for name in names:
            for camera in CAMERAS:
                client.client.call("simSetPanoramaResolution", camera, *resolution, name)
        for i in (0, 1):
            mesh = client.simSpawnObject(f"RiskReplayAppearance{i}", "Quadrotor1", pose([900, 900, -900]),
                                         airsim.Vector3r(*map(float, scale)), False, False)
            if not mesh:
                raise RuntimeError("Missing target appearance")
            stack.callback(client.simDestroyObject, mesh)
            if not np.allclose(xyz(client.simGetObjectScale(mesh)), scale, atol=.0001):
                raise RuntimeError("Visual actor scale differs from requested geometry")
            meshes.append(mesh)
        yield Renderer(client, names, meshes, log, resolution, refresh_on_switch)
