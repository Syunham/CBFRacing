"""Calibrate the appearance in two orthogonal views, then replay the frozen splits."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import airsim
import numpy as np
from scripts.simulation.airsim360_replay import create_renderer, pose
from scripts.simulation.collect_risk_flights import collect
from scripts.simulation.preview_split_s import set_camera


def measure(r, distance):
    name, = r.vehicles
    mesh = r.meshes[0]
    eye = np.array([8., 1., -25.])
    set_camera(r.client, name, eye)
    camera = "panorama_depth"
    w, h = 2048, 1024
    r.client.client.call("simSetPanoramaResolution", camera, w, h, name)
    dimensions = []
    for yaw in (0., np.pi/2):
        quat = (np.cos(yaw/2), 0., 0., np.sin(yaw/2))
        center = eye+[distance,0.,0.]
        if not r.client.simSetObjectPose(mesh, pose(center,quat), True):
            raise RuntimeError("Calibration target did not move")
        set_camera(r.client, name, eye)
        for _ in range(2):
            r.client.client.call("simTriggerPanoramaCapture", camera, name)
            resp, = r.client.simGetImages([airsim.ImageRequest(camera,0,True,False)],vehicle_name=name)
        d = (np.asarray(resp.image_data_float) if resp.image_data_float else np.frombuffer(resp.image_data_uint8,"<f4")).reshape(h,w)
        mask = (d > distance*.65) & (d < distance*1.35)
        y,x = np.nonzero(mask)
        a = ((x+.5)/w-.5)*(2*np.pi)
        b = ((y+.5)/h-.5)*np.pi
        ok = (np.abs(a)<.5)&(np.abs(b)<.4)
        if ok.sum()<100:
            raise ValueError(f"Insufficient calibration surface: {ok.sum()} pixels")
        lateral = (d[y,x]*np.cos(b)*np.sin(a))[ok]
        height = (d[y,x]*np.sin(b))[ok]
        dimensions.append([float(np.diff(np.percentile(v,[.2,99.8]))[0]) for v in (lateral,height)])
    return np.array([dimensions[1][0],dimensions[0][0],np.mean(np.array(dimensions)[:,1])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--width",type=int,choices=(256,512),default=256)
    parser.add_argument("--geometry",type=Path)
    parser.add_argument("--trajectory-stride",type=int,choices=(1,4),default=4)
    args=parser.parse_args()
    output=args.output
    output.mkdir(exist_ok=True,parents=True)
    geom = output/"geometry.json"
    if args.geometry:
        if geom.exists() and geom.read_bytes() != args.geometry.read_bytes():
            raise ValueError("Geometry differs from reference")
        if not geom.exists():
            shutil.copyfile(args.geometry,geom)
    if not geom.exists():
        os.environ["RISK_REPLAY_LOG"] = str(output/f"calibration-{time.time_ns()}.jsonl")
        with create_renderer(single=True) as renderer:
            target = np.array([.107,.107,.029])
            original = measure(renderer, 2.5)
            scale = target/original
            for _ in range(3):
                renderer.client.simSetObjectScale(renderer.meshes[0],airsim.Vector3r(*map(float,scale)))
                measured = measure(renderer,.6)
                if np.all(np.abs(measured-target) <= [.007,.007,.003]):
                    break
                scale *= target/measured
            else:
                raise RuntimeError(f"Geometry calibration did not converge: {measured}")
            profile = dict(model="Crazyflie 2.1 brushed", mesh="Quadrotor1", scale_xyz=scale.tolist(),
                           target_xyz_m=target.tolist(), measured_xyz_m=measured.tolist(), original_xyz_m=original.tolist(),
                           body_radius_m=.08, label_buffer_m=.05)
            geom.write_text(json.dumps(profile,indent=2))
    profile=json.loads(geom.read_text())
    source=Path("reports/flight_count_comparison/manifest_8.json")
    episodes=json.loads(source.read_text(encoding="utf-8-sig"))["episodes"]
    flights=[(e["episode_id"],e["split"],(source.parent/e["trajectory"]).resolve()) for e in episodes]
    collect(None,output,Path("reports/split_s_open_yard/placement.json"),flights,profile,args.trajectory_stride,(args.width,args.width//2))


if __name__=="__main__":
    main()
