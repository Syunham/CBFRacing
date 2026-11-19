"""Export synchronized AirSim chase and onboard views from a recorded flight."""

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

import airsim
import cv2
import numpy as np
from scripts.simulation.airsim360_replay import create_renderer, pose
from scripts.simulation.preview_split_s import place_course, set_camera, world_to_ned


def writer(stack, path, fps, size):
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not video.isOpened():
        raise RuntimeError(f"Cannot write {path}")
    stack.callback(video.release)
    return video


def export(folder):
    cv2.setNumThreads(1)
    data = np.load(folder/"flight/execution.npz")
    t, state = data["time"], data["state"]
    files = sorted((folder/"rgb").glob("*.npy"))
    stamps = np.array([int(f.stem)/1000 for f in files])
    if not len(files) or np.any(np.diff(stamps) > .081) or t[-1]-stamps[-1] > .081:
        raise ValueError("Missing onboard frames")
    placement_path = Path("reports/split_s_open_yard/placement.json")
    layout = json.loads(placement_path.read_text())
    course = json.loads((folder/"rendered_course.json").read_text())
    translation, heading = layout["translation_ned_m"], np.deg2rad(layout["heading_ned_deg"])
    direction = np.array([5., 8., -4.5])
    yaw = heading + np.arctan2(-direction[1],direction[0])
    pitch = np.arctan2(-direction[2], np.linalg.norm(direction[:2]))
    y,x = np.indices((360,640),dtype=np.float32)
    fwd = np.array([np.cos(yaw)*np.cos(pitch),np.sin(yaw)*np.cos(pitch),np.sin(pitch)])
    right = np.array([-np.sin(yaw),np.cos(yaw),0.])
    rays = fwd + (x[...,None]-319.5)*right/360 + (y[...,None]-179.5)*np.cross(fwd,right)/360
    u = ((np.arctan2(rays[:,:,1],rays[:,:,0])/(2*np.pi)+.5)*1024-.5).astype('f')
    v = ((np.arctan2(rays[:,:,2],np.hypot(rays[:,:,0],rays[:,:,1]))/np.pi+.5)*512-.5).astype('f')
    os.environ["RISK_REPLAY_LOG"] = str(folder/"movie_replay.jsonl")
    with ExitStack() as stack:
        geometry = folder/"vehicle_geometry.json"
        scale = json.loads(geometry.read_text())["scale_xyz"] if geometry.exists() else (1.,1.,1.)
        r = stack.enter_context(create_renderer(single=True,scale=scale))
        place_course(r.client, course, stack, placement_path.with_name("gate_texture.png"), translation, heading)
        primary, = r.vehicles
        if "ReplayB" in r.client.listVehicles():
            set_camera(r.client,"ReplayB",[50.,50.,-30.])
        camera = "panorama_original"
        r.client.client.call("simSetPanoramaResolution",camera,1024,512,primary)
        tracks = [writer(stack,folder/f"drone{i}_fpv.mp4",12.5,(768,384)) for i in (0,1)]
        chase = writer(stack,folder/"chase.mp4",12.5,(640,360))
        for frame, stamp in enumerate(np.arange(0,t[-1]+1e-8,.08)):
            idx = np.searchsorted(stamps,stamp+1e-6,side="right")-1
            cameras = np.load(files[idx])
            pair = state[np.abs(t-stamp).argmin()]
            targets = [world_to_ned(s[:3],translation,heading) for s in pair]
            eye = world_to_ned(pair[:,:3].mean(axis=0)-direction,translation,heading)
            for i,s in enumerate(pair):
                w,xq,yq,zq = s[6:10]*[1,1,-1,-1]
                c,h = np.cos(heading/2),np.sin(heading/2)
                q = [c*w-h*zq,c*xq-h*yq,c*yq+h*xq,c*zq+h*w]
                if not r.client.simSetObjectPose(r.meshes[i],pose(targets[i],q),True):
                    raise RuntimeError("Unable to place vehicle mesh")
                image = cv2.resize(cameras[i][:,:,::-1],(768,384))
                cv2.putText(image,f"Drone {i}   360 RGB   {stamp:.2f} s",(12,25),0,.65,(255,255,255),1,cv2.LINE_AA)
                tracks[i].write(image)
            set_camera(r.client,primary,eye)
            r.client.client.call("simTriggerPanoramaCapture",camera,primary)
            result, = r.client.simGetImages([airsim.ImageRequest(camera,0,False,False)],vehicle_name=primary)
            pano = np.frombuffer(result.image_data_uint8,np.uint8).reshape(512,1024,3)
            image = cv2.remap(pano,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
            for i, target in enumerate(targets):
                ray = target-eye
                scale = 360 / (ray@fwd)
                px = int(319.5+scale*(ray@right))
                py = int(179.5+scale*(ray@np.cross(fwd,right)))
                color = (80,215,255) if i == 0 else (255,215,90)
                offset = -24 if i == 0 else 12
                cv2.putText(image,f"D{i}",(px+offset,py-9),0,.45,color,1,cv2.LINE_AA)
            cv2.putText(image,f"Chase   {stamp:.2f} s",(12,25),0,.65,(255,255,255),1,cv2.LINE_AA)
            chase.write(image)
    print(f"Saved {frame+1} frames per view")


if __name__ == "__main__":
    export(Path("reports/spatial_potential")/sys.argv[1])
