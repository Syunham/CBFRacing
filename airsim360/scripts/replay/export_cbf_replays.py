"""Render four labelled three-view AirSim replays from the executed flight states."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import airsim
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.simulation.airsim360_replay import create_renderer, pose, xyz
from scripts.simulation.preview_split_s import place_course, set_camera, world_to_ned


ROOT = Path("reports/spatial_potential")
RUNS = {
    "cf21_origalloc_flow01": "原版航点    MPCC + OccupancyFlow",
    "cf21_origalloc_cbf01": "原版航点    MPCC + CBF-QP",
    "cf21_cbf055_nominal02": "视觉航点    MPCC + OccupancyFlow",
    "cf21_cbf055_filtered01": "视觉航点    MPCC + OccupancyFlow + CBF-QP",
}
WIDTH, HEIGHT = 1280, 704
CHASE_W, CHASE_H = 752, 548
FONT = "C:/Windows/Fonts/msyh.ttc"


def projection(heading):
    direction = np.array([5., 8., -4.5])
    yaw = heading+np.arctan2(-direction[1], direction[0])
    pitch = np.arctan2(-direction[2], np.linalg.norm(direction[:2]))
    fwd = np.array([np.cos(yaw)*np.cos(pitch), np.sin(yaw)*np.cos(pitch), np.sin(pitch)])
    right = np.array([-np.sin(yaw), np.cos(yaw), 0.])
    down = np.cross(fwd, right)
    y, x = np.indices((CHASE_H, CHASE_W), dtype=np.float32)
    focal = 420.
    rays = fwd+(x[...,None]-(CHASE_W-1)/2)*right/focal+(y[...,None]-(CHASE_H-1)/2)*down/focal
    u = ((np.arctan2(rays[:,:,1],rays[:,:,0])/(2*np.pi)+.5)*512-.5).astype("f")
    v = ((np.arctan2(rays[:,:,2],np.hypot(rays[:,:,0],rays[:,:,1]))/np.pi+.5)*256-.5).astype("f")
    return direction, fwd, right, down, focal, u, v


def chase_image(r, state, geom):
    direction, fwd, right, down, focal, u, v = geom
    positions = [world_to_ned(s[:3], r.translation, r.heading) for s in state]
    eye = world_to_ned(state[:,:3].mean(0)-direction, r.translation, r.heading)
    name, = r.vehicles
    set_camera(r.client, name, eye)
    c, h = np.cos(r.heading/2), np.sin(r.heading/2)
    for i, s in enumerate(state):
        w,x,y,z = s[6:10]*[1,1,-1,-1]
        q = [c*w-h*z,c*x-h*y,c*y+h*x,c*z+h*w]
        if not r.client.simSetObjectPose(r.meshes[i], pose(positions[i],q), True):
            actual=r.client.simGetObjectPose(r.meshes[i])
            a=actual.orientation
            if not (np.linalg.norm(xyz(actual.position)-positions[i])<.005 and
                    abs(np.dot(q,[a.w_val,a.x_val,a.y_val,a.z_val]))>.99995):
                raise RuntimeError(f"Cannot position D{i}: {xyz(actual.position)} vs {positions[i]}")
    for _ in range(2):
        r.client.client.call("simTriggerPanoramaCapture","panorama_original",name)
    result, = r.client.simGetImages([airsim.ImageRequest("panorama_original",0,False,False)],vehicle_name=name)
    pano = np.frombuffer(result.image_data_uint8,np.uint8).reshape(256,512,3)
    view = cv2.remap(pano, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    for i,pos in enumerate(positions):
        a = pos-eye
        px = int((CHASE_W-1)/2+focal*(a@right)/(a@fwd))
        py = int((CHASE_H-1)/2+focal*(a@down)/(a@fwd))
        color = [(75,225,255),(255,215,70)][i]
        shift = -35 if i==0 else 12
        cv2.line(view,(px+int(shift*.3),py-3),(px+shift,py-16),color,1,cv2.LINE_AA)
        cv2.putText(view,f"D{i}",(px+shift-7,py-18),0,.55,(20,20,20),3,cv2.LINE_AA)
        cv2.putText(view,f"D{i}",(px+shift-7,py-18),0,.55,color,1,cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(view,cv2.COLOR_BGR2RGB))


def depth_view(ranges):
    valid = np.isfinite(ranges) & (ranges > 0)
    # A fixed logarithmic scale preserves both the nearby airframe and the gates.
    intensity = np.zeros(ranges.shape, np.uint8)
    intensity[valid] = np.rint(255 * (1 - np.log1p(np.clip(ranges[valid], 0, 20)) / np.log1p(20)))
    return Image.fromarray(intensity)


def export(folder, renderer, geom, ffmpeg, title_text=None, include_depth=False):
    with np.load(folder/"flight/execution.npz") as f:
        times, states = f["time"], f["state"]
    report = json.loads((folder/"flight/summary.json").read_text())
    events = json.loads((folder/"flight/events.json").read_text())
    stamps = [np.asarray([e["time"] for e in es]) for es in events]
    width = WIDTH + 520 if include_depth else WIDTH
    output = folder/("rgb_depth_replay.mp4" if include_depth else "three_views_replay.mp4")
    # Preserve the last applied state even when it falls between regular video samples.
    indices = np.unique(np.r_[np.arange(0,len(times),4),len(times)-1])
    nearest = np.linalg.norm(states[:,0,:3]-states[:,1,:3],axis=1).argmin()
    if title_text is not None and not include_depth:
        indices = np.unique(np.r_[np.arange(0,len(times),2),len(times)-1])
    display_indices = indices.copy()
    closest_slot = np.abs(indices-nearest).argmin()
    if 0 < closest_slot < len(indices)-1:
        display_indices[closest_slot] = nearest
    base = Image.new("RGB",(width,HEIGHT),(18,25,33))
    title, label = (ImageFont.truetype(FONT,n) for n in (28,19))
    draw = ImageDraw.Draw(base)
    draw.text((12,7),title_text or RUNS[folder.name],font=title,fill="white")
    draw.text((1000,15),"AirSim  ·  轨迹回放",font=label,fill="#b7cedb")
    draw.text((8,66),"第三人称",font=label,fill="#e4edf1")
    draw.text((764,66),"0 号机  ·  第一人称全景",font=label,fill="#ffd951")
    draw.text((764,354),"1 号机  ·  第一人称全景",font=label,fill="#70dbff")
    if include_depth:
        for i,top in enumerate((66,354)):
            draw.text((1284,top),f"{i} 号机  ·  全景深度",font=label,fill="white")
        draw.text((1284,652),"深度越近越亮，20 m 以外及无效值为黑",font=label,fill="#b7cedb")
    with TemporaryDirectory(dir=folder) as temp:
        encoded = Path(temp)/"render.mp4"
        cmd=[ffmpeg,"-hide_banner","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
             "-s",f"{width}x{HEIGHT}","-r","25","-i","pipe:0","-an","-c:v","libx264",
             "-preset","veryfast","-crf","19","-pix_fmt","yuv420p","-threads","1","-movflags","+faststart",str(encoded)]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE) as process:
            for j,k in enumerate(display_indices):
                stamp = times[k]
                capture = renderer.capture(float(stamp),states[k])
                canvas = base.copy()
                canvas.paste(chase_image(renderer,states[k],geom),(6,96))
                for i,top in enumerate((96,384)):
                    canvas.paste(Image.fromarray(capture["rgb"][i]),(764,top))
                    if include_depth:
                        canvas.paste(depth_view(capture["depth"][i]),(1284,top))
                gates = [int(np.searchsorted(ts,stamp+1e-7,side="right")) for ts in stamps]
                distance = np.linalg.norm(states[k,0,:3]-states[k,1,:3])
                d = ImageDraw.Draw(canvas)
                d.text((12,652),f"{stamp:5.2f} s     机间距 {distance:.3f} m     过门   D0 {gates[0]}/14    D1 {gates[1]}/14",
                       font=label,fill="white")
                final = j==len(indices)-1
                if k==nearest and not final:
                    d.text((765,652),"最近交会",font=label,fill="#ffd951")
                if final:
                    if gates!=report["gates_passed"]:
                        raise ValueError("Replay end gate counts differ from execution")
                    status = "两圈完成" if report["success"] else "航次结束"
                    d.text((765,652),f"{status}    最小间距 {report['min_distance']:.4f} m",font=label,fill="#ffd951")
                repeat = 38 if final else round(times[indices[j+1]]*25)-round(times[indices[j]]*25)
                frame = canvas.tobytes()
                for _ in range(repeat):
                    process.stdin.write(frame)
                if include_depth and j % 100 == 0:
                    print(f"Rendered {j+1}/{len(display_indices)} poses",flush=True)
            process.stdin.close()
            if process.wait():
                raise RuntimeError("Video encoding failed")
        # Decode every frame before publishing the MP4.
        cap = cv2.VideoCapture(str(encoded))
        count = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok: break
                if image.shape[:2]!=(HEIGHT,width): raise ValueError("Unexpected output dimensions")
                count += 1
        finally:
            cap.release()
        if count!=round(times[-1]*25)+38: raise ValueError(f"Incomplete video: {count} frames")
        # A Windows tempfile directory has a private ACL which follows moved files.
        # Copy into the run folder first so the video inherits its normal access.
        staged=output.with_suffix(".partial.mp4")
        shutil.copyfile(encoded,staged)
        staged.replace(output)
        print(f"{output} {count} decoded frames",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("ffmpeg")
    p.add_argument("--run", help="Render one recorded flight instead of the four comparison runs")
    args=p.parse_args()
    cv2.setNumThreads(1)
    placement_path=Path("reports/split_s_open_yard/placement.json")
    layout=json.loads(placement_path.read_text())
    first=ROOT/(args.run or next(iter(RUNS)))
    course=json.loads((first/"rendered_course.json").read_text())
    scale=json.loads((first/"vehicle_geometry.json").read_text())["scale_xyz"]
    with TemporaryDirectory() as temp, ExitStack() as stack:
        os.environ["RISK_REPLAY_LOG"]=str(Path(temp)/"sensor.jsonl")
        r=stack.enter_context(create_renderer(single=True,scale=scale,resolution=(512,256)))
        r.heading=np.deg2rad(layout["heading_ned_deg"])
        r.translation=np.array(layout["translation_ned_m"])
        actors=set(r.client.simListSceneObjects())
        for obj in layout["removed_yard_fixtures"]:
            if obj in actors and not r.client.simDestroyObject(obj): raise RuntimeError(obj)
        parts=place_course(r.client,course,stack,placement_path.with_name("gate_texture.png"),r.translation,r.heading)
        if len(parts)!=28 or not all(part["textured"] for part in parts): raise ValueError("Gate placement failed")
        for name in ([args.run] if args.run else RUNS):
            folder=ROOT/name
            if course!=json.loads((folder/"rendered_course.json").read_text()) or scale!=json.loads((folder/"vehicle_geometry.json").read_text())["scale_xyz"]:
                raise ValueError("Replay geometry differs between runs")
            print("Rendering",name,flush=True)
            caption=None
            include_depth=False
            if args.run:
                cfg=json.loads((folder/"experiment.json").read_text())
                caption=("Minimum Snap 联合优化过门时间   ·   深度 CBF-QP"
                         if cfg.get("optimize_gate_times") and cfg.get("depth_pair") else RUNS.get(name,name))
                if cfg.get("snap_time_network") and cfg.get("depth_pair"):
                    caption="时间网络 + Minimum Snap   ·   深度 CBF-QP"
                if cfg.get("guan_time_network") and cfg.get("depth_pair"):
                    caption="Guan 时长网络与一步修正   ·   MPCC + 深度 CBF-QP"
                    include_depth=True
                if cfg.get("guarded_fast_fly") and cfg.get("depth_pair"):
                    caption="Fast-fly   ·   MPCC + 深度 CBF-QP"
                    include_depth=True
            export(folder,r,projection(r.heading),args.ffmpeg,caption,include_depth)


if __name__=="__main__":
    main()
