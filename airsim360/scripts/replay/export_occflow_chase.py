"""Third-person projection of the two separate replayed OccupancyFlow risk fields."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.simulation.airsim360_replay import create_renderer, AXES
from scripts.replay.export_cbf_replays import CHASE_H, CHASE_W, FONT, chase_image, projection
from scripts.simulation.preview_split_s import place_course


def read_json(path):
    return json.loads(path.read_text())


def risk_overlay(image, centers, state, renderer, geom):
    # Maximum of the unweighted 3-D field along each camera ray (visible scores >= .5).
    img = np.asarray(image)
    risk = np.zeros((CHASE_H, CHASE_W), np.float32)
    direction, fwd, right, down, focal = geom[:5]
    eye = state[:,:3].mean(0)-direction
    c, s = np.cos(renderer.heading), np.sin(renderer.heading)
    rot = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    basis = np.stack([right,down,fwd],axis=1)
    camera = ((np.asarray(centers).reshape(-1,3)-eye)*AXES) @ rot.T @ basis
    cx, cy = (CHASE_W-1)/2, (CHASE_H-1)/2
    r = .70
    for x,y,z in camera:
        if z <= -r:
            continue
        if z>r:
            px, py = focal*x/z+cx, focal*y/z+cy
            dx = focal*r*(1+abs(x/z))/(z-r)
            dy = focal*r*(1+abs(y/z))/(z-r)
            left,right_edge = max(0,int(np.floor(px-dx))), min(CHASE_W,int(np.ceil(px+dx))+1)
            top,bottom = max(0,int(np.floor(py-dy))), min(CHASE_H,int(np.ceil(py+dy))+1)
        else:
            left,right_edge,top,bottom=0,CHASE_W,0,CHASE_H
        if left>=right_edge or top>=bottom:
            continue
        u = (np.arange(left,right_edge)-cx)/focal
        v = ((np.arange(top,bottom)-cy)/focal)[:,None]
        dot = np.maximum(0.,u*x+v*y+z)
        dist = np.sqrt(np.maximum(0.,x*x+y*y+z*z-dot**2/(1+u*u+v*v)))
        score = 1/(1+np.exp(np.clip((dist-r)/.13,-50,50)))
        patch = risk[top:bottom,left:right_edge]
        np.maximum(patch,np.where(dist<=r,score,0.),out=patch)
    color = np.stack([np.full_like(risk,255), 204-170*np.clip(2*risk-1,0,1),
                      np.full_like(risk,26)],axis=-1)
    alpha = np.where(risk>=.5,.30+.42*(2*risk-1),0.)[...,None]
    return Image.fromarray(np.uint8(img*(1-alpha)+color*alpha))


def export(folder,renderer,geom,ffmpeg):
    cfg = read_json(folder/"experiment.json")
    audit = read_json(folder/"visual_comparison.json")
    if hashlib.sha256(Path(cfg["flow_prior"]).read_bytes()).hexdigest()!=cfg["flow_sha256"]:
        raise ValueError("OccFlow checkpoint mismatch")
    with np.load(folder/"flight/execution.npz") as f:
        times, states = f["time"],f["state"]
    ids = np.unique(np.r_[np.arange(0,len(times),4),len(times)-1])
    closest_frame = int(ids[np.argmin(abs(times[ids]-audit["closest_time_s"]))])
    final = folder/"occflow_chase_040.mp4"
    partial = final.with_suffix(".partial.mp4")
    fonts = [ImageFont.truetype(FONT,size) for size in (27,21,18)]
    header = Image.new("RGB",(2*CHASE_W,672),"#14202a")
    draw = ImageDraw.Draw(header)
    draw.text((16,5),f"OccFlow  ·  提前 0.40 s 的动态风险场     MPCC 权重 {cfg['weight']:.0f}",font=fonts[0],fill="white")
    for i in (0,1):
        draw.text((16+CHASE_W*i,46),f"{i} 号机观测   →   预测 {1-i} 号机未来占位",font=fonts[1],fill="#cce7fc")
    draw.text((16,635),"RGB-D 轨迹回放复算   |   视线最大风险：黄 0.5 → 红 1.0   |   低于 0.5 透明",font=fonts[2],fill="#cce7fc")
    cmd=[ffmpeg,"-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{2*CHASE_W}x672",
         "-r","25","-i","pipe:0","-an","-c:v","libx264","-threads","1","-preset","veryfast",
         "-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(partial)]
    # All sensor ticks are recomputed to preserve scene-flow history; movie is sampled every 80 ms.
    worker_cmd=[".conda-envs/risk-training/python.exe","-u","-m","scripts.replay.occflow_replay_worker",str(folder)]
    rendered=0
    with subprocess.Popen(worker_cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as worker, subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder:
        try:
            for k,(stamp,snapshot) in enumerate(zip(times,states)):
                obs=renderer.capture(float(stamp),snapshot)
                worker.stdin.write(f"{k}\n".encode())
                worker.stdin.write(obs["rgb"].astype(np.uint8,copy=False).tobytes())
                worker.stdin.write(obs["depth"].astype("<f4",copy=False).tobytes())
                worker.stdin.flush()
                points=json.loads(worker.stdout.readline())
                if k%4 and k!=len(times)-1:
                    continue
                bg=chase_image(renderer,snapshot,geom)
                canvas=header.copy()
                for i in (0,1):
                    pane=risk_overlay(bg,points[i],snapshot,renderer,geom)
                    canvas.paste(pane,(i*CHASE_W,80))
                    if not points[i]:
                        ImageDraw.Draw(canvas).text((16+i*CHASE_W,89),"本帧无动态点",font=fonts[2],fill="white",stroke_width=2,stroke_fill="#25374a")
                d=ImageDraw.Draw(canvas)
                distance=np.linalg.norm(snapshot[0,:3]-snapshot[1,:3])
                d.text((1000,12),f"t = {stamp:.2f} s      实际间距 {distance:.3f} m",font=fonts[1],fill="white")
                if k==closest_frame:
                    d.text((865,635),f"最近接 {audit['closest_time_s']:.3f} s   ·   {audit['min_distance_m']:.3f} m",font=fonts[2],fill="#ffd063")
                repeats=round(times[min(k+4,len(times)-1)]*25)-round(stamp*25) if k<len(times)-1 else 25
                if k==closest_frame:
                    repeats+=20
                encoder.stdin.write(canvas.tobytes()*repeats)
                rendered+=repeats
                if k%100==0: print(folder.name,k,len(times),flush=True)
        finally:
            worker.stdin.close()
            encoder.stdin.close()
        if worker.wait() or encoder.wait():
            raise RuntimeError("Risk computation or encoding failed")
    cap=cv2.VideoCapture(str(partial))
    count=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if frame.shape[:2]!=(672,2*CHASE_W):raise ValueError("Unexpected decoded size")
            count+=1
    finally:cap.release()
    if count!=rendered: raise ValueError(f"Only decoded {count}/{rendered} frames")
    partial.replace(final)
    print(final,count,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("ffmpeg")
    p.add_argument("folders",nargs="+",type=Path)
    args=p.parse_args()
    cv2.setNumThreads(1)
    placement=Path("reports/split_s_open_yard/placement.json")
    layout=read_json(placement)
    first=args.folders[0]
    course=read_json(first/"rendered_course.json")
    scale=read_json(first/"vehicle_geometry.json")["scale_xyz"]
    with TemporaryDirectory() as temp, ExitStack() as stack:
        os.environ["RISK_REPLAY_LOG"]=str(Path(temp)/"sensor.jsonl")
        r=stack.enter_context(create_renderer(single=True,scale=scale,resolution=(512,256),refresh_on_switch=True))
        r.heading=np.deg2rad(layout["heading_ned_deg"])
        r.translation=np.asarray(layout["translation_ned_m"])
        actors=set(r.client.simListSceneObjects())
        for obj in layout["removed_yard_fixtures"]:
            if obj in actors and not r.client.simDestroyObject(obj):raise RuntimeError(obj)
        parts=place_course(r.client,course,stack,placement.with_name("gate_texture.png"),r.translation,r.heading)
        if len(parts)!=28 or not all(item["textured"] for item in parts):raise RuntimeError("Gate placement failed")
        for folder in args.folders:
            if course!=read_json(folder/"rendered_course.json") or scale!=read_json(folder/"vehicle_geometry.json")["scale_xyz"]:
                raise ValueError("Mismatched replay scene")
            export(folder,r,projection(r.heading),args.ffmpeg)


if __name__=="__main__":
    main()
