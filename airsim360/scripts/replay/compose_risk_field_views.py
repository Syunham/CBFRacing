"""Append two frozen-network risk slices to each recorded, synchronized race video."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import deque

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
from scripts.perception.spatial_potential import camera_points, rotation
from scripts.perception.time_query_field import encode_views, load_rgbd

ROOT=Path(__file__).resolve().parents[2]/"reports/spatial_potential"
OUT=ROOT/"cf21_bendhead03_comparison"
FF=Path(sys.argv[1])
H,W=352,1536
PW,PH=420,280
LUT=(colormaps["inferno"](np.linspace(0,1,256))[:,:3]*255).round().astype("uint8")
SIDE,FWD=np.meshgrid(np.linspace(-6,6,121),np.linspace(6,-2,81))
F16,F18,F23=[ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",k) for k in (16,18,23)]
FG="#eff5fa"
MUTED="#becbd7"
CYAN="#4ce2ed"
GREEN="#96f6ae"
BG="#131e2b"

def serial(args):
    return [str(FF),"-hide_banner","-loglevel","error","-y",*map(str,args)]

def panel(values,ego,pose,forward,right,peer,future,course,stamp,lead=.4,cutoff=None):
    im=Image.new("RGB",(768,H),BG)
    heat=Image.fromarray(LUT[np.uint8(np.clip(values,0,1)*255)]).resize((PW,PH),Image.Resampling.BILINEAR)
    legend=LUT.copy()
    if cutoff is not None:
        display_values=np.asarray(Image.fromarray(values.astype("f")).resize((PW,PH),Image.Resampling.BILINEAR))
        mask=Image.fromarray(np.uint8(display_values<cutoff)*255)
        heat.paste((0,0,0),(0,0,PW,PH),mask)
        legend[np.linspace(0,1,len(legend))<cutoff]=0
    d=ImageDraw.Draw(heat,"RGBA")
    def xy(point):
        rel=point[:2]-pose[:2]
        return (float((rel@right+6)*(PW-1)/12),float((6-rel@forward)*(PH-1)/8))
    for dx in range(-6,7,2):
        x=(dx+6)*(PW-1)/12;d.line((x,0,x,PH),fill=(255,255,255,34))
    for dy in range(-2,7,2):
        y=(6-dy)*(PH-1)/8;d.line((0,y,PW,y),fill=(255,255,255,34))
    for j,(g,axis,half) in enumerate(zip(course["centers"],course["coordinate_axes"],course["half_extents"])):
        g=np.array(g);axis=np.array(axis)
        if abs(g[2]-pose[2])>half[1]:continue
        a,b=xy(g-axis*half[0]),xy(g+axis*half[0])
        if not (-38<a[0]<PW+38 and -38<a[1]<PH+38):continue
        d.line([a,b],fill=GREEN,width=2)
        for x,y in (a,b):d.rectangle((x-2,y-2,x+2,y+2),fill=GREEN)
        d.text((a[0]+3,a[1]-17),f"G{j+1}",font=F16,fill=GREEN,stroke_width=1,stroke_fill="#17222d")
    ox,oy=xy(pose);d.polygon([(ox,oy-6),(ox-5,oy+5),(ox+5,oy+5)],fill="white",outline="black")
    x,y=xy(peer);d.ellipse((x-5,y-5,x+5,y+5),outline=CYAN,width=2)
    if future is not None:
        x,y=xy(future);d.line((x-5,y-5,x+5,y+5),fill=CYAN,width=2);d.line((x-5,y+5,x+5,y-5),fill=CYAN,width=2)
    im.paste(heat,(14,48))
    d=ImageDraw.Draw(im)
    d.text((14,9),f"{ego} 号机   |   预测风险场（提前 {lead:.2f} s）",font=F23,fill=FG)
    x=454
    d.text((x,48),"低风险                            高风险",font=F16,fill=MUTED)
    im.paste(Image.fromarray(legend[None]).resize((240,12)),(x,78))
    for dx,s in [(0,"0"),(110,"0.5"),(228,"1")]:d.text((x+dx,94),s,font=F16,fill=MUTED)
    d.text((x,129),"▲  本机 · 机头朝上",font=F18,fill=FG)
    d.text((x,163),"○  邻机当前位置",font=F18,fill=CYAN)
    d.text((x,194),f"×  邻机 {lead:g} s 后实飞位置" if future is not None else "未来实飞位置超出记录",font=F18,fill=CYAN)
    d.text((x,226),"━  与切片相交的门洞",font=F18,fill=GREEN)
    d.text((x,267),f"水平切片  z = {pose[2]:.2f} m",font=F16,fill=MUTED)
    d.text((x,295),f"网格 2 m   ·   t = {stamp:.2f} s",font=F16,fill=MUTED)
    return im

@torch.inference_mode()
def render(idx):
    src=ROOT/f"cf21_bendhead03_run{idx}"
    config=json.loads((src/"experiment.json").read_text())
    path=Path(config["time_checkpoint"])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==config["time_checkpoint_sha256"]
    model=load_rgbd(path)
    course=json.loads((src/"rendered_course.json").read_text())
    with np.load(src/"flight/execution.npz") as f:times,states=f["time"],f["state"]
    logs={}
    for line in (src/"vision.jsonl").open():
        rec=json.loads(line)
        logs.setdefault((round(rec["decision_s"]/.02),rec["ego"]),rec)
    n=(len(times)-1)//4+1
    output=OUT/f"run{idx}_five_views.mp4"
    cmd=serial(["-threads","1","-i",OUT/f"run{idx}_three_views.mp4","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r","12.5","-i","-",
                "-filter_complex_threads","1","-filter_complex","[0:v][1:v]vstack=inputs=2:shortest=1[v]","-map","[v]","-c:v","libx264","-threads","1","-preset","veryfast","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",output])
    errors=[]; history=[deque(maxlen=4),deque(maxlen=4)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for k in range((n-1)*4+1):
            stamp=k*.02
            for h,ch in zip(history,("rgb","depth")):
                h.append(np.load(src/ch/f"{20*k:06d}.npy"))
                while len(h)<4:h.appendleft(h[0])
            if k%4:continue
            ctx=encode_views(model,*map(np.stack,history))
            canv=Image.new("RGB",(W,H),BG)
            for ego in (0,1):
                pose=states[k,ego]
                forward=rotation(pose).apply([1,0,0])[:2]
                forward/=np.linalg.norm(forward);right=np.array([forward[1],-forward[0]])
                world=np.broadcast_to(pose[:3],(*SIDE.shape,3)).copy()
                world[:,:,:2]+=SIDE[:,:,None]*right+FWD[:,:,None]*forward
                q=torch.from_numpy(camera_points(world.reshape(-1,3),pose).astype("float32"))[None]
                tau=torch.full((1,q.shape[1],1),.4)
                field=model.query_logits(ctx[ego:ego+1],q,tau).sigmoid().numpy().reshape(SIDE.shape)
                assert np.isfinite(field).all()
                # The recorded first node at each full refresh is at the camera origin or its warm-start estimate.
                origin=model.query_logits(ctx[ego:ego+1],torch.zeros(1,1,3),torch.zeros(1,1,1)).sigmoid().item()
                log=logs[k,ego];assert abs(log["time"]-stamp)<1e-6
                errors.append(abs(origin-log["node_risk"][0]))
                future=states[k+20,1-ego] if k+20<len(times) else None
                canv.paste(panel(field,ego,pose,forward,right,states[k,1-ego],future,course,stamp),(768*ego,0))
            data=canv.tobytes();proc.stdin.write(data);proc.stdin.write(data)
        for _ in range(25):proc.stdin.write(data)
        proc.stdin.close()
        if proc.wait():raise RuntimeError("Video encoder failed")
    print(idx,"frames",2*n+25,"origin check",np.percentile(errors,[50,95,100]),flush=True)
    assert max(errors)<.001
    return n*2+25

@torch.inference_mode()
def append_legacy_planes(race=None,cutoff=None,input_folder=None,lead=.1):
    import csv
    from scripts.replay.compose_fpv_motion_field import RACE, LegacyProjection, recorded_future
    race=RACE if race is None else race
    ms=round(lead*1000)
    original=(input_folder or race)/f"two_laps_fpv_cnn_gru_{ms}ms.mp4"
    suffix="_seven_views" if cutoff is None else "_seven_views_black_below_0p5"
    output=race/(original.stem+suffix+".mp4")
    config=json.loads((race/"experiment.json").read_text())
    model=LegacyProjection(race,lead) if config["time_checkpoint"] else LegacyProjection(lead=lead)
    course=json.loads((race/"rendered_course.json").read_text())
    with np.load(race/"flight/execution.npz") as flight:
        times,states=flight["time"],flight["state"]
    with original.with_suffix(".csv").open() as stream:
        recorded={(int(r["frame_time_ms"]),int(r["ego"])):r for r in csv.DictReader(stream)}
    cmd=serial(["-threads","1","-i",original,"-f","rawvideo","-pix_fmt","rgb24","-s","1520x352","-r","12.5","-i","-",
                "-filter_complex_threads","1","-filter_complex","[0:v][1:v]vstack=inputs=2:shortest=1[v]","-map","[v]","-c:v","libx264",
                "-threads","1","-preset","veryfast","-crf","17","-pix_fmt","yuv420p","-movflags","+faststart",output])
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as enc:
        for k in range(0,len(times),4):
            ids=np.maximum(0,np.arange(k-3,k+1))
            rgb,depth=(np.stack([np.load(race/ch/f"{round(times[i]*1000):06d}.npy") for i in ids]) for ch in ("rgb","depth"))
            contexts=encode_views(model.model,rgb,depth)
            canvas=Image.new("RGB",(1520,352),BG)
            for ego in (0,1):
                pose=states[k,ego]
                forward=rotation(pose).apply([1,0,0])[:2]
                forward/=np.linalg.norm(forward)
                right=np.array([forward[1],-forward[0]])
                world=np.broadcast_to(pose[:3],(*SIDE.shape,3)).copy()
                world[:,:,:2]+=SIDE[:,:,None]*right+FWD[:,:,None]*forward
                context=contexts[ego:ego+1]
                q=torch.from_numpy(camera_points(world.reshape(-1,3),pose).astype("f"))
                values=model.query(context,q).numpy().reshape(SIDE.shape)
                assert np.isfinite(values).all()
                future=recorded_future(times,states[:,1-ego,:3],k,lead)
                row=recorded[round(times[k]*1000),ego]
                if future is not None:
                    point=torch.from_numpy(camera_points(future[:3],pose).astype("f"))
                    assert abs(model.query(context,point).item()-float(row[f"risk_at_recorded_peer_{ms}ms"]))<1.e-5
                view=panel(values,ego,pose,forward,right,states[k,1-ego],future,course,times[k],lead=lead,cutoff=cutoff)
                canvas.paste(view.crop((0,0,760,352)),(760*ego,0))
            enc.stdin.write(canvas.tobytes())
        enc.stdin.close()
        if enc.wait():raise RuntimeError("Video encoder failed")
    check=subprocess.run(serial(["-threads","1","-i",output,"-progress","pipe:1","-f","null","-"]),capture_output=True,text=True,check=True)
    count=[int(line[6:]) for line in check.stdout.splitlines() if line.startswith("frame=")][-1]
    assert count==(len(times)+3)//4==len(recorded)//2
    print(output,count,"frames decoded")


if __name__=="__main__":
    torch.set_num_threads(1)
    if "--actual-legacy-flights" in sys.argv:
        from scripts.replay.compose_fpv_motion_field import render as render_fpv
        for i in (1,2):
            race=ROOT/f"cf21_bendhead03_run{i}"
            if "--black-below-half" in sys.argv:
                from tempfile import TemporaryDirectory
                lead=.03 if "--30ms" in sys.argv else .1
                with TemporaryDirectory() as temp:
                    render_fpv(str(FF), legacy=True, race=race, cutoff=.5, output_folder=Path(temp),lead=lead)
                    append_legacy_planes(race,cutoff=.5,input_folder=Path(temp),lead=lead)
                    if lead==.03:
                        csv_name="two_laps_fpv_cnn_gru_30ms.csv"
                        (race/csv_name).write_bytes((Path(temp)/csv_name).read_bytes())
            else:
                render_fpv(str(FF), legacy=True, race=race)
                append_legacy_planes(race)
        sys.exit(0)
    if "--legacy-planes" in sys.argv:
        append_legacy_planes()
        sys.exit(0)
    counts=[render(i) for i in (1,2)]
    pair=OUT/"two_runs_five_views.mp4"
    subprocess.run(serial(["-threads","1","-i",OUT/"run1_five_views.mp4","-threads","1","-i",OUT/"run2_five_views.mp4",
                   "-filter_complex_threads","1","-filter_complex","[0:v][1:v]vstack=inputs=2[v]","-map","[v]","-c:v","libx264",
                   "-threads","1","-preset","veryfast","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",pair]),check=True)
    for file,count in zip([OUT/"run1_five_views.mp4",OUT/"run2_five_views.mp4",pair],counts+[max(counts)]):
        check=subprocess.run(serial(["-threads","1","-i",file,"-progress","pipe:1","-an","-f","null","-"]),check=True,capture_output=True,text=True)
        actual=[int(l[6:]) for l in check.stdout.splitlines() if l.startswith("frame=")][-1]
        assert actual==count,(file,count,actual)
        print(file.name,actual,"frames decoded",flush=True)
