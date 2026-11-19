"""Compose synchronized three-view evidence from the two recorded flights."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]/"reports/spatial_potential"
OUT=ROOT/"cf21_bendhead03_comparison"
W,H=1536,424
FPS=12.5
BG=(19,30,43)
WHITE=(243,247,250)
MUTED=(166,186,200)
ACCENT=[(106,221,191),(255,196,125)]
FONT="C:/Windows/Fonts/msyh.ttc"
def face(size):
    return ImageFont.truetype(FONT,size)
F22,F19,F16=map(face,(22,19,16))

def event_note(t,k):
    if t<.20:return "起飞 · 两次初始状态、过门点和参考轨迹相同"
    if t<.40:return "0.20 s 参考更新  /  0.26 s 1 号机 MPCC 指令分叉"
    if t<1.32:return "0.40 s 1 号机重规划  /  0.60 s 第五个预瞄门的目标点首次不同"
    if 4.65<=t<=6.1:return "第一圈 5 → 6 门 · 两次的 0 号机飞行距离接近，平均速度出现差异"
    if t>=10.98:return "第二圈末段 · 对照两机实际过门时刻与规划到达时刻"
    return "同步回放 · 三幅画面使用同一仿真时刻"

def encoder(stack,ffmpeg,path,height):
    cmd=[str(ffmpeg),"-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
         "-s",f"{W}x{height}","-r","12.5","-i","-","-an","-c:v","libx264","-threads","1",
         "-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    def close():
        proc.stdin.close()
        if proc.wait():raise RuntimeError("H.264 encoding failed")
    stack.callback(close)
    return proc.stdin

class Flight:
    def __init__(self,stack,idx,stats):
        self.idx=idx
        self.stats=stats
        p=ROOT/f"cf21_bendhead03_run{idx}"
        self.captures=[]
        for filename in ("chase.mp4","drone0_fpv.mp4","drone1_fpv.mp4"):
            c=cv2.VideoCapture(str(p/filename),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
            if not c.isOpened() or abs(c.get(cv2.CAP_PROP_FPS)-FPS)>.001:raise ValueError(filename)
            self.captures.append(c);stack.callback(c.release)
        counts=[int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in self.captures]
        assert len(set(counts))==1
        self.frames=counts[0]
        with np.load(p/"flight/execution.npz") as a:
            self.times=a["time"];self.gap=np.linalg.norm(a["state"][:,0,:3]-a["state"][:,1,:3],axis=1)
        self.cross=np.asarray([r["crossing_s"] for r in stats])
        self.last=None

    def row(self,n,summary=False):
        frozen=n>=self.frames
        if not frozen and not summary:
            frames=[]
            for c in self.captures:
                ok,frame=c.read()
                if not ok:raise EOFError("Missing a synchronized frame")
                frames.append(frame)
            self.last=frames
        stamp=min(n,self.frames-1)/FPS
        t=self.times[-1] if summary else stamp
        row=Image.new("RGB",(W,H),BG)
        for i,f in enumerate(self.last):
            height=288 if i==0 else 256
            resize=cv2.resize(f,(512,height),interpolation=cv2.INTER_AREA)
            row.paste(Image.fromarray(cv2.cvtColor(resize,cv2.COLOR_BGR2RGB)),(i*512,79+(288-height)//2))
        d=ImageDraw.Draw(row)
        d.text((15,7),f"航次 {self.idx}   |   "+("两圈完成" if summary else f"仿真 {stamp:05.2f} s"),font=F22,fill=ACCENT[self.idx-1])
        d.text((354,10),f"两机中心距  {np.interp(t,self.times,self.gap):.3f} m",font=F19,fill=WHITE)
        both=np.where(self.cross.max(axis=1)<=t+1e-8)[0]
        if len(both):
            info=self.stats[both[-1]];gap=info["signed_ms"]
            leader=(f"{1 if gap>0 else 0} 号机先过 {abs(gap):.1f} ms" if abs(gap)>.05 else "两机同时过门（差 < 0.1 ms）")
            label=f"最近共同过门   第 {info['lap']} 圈 · 门 {info['gate']}    {leader}"
        else:label="尚未共同通过第一道门"
        d.text((701,10),label,font=F19,fill=WHITE)
        for i,label in enumerate(("第三人称追拍","0 号机 · 全景第一人称","1 号机 · 全景第一人称")):
            d.text((i*512+15,49),label,font=F16,fill=MUTED)
        note=event_note(t,self.idx) if not summary else "最终记录   第 6 门  %.1f ms   /   第 7 门  %.1f ms   ·   均为第二圈绝对时间差"%(abs(self.stats[12]["signed_ms"]),abs(self.stats[13]["signed_ms"]))
        d.text((15,379),note,font=F19,fill=WHITE)
        d.text((W-286,382),"末帧定格 · 终点按日志" if summary or frozen else "0.5× 回放 · 原始帧间隔 80 ms",font=F16,fill=MUTED)
        return np.asarray(row)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("ffmpeg",type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    stats=json.loads((OUT/"attribution.json").read_text())["gate_events"]
    with ExitStack() as stack:
        runs=[Flight(stack,i,s) for i,s in enumerate(stats,1)]
        both=encoder(stack,args.ffmpeg,OUT/"two_runs_three_views.mp4",2*H)
        individual=[encoder(stack,args.ffmpeg,OUT/f"run{i}_three_views.mp4",H) for i in (1,2)]
        count=max(r.frames for r in runs)
        for n in range(count+1):
            end=n==count
            rows=[r.row(n,end) for r in runs]
            payload=np.concatenate(rows).tobytes()
            for _ in range(25 if end else 2):
                both.write(payload)
            for stream,row,run in zip(individual,rows,runs):
                if not end and n>=run.frames:continue
                data=row.tobytes()
                for _ in range(25 if end else 2):
                    stream.write(data)
    for name,expected,height in [("two_runs_three_views",count*2+25,H*2),*((f"run{i}_three_views",r.frames*2+25,H) for i,r in enumerate(runs,1))]:
        cap=cv2.VideoCapture(str(OUT/(name+".mp4")),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        actual=0
        while True:
            ok,img=cap.read()
            if not ok:break
            assert img.shape==(height,W,3)
            actual+=1
        cap.release()
        assert actual==expected,(name,actual,expected)
        print(name,actual,"frames verified")

if __name__=="__main__":
    main()
