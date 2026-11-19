"""Reacquire world-frame candidate returns at the saved CBF simulation timestamps."""

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
from tempfile import TemporaryDirectory

import numpy as np
from scripts.perception.occupancy_flow import compact_surface, FlowPrior


def collect(folder, output, depth_only=False):
    output.mkdir(parents=True, exist_ok=False)
    cfg=json.loads((folder/"experiment.json").read_text())
    scale=json.loads((folder/"vehicle_geometry.json").read_text())["scale_xyz"]
    if not depth_only:
        if hashlib.sha256(Path(cfg["flow_prior"]).read_bytes()).hexdigest()!=cfg["flow_sha256"]:
            raise ValueError("Appearance checkpoint changed")
        prior=FlowPrior(cfg["flow_prior"])
    with np.load(folder/"flight/execution.npz") as f:
        times,states=f["time"],f["state"]
    clouds={}
    with TemporaryDirectory() as temp:
        cmd=[".venv/python.exe","-u","-m","scripts.simulation.spatial_render_worker",str(Path(temp)/"sensor.jsonl"),
             "--resolution","512","256","--scale",*map(str,scale)]
        if depth_only:
            cmd.append("--depth-only")
        with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                              env=dict(os.environ,SAVE_FLIGHT_VIEWS="0",SINGLE_FLIGHT_CAMERA="1")) as proc:
            try:
                if proc.stdout.readline().strip()!=b"READY":raise RuntimeError("AirSim not ready")
                for k,(stamp,snapshot) in enumerate(zip(times,states)):
                    proc.stdin.write((json.dumps([stamp,snapshot.tolist()])+"\n").encode())
                    proc.stdin.flush()
                    if not depth_only:
                        rgb=np.frombuffer(proc.stdout.read(2*256*512*3),np.uint8).reshape(2,256,512,3)
                    depth=np.frombuffer(proc.stdout.read(2*256*512*4),"<f4").reshape(2,256,512)
                    if depth_only:
                        np.savez_compressed(output/f"depth_{k:04d}.npz",depth=depth)
                        continue
                    for i in (0,1):
                        points,features=compact_surface(rgb[i],depth[i],snapshot[i],True)
                        clouds[f"{k}_{i}"]=points[prior.visible(features)>=.5,:3]
                    if k%50==0:print(k,len(times),flush=True)
            finally:proc.stdin.close()
            if proc.wait():raise RuntimeError("Renderer failed")
    if not depth_only:
        np.savez_compressed(output/"clouds.npz",**clouds)


if __name__=="__main__":
    collect(Path(sys.argv[1]),Path(sys.argv[2]),"--depth-only" in sys.argv[3:])
