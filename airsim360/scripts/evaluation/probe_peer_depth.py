"""Frozen-pose tests for missing depth at the peer, using the original renderer."""

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
import time

import numpy as np

from scripts.simulation.airsim360_replay import create_renderer
from scripts.simulation.preview_split_s import place_course


def main():
    run, out = map(Path, sys.argv[1:3])
    out.mkdir(exist_ok=False)
    os.environ["RISK_REPLAY_LOG"] = str(out/"sensor.jsonl")
    course = json.loads((run/"rendered_course.json").read_text())
    placement_file = Path("reports/split_s_open_yard/placement.json")
    placement = json.loads(placement_file.read_text())
    scale = json.loads((run/"vehicle_geometry.json").read_text())["scale_xyz"]
    traj = np.load(run/"flight/execution.npz")
    last = sorted((run/"rgb").glob("*.npy"))[-1]
    t = int(last.stem)/1000
    _, h, w, _ = np.load(last, mmap_mode="r").shape
    state = traj["state"][np.argmin(abs(traj["time"]-t))]
    y, x = np.indices((h,w)) + .5
    lon, lat = (x/w-.5)*2*np.pi, (y/h-.5)*np.pi
    rays = np.stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)],axis=-1)
    with ExitStack() as stack:
        renderer = stack.enter_context(create_renderer(True, scale, (w,h), refresh_on_switch=False))
        renderer.translation = np.asarray(placement["translation_ned_m"])
        renderer.heading = np.deg2rad(placement["heading_ned_deg"])
        actors = set(renderer.client.simListSceneObjects())
        for name in placement["removed_yard_fixtures"]:
            if name in actors:
                assert renderer.client.simDestroyObject(name)
        place_course(renderer.client, course, stack, placement_file.with_name("gate_texture.png"),
                     renderer.translation, renderer.heading)
        results = []

        def test(name, snapshot, order=(0,1)):
            ordered = snapshot[list(order)]
            pair = renderer.capture(t,ordered)
            counts = [None,None]
            rgb_errors = [None,None]
            for j, i in enumerate(order):
                delta = ordered[1-j,:3] - ordered[j,:3]
                qw, qv = ordered[j,6], -ordered[j,7:10]
                body = delta + 2*np.cross(qv, np.cross(qv,delta) + qw*delta)
                peer = body*[1,-1,-1]
                d = pair["depth"][j]
                dist = np.linalg.norm(d[...,None]*rays-peer,axis=-1)
                counts[i] = int((dist<.15).sum())
                if np.array_equal(snapshot,state):
                    u = int((np.arctan2(peer[1],peer[0])/(2*np.pi)+.5)*w)
                    v = int((np.arcsin(peer[2]/np.linalg.norm(peer))/np.pi+.5)*h)
                    roi = np.s_[v-9:v+10,u-9:u+10]
                    original = np.load(last,mmap_mode="r")[i][roi].astype(float)
                    rgb_errors[i] = float(abs(original-pair["rgb"][j][roi]).mean())
            results.append(dict(test=name,hits=counts,original_rgb_roi_mae=rgb_errors))
            print(name, counts, rgb_errors,flush=True)
            (out/"depth_tests.json").write_text(json.dumps(results,indent=2))

        position = renderer.position_pair
        def history(order=(0,1)):
            for file in sorted((run/"rgb").glob("*.npy"))[:-1]:
                stamp = int(file.stem)/1000
                previous = traj["state"][np.argmin(abs(traj["time"]-stamp))]
                renderer.capture(stamp,previous[list(order)])
        if "--trigger-check" in sys.argv[3:]:
            read = renderer.read
            for refresh in ["original","retrigger","advance_then_retrigger","repeat_pair"]:
                history()
                def refreshed(vehicle):
                    first = read(vehicle)
                    if refresh == "advance_then_retrigger":
                        renderer.client.simContinueForFrames(1)
                        time.sleep(.10)
                        if not renderer.client.simIsPause():
                            raise RuntimeError("Expected a frozen view")
                    return read(vehicle) if refresh in ("retrigger","advance_then_retrigger") else first
                renderer.read = refreshed
                if refresh == "repeat_pair":
                    test("first_of_repeated_pair",state)
                test(refresh,state)
                renderer.read = read
            return
        for extra in [0,1,3,8]:
            history()
            def hold(local,quats):
                for step in range(extra+1):
                    check = position(local,quats)
                return check
            renderer.position_pair = hold
            test(f"history_extra_pose_steps_{extra}",state)
            renderer.position_pair = position
            test(f"repeat_after_extra_{extra}",state)
        history((1,0))
        test("history_reverse",state,(1,0))
        test("baseline",state)
        test("swap_render_order_and_proxy",state,(1,0))
        renderer.meshes.reverse()
        test("swap_proxy_only",state)
        test("swap_render_order_only",state,(1,0))
        renderer.meshes.reverse()
        for angle in [90,180,270]:
            turned = state.copy()
            # Apply a yaw about camera 0's local up axis; peer 1 remains fixed.
            w0,x0,y0,z0 = state[0,6:10]
            c,s = np.cos(np.deg2rad(angle/2)),np.sin(np.deg2rad(angle/2))
            turned[0,6:10] = [w0*c-z0*s,x0*c+y0*s,y0*c-x0*s,z0*c+w0*s]
            test(f"turn_camera0_local_yaw_{angle}",turned)
        for factor in [1.5,2,3]:
            far = state.copy()
            far[1,:3] = state[0,:3] + factor*(state[1,:3]-state[0,:3])
            test(f"range_x{factor}",far)
        raised = state.copy()
        raised[:,:3] += [0,0,3]
        test("both_raised_3m",raised)
        test("baseline_repeat",state)


if __name__ == "__main__":
    main()
