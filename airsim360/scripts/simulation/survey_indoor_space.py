"""Measure the visible indoor volume from fixed, level preview cameras."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
from pathlib import Path

import airsim
import cv2
import numpy as np

from scripts.simulation.preview_split_s import set_camera
from scripts.simulation.airsim360_replay import xyz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eye", nargs=3, type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    c = airsim.MultirotorClient(timeout_value=30)
    paused = c.simIsPause()
    try:
        name, = c.listVehicles()
        c.simPause(True)
        c.simRunConsoleCommand("r.MotionBlurQuality 0")
        set_camera(c, name, np.array(args.eye))
        actual = c.simGetVehiclePose(name)
        cameras = ["panorama_original", "panorama_depth"]
        for camera in cameras:
            c.client.call("simSetPanoramaResolution", camera, 1024, 512, name)
        for _ in range(3):
            for camera in cameras:
                c.client.call("simTriggerPanoramaCapture", camera, name)
        rgb, depth = c.simGetImages([airsim.ImageRequest(cameras[0], 0, False, False),
                                    airsim.ImageRequest(cameras[1], 0, True, False)], vehicle_name=name)
        bgr = np.frombuffer(rgb.image_data_uint8, np.uint8).reshape(512, 1024, 3)
        d = np.frombuffer(depth.image_data_uint8, "<f4").reshape(512, 1024)
        q = actual.orientation
        np.savez(args.output.with_suffix(".npz"), position=xyz(actual.position),
                 orientation_xyzw=[q.x_val,q.y_val,q.z_val,q.w_val], depth=d)
        cv2.imwrite(str(args.output.with_suffix(".jpg")), bgr)
        print("horizon every 45 deg from forward", np.round(np.roll(d[256],-512)[::128],1))
        print("ceiling / floor m", d[0,512], d[-1,512])
    finally:
        c.simPause(paused)
        c.client.close()


if __name__ == "__main__":
    main()
