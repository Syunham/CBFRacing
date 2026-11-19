"""Attribute panorama depth silhouettes by moving one retained test mesh at a time."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
from pathlib import Path
import time
import airsim
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/drone_visibility_check'
OUT.mkdir(exist_ok=True)
client=airsim.MultirotorClient(timeout_value=8)
names=client.listVehicles()[:2]
models=['CodexTwoDroneModel_1','CodexTwoDroneModel_2']
poses={n:client.simGetObjectPose(n) for n in models}
vehicle_poses={n:client.simGetVehiclePose(vehicle_name=n) for n in names}
paused=client.simIsPause()
source=json.loads((ROOT/'reports/dual_flight_capture.json').read_text())
report={'vehicles':names,'retained_test_models':models,'cases':{},'comparisons':{}}
arrays={}
try:
    client.simPause(True)
    for name in names:
        frame=next(f for f in source['frames'] if f['vehicle_name']==name)
        s=frame['vehicle_state_before']
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*s['position_ned_m']),
            airsim.Quaternionr(*s['orientation_xyzw'])),True,vehicle_name=name)
    for label,hidden in [('both_models_present',[]),('model_1_moved_away',[models[0]]),('both_models_moved_away',models)]:
        for name in models:
            target=airsim.Pose(airsim.Vector3r(1000,1000,-1000),airsim.Quaternionr(0,0,0,1)) if name in hidden else poses[name]
            client.simSetObjectPose(name,target,True)
        report['cases'][label]={}
        for i,name in enumerate(names):
            for camera in ['panorama_original','panorama_depth']:
                client.client.call('simSetPanoramaResolution',camera,512,256,name)
                client.client.call('simTriggerPanoramaCapture',camera,name)
            rgb,depth=client.simGetImages([airsim.ImageRequest('panorama_original',0,False,False),
                airsim.ImageRequest('panorama_depth',0,True,False)],vehicle_name=name)
            image=np.frombuffer(rgb.image_data_uint8,np.uint8).reshape(256,512,3)
            array=np.frombuffer(depth.image_data_uint8,dtype='<f4').reshape(256,512).copy()
            arrays[(label,name)]=array
            cv2.imwrite(str(OUT/f'{label}_drone_{i+1}_rgb.png'),image)
            np.save(OUT/f'{label}_drone_{i+1}_depth.npy',array)
            report['cases'][label][name]={'rgb_timestamp_ns':int(rgb.time_stamp),'depth_timestamp_ns':int(depth.time_stamp)}
    for i,name in enumerate(names):
        a=arrays[('both_models_present',name)]
        b=arrays[('model_1_moved_away',name)]
        c=arrays[('both_models_moved_away',name)]
        report['comparisons'][name]={}
        for label,first,second in [('model_1',a,b),('model_2',b,c)]:
            delta=np.abs(first-second)
            changed=np.isfinite(delta)&(delta>.05)
            count,components,stats,centers=cv2.connectedComponentsWithStats(changed.astype(np.uint8),8)
            regions=[{'bbox_xywh':stats[j,:4].tolist(),'pixels':int(stats[j,4])} for j in range(1,count) if stats[j,4]>=3]
            report['comparisons'][name][label]={'changed_pixels':int(changed.sum()),'regions':regions}
            cv2.imwrite(str(OUT/f'{label}_drone_{i+1}_depth_difference_mask.png'),changed.astype(np.uint8)*255)
finally:
    for name,pose in poses.items(): client.simSetObjectPose(name,pose,True)
    for name,pose in vehicle_poses.items(): client.simSetVehiclePose(pose,True,vehicle_name=name)
    client.simPause(paused)
    client.client.close()
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report['comparisons'],indent=2))
