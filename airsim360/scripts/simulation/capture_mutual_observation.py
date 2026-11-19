"""Capture real simulator panoramas, with target-removal depth checks for both views."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import time
import airsim
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
OUT=ROOT/'captures'/f'mutual_observation_{stamp}'
OUT.mkdir(parents=True)
W,H=2048,1024
client=airsim.MultirotorClient(timeout_value=15)
report={'output_directory':str(OUT),'timestamp':stamp,'resolution':[W,H],
        'method':'Two controllable BP_Drone vehicles, frozen in simulation at separate airborne poses. No extra drone mesh fixtures. Real panorama RGB, with counterfactual depth visibility checks.',
        'views':{}}
names=[]

def vector(v): return [float(v.x_val),float(v.y_val),float(v.z_val)]

def capture(name,label):
    for camera in ['panorama_original','panorama_depth']:
        client.client.call('simTriggerPanoramaCapture',camera,name)
    rgb,depth=client.simGetImages([airsim.ImageRequest('panorama_original',0,False,False),
                                airsim.ImageRequest('panorama_depth',0,True,False)],vehicle_name=name)
    image=np.frombuffer(rgb.image_data_uint8,np.uint8).reshape(H,W,3).copy()
    if depth.image_data_float:
        array=np.asarray(depth.image_data_float,np.float32).reshape(H,W)
    else:
        array=np.frombuffer(depth.image_data_uint8,dtype='<f4').reshape(H,W).copy()
    cv2.imwrite(str(OUT/f'{label}_panorama.png'),image)
    np.save(OUT/f'{label}_depth.npy',array)
    return image,array,{'rgb_timestamp_ns':int(rgb.time_stamp),'depth_timestamp_ns':int(depth.time_stamp)}

try:
    assert client.ping()
    client.reset()
    client.simPause(True)
    for fixture in ['CodexTwoDroneModel_1','CodexTwoDroneModel_2']:
        if fixture in client.simListSceneObjects(): client.simDestroyObject(fixture)
    original=client.listVehicles()[0]
    other='MutualDroneB'
    if other not in client.listVehicles():
        assert client.simAddVehicle(other,'SimpleFlight',
            airsim.Pose(airsim.Vector3r(2.5,0,-3.6),airsim.to_quaternion(0,0,np.pi)),
            'DefaultQuadrotor')
    names=[original,other]
    poses={original:airsim.Pose(airsim.Vector3r(0,0,-3),airsim.to_quaternion(0,0,0)),
           other:airsim.Pose(airsim.Vector3r(0,0,-3.6),airsim.to_quaternion(0,0,np.pi))}
    for name in names:
        client.enableApiControl(True,name)
        client.armDisarm(True,name)
        client.simSetVehiclePose(poses[name],True,vehicle_name=name)
        for camera in ['panorama_original','panorama_depth']:
            client.client.call('simSetPanoramaResolution',camera,W,H,name)
    report['vehicles']=names
    report['scene_drone_actors']=client.simListSceneObjects('.*(Drone|Quad|FlyingPawn).*')
    client.simPause(False)
    holds=[client.moveByVelocityZAsync(0,0,-3 if i==0 else -3.6,180,
           yaw_mode=airsim.YawMode(False,0 if i==0 else 180),vehicle_name=n)
           for i,n in enumerate(names)]
    time.sleep(2)
    client.simPause(True)
    far=airsim.Pose(airsim.Vector3r(1000,1000,-1000),airsim.Quaternionr(0,0,0,1))
    visual_names={}
    visual_poses={}
    for i,name in enumerate(names):
        vehicle_pose=client.simGetVehiclePose(vehicle_name=name)
        position=vehicle_pose.position
        visual_pose=airsim.Pose(airsim.Vector3r(position.x_val+(2.5 if i else 0),position.y_val,position.z_val),vehicle_pose.orientation)
        mesh=client.simSpawnObject(f'MutualAppearance_{i+1}','Quadrotor1',visual_pose,airsim.Vector3r(1,1,1),False,False)
        assert mesh
        visual_names[name]=mesh
        visual_poses[name]=visual_pose
    report['method']='Real simulator panorama from two controllable vehicles. Because native BP_Drone bodies were not visible in the tested panoramas, each vehicle is represented by one Quadrotor1 appearance actor at its sampled global position. Observer appearance is parked away during its own capture. These are rendered mesh proxies, not native-body mutual visibility.'
    report['appearance_mapping']={n:{'actor':visual_names[n],'position_global_ned_m':vector(visual_poses[n].position),'scale':[1,1,1]} for n in names}
    for i,name in enumerate(names):
        target=names[1-i]
        client.simSetObjectPose(visual_names[name],far,True)
        client.simSetObjectPose(visual_names[target],visual_poses[target],True)
        label=f'drone_{i+1}'
        image,depth,metadata=capture(name,label)
        client.simSetObjectPose(visual_names[target],far,True)
        _,absent,_=capture(name,f'{label}_target_absent')
        client.simSetObjectPose(visual_names[target],visual_poses[target],True)
        changed=np.isfinite(depth)&np.isfinite(absent)&(absent-depth>.75)&(depth>.4)&(depth<6)
        count,components,stats,centers=cv2.connectedComponentsWithStats(changed.astype(np.uint8),8)
        regions=[{'bbox_xywh':stats[k,:4].tolist(),'pixels':int(stats[k,4]),'center_xy':centers[k].tolist()}
                 for k in range(1,count) if stats[k,4]>10]
        regions.sort(key=lambda x:x['pixels'],reverse=True)
        cv2.imwrite(str(OUT/f'{label}_target_visibility_mask.png'),changed.astype(np.uint8)*255)
        report['views'][label]={'observer':name,'target':target,**metadata,
                               'vehicle_pose_ned_m':vector(client.simGetVehiclePose(vehicle_name=name).position),
                               'target_visibility_regions':regions,'target_changed_pixels':int(changed.sum()),
                               'panorama_file':str(OUT/f'{label}_panorama.png')}
        if regions:
            # Analytical ERP window, with horizontal wrap; raw panorama remains unchanged.
            region=regions[0]
            x,y,w,h=region['bbox_xywh']
            crop_w=min(W,max(640,int(w*4)))
            crop_h=min(H,int(crop_w*2/3))
            center_x=x+w/2
            center_y=y+h/2
            left=int(center_x-crop_w/2)
            top=max(0,min(H-crop_h,int(center_y-crop_h/2)))
            crop=image[top:top+crop_h,(np.arange(crop_w)+left)%W]
            cv2.imwrite(str(OUT/f'{label}_observation.png'),crop)
            report['views'][label]['observation_crop']={'file':str(OUT/f'{label}_observation.png'),
                                                       'left_wrapped':left,'top':top,'width':crop_w,'height':crop_h}
        print(json.dumps(report['views'][label],ensure_ascii=False),flush=True)
    report['both_targets_verified']=all(v['target_changed_pixels']>10 for v in report['views'].values())
finally:
    for name in names:
        try: client.armDisarm(False,name)
        except Exception: pass
    client.client.close()
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    (ROOT/'reports/mutual_observation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('OUTPUT',OUT,flush=True)
