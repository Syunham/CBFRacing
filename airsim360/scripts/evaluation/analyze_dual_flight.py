"""Validate and bundle captured arrays; draw standalone scientific previews."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import hashlib
import os
from pathlib import Path
import cv2
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'logs/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
report=json.loads((ROOT/'reports/dual_flight_capture.json').read_text(encoding='utf-8'))
OUT=Path(report['output_directory'])
names=report['vehicles']
rgbs,depths,rgb_times,depth_times,positions=[],[],[],[],[]
validation=[]
for name in names:
    records=[f for f in report['frames'] if f['vehicle_name']==name]
    rgb_frames,depth_frames=[],[]
    for frame in records:
        bgr=cv2.imread(str(OUT/frame['rgb_file']))
        assert hashlib.sha256(bgr.tobytes()).hexdigest()==frame['rgb_sha256']
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        depth=np.load(OUT/frame['depth_file'],allow_pickle=False)
        assert hashlib.sha256(depth.tobytes()).hexdigest()==frame['depth_sha256']
        assert rgb.shape==(256,512,3) and rgb.dtype==np.uint8
        assert depth.shape==(256,512) and depth.dtype==np.float32
        assert rgb.std()>1 and np.isfinite(depth).any()
        rgb_frames.append(rgb)
        depth_frames.append(depth)
        validation.append({'vehicle':name,'frame':frame['frame'],'rgb_shape':list(rgb.shape),
                           'depth_shape':list(depth.shape),'depth_dtype':str(depth.dtype)})
    rgbs.append(np.stack(rgb_frames))
    depths.append(np.stack(depth_frames))
    rgb_times.append([f['rgb_timestamp_ns'] for f in records])
    depth_times.append([f['depth_timestamp_ns'] for f in records])
    positions.append([f['vehicle_state_before']['position_ned_m'] for f in records])
rgb=np.stack(rgbs)
depth=np.stack(depths)
valid=np.isfinite(depth)&(depth>0)&(depth<=1000)
np.savez_compressed(OUT/'dataset.npz',vehicle_names=np.asarray(names),rgb=rgb,depth=depth,
                    depth_analysis_mask=valid,rgb_timestamp_ns=np.asarray(rgb_times,dtype=np.int64),
                    depth_timestamp_ns=np.asarray(depth_times,dtype=np.int64),
                    position_ned_m=np.asarray(positions))
with np.load(OUT/'dataset.npz',allow_pickle=False) as bundle:
    assert np.array_equal(bundle['rgb'],rgb)
    assert np.array_equal(bundle['depth'],depth,equal_nan=True)

fig,axes=plt.subplots(2,2,figsize=(12,6.5),layout='constrained')
for i,name in enumerate(names):
    axes[i,0].imshow(rgb[i,0])
    axes[i,0].set_title(f'{name} - panorama RGB')
    masked=np.ma.masked_where(~valid[i,0],depth[i,0])
    colormap=plt.get_cmap('turbo').copy()
    colormap.set_bad('black')
    im=axes[i,1].imshow(masked,cmap=colormap,vmin=0,vmax=50)
    axes[i,1].set_title(f'{name} - panorama depth (raw units)')
    for axis in axes[i]: axis.set_axis_off()
fig.colorbar(im,ax=axes[:,1],shrink=.8,label='Raw depth; black = outside analysis mask')
fig.suptitle('Two-vehicle flight capture | 512 x 256 per image | sequential timestamped streams')
fig.savefig(OUT/'preview.png',dpi=160)
plt.close(fig)

rows=report['trajectory']
t0=rows[0]['host_monotonic_seconds']
times=np.asarray([r['host_monotonic_seconds']-t0 for r in rows])
fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
for name in names:
    xyz=np.asarray([r['vehicles'][name]['position_ned_m'] for r in rows])
    speed=np.asarray([np.linalg.norm(r['vehicles'][name]['velocity_ned_mps']) for r in rows])
    axes[0].plot(times,xyz[:,0],'.-',label=name)
    axes[1].plot(times,-xyz[:,2],'.-',label=name)
    axes[2].plot(times,speed,'.-',label=name)
for axis,title,ylabel in zip(axes,['X position','Height relative to vehicle NED origin','Speed'],['X (m)','-Z (m)','Speed (m/s)']):
    axis.set(title=title,xlabel='Elapsed host time (s)',ylabel=ylabel)
    axis.grid(alpha=.25)
axes[0].legend(fontsize=8)
fig.savefig(OUT/'flight_trajectory.png',dpi=160)
plt.close(fig)
temporal=[f['rgb_depth_timestamp_delta_ms'] for f in report['frames']]
summary={'primary_rgb_files':len(report['frames']),'primary_depth_files':len(report['frames']),
         'per_vehicle_frame_count':rgb.shape[1], 'bundle_rgb_shape':list(rgb.shape),
         'bundle_depth_shape':list(depth.shape),'all_depth_finite_fraction':float(np.isfinite(depth).mean()),
         'depth_analysis_valid_fraction':float(valid.mean()),
         'rgb_depth_timestamp_delta_ms_min':min(temporal),
         'rgb_depth_timestamp_delta_ms_max':max(temporal),
         'npy_png_readback_verified':True,'npz_lossless_readback_verified':True,
         'preview':str(OUT/'preview.png'),'trajectory_plot':str(OUT/'flight_trajectory.png'),
         'bundle':str(OUT/'dataset.npz'),'files':validation}
(OUT/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
readme='''# 双机并行飞行全景数据

每架无人机 4 组采样，总计 8 张 RGB PNG 和 8 份 float32 深度 NPY。

- `drone_1/` 对应 BP_Drone_C_1，`drone_2/` 对应 CodexDrone2。
- 每组的 `*_rgb.png` 是全景 RGB；`*_depth.npy` 保留原始浮点深度值。
- `*_depth_preview.png` 仅为 0..50 原始数值范围的伪彩预览，不能替代 NPY。
- `*_valid_mask.png` 是分析掩码：finite 且 0 < d <= 1000；不是经标定的传感器有效性标记。
- 原始深度包含 32256 等异常大值；未删除、替换或归一化原始数据。物理单位未独立标定。
- RGB/深度是顺序触发，各自带时间戳，本次两模态时间差约 33..123ms。不是严格同步真值。
- 返回的相机位姿字段均为零，不能用于重建相机外参。manifest 内另存采集前后飞行器状态。
- `dataset.npz` 是无损汇总：rgb [vehicle, frame, H, W, 3]、depth [vehicle, frame, H, W]，以及时间戳、NED 位置和分析掩码。
- `manifest.json` 含命令、轨迹、并行运动证据、帧元数据及降落/清理结果。
- `preview.png` 是两路 RGB/深度预览，`flight_trajectory.png` 是飞行轨迹与速度图。

```python
import numpy as np
data = np.load("dataset.npz", allow_pickle=False)
print(data["vehicle_names"])
print(data["rgb"].shape)    # (2, 4, 256, 512, 3), uint8, RGB
print(data["depth"].shape)  # (2, 4, 256, 512), float32
```
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='files'},indent=2))
