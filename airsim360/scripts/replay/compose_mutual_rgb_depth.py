"""Plot matched RGB/depth observation windows from saved simulator captures."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'logs/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors

manifest=json.loads((ROOT/'reports/mutual_observation.json').read_text(encoding='utf-8'))
parser=argparse.ArgumentParser()
parser.add_argument('--crop',action='store_true',help='Use the earlier local observation windows instead of complete panoramas')
args=parser.parse_args()
OUT=Path(manifest['output_directory'])
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False

fig=plt.figure(figsize=(12,8.4) if args.crop else (20.8,11.5),layout='constrained')
grid=fig.add_gridspec(2,3,width_ratios=[1,1,.035])
cmap=plt.get_cmap('turbo').copy()
cmap.set_bad('black')
norm=colors.Normalize(vmin=0,vmax=30,clip=True)
metadata={'source_manifest':str(OUT/'manifest.json'),
          'depth_display_range_raw':[0,30],
          'depth_mask':'finite and 0 < raw value <= 1000',
          'units':'Uncalibrated simulator raw depth units',
          'method':'Matched ERP crops' if args.crop else 'Complete 2048x1024 ERP panoramas, without cropping or angular reprojection; original RGB pixel values unchanged.',
          'views':{}}
for i,key in enumerate(['drone_1','drone_2']):
    view=manifest['views'][key]
    crop=view['observation_crop']
    raw_rgb=cv2.imread(str(OUT/f'{key}_panorama.png'))
    full_depth=np.load(OUT/f'{key}_depth.npy',allow_pickle=False)
    if args.crop:
        rows=slice(crop['top'],crop['top']+crop['height'])
        columns=(np.arange(crop['width'])+crop['left_wrapped'])%full_depth.shape[1]
        rgb=raw_rgb[rows,columns]
        depth=full_depth[rows,columns]
        assert np.array_equal(rgb,cv2.imread(str(OUT/f'{key}_observation.png')))
    else:
        rgb=raw_rgb
        depth=full_depth
        assert rgb.shape==(1024,2048,3) and depth.shape==(1024,2048)
    assert depth.shape==rgb.shape[:2]
    mask=np.isfinite(depth)&(depth>0)&(depth<=1000)
    if args.crop: np.save(OUT/f'{key}_observation_depth.npy',depth,allow_pickle=False)
    masked=np.ma.masked_where(~mask,depth)
    rgba=cmap(norm(masked),bytes=True)
    depth_label='observation' if args.crop else 'panorama'
    cv2.imwrite(str(OUT/f'{key}_{depth_label}_depth.png'),cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR))
    observer,target=('A','B') if i==0 else ('B','A')
    left=fig.add_subplot(grid[i,0])
    right=fig.add_subplot(grid[i,1])
    left.imshow(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
    im=right.imshow(masked,cmap=cmap,norm=norm,interpolation='nearest')
    label='RGB' if args.crop else '全景 RGB'
    depth_title='深度' if args.crop else '全景深度'
    left.set_title(f'{observer} 观察 {target} · {label}',fontsize=14 if args.crop else 20,pad=9)
    right.set_title(f'{observer} 观察 {target} · {depth_title}',fontsize=14 if args.crop else 20,pad=9)
    left.set_axis_off()
    right.set_axis_off()
    valid=depth[mask]
    metadata['views'][key]={'crop':crop if args.crop else None,'shape':list(depth.shape),
                           'analysis_valid_fraction':float(mask.mean()),
                           'depth_min_raw':float(valid.min()),'depth_max_raw':float(valid.max()),
                           'rgb_source_pixels_unchanged':True}
bar=fig.add_subplot(grid[:,2])
fig.colorbar(im,cax=bar,extend='max',label='深度原始值（显示截断于 30，单位未标定）')
fig.suptitle('双无人机互相观测：RGB 与深度' if args.crop else '双无人机互相观测：完整 360° 全景 RGB 与深度',fontsize=18 if args.crop else 26)
fig.supxlabel('真实仿真渲染 · 使用位置同步的可见外观模型 · '+('RGB 与深度采用相同全景裁剪区域' if args.crop else '完整等距柱状投影，未裁剪；每幅原始全景 2048×1024'),fontsize=10 if args.crop else 14)
stem='mutual_rgb_depth_2x2' if args.crop else 'mutual_panorama_rgb_depth_2x2'
target=OUT/f'{stem}.png'
fig.savefig(target,dpi=180 if args.crop else 220)
plt.close(fig)
metadata['combined_image']=str(target)
(OUT/f'{stem}.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
print(target)
print(json.dumps(metadata['views'],indent=2,ensure_ascii=False))
