"""Compare the same horizontal slice through the labeled future neighbor center."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(track, visual):
    with np.load(track/'example.npz') as a, np.load(visual/'example.npz') as b:
        if not np.array_equal(a['peer'],b['peer']):
            raise ValueError('Example frames do not match')
        values=[a['truth'],a['maps'][0],a['maps'][2],b['maps'][2]]
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig, axes=plt.subplots(1,4,figsize=(12.5,3.65),constrained_layout=True,sharey=True)
    for ax,v,title in zip(axes,values,('训练标签定义','旧时空网络','局部网络 · 带赛道先验','局部网络 · 无赛道查表')):
        im=ax.imshow(v,origin='lower',extent=(-2.02,2.02,-2.02,2.02),vmin=0,vmax=1,cmap='inferno',interpolation='nearest')
        ax.plot(0,0,'+',color='#4cffff',markersize=9,markeredgewidth=1.6)
        ax.add_patch(plt.Circle((0,0),.55,color='#4cffff',fill=False,ls=':',lw=1.2))
        ax.set_title(title)
        ax.set_xlabel('相对位置 X（米）')
        ax.set_xticks([-2,-1,0,1,2])
        ax.set_yticks([-2,-1,0,1,2])
    axes[0].set_ylabel('相对位置 Y（米）')
    fig.colorbar(im,ax=axes,label='风险分数',shrink=.8)
    fig.suptitle('0 号机在 5.76 秒的观测   |   十字为 0.4 秒后邻机真值，虚线为 0.55 米裕量',fontsize=13)
    fig.savefig(visual/'field_comparison.png',dpi=145)
    plt.close(fig)


if __name__=='__main__':
    plot(*(Path(p) for p in sys.argv[1:]))
