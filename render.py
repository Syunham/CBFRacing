"""Actual executed states and active network allocations, without invented motion."""
from context import *
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from PIL import Image
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})

REPORT=HERE/'report'
COLORS=['#467D99','#C27B52']

def draw_gates(ax,track):
    for j,(g,n) in enumerate(zip(track['centers'],track['normals'])):
        g=np.array(g);n=np.array(n);lat=np.cross([0.,0.,1.],n);up=np.array([0.,0.,1.])
        points=np.array([g+.75*(a*lat+b*up) for a,b in [(-1,-1),(1,-1),(1,1),(-1,1),(-1,-1)]])
        ax.plot(*points.T,color='#777777',lw=.8);ax.text(*(g+[0,0,.45]),str(j+1),fontsize=8)

def animate(r):
    target=REPORT/'rollout.gif'
    final=REPORT/'last_frame'
    if target.exists() and final.with_suffix('.pdf').exists() and final.with_suffix('.svg').exists():
        with Image.open(target) as im:
            count=im.n_frames
            for j in range(count):im.seek(j);im.load()
        return dict(run=r['name'],gif=target.relative_to(REPORT).as_posix(),frames=count,decoded=True)
    plans=[[],[]];files=iter(sorted((r['folder']/'plans').glob('*.npz')))
    for row in r['front']:
        if row['success']:
            with np.load(next(files)) as data:plans[row['agent']].append((row['time'],data['gates'].copy()))
    fig=plt.figure(figsize=(10,7));grid=fig.add_gridspec(2,1,height_ratios=[2.4,1],hspace=.12)
    ax=fig.add_subplot(grid[0],projection='3d');speed_ax=fig.add_subplot(grid[1]);draw_gates(ax,r['course'])
    speed=np.linalg.norm(r['x'][:,:,3:6],axis=-1)
    for i,c in enumerate(COLORS):speed_ax.plot(r['t'],speed[:,i],color=c,label=f'UAV {i+1}',lw=1.)
    speed_cursor=speed_ax.axvline(0,color='#888888',lw=.8)
    speed_ax.set(xlim=(0,r['t'][-1]+.05),ylim=(0,max(1.,speed.max()*1.1)),xlabel='Physical flight time (s)',ylabel='Speed (m/s)')
    speed_ax.spines[['top','right']].set_visible(False);speed_ax.grid(alpha=.2);speed_ax.legend(loc='lower right')
    np.savetxt(REPORT/'speed_source_data.csv',np.c_[r['t'],speed],delimiter=',',header='time_s,uav1_speed_mps,uav2_speed_mps',comments='')
    fig.subplots_adjust(left=.09,right=.97,bottom=.11,top=.91)
    world=np.array(r['course']['world_box']);ax.set(xlim=world[[0,3]],ylim=world[[1,4]],zlim=world[[2,5]],xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
    ax.set_box_aspect(world[3:]-world[:3]);ax.view_init(elev=30,azim=-58)
    lines=[];dots=[];bodies=[];allocations=[]
    for i,c in enumerate(COLORS):
        line,=ax.plot([],[],[],color=c,lw=1.3,label=f'UAV {i+1}');dot,=ax.plot([],[],[],'o',color=c,ms=4)
        body,=ax.plot([],[],[],color=c,lw=2);points,=ax.plot([],[],[],'x',color=c,ms=7,mew=1.6,label='Allocated regions' if i==0 else None)
        lines.append(line);dots.append(dot);bodies.append(body);allocations.append(points)
    ax.legend(loc='upper left',fontsize=8);label=fig.text(.5,.97,'',ha='center',fontsize=10)
    outcome='COMPLETE' if r['summary']['success'] else 'FAILED'
    fig.text(.5,.025,f'{r["name"]} | {outcome} | {r["summary"]["stop_reason"]}',ha='center',fontsize=7,wrap=True)
    frames=np.unique(np.r_[np.arange(0,r['t'][-1],.1),r['t'][-1]])
    def update(t):
        speed_cursor.set_xdata([t,t])
        k=min(max(int(np.searchsorted(r['t'],t,side='right')),1),len(r['t']));x=r['x'][k-1]
        for i in range(2):
            lines[i].set_data_3d(*r['x'][:k,i,:3].T);dots[i].set_data_3d(*x[i:i+1,:3].T)
            qw,qx,qy,qz=x[i,6:10]
            rotation=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy)],
                [2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-qw*qx)],
                [2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)]])
            cross=np.array([[-.15,0,0],[.15,0,0],[0,0,0],[0,-.15,0],[0,.15,0]])@rotation.T+x[i,:3];bodies[i].set_data_3d(*cross.T)
            active=[p for when,p in plans[i] if when<=t+1e-8]
            if active:allocations[i].set_data_3d(*active[-1].T)
        passed=[sum(e['time']<=t+1e-8 for e in ev) for ev in r['events']]
        label.set_text(f't={t:.2f}s | gates {passed[0]}/{r["summary"]["target_per_agent"]}, {passed[1]}/{r["summary"]["target_per_agent"]} | distance={np.linalg.norm(x[0,:3]-x[1,:3]):.3f}m | 2x playback')
        return lines+dots+bodies+allocations+[label]
    if not target.exists():
        animation=FuncAnimation(fig,update,frames=frames,interval=50,blit=False)
        animation.save(target,writer=PillowWriter(fps=20))
    update(r['t'][-1]);fig.savefig(final.with_suffix('.png'),dpi=300)
    fig.savefig(final.with_suffix('.svg'));fig.savefig(final.with_suffix('.pdf'))
    plt.close(fig);gc.collect()
    with Image.open(target) as im:
        count=im.n_frames
        for j in range(count):im.seek(j);im.load()
    return dict(run=r['name'],gif=target.relative_to(REPORT).as_posix(),frames=count,decoded=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');args=p.parse_args();folder=Path(args.folder).resolve();REPORT=folder
    data=np.load(folder/'execution.npz')
    r=dict(name=folder.name,folder=folder,t=data['time'],x=data['state'])
    for key,filename in [('summary','summary'),('course','course'),('events','events'),('front','frontends')]:
        r[key]=json.loads((folder/(filename+'.json')).read_text(encoding='utf-8'))
    print(animate(r))
