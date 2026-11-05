"""Task-loss learning with a differentiable trajectory surrogate, no labels."""
from cbfracing.networks.allocation import *
import argparse,copy

def scenes(count,seed):   # dataset
    rng=np.random.default_rng(seed);c=course();own=[];other=[];inputs=[];gs=[];ns=[];axes=[];duration=[]
    for k in range(count):
        index=int(rng.integers(0,7));ids=np.array([index,index+1,index+2])%7  # 3gates
        g=np.array(c['centers'])[ids];n=np.array(c['normals'])[ids];a=np.array(c['coordinate_axes'])[ids]  # centre nomal axis
        previous_axis=np.array(c['coordinate_axes'][(index-1)%7]);mix=rng.uniform(0,1)
        relative_axis=mix*a[0]+(1-mix)*previous_axis;relative_axis/=np.linalg.norm(relative_axis)  # to generate drones
        midpoint=g[0]-rng.uniform(1.,7.)*n[0]+rng.normal(0,.12,3);sep=rng.uniform(.8,1.8)*rng.choice([-1,1])
        pair=np.zeros((2,19));pair[:,6]=1.;pair[:,13:17]=MODEL['mass']*MODEL['gravity']/4
        pair[0,:3]=midpoint+.5*sep*relative_axis+rng.normal(0,.025,3);pair[1,:3]=midpoint-.5*sep*relative_axis+rng.normal(0,.025,3)
        speed=rng.uniform(2.,7.);mean=speed*n[0]+rng.normal(0,.2,3);dv=rng.normal(0,.25,3) 
        pair[0,3:6]=mean+.5*dv;pair[1,3:6]=mean-.5*dv  # initial states p v(rand) Q w f1-4 mu
        zz=np.array([[observation(pair[i],pair[1-i],g[j],n[j],a[j],g[j+1],n[j+1]) for j in range(2)] for i in range(2)])  # inputs
        inputs.append(zz);own.append(pair[:,:3]);other.append(pair[:,3:6]);gs.append(g[:2]);ns.append(n[:2]);axes.append(a[:2])
        duration.append([np.clip(np.linalg.norm(g[0]-midpoint)/speed,.65,3.),np.clip(np.linalg.norm(g[1]-g[0])/speed,.65,3.)])
    return {k:torch.tensor(np.array(v),dtype=torch.float64) for k,v in dict(z=inputs,p0=own,v0=other,g=gs,n=ns,a=axes,T=duration).items()}

def hermite(p0,p1,v0,v1,T,s):  # quick traj for fast initial training convergence
    s=s[None,None,:,None];T=T[:,None,None,None]
    pos=(2*s**3-3*s**2+1)*p0[:,:,None]+(s**3-2*s**2+s)*T*v0[:,:,None]+(-2*s**3+3*s**2)*p1[:,:,None]+(s**3-s**2)*T*v1[:,:,None]
    vel=((6*s*s-6*s)*p0[:,:,None]+(3*s*s-4*s+1)*T*v0[:,:,None]+(-6*s*s+6*s)*p1[:,:,None]+(3*s*s-2*s)*T*v1[:,:,None])/T
    acc=((12*s-6)*p0[:,:,None]+(6*s-4)*T*v0[:,:,None]+(-12*s+6)*p1[:,:,None]+(6*s-2)*T*v1[:,:,None])/(T*T)
    return pos,vel,acc

def objective(net,data,region_aware=True,center_weight=.015):
    z=data['z'];b=z.shape[0];offset=net(z.reshape(-1,24)).reshape(b,2,2)  # output
    points=data['g'][:,None,:,:]+offset[:,:,:,None]*data['a'][:,None,:,:]  # real gate positions
    speed=torch.linalg.vector_norm(data['v0'].mean(1),dim=-1).clamp(2,7)
    vg=data['n'][:,None,:,:]*speed[:,None,None,None]
    s=torch.linspace(0,1,41,dtype=torch.float64);positions=[];accelerations=[];radii=[]
    for j in range(2):
        start=data['p0'] if j==0 else points[:,:,0];vstart=data['v0'] if j==0 else vg[:,:,0].expand(-1,2,-1)
        pos,vel,acc=hermite(start,points[:,:,j],vstart,vg[:,:,j].expand(-1,2,-1),data['T'][:,j],s)
        positions.append(pos);accelerations.append(acc)
        r=pos[:,0]-pos[:,1];d=torch.sqrt(r.square().sum(-1)+1e-8);direction=r/d[:,:,None]
        # Worst-direction support of the two CPC aperture rectangles. This is
        # a surrogate robustness margin derived from physical region bounds,
        # not a supervised desired position or a preferred passing direction.
        h01=-2*s**3+3*s**2;h00=2*s**3-3*s**2+1
        support=.2*h01[None,:]*(abs((direction*data['a'][:,j,None,:]).sum(-1))+abs(direction[:,:,2]))
        if j:
            support+=.2*h00[None,:]*(abs((direction*data['a'][:,0,None,:]).sum(-1))+abs(direction[:,:,2]))
        radii.append(support if region_aware else torch.zeros_like(support))
    pos=torch.cat(positions,dim=2);acc=torch.cat(accelerations,dim=2);r=pos[:,0]-pos[:,1];d=torch.sqrt(r.square().sum(-1)+1e-8)
    support=torch.cat(radii,dim=1);robust=d-support
    collision=torch.relu(.55-robust).square().mean()
    # Continuous minimum along sampled curve chords prevents hiding a crossing
    # between display/training samples. Its geometry-only gradient may vanish
    # at a collinear collision; the motion cost resolves that degeneracy.
    dr=r[:,1:]-r[:,:-1];frac=(-(r[:,:-1]*dr).sum(-1)/(dr.square().sum(-1)+1e-8)).clamp(0,1)
    closest=r[:,:-1]+frac[:,:,None]*dr;minimum=torch.sqrt(closest.square().sum(-1)+1e-8)
    chord_support=torch.maximum(support[:,:-1],support[:,1:]);crossing=torch.relu(.55+chord_support-minimum).square().mean()
    motion=acc.square().sum(-1).mean()/100.
    centering=offset.square().mean()
    loss=150*(collision+crossing)+.05*motion+center_weight*centering
    metrics=dict(loss=float(loss.detach()),collision=float(collision.detach()),crossing=float(crossing.detach()),motion=float(motion.detach()),
        region_unsafe_fraction=float((robust.min(1).values<.55).double().mean()),minimum_nominal_distance=float(d.detach().min()),mean_abs_offset=float(offset.detach().abs().mean()))
    return loss,metrics

def main(a):
    out=HERE/'models'/a.name;out.mkdir(parents=True,exist_ok=False);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    net=GateAllocator().double();save_allocator(out/'initial.pt',net,seed=a.seed,updates=0,stage='random baseline')
    train=scenes(4096,20261029);valid=scenes(1024,20261030)
    # Store only unlabeled states/features; no allocation targets are generated.
    if not (HERE/'data/unlabeled_states.npz').exists():
        (HERE/'data').mkdir(exist_ok=True);np.savez_compressed(HERE/'data/unlabeled_states.npz',**{'train_'+k:v.numpy() for k,v in train.items()},**{'validation_'+k:v.numpy() for k,v in valid.items()})
    opt=torch.optim.Adam(net.parameters(),lr=.001);history=[];best=None;score=np.inf;start=time.perf_counter()
    with torch.no_grad():_,initial_metrics=objective(net,valid,a.region_aware,a.center_weight)
    for k in range(a.steps):
        ids=rng.choice(4096,192,replace=False);batch={key:value[ids] for key,value in train.items()};loss,metrics=objective(net,batch,a.region_aware,a.center_weight)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
        if k%100==0 or k==a.steps-1:
            with torch.no_grad():_,vm=objective(net,valid,a.region_aware,a.center_weight)
            row=dict(step=k,train=metrics,validation=vm);history.append(row)
            if vm['loss']<score:score=vm['loss'];best=copy.deepcopy(net.state_dict());best_step=k
            if k%500==0:print(a.name,row,flush=True)
    save_allocator(out/'last.pt',net,seed=a.seed,updates=a.steps,stage='Unlabeled differentiable Hermite task loss',region_aware=a.region_aware)
    net.load_state_dict(best);save_allocator(out/'allocator.pt',net,seed=a.seed,updates=a.steps,best_step=best_step,stage='Unlabeled differentiable Hermite task loss',region_aware=a.region_aware)
    with torch.no_grad():_,final=objective(net,valid,a.region_aware,a.center_weight)
    save_json(out/'metrics.json',dict(seed=a.seed,steps=a.steps,best_step=best_step,region_aware=a.region_aware,center_weight=a.center_weight,teacher_labels_used=False,
        pretrained_allocator_used=False,training_seconds=time.perf_counter()-start,initial=initial_metrics,final=final,history=history,
        scope='Differentiable trajectory surrogate with task collision/motion loss; not full rigid-body closed-loop training'))
    print('TRAINED',a.name,final,flush=True)

def cli(argv=None):
    selector=argparse.ArgumentParser(add_help=False)
    selector.add_argument('--task',choices=['allocation','cbf','time'],default='allocation',help='Train gate allocation, direct scalar CBF, or five-event times')
    selected,_=selector.parse_known_args(argv)
    if selected.task=='time':
        from cbfracing.training.events import build_parser,main as train_events
        train_events(build_parser(parents=[selector]).parse_args(argv))
    elif selected.task=='cbf':
        from cbfracing.training.safety_train import build_parser,main as train_cbf
        parser=build_parser(parents=[selector]);train_cbf(parser.parse_args(argv))
    else:
        p=argparse.ArgumentParser(parents=[selector]);p.add_argument('--name',default='from_scratch');p.add_argument('--seed',type=int,default=31);p.add_argument('--steps',type=int,default=2500);p.set_defaults(region_aware=True);p.add_argument('--center-weight',type=float,default=.015);main(p.parse_args(argv))

if __name__=='__main__':cli()
