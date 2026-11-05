"""Direct h learning with sign, finite-continuation and transition losses."""
from cbfracing.networks.barrier import *
import argparse,copy

def corpus(train_rollout,validation_rollout,output,extra=None,pose_anchors=False):
    train_path=Path(train_rollout).resolve()
    valid_path=Path(validation_rollout).resolve()
    paths=[train_path,valid_path]
    if extra:paths+=list(Path(extra).glob('*/execution.npz'));paths=[p.parent if p.suffix=='.npz' else p for p in paths]
    rows=[];manifest=[]
    for folder in paths:
        data=np.load(folder/'execution.npz');t=data['time'];x=data['state'];start=len(rows);split=int(folder==valid_path)
        distance=np.linalg.norm(x[:,0,:3]-x[:,1,:3],axis=-1)
        for k in range(0,len(t)-1,2):
            end=int(np.searchsorted(t,t[k]+.64-1e-9));complete=end<len(t)
            safe=bool(complete and distance[k:end+1].min()>.60);unsafe=distance[k]<.5
            for i in range(2):rows.append((features(x[k,i],x[k,1-i]),features(x[k+1,i],x[k+1,1-i]),float(t[k+1]-t[k]),safe,unsafe,0.,False,split))
        manifest.append(dict(folder=str(folder),sha256=hashlib.sha256((folder/'execution.npz').read_bytes()).hexdigest(),rows=len(rows)-start,validation=bool(split)))
    # Equal-motion geometry: both UAVs share velocity, attitude and input, so
    # their relative position remains constant under identical dynamics.
    for split,seed in [(0,1107),(1,2209)]:
        rng=np.random.default_rng(seed);count=18000 if split==0 else 4000
        for k in range(count):
            x=np.zeros(19);y=np.zeros(19);x[6]=y[6]=1.;x[13:17]=y[13:17]=MODEL['mass']*MODEL['gravity']/4
            r=rng.normal(size=3);r/=np.linalg.norm(r);d=rng.uniform(.12,1.15) if k%2 else rng.uniform(.2,6.)
            x[:3]=[0.,0.,3.];y[:3]=x[:3]-r*d;velocity=rng.uniform(-12,12,3);x[3:6]=y[3:6]=velocity
            if pose_anchors and k%3:
                roll,pitch=rng.uniform(-.6,.6,2);q=np.array([np.cos(roll/2)*np.cos(pitch/2),np.sin(roll/2)*np.cos(pitch/2),np.cos(roll/2)*np.sin(pitch/2),-np.sin(roll/2)*np.sin(pitch/2)])
                x[6:10]=y[6:10]=q;x[13:17]=y[13:17]=MODEL['mass']*MODEL['gravity']/4*rng.uniform(.8,1.3);x[5]=y[5]=rng.uniform(-2,2)
            z=features(x,y);rows.append((z,z,.02,d>.65,d<.5,float(np.tanh(1.2*(d-.58))),True,split))
        # Current collision labels with diverse velocity/attitude. No negative
        # recoverability label is inferred from a future collision or NLP error.
        for k in range(count//3):
            x=np.zeros(19);y=np.zeros(19);x[:3]=[0,0,3];r=rng.normal(size=3);r/=np.linalg.norm(r);y[:3]=x[:3]+rng.uniform(.15,.49)*r
            x[3:6]=rng.uniform(-15,15,3);y[3:6]=rng.uniform(-15,15,3);x[6:10]=rng.normal(size=4);x[6:10]/=np.linalg.norm(x[6:10])
            x[10:13]=rng.uniform(-5,5,3);x[13:17]=rng.uniform(.5,6.5,4)
            z=features(x,y);rows.append((z,z,.02,False,True,0.,False,split))
    arrays={name:np.array([r[j] for r in rows]) for j,name in enumerate(['z','next_z','dt','safe','unsafe','anchor_target','anchor','validation'])}
    output=Path(output);output.parent.mkdir(exist_ok=True,parents=True)
    if output.exists():raise FileExistsError(output)
    np.savez_compressed(output,**arrays)
    save_json(output.with_suffix('.json'),dict(trajectories=manifest,safe_window=.64,rows=len(rows),\
                                               label_rules='Complete observed safe continuation, current collision unsafe, others unknown; equal-motion states provide scale anchors',\
                                                validation='Distinct rollout and random seeds, same-course development validation'))
    return {k:torch.tensor(v,dtype=torch.float64) for k,v in arrays.items()}

def losses(net,d,derivative_weight=1.,unsafe_weight=5.):
    h=net(d['z']);hn=net(d['next_z']);safe=d['safe']>.5;unsafe=d['unsafe']>.5;anchor=d['anchor']>.5
    pos=torch.relu(.03-h[safe]).mean() if safe.any() else h.sum()*0
    neg=torch.relu(.03+h[unsafe]).mean() if unsafe.any() else h.sum()*0
    deriv=torch.relu(.005-(hn-h)/d['dt']-3*h)[safe].mean() if safe.any() else h.sum()*0
    fit=(h[anchor]-d['anchor_target'][anchor]).square().mean() if anchor.any() else h.sum()*0
    loss=5*pos+unsafe_weight*neg+derivative_weight*deriv+5*fit
    metrics=dict(loss=float(loss.detach()),safe_loss=float(pos.detach()),unsafe_loss=float(neg.detach()),derivative=float(deriv.detach()),scale_fit=float(fit.detach()),
        false_safe=int(((h>=0)&unsafe).sum()),unsafe=int(unsafe.sum()),false_unsafe=int(((h<0)&safe).sum()),safe=int(safe.sum()),anchor_mae=float(abs(h[anchor]-d['anchor_target'][anchor]).mean().detach()))
    return loss,metrics

def main(a):
    if a.dataset and a.rebuild:raise ValueError('Choose a saved dataset or rebuild the corpus')
    if a.steps<1:raise ValueError('Training steps must be positive')
    if a.rebuild and (not a.train_rollout or not a.validation_rollout):raise ValueError('--rebuild requires --train-rollout and --validation-rollout')
    data_path=HERE/'data'/f'{a.name}_dataset.npz' if a.rebuild else Path(a.dataset).resolve() if a.dataset else HERE/'data/direct_dataset_v2.npz'
    if not a.rebuild and not data_path.is_file():raise FileNotFoundError(data_path)
    out=HERE/'models'/a.name;out.mkdir(parents=True,exist_ok=False);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);net=DirectCBF().double()
    if a.initial:net.load_state_dict(load(a.initial)[0].state_dict())
    initial=copy.deepcopy(net.state_dict());torch.save(dict(width=32,state=initial,seed=a.seed,direct_output=True),out/'initial.pt')
    d=corpus(a.train_rollout,a.validation_rollout,data_path,a.extra,a.pose_anchors) if a.rebuild else {k:torch.tensor(v,dtype=torch.float64) for k,v in np.load(data_path).items()}
    tr=np.flatnonzero(d['validation'].numpy()==0);va=np.flatnonzero(d['validation'].numpy()==1);val={k:v[va] for k,v in d.items()};roll=tr[d['anchor'][tr].numpy()<.5]
    safe_roll=roll[d['safe'][roll].numpy()>.5]
    opt=torch.optim.Adam(net.parameters(),lr=.001);best=None;score=np.inf;history=[];tick=time.perf_counter()
    for k in range(a.steps):
        ix=np.r_[rng.choice(tr,384),rng.choice(safe_roll,128)];batch={key:v[ix] for key,v in d.items()}
        loss,m=losses(net,batch,a.derivative_weight,a.unsafe_weight);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
        if k%100==0 or k==a.steps-1:
            with torch.no_grad():_,vm=losses(net,val,a.derivative_weight,a.unsafe_weight)
            history.append(dict(step=k,train=m,validation=vm))
            if vm['loss']<score:score=vm['loss'];best=copy.deepcopy(net.state_dict());best_step=k
            if k%500==0:print(a.name,k,vm,flush=True)
    net.load_state_dict(best);torch.save(dict(width=32,state=best,seed=a.seed,steps=a.steps,\
                                              direct_output=True,output='h_theta=tanh(MLP(features)); no distance residual or learned margin',\
                                                source_initial=a.initial),out/'h.pt')
    import shutil
    (out/'source').mkdir(exist_ok=True)
    for path in [Path(__file__),HERE/'cbfracing/networks/barrier.py']:
        shutil.copy2(path,out/'source'/path.name)
    manifest_path=data_path.with_suffix('.json') if a.rebuild else data_path.parent/data_path.name.replace('direct_dataset','manifest').replace('.npz','.json')
    if manifest_path.exists():shutil.copy2(manifest_path,out/'dataset_manifest.json')
    save_json(out/'metrics.json',dict(seed=a.seed,steps=a.steps,best_step=best_step,derivative_weight=a.derivative_weight,unsafe_weight=a.unsafe_weight,\
                                      pose_anchors=a.pose_anchors,data_path=str(data_path),data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),\
                                        training_seconds=time.perf_counter()-tick,history=history,direct_output=True));check(net,out/'export_audit.json')
    print('TRAINED',a.name,score,flush=True)

def build_parser(parents=None):
    p=argparse.ArgumentParser(parents=parents or [],description='Train a direct scalar CBF; gate allocation and time initialization stay frozen')
    p.add_argument('--name',required=True);p.add_argument('--seed',type=int,default=7);p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--derivative-weight',type=float,default=3.);p.add_argument('--unsafe-weight',type=float,default=20.)
    p.add_argument('--pose-anchors',action='store_true',help='Add varied equal-motion poses when rebuilding the corpus')
    p.add_argument('--initial',help='Optional initial h checkpoint; omission starts from random weights')
    p.add_argument('--dataset',help='Frozen npz corpus; default is the bundled data/direct_dataset_v2.npz')
    p.add_argument('--rebuild',action='store_true',help='Build a new corpus from explicitly supplied rollouts')
    p.add_argument('--train-rollout');p.add_argument('--validation-rollout');p.add_argument('--extra',help='Additional rollout parent directory for --rebuild')
    return p

if __name__=='__main__':main(build_parser().parse_args())
