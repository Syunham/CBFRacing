"""Check the frozen network actually creates every saved CPC gate allocation."""
from context import *
from allocation import load_allocator,observation
import argparse

def verify(folder):
    folder=Path(folder);cfg=json.loads((folder/'config.json').read_text());front=json.loads((folder/'frontends.json').read_text());z=np.load(folder/'execution.npz');times=z['time'];states=z['state']
    if not cfg['allocator'] or cfg.get('allocation_zero',False):return None
    checkpoint=Path(cfg['allocator']);checkpoint=checkpoint if checkpoint.is_absolute() else HERE/checkpoint
    net=load_allocator(checkpoint);allocation_error=0.;target_error=0.;progress_error=0.;complementarity=0.;queries=[]
    files=sorted((folder/'plans').glob('*.npz'));assert len(files)==sum(r['success'] for r in front)
    for row,path in zip(front,files):
        if not row['success']:continue
        d=np.load(path);i=row['agent'];index=row['gate_index'];k=int(np.argmin(abs(times-row['time'])));other=states[k,1-i];own=d['X'][:,0];g,n,half,a=gate_pair(index)
        for j in range(2):
            g2,n2,_,_=gate_pair(index+j+1);obs=observation(own,other,g[j],n[j],a[j],g2[0],n2[0])
            with torch.no_grad():value=float(net(torch.tensor(obs[None])))
            allocation_error=max(allocation_error,abs(value-row['allocation'][j]));target_error=max(target_error,float(abs(d['gates'][j]-g[j]-value*a[j]).max()))
        mu=d['mu'];lam=d['lambda'];nu=d['nu'];assert mu.shape[0]==2 and d['gates'].shape==(2,3)
        progress_error=max(progress_error,float(abs(lam[:,1:]-lam[:,:-1]+mu).max()),float(abs(mu.sum(1)-1).max()),float(np.maximum(lam[0]-lam[1],0).max()))
        for j in range(2):
            lat=np.cross([0.,0.,1.],n[j]);lat/=np.linalg.norm(lat);up=np.cross(n[j],lat)
            point=d['gates'][j]+lat*d['gate_offsets'][0,j]+up*d['gate_offsets'][1,j]
            comp=mu[j]*(np.sum((d['X'][:3,1:].T-point)**2,axis=1)-nu[j]);complementarity=max(complementarity,float(abs(comp).max()))
        queries.append(dict(agent=i,index=index,time=row['time'],allocation=row['allocation'],mu_rows=2,total_time=float(d['z'][-1])))
    result=dict(network_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),queries=len(queries),
        all_points_recomputed_from_network=allocation_error<1e-9 and target_error<1e-9,allocation_error=allocation_error,target_geometry_error=target_error,
        cpc_progress_error=progress_error,cpc_complementarity_residual=complementarity,two_physical_events_only=True,
        no_waiting_time_output=True,source='Direct recomputation using saved physical own initial state, simultaneous observed neighbor PV and frozen checkpoint',query_details=queries)
    assert result['all_points_recomputed_from_network'] and progress_error<1e-5 and complementarity<3e-6
    save_json(folder/'learning_audit.json',result);print(folder.name,'learning verified',len(queries),'CPC queries',flush=True)
    return result

if __name__=='__main__':
    for f in sys.argv[1:]:verify(f)
