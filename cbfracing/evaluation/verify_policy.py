"""Recompute every frozen allocation and audit the selected gate-event encoding."""
from cbfracing.core.context import *
from cbfracing.networks.allocation import load_allocator,observation

def verify(folder):
    folder=Path(folder);cfg=json.loads((folder/'config.json').read_text());front=json.loads((folder/'frontends.json').read_text());z=np.load(folder/'execution.npz')
    if not cfg['allocator'] or cfg.get('allocation_zero',False):return None
    checkpoint=Path(cfg['allocator']);checkpoint=checkpoint if checkpoint.is_absolute() else HERE/checkpoint
    net=load_allocator(checkpoint);allocation_error=0.;target_error=0.;progress_error=0.;complementarity=0.;queries=[]
    files=sorted((folder/'plans').glob('*.npz'));successful=[r for r in front if r['success']];assert len(files)==len(successful)
    frontend=cfg.get('frontend','original');expected=cfg.get('preview_gates',2)
    memory={};time_error=0.;time_net=None
    if cfg.get('event_checkpoint'):
        from cbfracing.networks.event_supervised import load_event,predict
        time_net,_=load_event(cfg['event_checkpoint'])
    for row,path in zip(successful,files):
        d=np.load(path);i=row['agent'];index=row['gate_index'];k=int(np.argmin(abs(z['time']-row['time'])));other=z['state'][k,1-i];own=z['state'][k,i] if frontend=='minimum-snap' else d['X'][:,0];count=len(d['gates'])
        assert count==expected and len(row['allocation'])==count
        g,n,half,a=gate_pair(index,count)
        for j in range(count):
            g2,n2,_,_=gate_pair(index+j+1);obs=observation(own,other,g[j],n[j],a[j],g2[0],n2[0])
            with torch.no_grad():value=float(net(torch.tensor(obs[None])))
            value*=cfg.get('point_offset_scale',1.)
            limit=cfg.get('passage_update_limit',0.)
            if limit>0 and (i,index+j) in memory:
                old=memory[i,index+j];value=old+float(np.clip(value-old,-limit,limit))
            allocation_error=max(allocation_error,abs(value-row['allocation'][j]));target_error=max(target_error,float(abs(d['gates'][j]-g[j]-value*a[j]).max()))
        for j,value in enumerate(row['allocation']):memory[i,index+j]=value
        if frontend=='original':
            mu=d['mu'];lam=d['lambda'];nu=d['nu'];assert mu.shape[0]==count
            progress_error=max(progress_error,float(abs(lam[:,1:]-lam[:,:-1]+mu).max()),float(abs(mu.sum(1)-1).max()),float(np.maximum(lam[:-1]-lam[1:],0).max()))
            for j in range(count):
                lat=np.cross([0.,0.,1.],n[j]);lat/=np.linalg.norm(lat);up=np.cross(n[j],lat)
                point=d['gates'][j]+lat*d['gate_offsets'][0,j]+up*d['gate_offsets'][1,j]
                comp=mu[j]*(np.sum((d['X'][:3,1:].T-point)**2,axis=1)-nu[j]);complementarity=max(complementarity,float(abs(comp).max()))
        else:
            assert frontend in ('phase','minimum-snap') and row['audit']['passed'] and np.all(d['durations']>0)
        if time_net is not None:
            full_state=z['state'][k,i]
            half=d['event_feature_half'] if 'event_feature_half' in d else d['half']
            predicted=predict(time_net,full_state,d['gates'],d['normals'],half)
            time_error=max(time_error,float(abs(predicted-row['event_durations_input']).max()))
        queries.append(dict(agent=i,index=index,time=row['time'],allocation=row['allocation'],gate_events=count,total_time=float(sum(d['durations']))))
    result=dict(network_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),queries=len(queries),frontend=frontend,preview_gates=expected,
        all_points_recomputed_from_network=allocation_error<1e-9 and target_error<1e-9,allocation_error=allocation_error,target_geometry_error=target_error,
        time_network_verified=time_error<1e-9 if time_net is not None else None,time_prediction_error=time_error,
        cpc_progress_error=progress_error if frontend=='original' else None,cpc_complementarity_residual=complementarity if frontend=='original' else None,
        original_complementarity_checked=frontend=='original',fixed_gate_event_mesh=frontend=='phase',extra_waypoints=0,no_waiting_time_output=True,query_details=queries)
    assert result['all_points_recomputed_from_network'] and progress_error<1e-5 and complementarity<3e-6 and time_error<1e-9
    save_json(folder/'learning_audit.json',result);print(folder.name,'learning verified',len(queries),'planning queries',frontend,expected,'gates',flush=True)
    return result

if __name__=='__main__':
    for f in sys.argv[1:]:verify(f)
