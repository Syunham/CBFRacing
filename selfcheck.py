"""Check relocation, checkpoint identity and local task-loss derivatives."""
from train import *
import importlib

def main():
    local_names=['context','model','allocation','train','planning','controller','reference',
        'phase_planner','original_cpc','timing','seed','warm_seed','cli_profiles','barrier','safety_train','audit_neural']
    sources={}
    for name in local_names:
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        assert path.parent==HERE,(name,path)
        sources[name]=path.name
    provenance=json.loads((HERE/'PROVENANCE.json').read_text())
    assert hashlib.sha256((HERE/'models/allocation.pt').read_bytes()).hexdigest()==provenance['allocation_sha256']
    assert hashlib.sha256((HERE/'models/time_initializer.pt').read_bytes()).hexdigest()==provenance['time_initializer_sha256']
    assert hashlib.sha256((HERE/'models/direct_cbf_v2/h.pt').read_bytes()).hexdigest()==provenance['direct_cbf_sha256']
    from barrier import load as load_h,check as check_h
    check_h(load_h(HERE/'models/direct_cbf_v2/h.pt')[0],HERE/'checks/direct_cbf_export.json')
    net=load_allocator(HERE/'models/allocation.pt');state=initial_states();g,n,_,a=gate_pair(0)
    z=observation(state[0],state[1],g[0],n[0],a[0],g[1],n[1]);reverse=np.r_[-z[:6],z[6:]]
    with torch.no_grad():antisymmetry=float(abs(net(torch.tensor(z[None]))+net(torch.tensor(reverse[None]))).max())
    assert antisymmetry<1e-12
    torch.manual_seed(111);fresh=GateAllocator().double();batch=scenes(24,171)
    loss,_=objective(fresh,batch,True);loss.backward();parameter=fresh.net[0].weight;grad=parameter.grad.clone();errors=[]
    for row,col in [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(8,6),(12,15)]:
        original=float(parameter[row,col].detach());eps=1e-6
        with torch.no_grad():
            parameter[row,col]=original+eps;plus=float(objective(fresh,batch,True)[0])
            parameter[row,col]=original-eps;minus=float(objective(fresh,batch,True)[0]);parameter[row,col]=original
        errors.append(abs((plus-minus)/(2*eps)-float(grad[row,col])))
    assert max(errors)<2e-4
    result=dict(local_module_sources=sources,checkpoint_hashes_match=True,antisymmetry_error=antisymmetry,
        maximum_local_gradient_error=max(errors),gradient_entries=len(errors),scope='Local derivative entries and runtime relocation check')
    save_json(HERE/'checks/selfcheck.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
