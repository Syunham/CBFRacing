"""Compile this exact CBF OCP's derivatives and a Hessian-free correction kernel."""
from cbfracing.core.context import *
from cbfracing.control.rti import RTIMPCC
import subprocess,shutil
import argparse

def settings():
    os.environ.update(PHYSICAL_GATE_GUARD='1',CBF_HARD_STEPS='3',CBF_TAIL_WEIGHT='2000',RTI_DENSE_CBF='1',EXACT_GATE_APERTURE='1',RTI_BATCHED_AUDIT='1')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mx',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--samples',type=Path,nargs='+',required=True,help='Saved OCP npz files containing z/p or X/U/initial/od/weights/mu')
    args=ap.parse_args()
    if not all(p.is_file() for p in args.samples):ap.error('Every verification sample must exist')
    from validate import parse_args
    from cbfracing.core.runtime import configure
    configure(parse_args(['--name','compile','--frontend','minimum-snap','--rti-kernels','symbolic']))
    stem=os.environ['FAST_KERNEL_STEM']
    settings();s=RTIMPCC(0,n=20,cbf_checkpoint=str(HERE/'models/direct_cbf_v2/h.pt'),cbf_mode='hard',tube_recovery=True,dense_physics=True,gate_guard=os.getenv('OPT_LOCAL_GATE_WINDOW','0')=='1')
    correct=ca.Function('rti_correct',[s.z,s.p],[s.objective,s.g,ca.gradient(s.objective,s.z),ca.jacobian(s.g,s.z)])
    if args.mx:
        J=ca.jacobian(s.residuals,s.z)
        prep=ca.Function('rti_prepare',[s.z,s.p],[s.objective,s.g,ca.gradient(s.objective,s.z),ca.jacobian(s.g,s.z),2*J.T@J])
        ev=ca.Function('rti_evaluate',[s.z,s.p],[s.objective,s.g])
    else:prep=s.prepare;ev=s.evaluate;correct=correct.expand()
    key=s.native_key();path=HERE/'build';path.mkdir(exist_ok=True);old=os.getcwd();os.chdir(path)
    try:
        gen=ca.CodeGenerator(stem+'.c',dict(with_header=True));gen.add(prep);gen.add(ev);gen.add(correct);gen.generate()
    finally:os.chdir(old)
    cmd=[shutil.which('gcc'),'-O2','-shared','-o',str(path/(stem+'.dll')),str(path/(stem+'.c')),'-lm']
    tick=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
    record=dict(key=key,command=cmd,code=r.returncode,seconds=time.perf_counter()-tick,stderr=r.stderr)
    save_json(path/(stem+'_compile.json'),record)
    if r.returncode:raise RuntimeError(r.stderr)
    native=ca.external('rti_prepare',str(path/(stem+'.dll')));ne=ca.external('rti_evaluate',str(path/(stem+'.dll')));nc=ca.external('rti_correct',str(path/(stem+'.dll')))
    errors=[];times=[]
    samples=args.samples
    for f in samples:
        d=np.load(f)
        if 'z' in d and 'p' in d:z,p=d['z'],d['p']
        else:z,p=np.r_[d['X'].ravel(),d['U'].ravel()],np.r_[d['initial'],d['od'].ravel(),d['weights'],d['mu']]
        a=s.prepare(z,p);b=native(z,p);c=nc(z,p)
        error=max(float(abs(np.array(x)-np.array(y)).max(initial=0)) for x,y in zip(a,b))
        err_c=max(float(abs(np.array(x)-np.array(y)).max(initial=0)) for x,y in zip(a[:4],c))
        errors.append(max(error,err_c))
        row={}
        for name,fun in [('symbolic',s.prepare),('native',native),('correction',nc)]:
            ts=[]
            for _ in range(8):
                start=time.perf_counter();fun(z,p);ts.append(time.perf_counter()-start)
            row[name]=float(np.median(ts)*1000)
        times.append(row)
    record.update(max_absolute_error=max(errors),samples=len(errors),timings=times)
    assert max(errors)<1e-6,record
    save_json(path/(stem+'_compile.json'),record);print({k:v for k,v in record.items() if k!='timings'},flush=True)
