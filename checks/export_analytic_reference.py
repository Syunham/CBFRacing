"""Capture the pre-integration analytic NLP for a numerical regression check."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from context import *
from controller import RecoveryMPCC

if __name__=='__main__':
    output=HERE/'checks/analytic_nlp_before_direct.npz'
    if output.exists():raise FileExistsError(output)
    n=3;solver=RecoveryMPCC(0,n=n);rng=np.random.default_rng(20261030)
    x=rng.normal(size=(3,19*(n+1)+5*n));p=rng.normal(size=(3,19+70*(n+1)+14))
    f=solver.solver.get_function('nlp_f');g=solver.solver.get_function('nlp_g')
    np.savez_compressed(output,n=n,x=x,p=p,f=np.array([float(f(xx,pp)) for xx,pp in zip(x,p)]),g=np.array([np.asarray(g(xx,pp)).ravel() for xx,pp in zip(x,p)]),lbg=solver.lbg,ubg=solver.ubg,lbx=solver.lbx,ubx=solver.ubx)
    save_json(HERE/'checks/integration_before.json',{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['controller.py','planning.py','phase_planner.py','validate.py','train.py']})
    print('Saved pre-integration analytic NLP reference',output)
