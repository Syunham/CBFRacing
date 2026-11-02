"""Durable post-merge CLI validation; all numerical logs stay local."""
import sys,subprocess,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import save_json

if __name__=='__main__':
    branch=sys.argv[1];name='integrated_'+branch+'_20261030';target=14 if branch=='learned' else 2
    record=ROOT/'checks'/f'{name}.process.json'
    if record.exists():raise FileExistsError(record)
    command=[sys.executable,str(ROOT/'validate.py'),'--name',name,'--cbf',branch]
    if target!=14:command+=['--target-gates',str(target)]
    tick=time.perf_counter()
    with (ROOT/'checks'/f'{name}.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        save_json(record,dict(status='running',pid=child.pid,command=command))
        try:code=child.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            child.kill();child.wait();save_json(record,dict(status='wall_budget_exhausted',pid=child.pid,command=command));raise
    save_json(record,dict(status='finished',pid=child.pid,returncode=code,command=command,seconds=time.perf_counter()-tick))
    if code:raise RuntimeError('Inspect '+str(record.with_suffix('.log')))
    folder=ROOT/'outputs'/name
    for script in ['audit_neural.py','verify_policy.py','render.py']:
        with (ROOT/'checks'/f'{name}_{Path(script).stem}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/script),str(folder)],cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
    summary=json.loads((folder/'summary.json').read_text());assert summary['success'] and min(summary['gates_passed'])>=target
    if branch=='learned':assert json.loads((folder/'neural_audit.json').read_text())['full_goal_case_passed']
    print('INTEGRATION PASSED',branch,summary,flush=True)
