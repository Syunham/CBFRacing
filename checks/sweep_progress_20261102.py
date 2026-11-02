"""Bounded progress-weight trials; keep defaults and all physical constraints."""
import os,sys,json,time,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parent.parent
def run(weight):
    name=f'progress{weight}_20261102';out=R/'outputs'/name
    if out.exists():return
    command=[sys.executable,str(R/'validate.py'),'--name',name,'--progress',str(weight),'--max-time','25']
    row=dict(command=command,timeout_seconds=1800);start=time.perf_counter()
    with (R/'checks'/f'{name}.log').open('w',encoding='utf-8') as f:
        try:row['exit_code']=subprocess.run(command,cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
        except subprocess.TimeoutExpired:row['timed_out']=True
    row['wall_seconds']=time.perf_counter()-start
    if (out/'summary.json').exists():row['summary']=json.loads((out/'summary.json').read_text())
    (R/'checks'/f'{name}_exit.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    print(json.dumps(row),flush=True)
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,[300,400,500]))
