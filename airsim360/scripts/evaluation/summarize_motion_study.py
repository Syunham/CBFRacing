"""Collect offline and ordered gate metrics without selecting flights by their outcome."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import hashlib
from pathlib import Path
import numpy as np
from scripts.planning.utils.race_progress import progress

ROOT=Path('reports/spatial_potential')


def read(path):
    return json.loads(path.read_text())


def summarize():
    out=ROOT/'cf21_motion_study'
    out.mkdir(exist_ok=True)
    baseline=['cf21_bendhead03_run1','cf21_bendhead03_run2','cf21_local_future_two_laps01']
    files=[*baseline,*sorted(p.name for p in ROOT.glob('cf21_motion_*_race*') if (p/'flight/summary.json').exists())]
    races={}
    for name in files:
        p=ROOT/name
        if not (p/'flight/spatial_audit.json').exists():
            continue
        a=read(p/'flight/spatial_audit.json')
        cfg=read(p/'experiment.json')
        if name.startswith('cf21_motion_'):
            reference=read(ROOT/'cf21_local_future_two_laps01/experiment.json')
            settings=('separation','stop_separation','weight','passage_gain','passage_weight','passage_head_sha256',
                      'checkpoint_sha256','rti_cold_start','recover_frontend','guarded_fast_fly','lane_limit_m')
            if any(cfg[k]!=reference[k] for k in settings):
                raise ValueError(f'Non-model controls differ in {name}')
            if hashlib.sha256(Path(cfg['future_checkpoint']).read_bytes()).hexdigest()!=cfg['future_sha256']:
                raise ValueError(f'Checkpoint mismatch in {name}')
        c=a['ordered_crossing_times_s'];n=min(map(len,c))
        delta=1000*(np.asarray(c[0][:n])-c[1][:n])
        health=Path('logs')/(name+'-telemetry.jsonl')
        h=[json.loads(l) for l in health.read_text().splitlines()] if health.exists() else []
        races[name]=dict(ordered=a['ordered_crossings'],mean_ms=float(abs(delta).mean()) if n else None,
                         max_ms=float(abs(delta).max()) if n else None,gate_gaps_ms=delta.tolist(),
                         minimum_distance_m=a['minimum_separation_m'],frame_center_m=a['minimum_frame_clearance_m'],
                         strict_pass=a['passed'],progress=progress(p) if n==14 else None,
                         weights=cfg.get('future_checkpoint',cfg.get('time_checkpoint')),flight_audit_file=str(p/'flight/spatial_audit.json'),
                         max_cpu_c=max((r['cpu_c'] for r in h),default=None),new_whea=any(r['whea_record']>r['whea_baseline'] for r in h))
    models={n:read(ROOT/n/'results.json') for n in ('cf21_motion_attention02','cf21_motion_fragment01','cf21_motion_history01','cf21_motion_mixture01') if (ROOT/n/'results.json').exists()}
    report=dict(races=races,models=models,progress_definition='Travelled arc normalized between ordered gate crossings; both vehicles faster than 2 m/s, through first finish.')
    (out/'comparison.json').write_text(json.dumps(report,indent=2))
    return races,out


if __name__=='__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    races,out=summarize()
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    names={'cf21_bendhead03_run1':'原 CNN–GRU · 第一次','cf21_bendhead03_run2':'原 CNN–GRU · 第二次',
           'cf21_local_future_two_laps01':'上一版局部动态风险场'}
    labels={'attention02':'检索加权','fragment01':'轨迹片段修正','history01':'加入短时运动','mixture01':'三种运动假设'}
    for n,data in races.items():
        if n in names: label=names[n]
        else:
            model,index=n.removeprefix('cf21_motion_').split('_race')
            label=labels.get(model,model)+(' · 复跑' if index!='1' else '')
        ax.plot(np.arange(1,len(data['gate_gaps_ms'])+1),data['gate_gaps_ms'],marker='.',label=label)
    ax.axhline(0,color='0.35',lw=1)
    ax.axvline(7.5,color='0.7',ls=':')
    ax.set(xlabel='累计过门序号（第 8 门开始第二圈）',ylabel='0 号机减 1 号机过门时间（毫秒）',xticks=np.arange(1,15))
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.savefig(out/'gate_times.png',dpi=150);plt.close(fig)
