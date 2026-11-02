import sys,json,html
from pathlib import Path
R=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R))
from context import np,save_json
from audit_neural import verify
from verify_policy import verify as verify_policy
rows=[]
for name in ['integrated_drag_preview5_mu220_20261102','progress300_20261102','progress400_20261102','progress500_20261102']:
    p=R/'outputs'/name
    if not (p/'summary.json').exists():continue
    if not (p/'neural_audit.json').exists():verify(p);verify_policy(p)
    s=json.loads((p/'summary.json').read_text());c=json.loads((p/'config.json').read_text());a=json.loads((p/'audit.json').read_text());h=json.loads((p/'neural_audit.json').read_text());e=json.loads((p/'events.json').read_text())
    rows.append(dict(name=name,weight=c['progress'],success=s['success'] and a['full_goal_case_passed'],cbf_pass=h['actual_sampled_cbf_conditions_passed'],gates=s['gates_passed'],time=s['duration'],peak=s['max_speed'],min_distance=a['actual_dense_min_distance'],h_min=h['minimum_actual_h'],decay02=h['actual_discrete_violation_count'],decay08=h['actual_prediction_interval_violation_count'],lap_times=[[v[6]['time'],v[13]['time']-v[6]['time']] if len(v)>=14 else None for v in e],stop=s['stop_reason'],controller_p50_ms=s['controller_p50_ms']))
best=min([v for v in rows if v['success'] and v['cbf_pass']],key=lambda v:v['time'])
out=R/'outputs/progress_sweep_20261102';out.mkdir(exist_ok=True);save_json(out/'metrics.json',rows);save_json(out/'best.json',best)
table='\n'.join(f"|{r['weight']:g}|{r['gates']}|{r['success']}|{r['time']:.2f}|{r['peak']:.3f}|{r['min_distance']:.3f}|{r['cbf_pass']}|{r['stop']}|" for r in rows)
text=f'''# MPCC进度奖励继续增大实验

以220为基线，只改变进度奖励为300、400、500。原尺度Split-S双机两圈；五门phase前端、1.6秒MPCC、0.02秒反馈、表中无人机参数、机体系线性阻力、冻结网络、所有物理和CBF约束均保持。权重的作用依赖整套目标项及其尺度，本文500实验是当前实现的参数扫描。

|权重|各机门数|物理完赛|执行时长s|峰值m/s|最小间距m|实际CBF采样通过|停止原因|
|---|---|---|---:|---:|---:|---|---|
{table}

完整物理及CBF采样核验均通过的最快配置：{best['weight']:g}，两机两圈总用时{best['time']:.2f}s。失败轨迹的执行时长和峰值不纳入成功排名。求解失败表示当前求解过程未找到可接受解，不能据此证明原问题无可行解。

每个配置为相同初始状态的一次确定性验证。计算延迟未推进飞行，耗时含并发资源影响，不作为实时硬件基准。CBF检查是实际联合状态的离散采样核验。
'''
for r in rows:text+=f"\n权重{r['weight']:g}：各机圈时{r['lap_times']}；最小h={r['h_min']}；0.02/0.08秒下降率违规次数={r['decay02']}/{r['decay08']}；控制求解中位耗时{r['controller_p50_ms']:.1f}ms。\n"
(out/'REPORT.md').write_text(text,encoding='utf-8')
links=' · '.join(f'<a href="../{r["name"]}/rollout.gif">权重{r["weight"]:g}动图</a>' for r in rows if r['success'])
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Progress weight sweep</title><style>body{font:16px Arial;max-width:1300px;margin:25px auto;padding:15px}pre{white-space:pre-wrap;line-height:1.7}img{max-width:100%}</style><h1>MPCC进度奖励扫描</h1><p>'+links+'</p><img src="../'+best['name']+'/rollout.gif"><pre>'+html.escape(text)+'</pre>',encoding='utf-8')
print(json.dumps(dict(rows=rows,best=best),ensure_ascii=False))
