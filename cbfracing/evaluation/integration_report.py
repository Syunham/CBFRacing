"""Audit an integrated full flight and save a standalone result report."""
from cbfracing.core.context import *
from cbfracing.evaluation.audit_neural import verify as verify_safety
from cbfracing.evaluation.verify_policy import verify as verify_allocation
import argparse,html

def main(folder):
    folder=Path(folder).resolve();safety=verify_safety(folder);learning=verify_allocation(folder)
    cfg=json.loads((folder/'config.json').read_text());summary=json.loads((folder/'summary.json').read_text());physical=json.loads((folder/'audit.json').read_text())
    events=json.loads((folder/'events.json').read_text());data=np.load(folder/'execution.npz');t=data['time'];speed=np.linalg.norm(data['state'][:,:,3:6],axis=-1)
    finish=[];laps=[];means=[]
    for i,ee in enumerate(events):
        if len(ee)>=14:
            end=ee[13]['time'];finish.append(end);laps.append([ee[6]['time'],end-ee[6]['time']]);mask=t<end
            means.append(float(np.trapezoid(np.r_[speed[mask,i],np.interp(end,t,speed[:,i])],np.r_[t[mask],end])/end))
        else:finish.append(None);laps.append(None);means.append(None)
    provenance=json.loads((HERE/'PROVENANCE.json').read_text())
    hashes={name:hashlib.sha256((HERE/path).read_bytes()).hexdigest() for name,path in [('allocation','models/allocation.pt'),('time_initializer','models/time_initializer.pt'),('direct_cbf','models/direct_cbf_v2/h.pt')]}
    unchanged=all(hashes[k]==provenance[k+'_sha256'] for k in hashes)
    result=dict(passed=bool(safety['physical_and_sampled_cbf_passed'] and learning['all_points_recomputed_from_network'] and unchanged),
        summary=summary,physical=physical,neural=safety,learning=learning,finish_times=finish,lap_times=laps,mean_speeds=means,
        vehicle=cfg['dynamics'],preview_gates=cfg['preview_gates'],frontend=cfg['frontend'],progress_weight=cfg['progress'],
        event_checkpoint=cfg.get('event_checkpoint'),
        horizon_seconds=cfg['horizon']*cfg['dt'],feedback_seconds=cfg['feedback'],checkpoints_unchanged=unchanged,checkpoint_hashes=hashes)
    save_json(folder/'integration_result.json',result)
    text=f'''# CBFracing整合验证

质量{cfg['dynamics']['mass']}kg、臂长{cfg['dynamics']['arm']}m、惯量{cfg['dynamics']['inertia']}kg m²、单旋翼最大推力{cfg['dynamics']['thrust_max']}N、角速度范数{cfg['body_rate_norm_limit']}rad/s。MPCC和执行采用机体系线性阻力{cfg['dynamics']['drag']}kg/s。Minimum Snap生成运动学参考，由完整动力学MPCC执行并核验物理约束。

默认采用{cfg['preview_gates']}门预览、{cfg['frontend']}前端、C2空间参考、最多{cfg['reference_refine']}次内部参考重算、物理过门交点约束、MPCC进度奖励{cfg['progress']}。每通过当前门后重新规划。MPCC时域{cfg['horizon']*cfg['dt']}s、反馈{cfg['feedback']}s。共享穿越点网络逐门输出，没有增加额外航路点；三个checkpoint保持原值。

|检查|结果|
|---|---|
|完整物理和实际CBF采样核验|{result['passed']}|
|每机过门数|{summary['gates_passed']}|
|两机完成两圈总用时|{summary['duration']:.2f}s|
|最高速度（执行采样）|{summary['max_speed']:.4f}m/s|
|最高速度（加密核验）|{physical['max_speed']:.4f}m/s|
|最小机间距（加密核验）|{physical['actual_dense_min_distance']:.6f}m|
|最大角速度范数|{physical['max_rate_norm']:.6f}rad/s|
|最大旋翼推力|{physical['max_force']:.6f}N|
|实际最小h|{safety['minimum_actual_h']}|
|0.02s实际下降率违规次数|{safety['actual_discrete_violation_count']}|
|0.08s实际下降率违规次数|{safety['actual_prediction_interval_violation_count']}|
|冻结分配网络重算通过|{learning['all_points_recomputed_from_network']}，{learning['queries']}次规划|
|模型参数来源|本目录config.json|

各机平均速度：{means}m/s；各机两圈时间：{laps}s。

求解中位/95分位耗时：{summary['controller_p50_ms']:.1f}/{summary['controller_p95_ms']:.1f}ms，前端累计{summary['frontend_seconds']:.2f}s。计算延迟未推进物理仿真。结果为当前初始场景的数值验证，不构成连续时间不变集证明、实时部署或跨场景泛化证明。

phase前端固定航门事件网格并优化航段时长。original前端求解互补CPC。minimum-snap前端使用时间网络分配五段时长并求解凸QP。模型、配置与实际执行结果分别保存在本目录。

[动图](rollout.gif) · [完整审计](integration_result.json) · [速度数据](speed_source_data.csv)
'''
    (folder/'REPORT.md').write_text(text,encoding='utf-8')
    (folder/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>CBFracing整合验证</title><style>body{font:16px Arial,sans-serif;max-width:1300px;margin:25px auto;padding:15px}pre{white-space:pre-wrap;line-height:1.6}img{max-width:100%}</style><h1>CBFracing整合验证</h1><p><a href="integration_result.json">完整审计</a> · <a href="speed_source_data.csv">速度数据</a> · <a href="rollout.gif">动图</a></p><img src="rollout.gif"><pre>'+html.escape(text)+'</pre></html>',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],gates=summary['gates_passed'],duration=summary['duration'],peak=summary['max_speed'],minimum_distance=physical['actual_dense_min_distance'])),flush=True)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');main(parser.parse_args().folder)
