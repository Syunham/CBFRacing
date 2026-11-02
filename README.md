# CBFracing：门内分配学习 + CPC + MPCC/CBF

本目录包含解析CBF和直接输出学习CBF两个命令行分支，使用原尺度七门 Split-S、双机两圈。
源代码只导入本目录模块和 requirements.txt 中的第三方库，可以将整个目录复制到其他位置。

门内分配网络从随机权重训练，没有穿越点或先后次序的监督标签。保留共享反对称网络、
已知门几何及可微几何轨迹先验。附带的时间网络保持冻结，仅提供 CPC 初始猜测。

## 最短运行流程

在本目录打开 PowerShell，使用 Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe selfcheck.py

# 用附带的已验证模型重新进行双机两圈闭环验证
.\.venv\Scripts\python.exe validate.py --name demo
.\.venv\Scripts\python.exe audit.py outputs/demo
.\.venv\Scripts\python.exe verify_policy.py outputs/demo
.\.venv\Scripts\python.exe render.py outputs/demo

# 使用已验证的直接输出学习CBF v7分支
.\.venv\Scripts\python.exe validate.py --name learned_demo --cbf learned
.\.venv\Scripts\python.exe audit_neural.py outputs/learned_demo
.\.venv\Scripts\python.exe verify_policy.py outputs/learned_demo
.\.venv\Scripts\python.exe render.py outputs/learned_demo

# 从随机权重训练分配网络，再用新模型验证
.\.venv\Scripts\python.exe train.py --name scratch_s31 --seed 31 --steps 2500
.\.venv\Scripts\python.exe validate.py --name scratch_s31_eval --allocator models/scratch_s31/allocator.pt
.\.venv\Scripts\python.exe audit.py outputs/scratch_s31_eval
.\.venv\Scripts\python.exe render.py outputs/scratch_s31_eval
```

输出名必须使用新名字，已有模型和运行目录不会被覆盖。训练自动生成4096组训练状态和
1024组验证状态，batch为192组成对场景。`--target-gates 2` 可以运行短前缀检查；
默认每机14次过门。短前缀完成不代表两圈完成。
训练种子与轮次由 train.py 的命令行参数指定，config.json 用于数值规划器。

## 选择CBF分支

`validate.py --cbf analytic` 是原来的默认分支；`--cbf learned` 自动加载
`models/direct_cbf_v2/h.pt` 和 `learned_cbf_profile.json` 中的v7设置。
显式传入的命令行选项优先于分支默认值，实际生效配置保存在每次输出的config.json。

| 设置 | analytic（默认） | learned |
|---|---|---|
| 安全函数 | 固定距离函数 | 22→32→16→1 Tanh MLP直接输出完整hθ |
| 安全约束 | 距离硬约束，离散条件软惩罚 | hθ及离散条件均为硬约束 |
| 解析距离安全行 | 保留 | 移除 |
| 跟踪漏斗基础半宽 | 0.20m | 0.16m |
| 当前跟踪漏斗冲突恢复 | 关闭 | 保留未来漏斗，省去固定当前状态的漏斗行 |
| 控制间隔内部物理检查 | 原端点检查 | 额外检查5/10/15/20ms |
| MPCC单次迭代/秒预算 | 900/8 | 4000/35 |
| CPC初始化继续求解次数 | 0 | 最多2次 |

两个分支都保留完整刚体动力学、原尺度赛道、两门CPC事件、相同物理界和0.5m
机间距检查。当前门通过后重新规划，MPCC每0.02s更新。学习分支沿用已训练的
航门分配和冻结时间初值网络；新的安全网络直接输出hθ，没有可学习的额外裕度。

更换安全模型时使用：

```powershell
python validate.py --name custom_h_eval --cbf learned --cbf-checkpoint models/my_h/h.pt
```

## 训练分支

`train.py` 默认执行原航门分配训练，也可以显式传入 `--task allocation`。
`--task cbf` 执行直接安全函数训练，默认使用本目录附带的v2冻结数据集：

```powershell
# 随机初始化一个新的CBF；训练后需单独执行闭环验证
python train.py --task cbf --name my_h --seed 7 --steps 4000
python validate.py --name my_h_eval --cbf learned --cbf-checkpoint models/my_h/h.pt

# 重现已验证模型的第二轮训练
# initial.pt保存该轮开始前、已经过首轮训练的安全网络权重
python train.py --task cbf --name h_v2_retrain --seed 7 --steps 4000 --initial models/direct_cbf_v2/initial.pt --dataset data/direct_dataset_v2.npz

# 从随机权重完整重走两轮训练
python train.py --task cbf --name h_round1 --seed 7 --steps 4000 --dataset data/direct_dataset_v1.npz --unsafe-weight 5 --derivative-weight 1
python train.py --task cbf --name h_round2 --seed 7 --steps 4000 --dataset data/direct_dataset_v2.npz --initial models/h_round1/h.pt
```

CBF损失包含安全/危险符号、真实转移导数和等运动几何尺度锚点。v2默认危险项权重20、
导数项权重3。几何锚点属于固定训练先验，推理输出为整个神经函数。
数据和训练记录属于同赛道开发资料，包含公共状态前缀；完整闭环验证另行保存。

默认训练只读取附带的data/，运行与训练不依赖原实验项目。若要从新滚动数据重建
语料，显式提供 `--rebuild --train-rollout <目录> --validation-rollout <目录>`，
可增加 `--extra <包含其他运行的父目录>` 和 `--pose-anchors`。新语料另存到
`data/<训练名>_dataset.npz`，冻结v1/v2语料保持原样。

## 文件入口

| 文件 | 作用 |
|---|---|
| train.py | 随机初始化、无标签场景生成、任务损失训练 |
| safety_train.py / barrier.py | 直接安全函数训练、特征和优化器符号导出 |
| validate.py | 双机 CPC/MPCC 闭环执行，保存完整成功或失败日志 |
| cli_profiles.py / learned_cbf_profile.json | 解析/学习CBF命令行分支及默认设置 |
| render.py | 读取执行结果，生成 rollout.gif 和末帧 PNG/PDF/SVG |
| audit.py | 独立加密动力学回放、过门与安全间距检查 |
| audit_neural.py | 物理回放、实际学习h/离散残差和约束乘子核验 |
| verify_policy.py | 重算每次网络分配、CPC事件进度和互补残差 |
| selfcheck.py | 本地模块来源、checkpoint和局部训练梯度核验 |
| allocation.py | 共享24→48→48→1反对称分配网络 |
| context.py / config.json | 完整赛道、初始状态、CPC配置 |
| model.py | 完整13维刚体动力学及控制器增广状态模型 |
| planning.py / original_cpc.py | 网络分配区域、原始自由时间/事件 CPC |
| phase_planner.py / seed.py / warm_seed.py | CPC初值优化、几何初值、姿态退化修复 |
| controller.py | 完整非线性 MPCC及解析/学习CBF |
| reference.py | CPC空间参考和MPCC在线参数 |
| timing.py | 冻结的事件时间初值网络 |

## 保留的算法

每架无人机观测本机与邻机的位置速度及门几何，共享网络针对两个后续门分别输出
横向偏移 δ∈[−0.45,0.45]m。CPC以该偏移为门内区域中心，优化完整状态、推力、
最终时间和两行事件进度，不增加额外航路点。通过当前门后滚动规划。

训练使用两段可微Hermite曲线，损失为：

`L = 150*(collision + chord_collision) + 0.05*motion + 0.015*mean(delta**2)`

`motion = mean(||acceleration||²)/100`。避碰项以0.55m为阈值，叠加根据CPC区域
大小推导的几何裕量。梯度经过几何代理轨迹；实际CPC/MPCC用于闭环检验。
主训练损失未直接优化飞行时间。未收录没有带来额外改善的闭环参数搜索和历史变种。

解析分支的MPCC使用固定屏障 `h=||p_i-predicted_p_j||−0.55`，硬约束 `h>=0`，同时以
权重1e5惩罚负屏障、以权重2e4惩罚 `h[k+1]−exp(−3*0.08)*h[k]` 的负值。
邻机预测仅来自当前/历史位置速度；没有交换未来轨迹，也没有固定等待或优先级规则。
CPC中没有直接加入机间CBF。物理执行每0.02s更新；MPCC预测20×0.08s。

学习分支直接在MPCC中加入 `hθ[k]>=0` 和
`hθ[k+1]−exp(−3*0.08)*hθ[k]>=0`。每架无人机使用共享安全网络，输入本机状态及
当前/预测邻机位置速度。安全网络依据GCBF+的直接标量及符号/导数训练思路构建；
本实现为双机MLP与数值MPCC组合，详见
[GCBF+作者源码](https://github.com/MIT-REALM/gcbfplus/blob/main/gcbfplus/algo/module/cbf.py)。

无平移速度上限，旋翼推力0–7N、角速度范数上限9.4rad/s，仅最初静止，滚动时
不重置物理状态。最终检查的机间距阈值为0.5m。

## 附带的最新结果

`models/allocation.pt` 为seed31成功模型；`models/time_initializer.pt` 为冻结初值网络。
`results/example_seed31/` 保存此前已独立核验的两圈轨迹、29次CPC计划、全部控制输入、
审计结果和动图；配置中的checkpoint路径已重定位到本目录，模型文件内容保持一致。

该示例两机各过门14次，仿真飞行16.58s，加密最小间距0.76618m。
直接打开 `results/example_seed31/rollout.gif` 可查看；也可运行：

```powershell
.\.venv\Scripts\python.exe audit.py results/example_seed31
.\.venv\Scripts\python.exe verify_policy.py results/example_seed31
```

`results/training_metrics.json` 保留原训练记录。`PROVENANCE.json` 保存模型哈希及整理范围。
本目录未复制历史实验全集、虚拟环境、原始教师数据或ACADO二进制文件。
实际求解仍为离线数值验证，计算延迟未推进物理仿真；16.58s是飞行时间。

整理后已用本目录源码重新执行完整两圈，结果位于 `outputs/standalone_two_laps/`：
两机各14次过门，16.58s，最小间距0.76618m，独立物理回放及29次网络分配重算通过。
`checks/` 保留短训练、模块来源、梯度检查及重新执行日志。

## 2026-10-30分支整合核验

- 原解析分支的NLP目标、所有约束和边界与整合前保存的数值样本一致；三个分支/参数回归测试通过。
- `outputs/integrated_analytic_20261030/` 完成每机两门的短程兼容性验证。
- `outputs/integrated_learned_20261030/` 完成双机各14次过门、两圈飞行，仿真16.66s；
  独立加密最小间距约0.83246m，实际执行峰值12.8302m/s，物理回放和29次门内分配重算通过。
- 使用附带数据和首轮权重重新训练4000步，得到的CBF参数与已验证模型逐项相同。
- 两次整合运行均保存 `rollout.gif`、PNG/PDF/SVG末帧、原始状态、控制和审计JSON；
  `checks/integration_result_20261030.json` 汇总检查范围。三步训练仅用于入口检查，不作为训练效果结论。

可直接打开[学习分支完整动图](outputs/integrated_learned_20261030/rollout.gif)。

## 已知局限

分配输出常接近±0.45m边界，部分随机种子在代理目标上训练不佳。
完整轨迹通过依赖学习分配和数值安全层共同作用；当前结果限定于已测试赛道与状态。
数值加密检查尚未提供真实邻机运动下的连续时间安全证书。
