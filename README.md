# CBFracing

穿越点网络和时间网络生成五门参考，完整刚体 MPCC 使用学习 CBF 进行避碰。根目录的 train.py 和 validate.py 包含完整参数解析及主流程。

## 运行

在项目目录安装依赖，每次使用新的输出名称。

```powershell
python -m pip install -r requirements.txt

# 默认使用 Minimum Snap + 学习时间分配 + 凝聚 DAQP SQP-RTI
python validate.py --name snap_demo

# 显式选择前端及自训练时间检查点
python validate.py --name snap_custom --frontend minimum-snap --event-checkpoint models/my_times/events.pt

# 原有自由时间 phase 前端及原始互补 CPC
python validate.py --name phase_demo --frontend phase --start-mode gate1_aligned
python validate.py --name original_demo --frontend original --preview-gates 2 --mpcc-backend ipopt

# 完整审计、动图和速度曲线
python -m cbfracing.evaluation.integration_report outputs/snap_demo
python -m cbfracing.evaluation.render outputs/snap_demo
```

全部参数见 python validate.py --help。默认前端已选为新整合的 minimum-snap，保留 phase 和 original 命令行分支。Minimum Snap 默认载入 models/snap_short_residual_v2/events.pt，采用对准一号门的静止起点、进度奖励4200、弯道奖励900。--start-mode original 可以选择原始起点。phase 默认采用原始起点和220进度奖励。

RTI 使用缓存编译导数、动力学凝聚和 DAQP，最多16次更新，正则系数0.1，保留物理过门约束。前3个预测区间细分检查过门交点。Minimum Snap 分支直接使用RTI冷启动，phase和original分支首次由IPOPT初始化。可用 --rti-cold-start 或 --no-rti-cold-start 显式选择。

--rti-kernels auto 根据问题哈希加载 build/fast_physical_cross_dense.dll。改变预测长度、CBF模型或约束后，哈希不匹配会使用符号导数，并记录原因。--rti-kernels native 要求编译内核精确匹配，--rti-kernels symbolic 使用符号计算。提供的DLL对应Windows和当前CasADi配置。本机 .deps/ 包含DAQP 0.11.0的Python 3.12 Windows扩展，其他环境通过requirements.txt安装依赖。

实际20 ms执行区间通过非线性核验后施加控制，完整预测残差另外记录。求解失败后允许一次计入耗时的SQP可行性恢复，可用 --no-sqp-recovery 关闭。

## 时间网络训练

```powershell
# 从头监督训练五事件时间网络
python train.py --task time --frontend minimum-snap --name my_times --steps 2000

# 复现昨晚的短航段残差训练，冻结基础时间网络
python train.py --task time --frontend minimum-snap --architecture short --name my_short_times --log-bound .3 --steps 2000

# 使用自己整理的真实滚动数据
python train.py --task time --name custom_times --dataset data/custom_events.npz --steps 3000

# 穿越点和CBF训练
python train.py --name allocation_scratch --steps 2500
python train.py --task cbf --name custom_h --steps 4000
```

时间训练默认使用data/events_rollout.npz，包含60个训练样本和20个验证样本，按飞行运行划分。标签来自已核验轨迹的后续实际过门间隔。每个样本有54维状态与五门几何特征，以及5个正航段时长。自建npz需包含features[N,54]、targets[N,5]、train和val索引，两个索引集合应互斥。同一运行的样本应放在同一集合。

direct架构从头训练全部时间网络参数。short架构冻结models/snap_bounded_rollout_v1/events.pt，训练距离小于3 m的航段残差，修正比例约束在exp(±log_bound)内。输出保存到models/<name>/events.pt和training.json，随后通过--event-checkpoint调用。训练验证误差与完整闭环性能分别核验。

## 文件布局

|位置|内容|
|---|---|
|train.py|完整训练参数解析与分支选择、穿越点训练流程|
|validate.py|完整闭环参数解析、规划与控制执行流程|
|cbfracing/core/|动力学、赛道、配置及运行参数配置|
|cbfracing/networks/|穿越点、时间分配、学习CBF网络|
|cbfracing/planning/|Minimum Snap、phase、互补CPC、空间参考及初值|
|cbfracing/control/|MPCC问题、RTI、DAQP凝聚QP、邻机预测|
|cbfracing/training/|时间网络和CBF训练|
|cbfracing/evaluation/|物理与网络核验、绘图、结果报告|
|cbfracing/tools/|导数内核编译及给定样本的数值核验|
|models/、data/|检查点及训练数据|
|build/|与问题哈希绑定的编译内核|
|outputs/、results/|历史及新运行的独立结果|

每次运行保存实际状态、控制、逐次求解日志、前端规划数据及递归源码快照。已有检查点与历史结果保留。新增模型、数据和内核的来源记录在INTEGRATION_20261105.json。

## 整合验证

outputs/integration_snap_full_20261105/完成双机各两圈14门，峰值19.0116 m/s，两机第二圈5.9035/5.9098 s。执行数组与实验区推荐版本逐元素一致，物理、实际采样CBF及分配网络核验通过。该次MPCC中位/P95为24.93/53.29 ms，参考生成29次合计0.4461 s。

在本项目重新训练2000步短航段残差后，outputs/integration_trained_time_20261105/再次完成同一轨迹。实际计时受机器负载影响。

phase分支在对准一号门的起点完成了双机单门调用检查。原始起点配合本次RTI配置的两次测试在约0.7 s处求解失败，保留在outputs/integration_phase_smoke_20261105/与outputs/integration_phase_initialized_20261105/。完整两圈通过范围目前为上述Minimum Snap配置。

固定时长的Minimum Snap为运动学凸QP，完整MPCC包含非线性刚体动力学。预测采用20×0.08 s时域，每0.02 s执行一次。初始状态仅首次静止，后续携带实际状态，邻机预测基于当前与历史观测。前3个预测区间施加硬CBF下降率约束，后续使用惩罚，h的非负约束覆盖预测时域。

计算延迟单独统计，尚未推进物理仿真。上述计时与采样核验分别限定于当前测试的计算性能和安全性。
