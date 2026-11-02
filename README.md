# CBFracing：门内穿越点分配 + 滚动轨迹前端 + MPCC/CBF

默认配置已整合最新带线性阻力的五门预览方案。源代码只导入本目录模块和requirements.txt中的第三方库，可将整个目录复制到其他位置；模型和旧结果保持原样。

## 从这里开始

在本目录创建Python 3.12环境并安装requirements.txt，然后运行：

```powershell
python selfcheck.py
python -m unittest test_cli_branches -v
python validate.py --name demo_new
python render.py outputs/demo_new
python integration_report.py outputs/demo_new
```

输出名必须是新名字。validate.py默认选择学习CBF、5个后续物理航门、phase前端、C2参考、进度奖励220；每机通过14个门后结束。结果目录包含实际状态、控制、逐次规划、全部日志、动图、速度曲线及独立审计。

## 参数来源

config.json统一提供无人机参数和online_planning默认设置，CLI参数优先于默认设置。

|物理参数|当前值|
|---|---|
|质量|0.85 kg|
|臂长|0.15 m|
|惯量|diag(0.0025, 0.0021, 0.0043) kg m²|
|单旋翼推力|0–7 N|
|力矩系数|0.022|
|角速度约束|欧氏范数不超过10 rad/s|
|机体系线性阻力|diag(0.26, 0.28, 0.42) kg/s|
|平移速度上限|无|

阻力加速度为 `-R D Rᵀ v / m`，前端、MPCC预测及执行使用同一物理模型。当前未加入外部风速。角速度约束保持范数形式。

默认MPCC为20×0.08=1.6秒，反馈间隔0.02秒。进度奖励220、滞后误差权重10000、推力变化权重0.05、虚拟进度加速度权重2。学习CBF的alpha=3，h及离散下降率均为预测硬约束。

## 五门滚动规划

每架无人机观测本机状态、邻机位置速度和航门几何。冻结共享网络针对每个预览门分别输出横向穿越点偏移，范围为±0.45m；网络结构没有扩大，也没有增加额外空间航路点。时间网络为前两门提供初始猜测，其余门按距离构造初值，随后由数值前端优化时长和完整刚体轨迹。

默认phase前端将各物理航门放在各航段末端的固定网格节点，优化各航段时长。每通过当前门即用真实当前状态重规划；5门预览没有改变MPCC的1.6秒预测时域。仅最开始静止，滚动时不重置物理状态，也没有末端静止等式。

C2参考、最多两次内部参考更新、物理门平面交点约束与最新实验保持一致。前端没有机间CBF，学习CBF位于MPCC中；邻机预测来自当前及历史观测，不交换未来轨迹，没有显式等待或优先级规则。

保留原始互补CPC，可显式选择两门版本：

```powershell
python validate.py --name original_two_gate --frontend original --preview-gates 2 --preview-spline hermite --reference-refine 0 --progress 80 --no-physical-gate-guard
```

原始互补CPC同时优化总时间和事件进度，保留两门参数化；phase支持可配置多门预览。五门phase结果按其实际事件参数化报告。两个前端共享当前带阻力动力学。

## 解析与学习CBF分支

```powershell
# 当前默认：直接输出学习CBF
python validate.py --name learned_demo --cbf learned

# 保留解析距离CBF入口
python validate.py --name analytic_demo --cbf analytic

# 更换学习CBF或修改预览数量
python validate.py --name custom_demo --cbf learned --cbf-checkpoint models/my_h/h.pt --preview-gates 3
```

学习分支使用models/direct_cbf_v2/h.pt，结构22→32→16→1 Tanh，直接输出完整hθ，没有可学习的额外裕度。硬约束为hθ[k]≥0和hθ[k+1]≥exp(-3×0.08)hθ[k]。解析分支保留距离硬约束及软下降率惩罚。显式CLI选项优先于分支设置。

## 训练和主要入口

```powershell
python train.py --name scratch_s31 --seed 31 --steps 2500
python train.py --task cbf --name new_h --seed 7 --steps 4000
```

航门分配训练沿用无穿越点标签的几何代理损失；本次整合没有重新训练任何checkpoint。CBF默认读取附带冻结数据集，若物理条件或任务分布明显改变，需要另行采样及闭环验证。

|文件|作用|
|---|---|
|config.json / model.py|物理参数、阻力、默认规划配置及动力学|
|validate.py / cli_profiles.py|闭环执行、CLI覆盖及CBF分支|
|planning.py / phase_planner.py|滚动门内分配、可配置预览与自由航段时长规划|
|original_cpc.py|原始互补CPC前端|
|controller.py / reference.py|MPCC、CBF、物理门交点检查和空间参考|
|train.py / allocation.py|航门穿越点分配训练及共享网络|
|safety_train.py / barrier.py|直接安全函数训练及符号导出|
|integration_report.py|物理/CBF/分配核验与中文结果页|
|audit.py / audit_neural.py / verify_policy.py|独立物理回放、实际h/下降率和逐门网络重算|
|render.py|实际轨迹动图、完整速度曲线及PNG/PDF/SVG|
|selfcheck.py / test_cli_branches.py|本地模块与模型哈希、局部导数、动力学及配置回归|

verify_policy.py分别核验phase固定事件网格和original互补事件编码。audit.py使用每次运行保存的动力学参数回放历史结果，避免用新阻力核验旧零阻力轨迹。学习屏障值、离散下降率与物理避碰分别报告。

## 结果记录

本次完整整合验证保存于outputs/integrated_drag_preview5_mu220_20261102/。完成后打开该目录index.html，查看实际两圈结果、动图、速度和全部审计。

results/example_seed31及此前outputs下结果属于各自旧配置，保留用于追溯，不作为当前默认配置的测试结果。PROVENANCE.json保存冻结模型哈希，checks保存本次模型一致性、入口和历史回放检查。

所有飞行时长均为仿真物理时间，计算耗时单独记录且没有推进飞行。本项目当前为完整非线性数值验证后端，不宣称实时部署、跨场景泛化或全局时间最优。实际CBF采样检查不构成连续时间不变集证明。
