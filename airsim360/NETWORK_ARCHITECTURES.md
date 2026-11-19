# 双机竞速动态风险场网络结构与实验记录

截至 2026-11-12。本文覆盖本项目保留了实现、训练权重或实验记录的感知、运动预测和风险查询模型，并单列交接包中的状态基线与过门点辅助分支。下文的“闭环”指该模型输出实际进入当次规划或控制。用另一模型生成的轨迹离线绘制热图，归入回放评估。

## 旧版与当前版的区别

| | 旧时空网络 | 当前短时运动历史版 |
|---|---|---|
| 实际模型 | `RGBDField`，旧闭环权重 `cf21_time8_rgbd_50hz_resume02` | `cf21_motion_history01`，由可见点分类、运动预测和风险核组成 |
| 视觉输入 | 连续 4 帧 RGB-D。闭环从 512×256 采集，缩至 256×128 编码 | 连续 4 帧 512×256 RGB-D，提取紧凑表面和跨帧运动，结合本机状态 |
| 学习目标 | 给定位置与提前量，直接预测该点的风险。标签同时覆盖邻机和门框 | 根据可见点预测邻机未来位移。以预测位置为中心生成动态风险，静态深度风险单独计算 |
| 赛道信息 | 场景外观来自图像。查询点以当前相机为坐标系，无显式门号、圈数或世界位置输入 | 检索使用世界位置、相对位置、本机运动与起飞时间；时间特征在 1.8 s 封顶。无显式门号 |
| 预测形式 | 一个场景向量可反复查询不同位置和提前量，训练覆盖 0–1.6 s | 先预测 0–1.6 s、间隔 0.1 s 的 17 个三维位移；中间时刻插值，再查询空间风险 |
| 飞行时的邻机信息 | 图像观测 | 图像观测、表面点对应和训练航次片段。两条路线都不要求在线维护邻机身份和整机状态估计器 |

两版都能为 MPCC 提供随空间和时间变化的风险，但输出值的含义取决于标签与静态项。视频里选取的 0.10 s 是可视化提前量；MPCC 按轨迹节点的预计到达时间分别查询。[旧版闭环配置](reports/spatial_potential/cf21_bendhead03_run1/experiment.json) · [当前版闭环配置](reports/spatial_potential/cf21_motion_history01_race2/experiment.json)

## 一、从邻机状态输入到直接视觉查询

| 模型 | 输入、结构与输出 | 实验记录 |
|---|---|---|
| 状态时空场 `SpacetimeRisk` | 使用邻机位置、速度和因果加速度。将“查询点相对邻机匀速外推位置的偏移”、速度、加速度、提前量组成 10 维特征。Tanh MLP 为 **10→64→64→1**，最终得到风险值 | 交接包已有 16,000 步训练和数值闭环，是视觉迁移的起点。[模型](../risk_training_portable_20261109/cbfracing/networks/spacetime.py) · [交接记录](../risk_training_portable_20261109/CODEX_HANDOFF.md) |
| 带状态辅助头的 `VisualRisk` | 每帧 CNN **5→16→32→48**，池化至 2×8，再把 768 维投影到 64 维；与 15 维本机历史及观测有效信息拼接，GRU **79→32**。一支输出邻机相对位置、速度和加速度共 9 维；另一支用潜变量、相对查询坐标和提前量，经 **36→64→64→1** 输出风险 | 有状态阶段与风险阶段的合成数据训练、续训和本机 CUDA 冒烟记录。后续无邻机状态头的旧闭环使用下述 `RGBDField`。[实现](../risk_training_portable_20261109/risk_training/vision.py) · [本机训练配置](reports/risk_cuda_smoke/risk/config.json) |
| 全局 CNN–GRU `RGBDField` | 每帧为 RGB、径向深度、深度有效掩码共 5 通道。三层 3×3、步长 2 的卷积 **5→16→32→48**，各接 GroupNorm、SiLU；池化为 2×8。每帧 768 维，4 帧经单层 GRU 得到 64 维。拼入相机系三维查询点和提前量，MLP **68→96→64→1**，sigmoid 输出。共 **192,465** 个参数 | 实际训练过 1、4、8 航次对照、Crazyflie 尺寸重采集、半分辨率与原生 512×256、门心配对重采样。旧版两次七视角闭环均来自这一结构的半分辨率权重。查询点在解码阶段才进入网络。[模型](scripts/training/train_rgbd_field.py) · [航次数量对照](reports/flight_count_comparison/results.json) · [部署编码器](scripts/perception/time_query_field.py) |
| 查询点投影版 `ProjectedRGBDField` | 保留原 CNN–GRU，将相机系查询点投影到最新帧。额外读取 48 维局部卷积特征、5 维图像通道、查询深度残差和到 5×5 邻域表面的最近距离。解码器输入从 68 增至 **123**，后接 **96→64→1** | 8 航次离线对照。15,272 对门框/门心查询中，门框风险高于门心的比例从 **33.68%** 升至 **99.73%**；0.4 s 事件测试在阈值 0.5 下检出 28/28，空门误报 25/28。几何区分与未来进门预警应分别评价。[实现](scripts/training/train_rgbd_field.py) · [对照结果](reports/projected_rgbd_8/results.json) |
| 过门事件分类器 `GateWarning` | 将目标门移到全景中心，取最新帧 5 通道和 3 组相邻帧差分共 **20 通道**。四层步长 2 卷积 **20→24→32→48→64**，池化 2×4，分类头 **512→64→1**，含 dropout。输出“该门约 0.4 s 后有邻机进入”的事件分数 | 冻结投影版门框分支，单独训练预警头。测试含 28 对事件/空门样本；阈值 0.5 时，时序版检出 **15/28**、空门误报 **16/28**；独立训练的单帧版为 **17/28、16/28**。另有将时序版输入重复成同一帧的消融。[训练脚本](scripts/training/train_gate_warning.py) · [事件对照](reports/temporal_gate_warning_8/results.json) |

**同一 CNN–GRU 的数据实验。** 原生 512×256 实验增加了空门中心和“同一位置现在安全、0.4 s 后邻机到达”的配对查询。测试中，允许 2 次空门误报时，原采样检出 7/28、配对采样检出 5/28；允许 22 次误报时分别为 24/28 和 25/28。比较分辨率时还应核对采集图像大小与网络实际编码大小。[采样与工作点对照](reports/spatial_potential/cf21_native512_gatepairs_01/sampling_comparison.json)

## 二、在可见表面上建立风险场

这些模型保留 RGB-D 的空间对应关系。对于采用球形风险核的版本，单点风险为 `sigmoid((r − 距查询点的距离) / w)`。其中 r 是风险值降到 0.5 时的半径，w 控制边缘过渡。动态分支对有效表面点取最大风险，MPCC 在轨迹节点处还会考虑静态深度表面。[计算与控制接口](scripts/perception/local_future_field.py)

| 模型或组合 | 学了什么 | 使用情况 |
|---|---|---|
| 表面空间风险网络 `PotentialNet / SpatialField` | 从 4 帧颜色、颜色变化、深度、深度变化及邻域深度差构造每个可见点的 12 维特征，MLP **12→32→32→2** 输出半径、宽度。查询时在近邻深度点中取风险最大者。输入没有未来提前量 | 训练了原始、低频感知、Crazyflie 及 512 分辨率版本。曾直接作为空间风险项，也用于为视觉过门点选取横向偏移。[模型](scripts/perception/spatial_potential.py) · [训练](scripts/training/train_spatial_potential.py) · [实飞行仿真配置](reports/spatial_potential/cf21_512_unpaced12_space_w5600_cold01/experiment.json) |
| 紧凑表面外观分类器与 `OccupancyFlow` | 深度连通域筛出小表面，RGB、尺寸和深度对比等 12 维特征经 **12→32→1** 的 ReLU MLP 过滤候选点。跨帧表面对应给出局部运动；基础运动库用世界位置与速度构建 6 维键，取 12 个近邻的位移曲线加权平均。分类器由梯度训练，位移检索是非参数方法 | 初始 8 航次含 **70,590** 个分类点、**3,494** 个运动片段。尝试了扩充片段库、调整风险半径与短时缓冲、引导参考轨迹，以及在旧 CNN–GRU 风险上叠加局部运动风险。它们复用分类器，控制组合不同。[训练统计](reports/spatial_potential/cf21_flow_field_01/fit.json) · [运动与查询](scripts/perception/occupancy_flow.py) · [组合闭环](reports/spatial_potential/cf21_localrisk135_01/experiment.json) |
| 局部未来风险核 `LocalFutureField` | 每次最多保留 32 个可见点。以相对位置、表面运动、先验未来位移、本机速度和提前量组成 **13 维**输入，MLP **13→48→48→5** 输出三维中心修正、半径和宽度。用未来邻机中心、查询点风险软标签及排序损失训练 | `cf21_local_future01` 使用训练航次位移先验，完成两圈闭环。另一组 `cf21_local_visual_motion01` 以观测表面速度乘提前量作为位移起点，同为五输出头，做过离线训练和评估。验证航次的局部中心风险中位数分别为 **0.909、0.133**。[实现](scripts/perception/local_future_field.py) · [训练标签](scripts/training/train_local_future.py) · [检索先验结果](reports/spatial_potential/cf21_local_future01/evaluation.json) · [视觉速度外推结果](reports/spatial_potential/cf21_local_visual_motion01/evaluation.json) |

## 三、改进运动预测与风险形状

下列片段检索模型每次寻找 **16 个相似训练片段**。轨迹标签来自仿真真值，每条覆盖 0–1.6 s 的 17 个三维位移；训练检索时排除查询样本所在航次。闭环时，观测键由当时的 RGB-D 和本机状态生成。单中心的注意力、片段修正和历史版本用运动模型决定中心，沿用已有风险半径与宽度头，额外中心修正系数设为零。[样本与训练](scripts/training/train_context_motion.py) · [模型实现](scripts/perception/context_motion.py)

| 实验 | 结构与输入变化 | 验证与部署 |
|---|---|---|
| 检索注意力 `MotionAttention` | 13 维上下文含世界位置、相对位置、表面运动、本机速度与起飞时间。当前键与候选差值拼成 26 维。匹配 MLP **26→48→32→1** 学片段权重，修正 MLP **26→64→48→12** 给出四段三维残差速度，积分得到位移 | 13 条训练航次；有独立两圈闭环。[训练记录](reports/spatial_potential/cf21_motion_attention02/results.json) |
| 片段修正 `FragmentMotion` | 学习逐特征距离权重及修正矩阵。13 维版本的 MLP 为 **13→48→156**，输出重排为 13×12。先按初始条件差异逐条修正候选曲线，再加权平均；完全匹配时该片段修正为零 | 13 条训练航次；有两圈闭环。另试过围绕固定中心重新拟合球形风险轮廓，部署版本选中初始轮廓。[训练与校准记录](reports/spatial_potential/cf21_motion_fragment01/results.json) |
| **短时运动历史版** `cf21_motion_history01` | 同一片段修正结构扩至 **22→48→264**。新增 9 维为局部表面点簇速度、本机加速度和本机角速度，各 3 维。结合短时视觉运动与历史轨迹来预测中心 | 17 条训练、2 条验证、3 条测试；1,499 个匹配视觉观测，检索库含 8,354 个片段。同一权重完成两次两圈闭环。[训练清单](reports/spatial_potential/cf21_motion_history01/results.json) · [训练日志](logs/cf21_motion_history01.log) |
| 球形核宽度校准 `cf21_motion_width01` | 固定片段运动预测，调整原风险核的过渡宽度，保留原有球形几何 | 离线验证 Brier 误差从 **0.0820** 降至 **0.0815**，安全点平均风险从 **0.158** 升至 **0.164**。未列入两圈对照。[校准记录](reports/spatial_potential/cf21_motion_width01/calibration.json) |
| 方向性风险核 `DirectionalRisk` | 在冻结的运动预测上增加 **13→32→3** 的 MLP，输出横向半径修正、沿运动方向的尺度和边缘宽度，可形成沿飞行方向伸缩的椭球 | 离线训练。验证约束下最终选中 **epoch 0**，纵向尺度为 1。没有方向性模型改善闭环的实测结论。[实现](scripts/perception/directional_motion_risk.py) · [验证](reports/spatial_potential/cf21_motion_direction01/results.json) |
| 三种运动假设 `MixtureRisk` | 以 13 维片段版为基础，按 0.9 s 时的前进量把 16 个片段分成 **5、6、5** 三组。输入上下文、三组质量、分散程度、提前量，MLP **20→48→5** 输出三个混合权重以及共享半径、宽度。对单个表面点的三种未来风险加权求和 | 实际接入 MPCC 并完成两圈。验证集安全查询点误报率固定为 5% 时，近邻查询点召回从 **78.43%** 升至 **87.24%**。此处统计的是空间查询点，门洞进门事件按另一套样本计数。[实现](scripts/training/train_motion_mixture.py) · [验证与测试](reports/spatial_potential/cf21_motion_mixture01/results.json) |

## 四、实际闭环表现

下表来自同一份逐门审计。各行初始间距为 1.2 m，风险权重为 5600，两机各完成 14 道门。时间差是两机逐门到达时间差绝对值的均值；门框距离是两机全程中较小的机心到门框距离。严格通过条件同时要求两机中心间距超过 0.50 m、门框余量至少 0.25 m。[逐航次数据、配置及权重](reports/spatial_potential/cf21_motion_study/comparison.json)

| 参与控制的网络 | 平均过门时间差（ms） | 最小两机中心距（m） | 最小门框距离（m） | 严格通过 |
|---|---:|---:|---:|---|
| 旧全局 CNN–GRU，第 1 次 | 12.8 | 0.662 | 0.216 | 否 |
| 旧全局 CNN–GRU，第 2 次 | 63.3 | 0.670 | 0.224 | 否 |
| 局部未来风险核 | 479.2 | 0.866 | 0.232 | 否 |
| 检索注意力 | 183.7 | 1.200 | 0.242 | 否 |
| 片段修正，13 维 | 174.7 | 1.200 | 0.238 | 否 |
| 短时运动历史，22 维，第 1 次 | 222.4 | 1.200 | 0.256 | 是 |
| 短时运动历史，22 维，第 2 次 | 198.5 | 1.200 | 0.236 | 否 |
| 三种运动假设 | 187.9 | 1.200 | 0.230 | 否 |

在这些记录中，旧 CNN–GRU 的两次逐门同步性最好。新版改善了部分离线运动或风险指标，仍未稳定兼顾并排过门与门框余量。各版本训练数据量、检索先验和闭环观测分布均有变化，表中的差异不能全部归因于某一层网络。[旧版实际闭环视频](reports/spatial_potential/cf21_bendhead03_run1/two_laps_fpv_cnn_gru_100ms_seven_views.mp4) · [当前版实际闭环视频](reports/spatial_potential/cf21_motion_history01_race2/two_laps_fpv_dynamic_risk_field.mp4)

## 五、同一系统中的辅助模型与对照

- **视觉过门点分支。** 在门平面选取横向候选点，用空间风险评分结合侧向偏好产生基础偏移，再拟合共享的水平与竖直增益。后续使用闭环回归与航门转弯几何校正，输出 Fast-fly 的过门目标。该分支有独立的训练参数，输出单位是位置偏移。旧时空解码器和当前运动预测器均可与它配合。[初版拟合](scripts/training/fit_visual_passage.py) · [闭环拟合](scripts/training/fit_passage_rollouts.py) · [航门几何校正](scripts/training/fit_gate_clearance.py)
- **交接包的数值控制基线。** 包内另有 `ThreatNetwork`（22→32→16→1，即时风险）、`DirectCBF`（22 维状态特征至有符号屏障值）、`GateAllocator`（24→48→48→1，双机过门分配）和 `DurationNetwork`（26→96→96→2，前端时长初始化）。这些是已有控制组件；表中的视觉模型训练次数与两圈成绩仅归属于相应实验。[即时风险](../risk_training_portable_20261109/cbfracing/networks/threat.py) · [屏障](scripts/control/learned_barrier.py) · [分配器](scripts/planning/utils/allocation.py) · [时长网络](scripts/planning/utils/timing.py)
- **控制与采集消融。** 350、2800、5600 的风险权重，关闭动态项、仅保留静态项、用真值未来位移回放，以及切换视角后双采样，都改变了实验条件。预测提前量从 0.4 s 切换到 0.1 s 也只是改变查询参数。比较网络结构时应单独记录这些设置。[闭环入口](scripts/control/run_spatial_mpcc.py) · [固定记录回放](scripts/replay/replay_local_future_launch.py)
## 连续三维位置与时间的动态占用查询

`scripts/training/train_implicit_rgbd.py` 使用每架机 4 帧原生 512×256 全景 RGB-D、本机位姿和时间戳。标签为查询点至未来邻机中心的距离 ≤0.55 m。邻机真值仅用于离线采样、监督和评估。A、B、C 共用轻量编码器和数据划分，分别采用最新帧原位采样、多帧原位采样、多帧原位加两个可学习三维偏移。P 是原始 `ProjectedRGBDField` 在相同动态标签下的训练对照。

从项目根目录在 PowerShell 中运行完整的受监控训练：

```powershell
.\scripts\monitored_flight.ps1 -Name cf21_implicit_full01 -ImplicitExperiment train -ImplicitVariants P,A,B,C
```

默认按连续四帧取样。增加 `-ImplicitHistory long` 可切换至距现在约 0.20、0.08、0.02、0 秒的历史。每次使用新的运行名称，结果写入 `reports/spatial_potential/<Name>/`。各版以验证 BCE 选择权重，再按验证航次的空门误报预算固定阈值，分别记录独立测试航次的进门检出、空门误报、离门误报和冻结当前预测的对照。

对训练好的 C 版单独检查梯度或运行闭环：

```powershell
.\scripts\monitored_flight.ps1 -Name cf21_implicit_verify01 -VerifyImplicitCheckpoint reports/spatial_potential/cf21_implicit_full01/C/best.pt
.\scripts\monitored_flight.ps1 -Name cf21_implicit_flight01 -ImplicitCheckpoint reports/spatial_potential/cf21_implicit_full01/C/best.pt -ThreatWeight 5600 -RtiColdStart -PairedCameras
```

闭环在预测节点批量查询动态 logit 及世界坐标梯度，用局部一阶近似构造动态代价，并与已有 RGB-D 表面风险取最大值。每个节点按绝对预计到达时间减去最新视觉时间计算提前量，上限为训练时域 1.6 s。新图像每架机各编码一次。回放检查输出单核编码、缓存查询的 P50/P95，以及与有限差分和求解器符号梯度的对照。
