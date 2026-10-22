# 新128×2无辅助头：R24–R43固定池从零训练

2026-10-19结果更新：68041步与两组正式面板均已完成，选37423步；greedy53.775% [51.600%,55.900%]，S1024搜索56.625% [51.750%,61.375%]，详见[结果](results.md)。限定单训练seed与固定面板，未独立隔离各结构及训练设置的贡献。

同日按用户要求补测[30619步对R43的1000pair greedy](greedy_step30619/results.md)：47.775% [45.600%,49.925%]。沿用37423步的相同开局和policy seed，30619减37423的共同对手配对得分差为−6.000pp [−8.950,−2.925]pp；不是两个候选直接互打。30619为胜负MSE较低节点，但greedy不使用value，不能用该结果判断value或搜索棋力。该checkpoint从外置T7压缩包单独提取至runtime并核对原SHA，未恢复整批归档或训练。

2026-10-19按用户要求准备，并在有限验证后启动。入口[run.py](run.py)，配置[offline_128x2_relations_no_aux_r24_r43_20261019.json](../../configs/entity_attention_v1/offline_128x2_relations_no_aux_r24_r43_20261019.json)。复用原256×4实验的历史池、抽样、百分比完整留出、选点及对战流程；原实验配置和产物保留。

## 配方

新版128维、2层、4 heads、FFN512、dropout0，共583,682参数；`market_relations=true`和`value_readout="global_players_market"`启用市场及双方明保留牌的8维购买关系，以及全局/双方玩家/市场均值的value读取。[结构说明](../entity_attention_relations_20261019/README.md)。

这里的“不加两个辅助特征”指移除分差和剩余回合两个辅助预测头：`value_auxiliary=null`，模型实际不注册它们，loss仅为普通policy CE＋胜负value MSE，两者权重各1。PSW关闭；原始终局z与搜索访问分布不改标签。购买关系输入保留。

与[256×4从零训练](../entity_attention_256x4_aux_r24_r43_20261014/README.md)相比，仅改变模型配置、移除辅助头、学习率改为1e-4；其他配置逐字段相同：

| 项目 | 设置 |
| --- | --- |
| 数据 | 绝对R24–R43的20000局，原整局split |
| Train | 17971局／1451525局面 |
| 留出 | 2029局／163595局面；固定子集4096 |
| 初始化 | seed36500000随机参数，全新AdamW；不读取R43权重或动量 |
| 优化 | lr1e-4，B256，weight decay1e-4，clip1，FP32 |
| 抽样 | seed34410000，单池均匀有放回，平均曝光12次，共68041步 |
| 验证seed | 34420000，与原实验相同的固定4096局面 |
| 缓存 | 8GiB CompactGameCache |
| 保存 | 每1000步保存恢复点；每5%额外保存并评估 |
| 验证 | 0步仅固定4096；3403、6805……68041等20节点测固定4096和完整留出 |
| 选点 | 20个百分比节点中，完整留出policy KL＋胜负value MSE最小的完整checkpoint；同分取更早步 |

原配置中的`auxiliary_initialization_seed=36100000`保留，但无辅助模式不使用它，也不重设主网初始化后的Torch RNG。初始化元信息将其记为null。`history_batch_size`在固定单池入口不生效。

训练完成后串行评测选中模型对原R43：greedy1000pair／CPU8worker，随后双方native S1024/N1搜索200pair／CUDA compiled，根噪声0、T0、c_puct1.5。开局和bootstrap seed沿用256实验：greedy从36600000、搜索从36700000起。对局不参与选点。

## 启动与恢复

从仓库根目录执行，同一命令自动从最近已提交checkpoint恢复，接着完成选点与未完成对战：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u artifacts/entity_attention_128x2_relations_no_aux_r24_r43_20261019/run.py run
```

首次后台启动身份记录在`launch.json`；脚本自行记录阶段与退出状态，不需要持续轮询。`train`只训练和选点；`prepare`只冻结计划；`status`按当前原始产物刷新汇总。不要并行重复启动同一输出目录。配置/计划/原始数据身份冻结，恢复时沿用原参数。

训练产物在`runs/entity_attention_128x2_relations_no_aux_r24_r43_20261019/`：逐步`training_metrics.jsonl`、抽样引用账本、`validation_curve.jsonl`、`full_validation/step-*.json`、checkpoints及`selection.json`。指标含policy CE、目标/预测熵、KL、胜负MSE、总loss与梯度裁剪信息，没有辅助loss。正在训练时，`progress.json`记录已提交恢复点；本目录`summary.json`／`results.md`在阶段结束或status调用时刷新。

阶段日志及live状态在`runtime/entity_attention_128x2_relations_no_aux_r24_r43_20261019/`；模型与重型对局sidecar不随Git分发。对战配置及完成汇总在本目录`evaluation_configs/`与`panels/`。原R43仅作对手，SHA为`b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14`。

## 有限验证

[preflight.py](preflight.py)只使用原R24一个train归档的前4条真实样本，在独立runtime目录做CPU两步／CUDA一步检查。它验证真实无辅助网络、随机参数与空optimizer、lr1e-4、有限loss、checkpoint恢复及frozen加载；CPU还验证恢复后下一步权重和AdamW状态逐位一致，不改正式训练目录。

相关pytest覆盖None/双辅助模式的百分比验证及精确恢复、旧辅助默认兼容、完整留出选点、配置仅改变指定项、报告不依赖辅助指标、两组评测配置及模型本身。命令与结果保存于`tests.xml`、`cpu_preflight.json`、`cuda_preflight.json`和`delivery_check.json`；这些是工程验证，不是正式学习或棋力结论。正式执行版本和命令另见`execution.json`与各会话源码快照。

本次[102项CPU测试](tests.xml)全部通过；[CPU检查](cpu_preflight.json)确认恢复后下一步权重/AdamW逐位一致，[CUDA检查](cuda_preflight.json)确认真实4样本lr1e-4更新与保存加载通过，两个辅助头实际不存在。[冻结计划](plan.json)核对20000局、68041步及20个百分比验证节点。[交付核验](delivery_check.json)保存配置与原始检查身份。

## 2026-10-19冷产物位置

非主要checkpoint、逐步指标及抽样账本已集中到`archive/20261019/runs/`下相同run结构；关键模型与验证／选点文件留在原位置，原文件路径保留相对链接。见[逐文件迁移清单、保留策略及恢复说明](../repository_archive_20261019/README.md)。只读诊断可在恢复挂载后访问；精确续训前须按清单撤回该run的迁移，恢复原位真实账本／日志和权重，避免续训改写冻结归档。本次没有启动或恢复训练。
