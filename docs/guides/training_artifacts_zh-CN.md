# 训练指标、抽样记录与checkpoint

## Git与原始数据的保存边界（2026-10-27）

下文“保存”指实验落盘与恢复需要，不代表进入Git。Git保存源码、正式配置、命令、种子、模型／数据身份、紧凑指标及结论；checkpoint、完整训练池、逐步`training_metrics.jsonl`、学习曲线JSONL、完整留出结果／NPZ和大留出清单保存在`runs/`、`runtime/`或外部归档。小型会话身份JSON可以纳管；逐样本引用及大型局清单单独保存。

同一JSON混合指标与逐样本明细时，另存聚合摘要并登记原件SHA、字节数及实际位置。数据或留出manifest作为恢复输入时，新机器必须先恢复原件；摘要不能代替输入。历史原件不改写，当前进程仍可读写原路径。[全仓库规则](../../artifacts/README.md)、[本次停止跟踪清单](../../artifacts/repository_git_cleanup_20261027/README.md)。

2026-10-24后续：流式会话累计现作为新建离线默认行为，与4 worker共享只读供数及传输优化一起接入后两组；缓存按池大小配置。逐步主指标仍正常写入，历史原件不改写。当前三组均已发布优化入口，本次未启动；[默认与恢复](offline_execution_zh-CN.md)，[后两组交付](../../artifacts/offline_training_defaults_20261024/README.md)。

2026-10-24，源码中的`run_training_round`停止在内存保留全部step字典，改为流式累计本次会话步数、位置数、loss总和及首尾值；主`training_metrics.jsonl`仍逐步写入，汇总返回字段保持。当前104k仅policy组已有独立冻结发布并接入脚本，停止于115000步、未重启；其他既有冻结实验需另行发布才会采用。[验证与恢复命令](../../artifacts/training_streaming_20261024/README.md)。

2026-10-23新准备的[R21–R37仅短期value离线实验](../../artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023/README.md)使用本页轻量记录实现：逐步主指标、参考身份和SHA，不写大抽样账本。离线周期点沿用此前每1000步，另保留全部5%节点；下文500步周期仅适用于所述在线分支。

2026-10-23按用户要求，当前`random_budget_iteration`、`hybrid_train`及`offline_objectives`／`offline_mixed_budget`使用轻量抽样记录：不再写policy、value两份`.bin`与两份`.steps.jsonl`，恢复时也不复制整份主指标到`.metrics.jsonl`。每步指标统一保存在`training_metrics.jsonl`；历史记录及T7归档不删除、不重写。

## 保留的记录

- `training_metrics.jsonl`：每步policy CE／KL／熵、value MSE、梯度范数、采样来源计数和耗时；这里是当步batch指标，不是固定留出结果或100步平均。
- `usage/`或`references/`内的小型会话JSON：每路batch大小、数据身份与`az-reference-sha256-only-v1`格式标识。联合会话JSON的`sources`指向真实存在的身份JSON，不再指向未生成的`.bin`。
- 主指标的`reference_sha256`与checkpoint的`last_batch_sha256`：保留原算法计算的batch摘要；不保存逐样本顺序，不能凭摘要反推出样本或精确覆盖率。
- 配置、代码快照、数据池／split／留出清单、模型、AdamW、采样及Torch RNG：继续保存。断点恢复不依赖完整抽样账本。

均匀有放回子池有N个可抽样局面、累计抽M次时，平均曝光为M/N，期望覆盖率为`1-(1-1/N)^M`；混合采样按子池分别计算。实际来源计数仍可由主指标汇总，不把统计期望当作实际逐局面覆盖率。完整顺序、逐局面次数不再默认产生。

旧的显式记录／覆盖审计链路保留兼容：`ReferenceWriter`的旧调用与旧`alphazero_loop`／`offline_train`仍支持原始记录；`HybridReferenceWriter(..., persist_records=True)`可供专项审计显式启用。本次改变的是上述当前在线／近期离线链路的默认行为。已经冻结的历史源码和已结束实验不会被新工作区代码自动升级；恢复旧实验须依其冻结版本解释记录。

## 在线checkpoint与留出频率

当前[R31起点10k新局／60k池分支](../../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/README.md)的周期恢复点改为每500步；每轮七个固定诊断点仍为`0,100,250,500,1000,2000,3000`。保存集合取并集，重叠节点只写一次：

`0,100,250,500,1000,1500,2000,2500,3000`

0步保存为`initial.pt`，其余为`step-*.pt`，轮末另外保存现有`final.pt`，正常完整一轮共10个`.pt`文件、9个不同训练节点。显式限制会话步数或中断后续跑时，仍可能额外保存会话终点，确保有限调用的训练进度可恢复。旧轮的50步checkpoint不自动清理。

固定2048位置小面板和每六轮固定此前六轮完整validation的大面板，仍只在上述七个诊断点评测；1500、2500步仅保存周期恢复点。每步训练batch指标、每50步控制台进度输出保持。保存周期变化不改变训练更新、抽样、留出划分、教师升级或棋力评测节奏。

恢复允许仅更换`checkpoint_every_steps`，原始`runs/config.json`及旧checkpoint中的配置保持当时身份；新会话／轮次配置及新checkpoint记录实际周期。学习率、batch等训练配置仍按现有一致性检查，不因此放开。

## 中断记录及恢复边界

恢复时主指标会删除最后checkpoint之后的未提交步骤，再从该节点追加，因此主指标足以分析最终提交的训练轨迹。停写`.steps.jsonl`后，不再默认保存被裁掉尝试的逐步耗时，不能精确复算这部分重跑成本；旧`alphazero_analysis`缺少会话记录时应报告不可用，不能以零补齐。

当前在线命令仍为`bash scripts/run_r31_10k_p60k_iteration_20261023.sh`，已改用独立`source_recording_checkpoint_20261023`快照。旧六轮调度快照、模型、native、牌池和固定评测器保持原身份；发布差异、工程验证及只读检查见[交付记录](../../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/recording_checkpoint_20261023/README.md)。本次不自动恢复训练，也不启动采集或棋力赛。
