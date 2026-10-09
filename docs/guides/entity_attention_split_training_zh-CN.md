# Policy近期窗口与detach value续训

2026-10-09已接入可选[行为搜索复用M4标签的16轮分支](entity_attention_avg4_iteration_zh-CN.md)：保留本配置200局／100步／lr1e-4及分头窗口，行为1份＋追加3份完整S1024/N1，2/4/8/12/16轮各对冻结16500起点100pair。追加标签已改8个CPU worker＋单GPU owner，已完成12轮／17700步，前三个完整对战面板未确认稳定收益，第12轮对战未完成；固定历史采样与梯度诊断已完成，暂不改正式配额。下文“平均标签未接入”描述旧实验当时状态；本文件的原命令仍运行单次标签配置。

2026-10-08工程更新：[连续自迭代对照套件](entity_attention_iteration_suite_zh-CN.md)已新增冻结的`learning_rate_override`与可选`policy_recent_rounds`，实际AdamW lr与checkpoint配置同步，保留模型/moments/RNG并支持严格恢复。当前长实验精简为按B→C→D运行三组各seed1×16轮、仅测冻结起点趋势；B已完成前两轮／16600步及首次评测，C/D未运行。lr覆盖已实现，低lr多轮组未运行；本文件的原正式配置与历史结果保持原适用范围。

同日完成[冻结encoder、只练critic的B2账本回放](../../artifacts/entity_attention_critic_only_replay_20261008/README.md)：从16550保留AdamW状态重放B2原有50步value批次，整15代3000局评估；训练MSE 0.813065→0.798345，留出0.802914→0.789075（Δ=-0.013839，整局配对95%区间[-0.017369,-0.010276]），正式16600留出为0.808967。encoder/policy参数及其优化器状态均未改变。支持policy驱动的共享表征变化干扰value拟合，确认固定表征下critic可以学习；最新B2只有12局留出，单代改善仍未确认，未测棋力。本次是独立诊断，没有改动正式训练配置。

随后完成[16500／16600 encoder的相同新critic拟合对照](../../artifacts/entity_attention_encoder_critic_refit_20261008/README.md)：critic相同随机初始化、AdamW动量清零、同一15代数据／500个batch，记录0/50/200/500步。500步训练MSE 0.803989／0.803363，留出0.792473／0.792415，留出差值95%区间[-0.001835,+0.001707]；200步后曲线基本重合，未发现16600整体value可拟合性明显下降。更符合critic未及时适应共享表征变化的解释，500步仍未证明拟合上限；未更换正式模型或评估棋力。

最新完成[原16600 critic冻结主干适配](../../artifacts/entity_attention_critic_adaptation_strength_20261008/README.md)：保留原critic和AdamW状态，额外0/50/200/500步的完整留出MSE 0.808967／0.789053／0.787500／0.786814，policy/encoder及其优化器状态不变。四个诊断模型已保存；搜索对战脚本准备了三节点各100pair筛查、选定节点独立2000pair确认、前中期／后期替换各1000pair，后续三个100pair DEV已完成，50／200／500额外步数得分51.75%／46.0%／45.75%，区间均跨50%；HEAD及阶段替换确认未完成。

2026-10-06实现，2026-10-07更新运行状态。用户确定的新方案已接通正式`hybrid_train`训练、fork、断点恢复、来源账本及固定PPO评测。配置为`configs/entity_attention_v1/hybrid_split_policy_value.json`，从已停止的S1024/N1分支第50新轮、14500步创建独立分支。用户实跑已完成20新轮、16500步；第10/15/20新轮固定PPO得分70.75%/66.50%/64.00%，各200局有效，点估计尚不证明长期增益或退化。最新模型的[四组目标噪声实验](../../artifacts/entity_attention_target_noise_20261007/README.md)确认世界抽样影响大于根噪声，未改动此训练配置。

此前[最新单轮份额12.5%→18.75%→25%的候选设计](../archive/entity_attention_fresh_share_zh-CN.md)保留为历史设计记录；本次采用下面的分头窗口，不执行旧阶梯，也不重启原离线数据日程。固定数据[policy/value消融](../../artifacts/entity_attention_policy_value_ablation_20261006/README.md)提供动机，本次有限验证确认实现与恢复，尚未证明长期棋力收益或最佳窗口。

2026-10-07新增[三种搜索教师的冻结采集与短训对照](../../artifacts/entity_attention_chance_training_probe_20261007/README.md)：同一16500步起点，各采200局并固定训练500步，沿用value detach和独立clip1。该诊断为100%本组新train数据，不混入正式800/3000局历史窗口，不修改本配置或自动恢复正式长训。

2026-10-07报告复核新增[平均标签短训结果与自迭代分析](../reports/entity_attention_search_teachers_zh-CN.md#九如何改善当前自迭代先改变标签再判断剩余限制)：平均标签学生相对单次标签学生有拟合与直接对打优势，尚未确认超过起点。正式分头20轮的本轮留出policy KL每轮更新后均上升；描述性统计及输入哈希见[修订统计](../../artifacts/entity_attention_target_averaging_20261007/review_statistics.json)。推荐下一观察段先保留200局/100步、policy近4轮/value3000局及独立裁剪，只改变标签生成；该分析时点尚未接入平均标签；后续行为复用M4已完成12轮，见[M4指南](entity_attention_avg4_iteration_zh-CN.md)。以下命令仍对应原单次标签配置。

2026-10-08新增[正式回放学习率×步数探针](../../artifacts/entity_attention_formal_lr_step_20261007/README.md)：不动本配置的采样、detach、batch 与裁剪，从16500步起点逐条重放round-019已记录的正式账本，三条轨迹只把优化器lr改为2.5e-5/5e-5/1e-4并记录25/50/100步节点，边界节点用同口径第二固定账本延伸到125/150步。从round-018重放100步与正式round-019检查点逐张量一致（模型与AdamW max_diff全0），起点诊断与正式validation一致。2.5e-5的recent留出KL随步数继续下降到0.40525@150（起点0.42109、1e-4@100=0.41977），但三档lr的policy梯度99–100%步被裁剪，且50-pair面板全部区间跨0；DEV最高节点2.5e-5@150=60.0%在独立HEAD种子上41.5%（对照46.5%），未复现。因此**不据此改动本配置**；冻结的算法级lr覆盖已随趋势套件实现；低lr的C组未运行，长期收益未验证。

## 训练与采样

| 项目 | 本配置 |
| --- | --- |
| 起点 | `runs/entity_attention_hybrid_s1024_n1_from_round9_seed1/rounds/round-049/checkpoints/final.pt`，14500步 |
| 每轮 | 上一轮最新模型采200局，训练100步；教师始终S1024/N1，权重每轮更新 |
| Policy | 最近4个实际采集轮，约800归档局；仅train split，按局面均匀、有放回抽256个样本 |
| Value | 3000归档局FIFO内，抽128个近期4轮train样本＋128个其余较早train样本；两桶互斥 |
| 模型 | 共享attention trunk；policy正常反传，value head输入`h.detach()`；value批次不建立trunk反向图 |
| 优化 | 保留一个AdamW及原参数顺序、moments；lr和weight_decay均1e-4，两路loss权重均1 |
| 裁剪 | trunk＋policy专用参数与critic参数分别clip1.0；没有全模型联合裁剪 |
| 样本成本 | 每步policy256＋value256，共512个抽样引用、两次state编码；每轮仍只采200新局 |
| 留出 | 整局split；本轮及近期验证使用完整留出局，训练池均排除validation；forced单候选的policy_weight仍为0 |
| 采集执行 | 1 GPU owner、8 search workers；共享槽2、固定地址native CUDA Graph、owner单缓冲；B[16,32,64,128,200]、C[16,32,48,64]，FP32、batch上限200 |
| 内存 | 配置冻结总16 GiB缓存预算；本机命令用执行覆盖8 GiB，其中在线两桶共享同一个4 GiB LRU，离线固定诊断使用另一缓存 |
| 棋力 | 每5个新轮次，固定PPO lifetime5,512,112，双方S1024/N1、100 paired seeds/200有效局；与已有第50轮61.75%补测使用相同game/policy/search/bootstrap种子 |

800/3000均包含留出局，实际训练局数、局面数随split和局长而变。“最近4轮”依据活动回放的采集轮索引选择，包含fork继承的数据，避免先从零攒800局。均匀局面采样意味着最新教师约占policy的25%、value的12.5%，比例不会精确固定为这两个数。验证起点的真实份额为24.13%和12.07%；每轮`replay_snapshot.json.split_training.teachers`会记录各教师身份与按局面数计算的期望比例。

两路各自用本batch的有效权重和归一化loss，不能把512个样本拼起来同时训练两种loss。Value不反传trunk，但policy更新仍会改变共享特征，所以value函数并不会固定；更旧终局标签也来自旧策略，不等于当前策略下完全无偏的value。跨轮保留的AdamW动量包含历史联合训练信息，没有重置。

## 数据替换与观察

第1新轮从14500步到14600步，最近4轮由父分支最后3个采集轮＋本轮组成；第4新轮起policy窗口全部来自新分支。Value的3000局窗口每轮替换200局，第15新轮起全部来自新分支，之后继续滑动。`--forever`没有总轮数上限。

`training_metrics.jsonl`记录policy CE/KL、value MSE、各自权重和、两组裁剪前范数/是否裁剪、每路batch256和总batch512。训练loss是两个不同采样分布上的loss之和，只能作辅助诊断；按`source_positions.policy_round_*`与`value_round_*`核对实际教师使用量。

每轮`validation_before.json`和`validation_after.json`含`current`、`recent`，分别对应本轮和近4轮整局留出；同轮前后比较使用相同局面。After另有原`offline_fixed`和`online_fixed`，用于历史保留能力，不代表最新教师目标；跨轮recent目标会变化，也不能直接把不同轮CE大小当棋力。

`source_usage.json.split_training`汇总教师SHA、各桶规模、期望比例及实际按头/采集轮抽样量；`usage/policy_recent`、`value_recent`、`value_history`分别保存可恢复二进制引用账本。原`quotas:[0,256]`表示每路全在线，split字段和`batch_size:512`记录实际双路用量。

评测写入`rounds/round-XXX/evaluation/mcts_vs_fixed_anchor.json`。PPO使用actor_projection=true，attention使用false，各自独立设置；对手checkpoint SHA在配置中冻结。Score不决定是否更新、不会回滚；已有工程失败检查仍会要求恢复完成有效评测。用固定PPO score、配对区间和近期留出变化综合判断，不因一次高分宣称进步。

## 启动与恢复

从仓库根目录首次启动：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src TORCHINDUCTOR_COMPILE_THREADS=2 python -u -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_split_policy_value.json \
  --fork-from runs/entity_attention_hybrid_s1024_n1_from_round9_seed1 --fork-round 50 \
  --output runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1 \
  --replay-cache-mib 8192 --forever
```

中断后恢复同一输出：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src TORCHINDUCTOR_COMPILE_THREADS=2 python -u -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_split_policy_value.json \
  --output runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1 \
  --resume --replay-cache-mib 8192 --forever
```

若希望先跑15个新轮次，把`--forever`替换为`--online-rounds 15`，完成后再用resume/forever接续；轮数是该新分支的绝对目标。首次命令会复制checkpoint和元信息，原始NPZ仍引用父run路径，需要保留父数据。不会重新离线训练或重置优化器/RNG。

只检查配置和源数据：

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_split_policy_value.json \
  --fork-from runs/entity_attention_hybrid_s1024_n1_from_round9_seed1 --fork-round 50 --check-config
```

实现和有限验证见[交付记录](../../artifacts/entity_attention_split_training_20261006/README.md)。正式输出未被验证任务创建；验证只写独立`runs/entity_attention_split_training_validation_20261006`，没有采集新对局。
