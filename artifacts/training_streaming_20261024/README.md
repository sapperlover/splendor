# 流式训练统计与主进程传输优化

2026-10-27后续状态：这里记录的停止／准备／仅当前组适用属于当时工程发布范围；后续104k仅policy、双辅助及44k仅value均采用已发布优化完成训练、选点和两组对战，见[完成汇总](../../docs/reports/auxiliary_offline_results_20261027_zh-CN.md)。原工程验证数值与冻结发布不改写。

2026-10-24后续：后两组现已另建发布并全部接入相同三项优化，本次均18GiB／4 worker；后三项设为未来新建离线默认，缓存按池大小配置。该变更见[后两组与离线默认交付](../offline_training_defaults_20261024/README.md)。本目录发布与原验证保留，下面仅policy的适用范围及后两组未改为上一次任务时点记录。

2026-10-24，按用户要求停止当前104k仅policy辅助组及外层串行队列。最后提交点为 **115000步**，本次只准备优化后的源码与恢复命令，未恢复正式训练。停止身份见[stop_request.json](stop_request.json)。

## 改动

- `run_training_round`不再将每个step字典保存在`step_records`列表。只累计本次会话步数、位置数、loss总和及首尾loss，内存占用不随步数增长；恢复后仍只统计本次会话。每步`training_metrics.jsonl`、回调、checkpoint、验证和返回字段保留。均值为同一批loss的算术平均；浮点加法的末位舍入可与Python内置`sum`略有差异。
- policy/value分离训练时，policy分支只计算policy logits；value分支只上传value标签，不再复制用不到的policy标签。continuation value／辅助分支使用已有state-only格式，跳过候选动作特征与候选mask的H2D传输，移除其无用候选检查同步。状态数组直接引用原batch，无额外NumPy深拷贝；worker端仍使用原collator／IPC。
- 主指标和辅助指标合并为一次标量批量回读；每100步的各任务encoder梯度范数也合并回读。有限性检查使用GPU张量合并，保留一次显式检查；非有限输出和非有限梯度仍在optimizer更新前报错。

保留18GiB共享只读池、4个供数worker和独立256MiB验证缓存。模型、AdamW、学习率、标签、抽样顺序、RNG消费边界及验证／早停节奏保持。本次发布仅接入当前policy104k组，后续双辅助与44k组仍使用各自原冻结发布和配置。

消除`step_records`解决了明确的线性留存来源，并不代表进程RSS绝对不变：共享页触及、Python／Torch分配器和首次验证缓存仍会造成有界预热变化。

## 验证

- [tests.xml](tests.xml)：87项相关测试通过。包含恢复后的会话统计、零新增步、10000次模拟更新仅存活至多两个step字典、state-only输出／梯度逐位一致、四种非有限故障在参数更新前退出，以及原共享只读／RNG／恢复回归。
- [gpu_smoke.json](gpu_smoke.json)：RTX4050、68局真实训练数据、B256＋256、4 worker、115000步真实checkpoint。新旧实现连续8次更新的模型、梯度、AdamW、所有指标及CPU／CUDA／抽样RNG逐位一致，显式覆盖第1、8次的全部任务梯度诊断。
- 同一短测采用旧→新→新→旧顺序，每段20次预热＋100次计时。汇总旧约26.61步/秒，新约27.11步/秒，仅约 **1.9%**，单段波动重叠，不能认定已有明显稳定提速，更不能外推全池吞吐。此前中间实现和并发CPU回归时的结果分别保留在`gpu_smoke_initial.json`、`gpu_smoke_with_cpu_regression.json`，后者不作为速度结论。
- profiler普通步的`aten::_local_scalar_dense`计数197→179，其中含CPU上的AdamW step计数器，不能当成CUDA同步次数。
- [verification.json](verification.json)：实际入口、冻结源码123个文件、恢复identity及固定留出digest核对通过；251个历史checkpoint／完整留出／旧发布文件哈希保持。87项回归中的非有限故障测试触发了原报错路径的requires-grad标量转换提示，不影响更新前失败语义。

## 恢复命令

仓库根目录，恢复当前组的完整训练→选点→greedy1000pair→搜索A500pair：

```bash
bash scripts/run_attention128_policy104k_20261024.sh
```

恢复原串行链，当前组全部成功后继续双辅助、44k仅value组：

```bash
bash scripts/run_attention128_auxiliary_queue_20261024.sh
```

二者任选其一。脚本自行使用conda环境及正确PYTHONPATH；当前组自动恢复最后提交的115000步模型／AdamW／RNG，未提交日志按原机制回退重算。恢复时仍需重新预热共享池。`train`参数只训练与选点，不运行棋力面板，不构成串行链成功边界。

当前脚本已指向本目录[resume_policy.py](resume_policy.py)，实际源码为`runtime/entity_attention_128x2_opponent_policy_r21_r37_20261024/source_streaming_20261024`；来源、差异和所有文件SHA见[source.json](source.json)。18GiB／4 worker配置沿用上一发布。原计划、原源码和checkpoint保持，未修改旧冻结发布。

## 后续可优化方向

若继续优化，应优先单独测量固定容量的pinned-memory staging、`non_blocking` H2D与下一批双缓冲重叠，并合并小块传输。只设置`non_blocking=True`不能保证现有普通NumPy／pageable内存实现传输与计算重叠；也不能将18GiB整个历史池锁页。当前尚未实现这条路径。

本次短测收益小，不能据此继续增加worker。进一步分析需要分开测CPU提交、IPC、GPU计算和实际同步；编译／融合优化器也可能减少Python及小kernel开销，但属于另外的数值与恢复验证范围，本次未切换。
