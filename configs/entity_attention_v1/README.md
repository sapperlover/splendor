# Entity attention v1 配置

操作和结果说明见 [专题指南](../../docs/guides/entity_attention_offline_zh-CN.md)。所有命令从仓库根目录执行。

| 配置 | 用途与前置条件 |
| --- | --- |
| `offline_attention.json` | 新结构随机初始化，全部十轮原 split，batch256、12,045步、4 GiB LRU；写 `runs/entity_attention_offline` |
| `offline_deep_sets.json` | 旧结构也随机初始化，与新结构同数据、seed、优化及抽样预算；写 `runs/deep_sets_offline` |
| `offline_smoke.json` | 首轮前6局、batch16、8步、验证16、64 MiB LRU；由有限验证脚本使用 |
| `selfplay_smoke.json` | 新 smoke 导出模型、native_cuda_pool、2局、S16/N1、compiled/reused、projection=false |
| `evaluation_smoke.json` | 新 smoke 对旧θ5，native_batched、1 pair、S16/N1；分别配置新false/旧true projection |
| `mcts_vs_theta5.json` | 正式离线 final 对旧θ5，双方S1024/N1、50 pairs；需要先生成新 final |
| `greedy_vs_theta5.json` | 正式新旧裸网 greedy、50 pairs；需要先生成新 final |
| `greedy_vs_ppo_5_5m_100pairs.json` | 新离线final vs PPO lifetime5,512,112，双方greedy、100 pairs/200局 |
| `mcts_s1024n1_vs_ppo_5_5m_100pairs.json` | 同一双方checkpoint，双方MCTS S1024/N1、100 pairs/200局，compiled/reused、新false/旧true projection |
| `offline_step{1000,5000}_greedy_vs_ppo_5_5m_100pairs.json` | 离线训练1000/5000步对PPO lifetime5,512,112，双方greedy，各100 pairs |
| `offline_step{1000,5000}_mcts_s1024n1_vs_ppo_5_5m_100pairs.json` | 相同离线节点与PPO双方S1024/N1，各100 pairs；与final面板复用相同game/policy/search/bootstrap seed |
| `online_3x200.json` | 早期短阶段候选保留，不是近期M4配置 |
| `online_10x1000.json` | 历史自迭代：离线final起点，10轮×1000局、同时调度1000局、S1024/N1；每轮对固定起点搜索/greedy各50 pairs；[命令与验证](../../docs/guides/entity_attention_self_iteration_zh-CN.md) |
| `collection_static_graph.json` | 可选采集预设：W8/A1000/B500、shared_slots=2、固定B64/256/500、native-cudagraph、owner单缓冲；只验证短窗口收益，[报告与命令](../../docs/reports/entity_attention_execution_speed_zh-CN.md) |
| `hybrid_static_graph.json` | 从随机初始化离线5000步，逐渐混入自迭代；每轮1000局/500更新，第16轮起纯在线；共享槽2＋固定地址Graph、owner单缓冲；replay总上限16 GiB，混合时两池各8 GiB；[启动与恢复](../../docs/guides/entity_attention_hybrid_zh-CN.md) |
| `hybrid_cpu_smoke.json` | 独立有限CPU链路验证：离线4步、每轮2局/2更新、batch8、S2/N1；不用于正式训练或棋力结论 |
| `hybrid_fork_s512_n2.json` | 从第九轮9500步分支，继承AdamW/RNG；S512/N2、100局/100步、在线窗口2000局、active100/B128、共享槽2/固定地址Graph/owner单缓冲；[命令与数据替换](../../docs/guides/entity_attention_hybrid_fork_zh-CN.md) |
| `hybrid_fork_s1024_n2.json` | 同上，仅采集总模拟改为1024，每世界512；其余训练/采样/评测参数相同 |
| `hybrid_fork_s1024_n1.json` | 从原第九轮9500步恢复，S1024/N1、AdamW/RNG接续，200局/100步、3000局/15个教师；B200/Graph[16,32,64,128,200]，共享槽2/owner单缓冲，每5轮对theta_5000评测；已完成50新轮/14500步后停止。原resume仍全在线池均匀采样，不执行最新25%；[状态与命令](../../docs/guides/entity_attention_hybrid_s1024_n1_zh-CN.md) |
| `hybrid_split_policy_value.json` | 从当前N1第50新轮/14500步fork；policy近4轮，detach value保留3000局/近4轮50%，各batch256、独立clip1；200局/100步，固定PPO双方S1024/N1每5轮200局；[启动与恢复](../../docs/guides/entity_attention_split_training_zh-CN.md) |
| `iteration_avg4_behavior_20261009.json` | 16500步分支，行为＋3份完整S1024/N1平均policy标签；16轮200局／100步／lr1e-4，原policy4轮／value3000局；追加标签8个CPU worker／总256根／单GPU batch200；每轮256局面诊断，2/4/8/12/16轮对冻结起点各100pair，无硬RAM／磁盘门槛；[启动／恢复](../../docs/guides/entity_attention_avg4_iteration_zh-CN.md)，已完成12轮／17700步，前三个完整对战面板未确认稳定收益，第12轮对战未完成；固定历史采样与梯度诊断已完成，暂不改正式配额 |
| `iteration_trend_20261008.json` | 未完成趋势方案：共同16500步起点，按B→C→D完成三组各seed1×16轮；2/4/8/12/16轮对冻结起点100pair，4GiB缓存/16GiB内存上限，无磁盘门槛；[同一启动/恢复命令](../../docs/guides/entity_attention_iteration_suite_zh-CN.md) |
| `online_smoke.json` | 本次两轮GPU短测：每轮4局、2步、S16/N1，搜索/greedy各1 pair；独立runtime输出 |

配置交付时没有启动正式任务；随后新结构离线已完成12045步及六个PPO搜索/greedy面板，原混合run完成9轮、小步N1分支完成50轮。本次[共同PPO补测配置与结果](../../artifacts/entity_attention_strong_anchor_20261006/README.md)独立归档，没有改写正式训练配置。旧[最新单轮份额](../../docs/archive/entity_attention_fresh_share_zh-CN.md)保留为历史设计，其JSON不能作为CLI配置；分头窗口/detach分支后续已完成20新轮／16500步；上述40项回归和短GPU验证是当时的工程检查，不代表长期收益确认。

两个离线配置保持原冻结4 GiB值，实际运行建议加`--replay-cache-mib 8192`；resume同样可加此执行参数，无须修改配置。`history_batch_size`在离线入口无作用；离线只有一个覆盖全选定train positions的池。新结构不支持actor projection，混合评测通过每方`inference`明确区分。

[旧四组配置](../archive/entity_attention_iteration_suite_20261008.json) 已移入 `configs/archive/`，是被替代的方案历史快照，仅供复核旧准备记录；当前入口不再支持其协议。
