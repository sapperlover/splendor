# Entity attention v1 配置

2026-10-22新增[random_budget_r20_s256n4_noble12_20261022.json](random_budget_r20_s256n4_noble12_20261022.json)：继承raw R20状态、AdamW/RNG及24k活动池，从第21轮无限续跑；25%高预算S256N4（基础simulations仍为总1024）、75%低预算S128N1（low_simulations128、low_determinizations1），新采集用12贵族。其余有效在线配置沿用原分支，旧21轮残留已授权删除。[配置身份、验证与恢复](../../artifacts/entity_attention_r20_s256n4_noble12_iteration_20261022/README.md)。下文原在线分支“完成17轮”是历史时点，实际已完成20轮。

后续状态（2026-10-21）：下方attention128同8万局对照已由用户启动，当前训练仍在运行；“正式训练未启动”保留为首次配置交付时的状态。当前模型、数据及冻结源码在[本次存储审计](../../artifacts/checkpoint_storage_audit_20261021/README.md)中列为保留，未中断或改配置。

2026-10-21新增[同MLP的8万局attention128从零离线对照](../../artifacts/entity_attention_128x2_offline_80k_20261021/README.md)：`offline_128x2_no_aux_80k_lr1e4_20261021.json`使用现有128×2／4头／FFN512购买关系预设、583682参数，lr1e-4、16GiB缓存、policy高预算/value全预算各256、270557步，每5%同固定子集与完整留出。除模型及独立输出路径外与MLP配置相同，正式训练未启动；命令`bash scripts/run_attention128_offline_20261021.sh`只训练＋KL+MSE选点，不自动对战。

新增[random_budget_37423_p6k_v24k_20261019.json](random_budget_37423_p6k_v24k_20261019.json)：37423步继承AdamW、lr2e-5，每轮4k局／1000步，25% S1024＋75% S128，FP/PTP关闭；原R24–43共20k局接入24k池，旧局随机淘汰，旧policy固定25%子集并控制累计新局quota为1/6、1/3…。[启动／恢复及限制](../../artifacts/entity_attention_random_budget_iteration_20261019/README.md)。2026-10-20已完成17轮，第18轮中断未运行。采集为B500/pipeline2；后续搜索评测改为500pair并复用采集execution的其他参数（12 worker、1000活跃局、B500及native CUDA Graph），搜索owner单槽，双方仍全程S1024、无根噪声。原命令可恢复；旧评测保留200pair，已准备面板沿用冻结配置。

新增[collection_randomized_budget.json](collection_randomized_budget.json)：新版128×2的37423步教师，25% S1024＋75% S128，根Forced Playouts k=2＋Policy Target Pruning；仅高预算policy监督、全部步骤value监督。待运行采集示例；[用法与在线训练配置位置](../../docs/guides/search_training_zh-CN.md)。

2026-10-19状态核对：新版128×2、无辅助／仅回合消融均已完成，结果见各专题；256×4双辅助保持中断。7个离线run的中间checkpoint与详细账本已[集中归档](../../artifacts/repository_archive_20261019/README.md)，配置JSON与冻结路径不变；只读复核需恢复归档挂载；恢复训练前按清单撤回对应run的迁移，避免写入冻结归档。

2026-10-19新增[新版128×2、无辅助头、lr1e-4从零训练](../../artifacts/entity_attention_128x2_relations_no_aux_r24_r43_20261019/README.md)：`offline_128x2_relations_no_aux_r24_r43_20261019.json`保留新版购买关系与value读取，R24–43原20000局、曝光12次/68041步、每5%完整留出，选KL＋MSE最小完整模型后对R43测1000pair greedy／200pair S1024。其余配置逐字段沿用256实验；`offline_scratch.value_auxiliary=null`实际不注册两个辅助头。

2026-10-19新增[128×2](model_relations_128x2.json)和[256×4](model_relations_256x4.json)购买关系预设：`market_relations=true`、`value_readout="global_players_market"`，市场与双方明保留牌共用8维公开购买关系＋全局/双方玩家/市场均值的value读取；无辅助参数分别583,682／3,828,226。它们是`model_config`片段，旧配置不自动切换；辅助升级会保留两项开关。明保留扩展复用原开关和参数，盲保留不追加关系。[结构、兼容性和有限验证](../../artifacts/entity_attention_relations_20261019/README.md)。

操作和结果说明见 [专题指南](../../docs/guides/entity_attention_offline_zh-CN.md)。所有命令从仓库根目录执行。

| 配置 | 用途与前置条件 |
| --- | --- |
| `offline_256x4_aux_r24_r43_20261014.json` | 256×4随机初始化、双辅助各0.25、PSW关闭，原R24–43的20000局，曝光12次／68041步，每5%完整留出评估并按KL＋胜负MSE选完整checkpoint，随后对R43测1000pair greedy／200pair S1024；[运行与续跑](../../artifacts/entity_attention_256x4_aux_r24_r43_20261014/README.md)。本配置使用 `splendor.offline_scratch`，不能交给旧 `offline_objectives` CLI。 |
| `model_256x4.json` | 新模型的 `model_config` 片段：hidden256、heads8、layers4、FFN1024、dropout0，3,628,034参数；嵌入新训练配置或交给`build_actor_critic`，不是完整训练CLI配置。历史默认仍128×2，旧R43不能直接按大模型续训。[结构与有限验证](../../artifacts/entity_attention_256x4_20261014/README.md) |
| `offline_r24_r43_policy_surprise_20261013.json`、`offline_r24_r43_value_auxiliary_20261013.json` | 原R43起点，绝对R24–43的20000局固定池，各22681步；50% PSW与双辅助头（各权重0.25）分别训练。训练完成，随后独立执行final对原R43各1000pair greedy：51.350%／53.900%，辅助组区间高于50%。[命令](../../artifacts/entity_attention_r24_r43_offline_objectives_20261013/README.md)、[结果](../../artifacts/entity_attention_r24_r43_offline_objectives_20261013/results.md) |
| `r9_online_lr2e5_1000_forever_20261011.json`、`hybrid_r9_online_lr2e5_1000_forever_20261011.json` | 原第9轮／38898步续训，继承第4–9轮6000局及原split；纯在线、1000局／1000步／lr2e-5／batch256、B500采集。1000步完整模型无条件升级；每轮greedy、绝对第12／15……轮搜索各200pair；无限循环，同一命令启动／恢复。已完成截至绝对第43轮的34轮续训；历史命令保留，重新执行将恢复／继续运行。[流程与命令](../../docs/guides/entity_attention_p4_pool_forever_plan_20261011_zh-CN.md) |
| `p4_pool_iteration_20261010.json`、`hybrid_p4_pool_20261010.json` | P4/34398步及连续AdamW，9轮1000局/500步，旧池／在线224／32渐变至第6轮32／224，第7轮起仅在线近6000局；采集B500、Graph档16/32/64/128/192/256/384/500。教师无条件升级；主线、同预算对照、全量留出及14面板已全部完成，终点搜索胜过三个已测对手，限定单seed/DEV，未确定最优参数。[完整报告](../../docs/reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)、[数据与命令](../../docs/guides/entity_attention_p4_pool_iteration_zh-CN.md) |
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
| `iteration_avg4_behavior_20261009.json` | 16500步分支，行为＋3份完整S1024/N1平均policy标签；16轮200局／100步／lr1e-4，原policy4轮／value3000局；追加标签8个CPU worker／总256根／单GPU batch200；每轮256局面诊断，2/4/8/12/16轮对冻结起点各100pair，无硬RAM／磁盘门槛；[启动／恢复](../../docs/guides/entity_attention_avg4_iteration_zh-CN.md)，已完成12轮／17700步，原早期面板未确认稳定收益；后续r4、r12各300pair扩样完成，分别50.83%、49.17%，区间均跨50%；固定历史采样与梯度诊断已完成，暂不改正式配额 |
| `iteration_trend_20261008.json` | 未完成趋势方案：共同16500步起点，按B→C→D完成三组各seed1×16轮；2/4/8/12/16轮对冻结起点100pair，4GiB缓存/16GiB内存上限，无磁盘门槛；[同一启动/恢复命令](../../docs/guides/entity_attention_iteration_suite_zh-CN.md) |
| `online_smoke.json` | 本次两轮GPU短测：每轮4局、2步、S16/N1，搜索/greedy各1 pair；独立runtime输出 |

配置交付时没有启动正式任务；随后新结构离线已完成12045步及六个PPO搜索/greedy面板，原混合run完成9轮、小步N1分支完成50轮。本次[共同PPO补测配置与结果](../../artifacts/entity_attention_strong_anchor_20261006/README.md)独立归档，没有改写正式训练配置。旧[最新单轮份额](../../docs/archive/entity_attention_fresh_share_zh-CN.md)保留为历史设计，其JSON不能作为CLI配置；分头窗口/detach分支后续已完成20新轮／16500步；上述40项回归和短GPU验证是当时的工程检查，不代表长期收益确认。

两个离线配置保持原冻结4 GiB值，实际运行建议加`--replay-cache-mib 8192`；resume同样可加此执行参数，无须修改配置。`history_batch_size`在离线入口无作用；离线只有一个覆盖全选定train positions的池。新结构不支持actor projection，混合评测通过每方`inference`明确区分。

[旧四组配置](../archive/entity_attention_iteration_suite_20261008.json) 已移入 `configs/archive/`，是被替代的方案历史快照，仅供复核旧准备记录；当前入口不再支持其协议。
## 2026-10-19：小模型辅助目标对照

新增`offline_r24_r43_no_aux_20261019.json`与`offline_r24_r43_turns_only_20261019.json`，除`value_auxiliary`外逐字段沿用旧小模型双辅助配置：无辅助为null，仅剩余回合为score_loss_weight0／turns_loss_weight0.25。两组R43 warm start各22681步，串行训练及final对R43的1000pair greedy／200pair S1024见[专题入口](../../artifacts/entity_attention_r24_r43_aux_ablation_20261019/README.md)。默认旧双辅助行为保留；单回合模型配置`predict_score=false`且不注册分差头。
