# 实验导航

2026-10-09整理。机器可读身份、范围和证据SHA见[experiment_index.json](experiment_index.json)。有效记录不等于已证明棋力提升；未完成实验中可能有有效的已完成子结果。

较早M4与固定数据诊断见[M4整合报告](../docs/reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md)，操作入口见[docs导航](../docs/index.md)。

2026-10-10完成[P4起点九轮自迭代](../docs/reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)：终点S1024/N1对P4／同预算旧池对照／P12＋V4得分58.750%／61.625%／58.125%，各200pair、区间均高于50%；含完整拟合矩阵、value阶段数据、资源口径和纯CPU复算证据，限定单seed与固定DEV。

2026-10-10新增[大池分支组合与greedy对战队列](entity_attention_recent_pool_crossplay_20261010/README.md)：三组搜索＋十组greedy，各200pair；[结果与进度](entity_attention_recent_pool_crossplay_20261010/results.md)和机器索引状态自动更新。未完成组不作为棋力证据。

2026-10-11新增[固定原第30轮完整value的三个policy对照＋第24轮对第30轮](entity_attention_r30_fixed_value_arena_20261011/README.md)：四组S1024/N1各200pair串行；[结果与进度](entity_attention_r30_fixed_value_arena_20261011/results.md)和机器索引随整组完成自动更新。

2026-10-11新增[原第30轮搜索预算对照](entity_attention_r30_search_budget_arena_20261011/README.md)：S1024/N1对无搜索greedy，随后S4096/N1对S1024/N1，各200pair；[结果与进度](entity_attention_r30_search_budget_arena_20261011/results.md)和机器索引随整组完成自动更新。

2026-10-11新增[第30轮对第24轮的1000pair greedy直接对战](entity_attention_r30_r24_greedy_1000pair_20261011/README.md)：原配置完整网络、双方无搜索、交换先后手共2000局；[结果](entity_attention_r30_r24_greedy_1000pair_20261011/results.md)与机器索引自动更新。

2026-10-11新增[第29轮终点：policy缩池／value扩池BCD](entity_attention_r30_head_pools_20261011/README.md)：使用截至第30轮的已有在线数据，B=3k/6k、C=6k/9k、D=3k/9k；各1000步联合训练、共同逐代留出指标及对P4V4的200pair greedy，无新采集或搜索对战。[结果与进度](entity_attention_r30_head_pools_20261011/results.md)与机器索引随链路完成更新。

新增[原第30轮policy与无根噪声S1024的一致性](entity_attention_r30_search_alignment_20261011/README.md)：124局完整留出、8283个有效policy局面，当前完整网络重新搜索两个独立N1世界seed；测Top-1一致率、访问KL和搜索自身随机差异，含前中期／后期划分。[结果](entity_attention_r30_search_alignment_20261011/results.md)完成后自动写入，不作棋力提升证明。

## 未完成实验／确认

| 实验身份 | 阅读入口 |
| --- | --- |
| `mcts-sweep` | [MCTS 50-pair 对照的实际状态](../docs/reports/mcts_sweep_zh-CN.md) |
| `entity-attention-chance-depth-2026-10-07` | [大S深度初筛（用户取消后续）](entity_attention_chance_depth_20261007/README.md) |
| `entity-attention-iteration-trend-seed1-2026-10-08` | [B → C → D 单种子趋势实验](entity_attention_iteration_trend_seed1_20261008/README.md) |
| `entity-attention-critic-adaptation-strength-2026-10-08` | [16600原critic适配与后续搜索棋力检验](entity_attention_critic_adaptation_strength_20261008/README.md) |
| `entity-attention-avg4-behavior-preparation-2026-10-09` | [行为复用M4连续训练：记录](entity_attention_avg4_behavior_20261009/README.md) |

## 有效记录

| 实验身份 | 阅读入口 |
| --- | --- |
| `entity-attention-r30-value-weight-2026-10-11` | [第29轮终点：value权重1／0.25／0的训练与评测已完成](entity_attention_r30_value_weight_20261011/README.md)、[整网结果及归因限制](entity_attention_r30_value_weight_20261011/analysis.md) |
| `entity-attention-r9-lr-step-arena-2026-10-11` | [lr2e-5×1000步 vs lr1e-4×500步：完成200 pair直接搜索对战](entity_attention_r9_lr_step_arena_20261011/README.md)、[53.250%，95% CI [48.500%, 58.000%]](entity_attention_r9_lr_step_arena_20261011/results.md) |
| `entity-attention-r9-ofat-2026-10-10` | [第九轮步数／学习率／在线池／batch单变量完成结论](entity_attention_r9_ofat_20261010/analysis.md)、[100–2000步全部55点矩阵](entity_attention_r9_ofat_20261010/steps_100_2000.md)、[完整曲线与区间](entity_attention_r9_ofat_20261010/results.md) |
| `entity-attention-p4-p9-crossplay-2026-10-10` | [新P9＋旧V4、旧P4＋新V9对起点，完成各200pair](entity_attention_p4_p9_crossplay_20261010/README.md)、[完整结果](entity_attention_p4_p9_crossplay_20261010/results.md) |
| `entity-attention-r9-step-extension-2026-10-10` | [第九轮固定6000局窗口：500／1000／2000／5000步留出拟合](entity_attention_r9_step_extension_20261010/README.md)、[结果表与差值区间](entity_attention_r9_step_extension_20261010/results.md) |
| `entity-attention-p4-pool-iteration-2026-10-10` | [九轮完整报告](../docs/reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)、[结果与复算证据](entity_attention_p4_pool_iteration_20261010/README.md) |
| `az-loop-seed1` | [AlphaZero seed1：五轮训练与逐代评测](../docs/reports/alphazero_seed1_zh-CN.md) |
| `native-inference-08` | [Native 推理吞吐：有效基线与结论边界](../docs/reports/inference_throughput_zh-CN.md) |
| `transfer-profile-09` | [Native 推理吞吐：有效基线与结论边界](../docs/reports/inference_throughput_zh-CN.md) |
| `gpu-pool-09` | [多搜索 worker 供给单 GPU：窗口与整轮结果](../docs/reports/multiworker_gpu_zh-CN.md) |
| `pipeline-profile-2026-10-02` | [采集、learner 与固定评测：分项时间统计](../docs/reports/pipeline_profile_zh-CN.md) |
| `end-to-end-speed-2026-10-03` | [端到端速度优化与完整验收](../docs/reports/end_to_end_speed_optimization_zh-CN.md) |
| `entity-attention-v1-2026-10-04` | [实体注意力模型与离线搜索经验学习](../docs/guides/entity_attention_offline_zh-CN.md) |
| `offline-training-cache-speed-2026-10-04` | [离线训练缓存瓶颈与执行参数优化](offline_training_speed_20261004/README.md) |
| `entity-attention-self-iteration-preparation-2026-10-04` | [新模型自迭代：10×1000](../docs/guides/entity_attention_self_iteration_zh-CN.md) |
| `entity-attention-execution-speed-2026-10-05` | [新模型固定形状、kernel启动与独立双缓冲](../docs/reports/entity_attention_execution_speed_zh-CN.md) |
| `entity-attention-search-screen-2026-10-05` | [Attention 自迭代教师 S/N 有限筛选（2026-10-05）](entity_attention_search_screen_20261005/README.md) |
| `entity-attention-hybrid-fork-2026-10-05` | [从第九轮创建 S512/N2 与 S1024/N2 训练分支](../docs/guides/entity_attention_hybrid_fork_zh-CN.md) |
| `entity-attention-worker-cleanup-2026-10-05` | [OOM 后孤儿进程清理与 owner 死亡退出修复](entity_attention_worker_cleanup_20261005/README.md) |
| `entity-attention-hybrid-graph200-2026-10-06` | [200局采集的Graph尺寸与恢复验证](entity_attention_graph200_20261006/README.md) |
| `entity-attention-memory-window-2026-10-06` | [S1024/N1 第31轮 RAM OOM：修复与历史窗口分析](entity_attention_memory_window_20261006/README.md) |
| `entity-attention-strong-anchor-2026-10-06` | [原第9轮与第50轮：共同PPO锚点补测及最新数据方案](entity_attention_strong_anchor_20261006/README.md) |
| `entity-attention-policy-value-ablation-2026-10-06` | [Entity attention：固定数据 policy / value 消融（2026-10-06）](entity_attention_policy_value_ablation_20261006/README.md) |
| `entity-attention-split-training-2026-10-06` | [分头窗口与detach value实现验证](entity_attention_split_training_20261006/README.md) |
| `entity-attention-target-noise-2026-10-07` | [冻结 attention 教师的目标噪声 A/B/C/D 实验](entity_attention_target_noise_20261007/README.md) |
| `entity-attention-chance-search-2026-10-07` | [随机节点／信息历史搜索：同预算 A/B/C/D 复测](entity_attention_chance_search_20261007/README.md) |
| `entity-attention-chance-strength-2026-10-07` | [同一冻结模型：随机节点与确定化搜索的100 pair棋力对照](entity_attention_chance_strength_20261007/README.md) |
| `entity-attention-chance-pw-2026-10-07` | [Chance progressive widening：实现与有限诊断](entity_attention_chance_pw_20261007/README.md) |
| `entity-attention-chance-pw-noise-2026-10-07` | [CPW k=0.25、α=0.5：A/B/C/D目标稳定性复测](entity_attention_chance_pw_noise_20261007/README.md) |
| `entity-attention-chance-pw-strength-2026-10-07` | [CPW k=0.25、α=0.5：同模型100 pair棋力对照](entity_attention_chance_pw_strength_20261007/README.md) |
| `entity-attention-chance-pw-greedy-2026-10-07` | [CPW k=0.25、α=0.5：对同权重greedy的100 pair](entity_attention_chance_pw_greedy_20261007/README.md) |
| `entity-attention-chance-training-probe-2026-10-07` | [随机节点／CPW 搜索与冻结教师短训](../docs/reports/entity_attention_search_teachers_zh-CN.md) |
| `entity-attention-target-averaging-2026-10-07` | [随机节点／CPW 搜索与冻结教师短训](../docs/reports/entity_attention_search_teachers_zh-CN.md) |
| `entity-attention-step-dose-2026-10-07` | [步数、有效学习率、value 与标签平均：200 局短训剂量对照](entity_attention_step_dose_20261007/README.md) |
| `entity-attention-formal-lr-step-2026-10-08` | [正式回放账本上的学习率×步数短训探针](entity_attention_formal_lr_step_20261007/README.md) |
| `entity-attention-iteration-suite-validation-2026-10-08` | [最终套件有限 CUDA 验证](entity_attention_iteration_suite_validation_v2_20261008/README.md) |
| `entity-attention-stage-search-2026-10-08` | [只搜后期／只搜前中期：同权重 greedy 对照](entity_attention_stage_search_20261008/README.md) |
| `entity-attention-critic-only-replay-2026-10-08` | [固定encoder，只练critic：B2的50步账本回放](entity_attention_critic_only_replay_20261008/README.md) |
| `entity-attention-encoder-critic-refit-2026-10-08` | [冻结16500与16600 encoder，重新拟合相同critic](entity_attention_encoder_critic_refit_20261008/README.md) |
| `decision-probe-v1-2026-10-09` | [固定 256 个独立真实局面的决策诊断](decision_probe_v1_20261009/README.md) |
| `entity-attention-policy-history-fixed-m4-2026-10-09` | [固定M4数据：历史采样与policy梯度诊断](entity_attention_policy_history_20261009/README.md) |
| `entity-attention-avg4-independent-delta-kl-2026-10-09` | [行为+M4 分支：独立 M4 配对 ΔKL 诊断（2026-10-09）](entity_attention_avg4_delta_kl_20261009/README.md) |
| `entity-attention-m4-diagnostics-conclusions-2026-10-09` | [M4 连续训练与固定数据诊断：完整结论（2026-10-09）](../docs/reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md) |
| `entity-attention-clip-sweep-2026-10-09` | [policy 组 clip 阈值回放探针（2026-10-09）](entity_attention_clip_sweep_20261009/README.md) |
| `entity-attention-scaleup-2026-10-09` | [不确定结果扩样实验（2026-10-09）](entity_attention_scaleup_20261009/README.md) |
| `entity-attention-hybrid-kl-trend-2026-10-09` | [5000步起点→9轮：固定留出、9代矩阵与基线复核（2026-10-09）](entity_attention_hybrid_kl_trend_20261009/README.md) |
| `entity-attention-hybrid-offline9500-2026-10-09` | [离线9500重放、固定留出与200pair面板（2026-10-09）](entity_attention_hybrid_offline9500_20261009/README.md) |
| `entity-attention-recent-pool-2026-10-09` | [29,000局 attention S1024/N1：34398步得分86.125%，终点55.50%；最终policy＋34398步value组合200pair得分92.25%，较34398步的配对差CI高于零](entity_attention_recent_pool_20261009/README.md) |

## 历史有效记录

| 实验身份 | 阅读入口 |
| --- | --- |
| `python-engine-rules-v2` | [Python 规则引擎历史验收](../docs/reports/engine_validation_zh-CN.md) |
| `ppo-5m-seed1` | [PPO 5M 单 seed 训练报告](../docs/reports/ppo_5m_zh-CN.md) |
| `native-early-panel` | [Native 推理吞吐：有效基线与结论边界](../docs/reports/inference_throughput_zh-CN.md) |
| `entity-attention-hybrid-s1024-n1-2026-10-06` | [原第九轮 S1024/N1 小步更新配置验证](entity_attention_hybrid_s1024_n1_20261006/README.md) |
| `entity-attention-hybrid-s1024-n1-replay3000-2026-10-06` | [原第九轮 S1024/N1 小步更新配置验证](entity_attention_hybrid_s1024_n1_20261006/README.md) |

## 失效或被替代

| 实验身份 | 阅读入口 |
| --- | --- |
| `inference-06-07` | [Native 推理吞吐：有效基线与结论边界](../docs/reports/inference_throughput_zh-CN.md) |
| `entity-attention-iteration-suite-2026-10-08` | [连续自迭代实验套件交付（历史方案）](entity_attention_iteration_suite_20261008/README.md) |
| `entity-attention-iteration-trend-2026-10-08` | [B → C → D 趋势实验交付（历史双种子准备）](entity_attention_iteration_trend_20261008/README.md) |

## 未核验记录

| 实验身份 | 阅读入口 |
| --- | --- |
| `native-multi-round-training` | [AlphaZero 式 policy/value 循环](../docs/guides/alphazero_zh-CN.md) |
| `az-theta5-10x1000-seed1-preparation` | [旧 θ5 起点的 10×1000 AZ 训练准备](../docs/archive/next_az_training_zh-CN.md) |

索引聚焦已复核的专题；未列入的既有基线和搜索汇总仍在原目录，未列入不代表失效。每个目录内的README、配置、原始汇总与分析入口共同限定其版本。目录身份和恢复产物不因本次导航整理而重建。
