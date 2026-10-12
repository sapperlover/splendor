# 文档导航

本文按阅读目的组织。状态以 [当前状态](current_project_state_zh-CN.md) 导航，以对应代码／配置和 [实验索引](../artifacts/experiment_index.json) 的原始证据核验。历史准备说明与被替代设计在 [archive](archive/README.md)，其日期不表示当前实现状态。

## 规则与契约

- [双人标准规则](splendor_standard_rules_zh-CN.md)，卡牌与常量见 `data/`。
- [引擎与 Observation](specs/engine_contract_zh-CN.md)。
- [特征 schema](specs/feature_schema_zh-CN.md)。
- [学习数据与模型契约](specs/learning_contract_zh-CN.md)。
- [PPO 训练、历史池与评测协议](specs/ppo_training_protocol_zh-CN.md)。

## 操作入口

| 目的 | 文档 |
| --- | --- |
| 搜索算法、后端与推理 | [搜索](guides/search_zh-CN.md)、[native 构建](guides/native_search_zh-CN.md)、[多 worker GPU](guides/multiworker_gpu_zh-CN.md) |
| 数据格式、采集与恢复 | [自对弈数据](guides/selfplay_data_zh-CN.md) |
| 训练与恢复 | [AZ 循环](guides/alphazero_zh-CN.md)、[实体注意力离线学习](guides/entity_attention_offline_zh-CN.md) |
| 近期大池离线训练 | [29,000局 attention S1024/N1固定池](../artifacts/entity_attention_recent_pool_20261009/README.md)，34398步对θ5000得分86.125%，最终完整模型55.50%；最终policy＋34398步value的双网络组合200pair得分92.25%，较34398步提高6.125个百分点，配对差CI [+2.000,+10.250] |
| P4起点九轮自迭代（完成） | [完整报告](reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)：终点搜索对P4／同预算旧池对照／P12＋V4得分58.750%／61.625%／58.125%，各200pair且区间高于50%；[流程与命令](guides/entity_attention_p4_pool_iteration_zh-CN.md)，旧池224／在线32渐变至第6轮32／224，第7轮起纯在线 |
| 第9轮终点长期在线续训（入口已实现，正式未启动） | [流程与统一启动／恢复命令](guides/entity_attention_p4_pool_forever_plan_20261011_zh-CN.md)：从38,898步继承原第4–9轮6000局窗口，1000局／1000步／lr2e-5，不用旧离线池，各轮1000步完整模型升级教师；每轮greedy、绝对第12／15……轮搜索，各200pair |
| 大池模型组合与greedy对战队列 | [三组搜索＋十组greedy，各200pair](../artifacts/entity_attention_recent_pool_crossplay_20261010/README.md)，串行、可中断恢复；[进度与结果](../artifacts/entity_attention_recent_pool_crossplay_20261010/results.md)自动更新 |
| 最近的连续实验 | [行为复用 M4](guides/entity_attention_avg4_iteration_zh-CN.md)，12／16轮完成，当前停止；[分头窗口](guides/entity_attention_split_training_zh-CN.md)，20轮完成 |
| 较早训练分支的复现 | [10×1000](guides/entity_attention_self_iteration_zh-CN.md)、[5000步混合分支](guides/entity_attention_hybrid_zh-CN.md)、[第九轮 N2 fork](guides/entity_attention_hybrid_fork_zh-CN.md)、[N1 小步分支](guides/entity_attention_hybrid_s1024_n1_zh-CN.md) |
| 未完成的 B→C→D 趋势实验 | [套件流程](guides/entity_attention_iteration_suite_zh-CN.md)，B仅完成前两轮，C／D未运行 |
| 棋力与快速诊断 | [配对评测及网页](guides/evaluation_zh-CN.md)、[固定256真实局面](guides/decision_probe_zh-CN.md) |
| 配置与运行工具 | [configs](../configs/README.md)、[scripts](../scripts/README.md) |

## 最近的实验结论与证据

最新读[P4起点九轮自迭代完整报告](reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)：完整对照支持本次在线方案的棋力收益，但不单独证明教师换代贡献或最优参数。较早的 [M4 与固定数据诊断完整结论](reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md) 统合以下实验，结论限定对应分支，不能用本次新方案回填旧分支收益。

| 问题 | 专题与原始记录 |
| --- | --- |
| P4新分支第九轮的步数／lr／池大小／batch | [10组单变量完成分析](../artifacts/entity_attention_r9_ofat_20261010/analysis.md)、[100–2000步全部55点矩阵](../artifacts/entity_attention_r9_ofat_20261010/steps_100_2000.md)、[完整曲线与配对区间](../artifacts/entity_attention_r9_ofat_20261010/results.md)：同一第八轮终点及AdamW，固定第九轮完整留出；较小lr对policy有正面信号，1k池加重value过拟合；单个候选的后续棋力补测见下一条 |
| 第九轮小学习率候选的直接搜索棋力 | [lr2e-5×1000步 vs lr1e-4×500步](../artifacts/entity_attention_r9_lr_step_arena_20261011/README.md)、[完成结果](../artifacts/entity_attention_r9_lr_step_arena_20261011/results.md)：新开局200 pair，双方S1024/N1；前者得分53.250%，95% CI [48.500%, 58.000%]，尚未确认棋力优势 |
| 平均标签与连续自迭代 | [早期固定数据对照](../artifacts/entity_attention_target_averaging_20261007/README.md)、[M4 12轮记录](../artifacts/entity_attention_avg4_behavior_20261009/README.md) |
| 历史配额、训练／留出、梯度与 AdamW | [固定 A／B／C 对照](../artifacts/entity_attention_policy_history_20261009/README.md) |
| 独立新标签及未抽中 Train 局面 | [fresh ΔKL 修订与补测](../artifacts/entity_attention_avg4_delta_kl_20261009/README.md)；撤回原“抽中显著改善／未抽中稳定退化”解释 |
| 步数、学习率与裁剪 | [单轮剂量](../artifacts/entity_attention_step_dose_20261007/README.md)、[正式账本 lr×步数](../artifacts/entity_attention_formal_lr_step_20261007/README.md)、[clip 探针](../artifacts/entity_attention_clip_sweep_20261009/README.md)；50步仅是初筛较高点估计 |
| value 能否拟合及能否提升棋力 | [policy／value 消融](../artifacts/entity_attention_policy_value_ablation_20261006/README.md)、[critic-only](../artifacts/entity_attention_critic_only_replay_20261008/README.md)、[两版 encoder 重拟合](../artifacts/entity_attention_encoder_critic_refit_20261008/README.md)、[critic 适配与已完成 DEV](../artifacts/entity_attention_critic_adaptation_strength_20261008/README.md) |
| 搜索在哪些阶段有收益 | [只搜前中期／后期／全程](../artifacts/entity_attention_stage_search_20261008/README.md) |
| 搜索噪声、随机节点与 CPW 教师 | [系列报告](reports/entity_attention_search_teachers_zh-CN.md)，从报告定位各原始实验 |

## 基线与性能报告

- [引擎历史验收](reports/engine_validation_zh-CN.md)、[PPO 5M seed1](reports/ppo_5m_zh-CN.md)、[AZ 五轮与扩展评测](reports/alphazero_seed1_zh-CN.md)、[MCTS sweep 完成情况](reports/mcts_sweep_zh-CN.md)。
- [推理吞吐有效基线及失效结果](reports/inference_throughput_zh-CN.md)、[多 worker 的窗口／整轮比较](reports/multiworker_gpu_zh-CN.md)。
- [流程时间口径](reports/pipeline_profile_zh-CN.md)、[端到端性能验收](reports/end_to_end_speed_optimization_zh-CN.md)、[实体注意力执行优化](reports/entity_attention_execution_speed_zh-CN.md)。性能结果限定机器、模型、形状与版本，没有确认所有负载下的最佳 batch／worker。

## 维护与历史资料

[产物约定](../artifacts/README.md) 说明哪些原件纳管、哪些保留本地。[2026-10-09 整理记录](../artifacts/repository_cleanup_20261009/README.md) 给出路径迁移与结论修订；更早的本地文档包在 `runtime/document_archive/2026-10-01/`。`prompt/` 为被忽略的历史任务资料，只有本次明确采用时才是需求。
