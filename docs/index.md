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
| 最近的连续实验 | [行为复用 M4](guides/entity_attention_avg4_iteration_zh-CN.md)，12／16轮完成，当前停止；[分头窗口](guides/entity_attention_split_training_zh-CN.md)，20轮完成 |
| 较早训练分支的复现 | [10×1000](guides/entity_attention_self_iteration_zh-CN.md)、[5000步混合分支](guides/entity_attention_hybrid_zh-CN.md)、[第九轮 N2 fork](guides/entity_attention_hybrid_fork_zh-CN.md)、[N1 小步分支](guides/entity_attention_hybrid_s1024_n1_zh-CN.md) |
| 未完成的 B→C→D 趋势实验 | [套件流程](guides/entity_attention_iteration_suite_zh-CN.md)，B仅完成前两轮，C／D未运行 |
| 棋力与快速诊断 | [配对评测及网页](guides/evaluation_zh-CN.md)、[固定256真实局面](guides/decision_probe_zh-CN.md) |
| 配置与运行工具 | [configs](../configs/README.md)、[scripts](../scripts/README.md) |

## 最近的实验结论与证据

先读 [M4 与固定数据诊断完整结论](reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md)。它统合以下实验，并说明统计修正、适用范围和下一步候选；没有确定最优训练参数。

| 问题 | 专题与原始记录 |
| --- | --- |
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
