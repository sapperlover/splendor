# 文档导航

2026-10-04新增：[实体注意力模型与离线训练](guides/entity_attention_offline_zh-CN.md)，有限GPU验证完成，用户已启动正式拟合；数据加载占主要耗时，已提供8 GiB运行时缓存覆盖及resume命令。[旧θ5起点10×1000配置与命令](guides/next_az_training_zh-CN.md)、[记录字段/分母清单](guides/next_az_record_fields_zh-CN.md) 保留准备时记录；该十轮源 run 已完成。

docs 只保留需要阅读和维护的规范、操作指南、可复核报告。入口为 [当前状态](current_project_state_zh-CN.md)，实验分类与依据见 [实验索引](../artifacts/experiment_index.json)。文件中的实验日期与适用版本独立于本次整理日期。

## 规则与现行契约

- [标准规则与规范数据](splendor_standard_rules_zh-CN.md)：唯一玩法依据，包含双人本地 forced-skip 增补。
- [引擎、Observation 与阶段机](specs/engine_contract_zh-CN.md)：公开边界、私有状态、replay 和 native 差异。
- [数值特征 schema](specs/feature_schema_zh-CN.md)：维度、mask、缩放、隐藏信息和候选映射。
- [Actor–Critic、学习环境与 PPO](specs/learning_contract_zh-CN.md)：网络、transition、GAE、更新与 checkpoint。
- [PPO 训练、评测与历史池协议](specs/ppo_training_protocol_zh-CN.md)：5C run-format v4、种子域、DEV/promotion/FINAL。

## 操作指南

- [冻结网络搜索](guides/search_zh-CN.md)：PUCT 与固定深度算法/API。
- [自对弈数据与恢复](guides/selfplay_data_zh-CN.md)：v1/v2 数值归档、compact trace、checked 重放。
- [AlphaZero 循环](guides/alphazero_zh-CN.md)：replay、损失、跨轮状态、恢复和命令。
- [实体注意力与离线学习](guides/entity_attention_offline_zh-CN.md)：模型架构、全量所选 train 抽样、验证/恢复和运行命令。
- [新模型10×1000自迭代](guides/entity_attention_self_iteration_zh-CN.md)：1000局同时调度，固定起点双面板，正式命令与有限CPU/GPU验证。
- [原生后端与推理配置](guides/native_search_zh-CN.md)：构建、后端、batch/传输与版本兼容。
- [多 worker GPU](guides/multiworker_gpu_zh-CN.md)：共享槽、容量、故障与恢复。
- [固定配对评测](guides/evaluation_zh-CN.md)：计分、种子、bootstrap、sweep 和结果解释。

## 可复核报告

- [Python 引擎历史验收](reports/engine_validation_zh-CN.md)。
- [PPO 5M seed1](reports/ppo_5m_zh-CN.md)。
- [AZ 五轮与逐代扩展评测](reports/alphazero_seed1_zh-CN.md)。
- [推理吞吐有效基线与失效结论](reports/inference_throughput_zh-CN.md)。
- [多 worker GPU 的窗口与整轮对照](reports/multiworker_gpu_zh-CN.md)。
- [采集、learner 与固定评测的分项统计](reports/pipeline_profile_zh-CN.md)。
- [端到端性能优化与完整1000局/50 pairs验收](reports/end_to_end_speed_optimization_zh-CN.md)。
- [MCTS sweep 实际完成状态](reports/mcts_sweep_zh-CN.md)。

## 历史归档和维护边界

旧工作包、实施大纲、修复流水账以及混杂的性能附录已完整打包到 `runtime/document_archive/2026-10-01/docs_before_cleanup.tar.gz`，逐文件 SHA 位于相邻 `manifest.json`。其中仍有效的接口/算法已提取到以上文档；原件不继续作为 docs 或 Git 中的现行资料。归档仅存在本机，迁移需备份。

`prompt/` 是被忽略的本地历史任务资料，只有用户本次明确采用时才是需求，docs 不依赖它才能读懂。实验原件保留策略见 [artifacts 约定](../artifacts/README.md)，配置用途见 [configs 导航](../configs/README.md)。保留原实验目录身份，日志、checkpoint 和恢复 sidecar 不在本次整理中重组。
