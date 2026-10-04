# 当前项目状态

既有条目证据核对：2026-10-01；2026-10-03 新增流程统计及完整性能验收导航。本文仅作导航；规则看标准规则与 data，契约看对应 specs，实现与实验状态优先核对当前代码/config 和可复核原始产物，再读对应版本报告。总览不凌驾于专题协议或报告。

| 能力或实验 | 已确认状态 | 阅读入口 |
| --- | --- | --- |
| 双人基础规则、Observation、特征与 PPO | 已实现；Python checked 为参照，native 另有 C++ 实现 | [契约](specs/engine_contract_zh-CN.md) |
| PPO seed1 | 5M run-local decisions 完成；DEV best 为 lifetime 4,662,112，latest 5,512,112；未据此宣称独立 FINAL best | [报告](reports/ppo_5m_zh-CN.md) |
| 固定深度 / PUCT 搜索 | Python 与 native 执行入口可用；不完全信息近似 | [搜索指南](guides/search_zh-CN.md) |
| AZ seed1 | Python batched_cpu 5 轮 × 500 局、2,982 优化步完成；θ1–θ5 扩展面板已保存 | [报告](reports/alphazero_seed1_zh-CN.md) |
| native 单 owner 推理 | 08 正确输入窗口、09 传输画像、500 局比较已保存；按负载区分推荐 | [报告](reports/inference_throughput_zh-CN.md) |
| 多 worker GPU pool | 已实现，有窗口和每档一次完整 500 局；整轮收益尚无重复稳定性证据 | [报告](reports/multiworker_gpu_zh-CN.md) |
| 流程时间统计 | 历史有限learner/评测/pool分项与口径修正保留；其“未实施优化”仅适用当时版本 | [分项报告](reports/pipeline_profile_zh-CN.md) |
| 完整性能优化 | 两组各1000局及checked replay、各50 pairs、8+128步固定learner对照完成；采集1.239×、learner总成本8.082×、评测3.292×；每组一次 | [完整报告](reports/end_to_end_speed_optimization_zh-CN.md) |
| MCTS sweep | 01/02/05 完整，03 不完整，04 无汇总；非全部完成 | [报告](reports/mcts_sweep_zh-CN.md) |

不能从 smoke 推断棋力，从稳态窗口推断整轮收益，或从单 seed/多个被查看面板推断总体稳定提升。当前没有 native 多轮生产训练、多训练 seed 或扩容 ablation 的报告依据。

完整文档导航见 [index](index.md)，实验状态与证据见 [experiment_index](../artifacts/experiment_index.json)。旧工作包/实施计划已打包至忽略的 `runtime/document_archive/2026-10-01/docs_before_cleanup.tar.gz`；归档旁的 manifest 保存逐文件 SHA。这是本地历史资料，不是现行需求。
