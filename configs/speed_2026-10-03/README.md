# 完整性能对照配置

`collection_baseline.json` / `collection_optimized.json` 是各1000局、PPO latest、S1024/N1、W8/A500/B500/micro64/slot1/wait5ms的冻结对照；优化只增加64档和明确编译失败策略，等待/gather改动来自生产实现。baseline必须使用原源码快照，不因文件名自动选择旧实现。

`evaluation_baseline.json` / `evaluation_optimized.json` 是固定θ5/θ4、50 pairs、同game/policy seeds、无噪声/温度0的冻结对照；前者CPU eager8 workers，后者唯一CUDA owner、active100/B64、16/32/48/64 compiled/reused、wait0。

`alphazero_cached_example.json` 是新目录用的生产示例：2GiB有界replay cache、1000局pool采集、native MCTS vs parent批量评测。已验证极小新路径闭环；没有用此示例开展新的多轮长训，不能替换旧run配置后resume。`*_smoke`、`evaluation_short_b*`仅为有限回归/选形状，不用于正式吞吐排名。

正式实际结果与限制见[性能专题](../../docs/reports/end_to_end_speed_optimization_zh-CN.md)，冻结source/模型SHA、产物与命令见[验收清单](../../artifacts/end_to_end_speed_2026-10-03/README.md)。这些本地checkpoint/NPZ不随Git提供。
