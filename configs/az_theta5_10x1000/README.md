# 旧 θ5 起点的训练准备配置

准备记录日期为2026-10-04，当时为 prepared_not_started；后续源run轮0–9已完成。详见[历史准备与命令](../../docs/archive/next_az_training_zh-CN.md)、[字段清单](../../docs/archive/next_az_record_fields_zh-CN.md)。配置及其冻结身份不改。

- `formal.json`：独立10×1000、最近6000局、4GiB、旧θ5权重起点、S1024/N1、每轮MCTS/greedy各50pairs、opt-in完整记录。
- `smoke.json`：独立2×6、S4、最多2步/轮、各1pair、固定集16，用于native调用链预检；准备交付时未启动，此处不补认后续smoke验收。
- `round-002/004-vs-theta5/ppo.json`：人类第3/5轮固定开发锚点各200pairs，同一DEV输入。
- `round-009-vs-theta5/ppo.json`：第10轮预留面板各200pairs。
- `round-009-vs-*-extension500.json`：可选独立500pairs扩展，不默认执行。

MCTS评测唯一CUDA owner、active100/B64、compiled16/32/48/64、wait0；greedy不传native_batched。阶段外部配置要求对应checkpoint可读、GPU可用，不能仅据配置解析通过认定面板已执行。

`scripts/prepare_next_az.py` 默认只检查，`--write`只重建以上准备配置/身份与种子表，不启动实验。正式启动后不修改冻结配置；继续使用同一配置和`--resume`。完整SHA与种子审计位于`artifacts/az_theta5_10x1000_seed1/`。
