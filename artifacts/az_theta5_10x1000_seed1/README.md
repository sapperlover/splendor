# 旧 θ5 续训 10×1000：完成记录与历史准备产物

2026-10-12复核：**十轮正式训练已完成，10000局、12050次更新、20个常规50-pair面板均已完成。** 原 `prepared_not_started` 只对应2026-10-04的交付准备阶段，不能再代表该run当前状态。完整事实、锚点对照和限制见[完成报告](../../docs/reports/az_theta5_10x1000_zh-CN.md)、[原始训练汇总](summary.json)和[本次只读复核](completion_audit_20261012.json)。

追加的第3／5／10轮对旧θ5搜索锚点各200pair，得分41.50%／42.625%／46.625%；终点95% CI [41.875%,51.250%]，未确认终点超过旧θ5。第10轮对PPO追加面板只完成100/200pair，59.25%为未完成面板的点估计，原文件 `passed=false`、无正式配对区间；不能写成完成200pair或已确认胜过PPO。常规终点greedy对旧θ5为39.50% [30.00%,49.50%]，搜索与greedy分别解释。

10000局原始 `selfplay/` 已归档至 `archive/20261012/runs/az_theta5_10x1000_seed1/rounds/`，旧路径保留相对符号链接；模型与本目录结论留在原处。归档位置与复核入口见[归档结论索引](../repository_archive_20261012/conclusions.md)。

## 准备阶段的历史说明

以下说明保留2026-10-04交付时的范围。49项有限CPU回归通过，独立native CUDA smoke当时未执行；该测试计数不代表本次重新运行，也不能替代4GiB扩容或4vs8比较的验收。

`preparation_manifest.json`记录真实源码/dirty身份、配置SHA、机器信息和两个起点checkpoint身份；`seeds.json`包含全部10轮派生和面板输入，`smoke_seeds.json`为独立小验证。计划和统计分母见[训练指南](../../docs/archive/next_az_training_zh-CN.md)及[字段清单](../../docs/archive/next_az_record_fields_zh-CN.md)。

正式运行已在这里保存每轮dataset/current_before_after/fixed_curve/usage JSON、summary、训练和固定验证CSV；原NPZ、trace、checkpoint、二进制引用/计数、TensorBoard及恢复sidecar使用忽略的runs路径，部分原件现依归档清单定位。运行前准备清单不包含后来生成的结果，不能用准备清单否定这些完成记录。

`decision_template.json`只是人工查看记录模板，未做第3/5/10轮判断，也未默认选模/扩样。运行后每次查看追加到decisions.jsonl，保留时间、模型/面板、依据和是否延后评测。
