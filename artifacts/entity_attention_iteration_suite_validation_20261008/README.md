# 初版套件有限 CUDA 验证

2026-10-08。两条分支，各2轮、每轮4局/S2、每轮2个更新步骤；共16新局、8更新步骤、7个1-pair面板（14评测局）。调用真实GPU采集、训练与固定配对评测；每轮检查lr/checkpoint、抽样账本与留出隔离。模型起点与正式方案相同，但所有科学参数缩小，仅作工程验证。

`analysis.json.status=complete`；实测阶段进程组峰值3.565GiB。已使用完全相同命令重复启动，只有preflight/analysis执行，无新增采集、训练或评测。该版本源码哈希见 `manifest.json`，它早于最终增加父进程退出联动的版本；不继续用当前源码恢复本初版目录，后续验证独立保存于 [v2](../entity_attention_iteration_suite_validation_v2_20261008/README.md)，原产物保留。

本验证的成绩没有统计解释，不能作为任一候选改善棋力的证据。正式长实验见 [交付记录](../entity_attention_iteration_suite_20261008/README.md)。
