# 连续自迭代实验套件交付（历史方案）

2026-10-08后续：本四组方案已由[B→C→D趋势方案](../entity_attention_iteration_trend_20261008/README.md)替代，未启动正式长训练。下文配置、流程和验证适用于旧版本；旧manifest及原始验证结果保留，当前入口不再执行其协议。

2026-10-08。**正式长实验未运行**；本目录保存旧方案的启动检查与冻结快照，没有正式长期结果。后续趋势协议及其命令见 [流程指南](../../docs/guides/entity_attention_iteration_suite_zh-CN.md)，主配置见 [iteration_suite_20261008.json](../../configs/archive/entity_attention_iteration_suite_20261008.json)。

四组（200局/100步、200局/50步、100步低lr、50步仅policy最新1轮）×两套采集种子×16轮，128训练轮；25,600新局、9600更新步骤、5600 paired seeds/11,200评测局。三个主比较固定在第16轮，分别报告训练种子结果、95%与98.333%家族区间，PPO非劣界限−5pp。第8轮只观察，不挑checkpoint。

工程交付：冻结算法级lr覆盖及checkpoint一致记录、policy独立近期窗口、串行逐轮进程、user cgroup 16GiB硬上限/零swap、资源保留检查、文件锁、断点恢复和引用账本核验。缓存总4GiB，采集活动局200/GPU batch128，评测batch64/eager FP32。

有限验证记录见 `verification.json`。第一次 CUDA 完整链使用 [独立验证目录](../entity_attention_iteration_suite_validation_20261008/README.md)，最终带父进程退出联动的版本见 [最终验证](../entity_attention_iteration_suite_validation_v2_20261008/README.md)。这些只验证工程调用链，不估计任何正式组的棋力或20小时稳定性。

运行产物：模型与数据在 ignored `runs/entity_attention_iteration_suite_20261008/`；日志、状态、冻结源码在 ignored `runtime/entity_attention_iteration_suite_20261008/`；现存长期材料为 `training_configs/`、`manifest_pre_v2_freeze.json`、`code_version_pre_v2_freeze.json`、`preflight.json`、`verification.json` 和 `resources/`；没有正式轮次／对战分析。评测相邻 `*.games/` 保持忽略。

2026-10-08 后续：实现加入 v2 趋势协议（`iteration_trend_20261008.json`）后，本目录交付时冻结的 `manifest.json`/`code_version.json` 已归档为 `manifest_pre_v2_freeze.json`/`code_version_pre_v2_freeze.json`；正式训练从未启动（`runs/entity_attention_iteration_suite_20261008/` 不存在），当前runner不支持该四组协议，不能用此配置重新启动。磁盘下限按要求移除（初次 11 GiB 与运行 3 GiB 只采样记录、不再阻断），内存与显存约束不变。`verification.json` 保留为加入 v2 之前的工程验证记录。
