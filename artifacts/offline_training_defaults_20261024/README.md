# 后两组执行接入与离线训练默认

2026-10-27后续状态：这里记录的停止／准备／仅当前组适用属于当时工程发布范围；后续104k仅policy、双辅助及44k仅value均采用已发布优化完成训练、选点和两组对战，见[完成汇总](../../docs/reports/auxiliary_offline_results_20261027_zh-CN.md)。原工程验证数值与冻结发布不改写。

2026-10-24按用户要求，将流式统计、4 worker共享只读供数、传输／同步优化全部接入后两组，并设为以后新建离线训练的默认。缓存继续按池大小设置，本次后两组均为18GiB。**本次只准备和短验证，未启动正式训练。**

## 当前三组

| 组 | 实际执行发布 | 缓存 | worker |
| --- | --- | --- | --- |
| 104k仅policy | 上一版`source_streaming_20261024`，保持原件 | 18GiB | 4 |
| 104k双辅助 | 新`source_offline_defaults_20261024`，见[both104k/source.json](both104k/source.json) | 18GiB | 4 |
| 44k仅value | 新`source_offline_defaults_20261024`，见[value44k/source.json](value44k/source.json) | 18GiB | 4 |

三组均启用流式会话累计和传输／同步优化。后两组原16GiB配置、原冻结源码、plan、整局留出和面板模板保留；新配置仅调整缓存和worker，执行source另行记录。原种子、从零初始化、新AdamW、两路B256、lr1e-4、辅助权重、曝光数、5%完整主KL+MSE早停／选点及R37面板保持。

`run_variant.py`复用原共享runner，用旧配置验证原冻结计划，再选择新执行配置和源码；不会改写旧plan来接受新身份。实际训练会话中的source身份指向本次对应的source记录。

## 未来默认与兼容性

三个基础配置类`OfflineConfig`、`OfflineObjectiveConfig`、`ScratchOfflineConfig`默认`training_workers=4`，派生的混合预算、早停、续训和continuation入口继承。共享供数补齐单batch、分离batch、旧终局辅助标签和PSW；continuation及progress标签仍在worker生成。验证使用独立256MiB缓存；旧`offline_train`的最终汇总分别报告训练池`replay_cache`和`validation_cache`，避免将验证容量当成训练池容量。

公共训练器流式累计统计；无辅助的分离value、continuation和旧progress分支都使用state-only接口。主／辅助指标及任务梯度范数合并回读，旧终局辅助的重复标量读取也合并。NaN、Inf及非有限梯度仍在optimizer更新前失败。

worker不进入训练身份，恢复可改变；旧`offline_train`配置无该字段也可恢复。`HybridConfig`保留省略默认字段的旧序列化，并把离线worker数视为执行设置，避免增加离线配置字段后破坏现有混合训练恢复身份。`training_workers=0`显式选原串行路径。默认4 worker需所选缓存容纳完整压缩训练池与标签；容量不足明确报错并回收，不自动退回LRU。

缓存容量未统一提高为18GiB，各入口原默认容量保持；新实验应按池大小配置。完整说明见[离线执行指南](../../docs/guides/offline_execution_zh-CN.md)。

## 验证

- [tests.xml](tests.xml)：181项相关回归通过，覆盖attention／set-MLP、普通共享／分离batch、旧辅助、PSW、早停、续训、continuation、旧离线入口和混合训练兼容性。新增默认worker路径与串行结果、旧配置恢复、模型／AdamW／RNG逐位一致及退出回收验证通过。
- [additional_tests.xml](additional_tests.xml)：补充2项，检查共享池容量不足明确失败并回收loader，以及state-only progress输出／梯度一致。
- [cache_reporting_tests.xml](cache_reporting_tests.xml)：旧离线入口最终缓存统计补充回归，通过串行／默认4 worker及历史配置恢复，核对训练8MiB和验证256MiB分别报告。初次接口误用已修正，失败记录保留于`cache_reporting_initial_tests.xml`；改前发布清单及GPU中间记录作为本次迭代原件保留，最终GPU结果为`gpu_smoke.json`。
- [gpu_smoke.json](gpu_smoke.json)：RTX4050、真实B256＋256，双辅助取68局、44k仅value取44局；各自从原正式seed初始化，对照原冻结训练器串行3次更新与新4 worker＋流式训练调用链3次更新。模型、梯度、AdamW、全部当步指标及CPU／CUDA／抽样RNG逐位一致，第1步包含encoder分任务梯度诊断。这是工程验证，无棋力或完整池速度结论。
- [verification.json](verification.json)：两组实际入口、各123个执行文件、原计划、replay identity和固定留出digest核对通过，后两组仍无正式checkpoint；749个历史原件哈希保持，当前policy入口仍能核对和恢复。
- 容量核算：原完整policy共享区17869150492字节，双辅助新增14字节／位置及最多189字节／局对齐，保守上界17991526977字节（约16.76GiB），小于18GiB。44k仅value是其中子集且移除对手标签，也在此保守上界内。此处不是总进程内存上限，也不是两份完整池重新预热实测。

回归中的非有限故障测试沿用原错误详情标量转换提示；沙箱内一个CUDA缓存清理测试出现NVML初始化提示，测试通过，实际CUDA短验证已单独在目标GPU完成。

## 命令和产物

恢复当前policy组并按原成功边界继续后两组：

```bash
bash scripts/run_attention128_auxiliary_queue_20261024.sh
```

后两组单独入口分别为`bash scripts/run_attention128_both104k_20261024.sh`和`bash scripts/run_attention128_value44k_20261024.sh`。两者支持prepare／run／train／status；`train`只训练与选点，不构成队列完整成功边界。

新配置：

- [双辅助实际配置](../../configs/entity_attention_v1/offline_continuation_r21_r37_prefetch_20261024.json)
- [44k仅value实际配置](../../configs/entity_attention_v1/offline_shortterm_value_r21_r31_vs_r37_prefetch_20261024.json)

原件保护与改前备份见`before.json`和`before/`；发布构建脚本为`prepare_release.py`，入口适配为`run_variant.py`，有限GPU验证命令封装在`gpu_smoke.py`。最终进程状态见`stopped_verification.json`。
