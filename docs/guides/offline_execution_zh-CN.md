# 离线训练执行默认配置

2026-10-27执行结果更新：采用本页优化发布的104k仅policy、双辅助及44k仅value均已完成训练、选点和原两组评测，后续policy＋得分、无辅助也已完成；见[完成汇总](../reports/auxiliary_offline_results_20261027_zh-CN.md)。下文“本次未启动”指2026-10-24发布时点，恢复命令不变。

2026-10-25存储更新：[原80000局池及四组已完成辅助实验的普通checkpoint／逐步日志已外置](../../artifacts/repository_archive_20261025/README.md)。原80k池的离线训练／全量复核需先恢复NPZ；旧普通节点／曲线复核需恢复对应模型／日志，续写前恢复为真实文件。R21–R37的104k池及当前在线初始64k／新采集数据、关键模型保持本机；本页训练执行默认不变。

2026-10-24按用户要求，新建离线训练统一采用以下默认行为。它们改变供数、传输和统计实现，不改变目标、样本顺序或优化器设置。

| 项目 | 默认行为 |
| --- | --- |
| 会话统计 | 流式累计步数、位置数、loss总和及首尾值；不在内存保留全部step字典 |
| CPU供数 | `training_workers=4`，spawn进程，共享同一份只读压缩训练池，最多预取4组batch |
| 传输与同步 | 分离value／辅助分支只上传状态与所需标签；指标及任务梯度范数批量回读；更新前有限性检查合并执行 |
| 缓存 | 按各实验池大小配置，**18GiB不是通用默认值**；`cache_mib`或旧入口的`replay_cache_bytes`保持独立设置 |

以上worker默认覆盖当前源码的`offline_train`、`offline_objectives`、`offline_scratch`、`offline_mixed_budget`、`offline_early_stopping`、`offline_finetune`、`offline_continuation`。支持普通共享batch、高预算policy／全预算value分离batch、PSW、旧终局辅助和continuation辅助。`training_workers=0`显式选择原串行供数，适合小型工程验证；不自动降级。流式统计与已接入的传输优化是公共训练器内建行为，无需再设开关。

## 缓存和恢复

4 worker路径先完整预热压缩训练数据和所需辅助标签，写入一个Linux memfd；worker只读映射同一份数组，不各建一份大缓存。所选容量必须容纳**整个压缩训练池、标签及64字节对齐开销**；不足时明确报错并回收进程。它是固定训练池，不是自动淘汰的LRU。验证使用独立256MiB流式缓存，避免驱逐训练数据。

缓存不是整个进程树的内存上限。容量选择应结合预热记录中的`arena_bytes`、batch与worker私有内存以及系统可用内存。RSS会重复计入共享映射，应结合PSS观察。本次104k双辅助和44k仅value组均设18GiB；双辅助共享区的保守上界约16.76GiB，44k组在此更保守上界内。此为原完整policy池实测加新增标签／对齐的容量核算，未重新预热两份完整池。

主进程保持原抽样顺序，checkpoint仅提交已消费batch的采样RNG；中断时未消费的预取丢弃，恢复重算。`training_workers`是执行参数，不进入训练身份；现有目标训练器保留旧数据／checkpoint digest，旧`offline_train`也允许省略该字段的历史配置恢复。实际执行worker与缓存记录在会话source或运行记录中。

## 冻结发布与当前队列

工作区默认不会覆盖已冻结的历史执行源码。新建实验应从当前代码发布，并核对实际配置／源码中的默认；历史实验按自己的冻结版本解释。

本次已分别更新当前队列后两组的执行发布、实际配置和启动脚本，原计划、种子、模型初始化、AdamW、整局split、辅助权重、验证／早停和评测面板保持；原冻结发布作为历史原件保留。当前policy组继续使用上一次已验证的优化发布，后两组采用本次发布。[发布、原始验证与边界](../../artifacts/offline_training_defaults_20261024/README.md)。

本次未启动正式训练。仓库根目录手动恢复原串行链：

```bash
bash scripts/run_attention128_auxiliary_queue_20261024.sh
```

只运行后两组某一组的完整流程，可分别使用：

```bash
bash scripts/run_attention128_both104k_20261024.sh
bash scripts/run_attention128_value44k_20261024.sh
```

单组入口可用`prepare`核对冻结计划及实际发布，不开始训练。总队列仍在前一组训练、选点、greedy1000pair和搜索A500pair全部成功后交接；失败／中断停止。
