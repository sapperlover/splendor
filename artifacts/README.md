# 长期实验交付物

本目录保存报告引用、值得长期复核的配置快照、汇总与统计表，默认纳入 Git。阅读入口为[实验导航](index.md)，机器可读身份与SHA见[实验索引](experiment_index.json)，按 valid、historical_valid、invalid_or_superseded、incomplete、unverified 明确适用范围；有效不等于“已证明棋力提升”。

日志、失败现场、cache、分片运行状态和可恢复 `*.games/` sidecar 放在忽略的 `runtime/` 或相邻被忽略的 sidecar 目录。训练 NPZ、checkpoint、TensorBoard 放在忽略的 `runs/`。新报告引用的必要汇总不能只存于 runtime；重型 sidecar/模型不因此纳管。

失效或被替代的原始测量移入 runtime 本地保留，随仓库只保留失效原因、原/新位置、哈希与替代依据；历史有效的实验交付物继续保留，限定版本。未完成评测 summary 可在原恢复路径保留并明确 incomplete，不能当成完成结果。这两类与有效基线分开索引。

既有 `work_package_*` 产物目录是历史实验身份，暂保留路径以兼容 manifest 和工具。docs 中的工作包原文已本地打包，不作为现行说明。归档位于 `runtime/document_archive/2026-10-01/`，被 Git 忽略；迁移环境需要另行备份。

[2026-10-01 保全清单](retention_manifest_2026-10-01.json) 记录新增原件副本的来源、字节数与 SHA-256。副本 JSON 内保留测量时绝对路径，不能据此认定复制后的目录缺失；外部清单定位长期副本。不用当前代码版本或硬件补填缺失的历史信息。

## 原始证据与目录维护

小型诊断NPZ、逐局面统计、学习曲线JSONL、来源局面和源码patch如用于复算有效结论，应随实验保存；它们与正式训练归档NPZ、缓存和运行日志区分。`oom_kernel.txt`是OOM归因的唯一内核摘录，保留为该历史事件的必要原始证据；只有重复测试stdout移到本地runtime，JUnit／结构化结果继续纳管。

新的临时日志、会话资源采样和生成中间件写`runtime/`；完成实验的关键summary、逐根／逐局配对统计及身份写回对应实验目录。目录名是实验身份，不按是否热门或文件数量合并；重复的随机教师root bank也保留各实验冻结身份，避免改写既有hash依赖。

[本次整理清单](repository_cleanup_20261009/manifest.json)记录路径、原SHA、文档修订前后SHA与本地保全位置。历史JSON里的绝对路径／配置路径／源码hash代表当时身份，迁移后依清单定位，不用当前文件hash回填历史测量。`evidence_hashes`表示当前阅读／复核入口的内容，旧值在整理清单中保留。
