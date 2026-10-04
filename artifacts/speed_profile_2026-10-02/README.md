# 流程时间归因：2026-10-02 测量，2026-10-03 整理

结论、计时口径、有限命令与限制见 [专题报告](../../docs/reports/pipeline_profile_zh-CN.md)。本目录只交付统计与归因，没有实施性能优化或替换生产配置。

有效证据：

- `history.json`：历史完整 AZ 五轮、完整 pool 500 局的时间重新统计，含原件 SHA。
- `learner_r0_256mib_v2.json`：首轮 replay，8 步预热＋24 步测量。
- `learner_r4_256mib_v2.json`、`learner_r4_2048mib_v2.json`：第五轮同引用序列、同 parent optimizer 的有限缓存对照。
- `replay_r4_cprofile.json`：额外两批只读 NPZ 函数画像，profiler 开销不混入 learner 时间占比。
- `evaluation_native_config.json`、`evaluation_native_cpu_v2.json`：θ5/θ4 native S=1024/N=1 的一个完整 CPU pair，非棋力面板。
- `collection_opening_b500.json`：35 秒生产形状开局窗口，没有完成局。
- `collection_full16_b16.json`：16 局完整诊断，包含阶段实际叶宽、两层 padding、尾段和 checked replay。
- `summary.json`：紧凑汇总、正确性核对和验证命令。
- `manifest.json`：有效/被替代文件分类、SHA 和文件写入时间；写入时间不冒充精确测量起止。

无 `_v2` 的三个 learner 初版原件保存在忽略的 `runtime/speed_profile_2026-10-02/superseded/`：run 根使用相对路径，导致派生 replay 验证缓存失效，准备阶段全量重验证。本次结论采用修正后的 `_v2` 文件；初版原件字节与哈希保持不变，不参与准备成本和正式速度结论，manifest 记录原位置与保留位置。

完整命令已在各 JSON 的 `provenance.command` 保存。当前脚本为 `scripts/profile_speed_pipeline.py`；有效新测量的执行脚本副本及 SHA 位于各自 `runtime/speed_profile_2026-10-02/<case>/profile_speed_pipeline.py`，支持核对诊断包装版本。首版历史汇总仅做静态原件统计。

运行日志、逐批记录、完整 `.pstats`、生成的游戏与诊断 checkpoint 在忽略的 `runtime/speed_profile_2026-10-02/`。旧 `runs/az_loop_seed1/`、`runtime/module_b/` 原件未改动。代码基点是 Git HEAD `f213e637…`，新增脚本/文档未提交，不能把 HEAD 标签当作诊断脚本本身的身份。
