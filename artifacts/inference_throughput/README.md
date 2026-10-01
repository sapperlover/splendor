# 推理与采集吞吐交付物

统一报告：[单 owner / 传输基线](../../docs/reports/inference_throughput_zh-CN.md)、[多 worker GPU](../../docs/reports/multiworker_gpu_zh-CN.md)。所有状态及适用范围见 [实验索引](../experiment_index.json)。

- `08/`：正确 row-stride 后的单 owner 稳态窗口、CPU eager/compiled、slot 对照、汇总和 500 局生产形状；完整 manifest/summary 副本与原件关联见保全清单。
- `09/`：2026-09-30 的传输与 replay 分项微基准，不能替代完整采集吞吐。
- `multiworker_gpu/`：七份窗口、两组完整 500 局 manifest/summary 和比较摘要；单轮收益不当作重复稳定收益。
- `invalid_records.json`：06 旧测量的原位置、本地保留位置、哈希和失效/被替代状态。

06 原件现在位于 `runtime/inference_throughput/06_archived/`；07 错误输入窗口仍位于 `runtime/inference_throughput/07/`。中间分片继续位于 `runtime/inference_throughput/06_stage_logs/`。这些只作本地审计，不是当前基线；迁移机器时自行备份。规则与上层 artifacts README 一致。
