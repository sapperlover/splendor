# inference_throughput 交付物

本目录只保留可长期引用、能复核报告结论的汇总与基线：

- `08/summary_08_2026-09-29.json` 及 `08/*.json`：当前唯一有效的推理吞吐基线（08 修复多行布局后的重测）。
- `summary_2026-09-29.json`、`end_to_end_2026-09-29.json`、`pure_forward_{cpu,cuda}.json`、`profile_2026-09-29.json`：
  06 工作包的汇总/完整对局/微基准，被 `docs/native_inference_throughput_zh-CN.md` 引用（历史口径，文档已标注失效范围）。其中 `profile_2026-09-29.json` 的输入准备误标为 H2D，推导的 D2H 占比无效，需重跑 profiler；原文件保留。

不入库的本地数据（`runtime/` 整体被 Git 忽略）：

- `runtime/inference_throughput/07/`：07 的原始窗口与 `summary_07.json`。它们产自 CUDA reused 多行布局缺陷，
  数字与结论均无效，只作本地审计保留。
- `runtime/inference_throughput/06_stage_logs/`：06 的 `stage_*` / `baseline_*` / `rerun_*` / `profile_fixed_*` /
  `profile_pipeline_*` 等中间分片，可由脚本重生成，不属于交付物；`summary_2026-09-29.json` 的 `source`
  字段引用这些文件名，对应文件在同一目录本地保留。

规则：交付物随仓库长期保留；被取代或失效的运行数据移入 `runtime/` 本地保留，不新开逐文件 `.gitignore`
白名单（与 `artifacts/README.md` 一致）。
