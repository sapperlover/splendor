# PPO 5M 单 seed 训练报告

本文记录已完成 `5c_main_seed1` 的冻结 `single_process` 运行。后续 worker-local 优化不追溯改变该 run 的耗时和结果。具体训练、评测与历史池规则见 [实验协议](../specs/ppo_training_protocol_zh-CN.md)。

## 1. 结论摘要

`5c_main_seed1` 已正常完成计划中的 `5,000,000` 个 run-local learning decisions。
训练过程共完成 `698` 次本阶段 PPO update、`110,344` 局完整训练对局，
没有 engineering truncation、NaN/Inf、IllegalAction 或 invariant failure。

最终应明确区分两个 checkpoint：

- **best-dev**：`local_004150000_lifetime_004662112.pt`，是该 PPO run 的 DEV best；后续搜索和 AZ 实验另有各自教师与评测身份；
- **latest**：`local_005000000_lifetime_005512112.pt`，是 5M 边界的最新训练状态，但没有取代 best-dev。

best-dev 在固定 Formal DEV 面板上的关键结果为：

| 对手 | 得分率 | W/D/L | 先手得分率 | 后手得分率 |
|---|---:|---:|---:|---:|
| ScorePolicy | 70.375% | 281 / 1 / 118 | 69.00% | 71.75% |
| DevelopmentPolicy | 78.625% | 314 / 1 / 85 | 78.50% | 78.75% |
| 5C 起点 `512112` | 75.375% | 301 / 1 / 98 | 75.50% | 75.25% |

固定启发式综合指标：

```text
heuristic_score = 0.5 * ScorePolicy score rate
                + 0.5 * DevelopmentPolicy score rate
                = 74.500%
```

相比起点 checkpoint 在同一套 Formal seeds 上的 `41.8125%`，best-dev 提高了
`32.6875` 个百分点。这是该固定 DEV 面板上的提升；重复选模后的 DEV 分数不是独立 FINAL 的无偏结果。

但是，本报告中的“最优”仍指 **DEV 协议下的 best-dev**。预留的
`FINAL` seed domain 尚未使用，也没有进行 3 个完整训练 seed 的重复实验，
因此不应把当前 DEV 结果当作无偏的最终试验结论。

## 2. 运行身份与可复核信息

| 项目 | 值 |
|---|---|
| run ID | `5c_main_seed1` |
| 阶段 | `5c_main_5m_self_play` |
| 结束状态 | `complete` |
| training seed | `1` |
| run-local decisions | `5,000,000` |
| 最终 lifetime decisions | `5,512,112` |
| 本阶段 / lifetime PPO updates | `698 / 763` |
| 源 checkpoint | `step_000512112.pt` |
| 规则版本 | `splendor-base-two-player-v2-forced-skip` |
| feature schema | `splendor-observation-action-v1` |
| RNG | `splitmix64-v1` |
| 训练设备 | NVIDIA GeForce RTX 4050 Laptop GPU |
| PyTorch / CUDA runtime | `2.14.0+cu130 / 13.0` |
| Python | `3.12.14` |
| 运行中 resume 次数 | `1` |
| 活跃计时 | `48,104.68 s` = 13 h 21 min 45 s |
| 准备到完成的时间跨度 | 约 14 h 10 min（包含中断/恢复间隔） |

checkpoint 校验：

| checkpoint | SHA-256 | 大小 |
|---|---|---:|
| best-dev `local_004150000_lifetime_004662112.pt` | `255db59c4aaa356ecbf8673946ba037cfde2b1146cebd2e386fb2ff8a7e2c50c` | 3.6 MiB |
| latest `local_005000000_lifetime_005512112.pt` | `009d17d2e9bc0d879142d7ca3ff1777a6f8caa9dda9243c6a2d5eb3538b5d5b0` | 3.6 MiB |

权威输出：

- 完成状态：`runs/5c_main_seed1/status.json`
- 运行总结：`runs/5c_main_seed1/summary.json`
- 完整指标：`runs/5c_main_seed1/metrics.jsonl`
- best-dev 指针：`runs/5c_main_seed1/checkpoints/best_dev.json`
- best-dev 变更历史：`runs/5c_main_seed1/checkpoints/best_dev_history.jsonl`
- latest 指针：`runs/5c_main_seed1/checkpoints/latest.json`


## 长期复核入口

`artifacts/work_package_5c/run_snapshot/` 保留原始 config、status、summary、best-dev 指针与 best Formal candidate 汇总的字节副本；[保全清单](../../artifacts/retention_manifest_2026-10-01.json) 记录来源和哈希。原冻结 run 与 checkpoint 没有移动。

既有 `formal_candidates.csv`、`ppo_updates.csv`、100k/250k 行为统计和 historical matchup matrix 保留在 `artifacts/work_package_5c/`，`export_manifest.json` 记录导出来源。文件名中的 `historical_final_pool` 指最终历史池，不等于 FINAL seed 验收。完整旧报告已本地打包；当前报告只保留可复核结论与入口。
