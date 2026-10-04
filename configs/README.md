# 配置入口

配置按现有实现家族保留稳定路径，避免改变历史 manifest、实验身份和恢复命令；同目录中的 production、smoke、benchmark 用文件名和下表区分。

| 用途 | 路径 | 状态 |
| --- | --- | --- |
| Python AZ 主配置 | `alphazero_loop.json` | 对应已有 5x500 完成 run，原冻结 config 在 runs 中 |
| Python AZ smoke | `alphazero_loop_smoke.json` | 有限闭环验证 |
| 历史 v1 自对弈 | `mcts_selfplay_round000.json`、`mcts_selfplay_smoke.json` | 读取兼容保留 |
| 固定搜索评测 | `mcts_vs_*.json`、`mcts_50pair_sweep/` | sweep 部分完成，见报告 |
| native 生产与 smoke | `native_search/alphazero_loop_*`、`selfplay_*` | CPU/CUDA/pool 分别选取，不等于多轮长训已验收 |
| native 性能基准 | `native_search/throughput_*`、`selfplay_cuda_baseline_500.json` | 基准形状与完整采集分别解释 |
| 端到端性能对照与新 AZ 示例 | `speed_2026-10-03/` | 各1000局采集、各50 pairs固定θ5/θ4，以及2GiB cache / native批量评测示例；结果见[专题](../docs/reports/end_to_end_speed_optimization_zh-CN.md) |
| Python 搜索历史对照 | `search_accel/` | 有价值的复现配置；非 native 通用推荐 |

逐代 AZ 评测的生成配置默认写入 `runtime/az_loop_seed1_evals/configs/`，可从脚本再生成，正式 configs 不存其运行副本。旧 configs/az_loop_seed1_evals 忽略规则保留，兼容显式指定旧输出目录的命令。原冻结 run 不移动或重写。

目录、命令和结果入口见 [docs 导航](../docs/index.md)。新运行先核对 checkpoint 和目标设备；本仓库的大模型与训练 NPZ 不随 Git 提供。
