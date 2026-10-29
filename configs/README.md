# 配置入口

2026-10-24新增[离线执行默认](../docs/guides/offline_execution_zh-CN.md)：当前源码各离线入口缺省使用4 worker、流式统计及传输优化，缓存按池大小设置。后两组实际执行配置为18GiB／4 worker，原16GiB配置保留历史身份；[发布及配置链接](../artifacts/offline_training_defaults_20261024/README.md)。

短期value／两层对手policy的两份44k离线配置在[attention目录](entity_attention_v1/README.md)，使用`offline_continuation`入口；[定义及命令](../docs/guides/continuation_auxiliary_zh-CN.md)。现行在线配置未启用，未启动正式训练。

当前R31在线配置的`learner.checkpoint_every_steps=500`，七个`validation_steps`不变、独立强制保存；当前在线／近期离线默认停写双份抽样与逐步明细，见[训练记录规则](../docs/guides/training_artifacts_zh-CN.md)。旧冻结配置保留原周期。

配置按现有实现家族保留稳定路径，避免改变历史 manifest、实验身份和恢复命令；同目录中的 production、smoke、benchmark 用文件名和下表区分。

| 用途 | 路径 | 状态 |
| --- | --- | --- |
| 当前日常greedy／搜索A | [greedy](entity_attention_v1/evaluation_greedy_r43.json)、[搜索A](entity_attention_v1/evaluation_search_a_r43.json) | 固定R37对R43示例；CPU8／1000 pair与GPU A／500 pair；[参数与命令](../docs/guides/evaluation_zh-CN.md) |
| Python AZ 主配置 | `alphazero_loop.json` | 对应已有 5x500 完成 run，原冻结 config 在 runs 中 |
| 旧θ5起点10×1000 | `az_theta5_10x1000/` | 原准备配置/命令保留，十轮源run已完成；不据此替代其棋力评测结论 |
| 实体注意力与离线对照 | `entity_attention_v1/` | 新旧离线、smoke、混合面板；支持8 GiB离线缓存执行覆盖；[离线专题](../docs/guides/entity_attention_offline_zh-CN.md) |
| 新模型10×1000自迭代 | `entity_attention_v1/online_10x1000.json` | 同时调度1000局，每轮对固定起点搜索/greedy各50 pairs；有限CPU/GPU验证通过，已有轮0–2完成记录；历史分支，不是近期M4入口；[命令](../docs/guides/entity_attention_self_iteration_zh-CN.md) |
| Python AZ smoke | `alphazero_loop_smoke.json` | 有限闭环验证 |
| 历史 v1 自对弈 | `mcts_selfplay_round000.json`、`mcts_selfplay_smoke.json` | 读取兼容保留 |
| 固定搜索评测 | `mcts_vs_*.json`、`mcts_50pair_sweep/` | sweep 部分完成，见报告 |
| native 生产与 smoke | `native_search/alphazero_loop_*`、`selfplay_*` | CPU/CUDA/pool 分别选取，不等于多轮长训已验收 |
| native 性能基准 | `native_search/throughput_*`、`selfplay_cuda_baseline_500.json` | 基准形状与完整采集分别解释 |
| 端到端性能对照与新 AZ 示例 | `speed_2026-10-03/` | 各1000局采集、各50 pairs固定θ5/θ4，以及2GiB cache / native批量评测示例；结果见[专题](../docs/reports/end_to_end_speed_optimization_zh-CN.md) |
| Python 搜索历史对照 | `search_accel/` | 有价值的复现配置；非 native 通用推荐 |

逐代 AZ 评测的生成配置默认写入 `runtime/az_loop_seed1_evals/configs/`，可从脚本再生成，正式 configs 不存其运行副本。旧 configs/az_loop_seed1_evals 忽略规则保留，兼容显式指定旧输出目录的命令。原冻结 run 不移动或重写。

目录、命令和结果入口见 [docs 导航](../docs/index.md)。新运行先核对 checkpoint 和目标设备；本仓库的大模型与训练 NPZ 不随 Git 提供。
