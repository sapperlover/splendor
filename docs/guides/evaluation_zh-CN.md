# 固定配对评测与结果解释

2026-10-04新增：[新AZ每轮/阶段/预留面板](next_az_training_zh-CN.md)，固定种子和paired bootstrap分开DEV/最终用途。`scripts/run_recorded_evaluation.py`复用本入口并补外部单面板会话/资源账本，不调度训练，也不自动暂停或选模；本次仅准备未运行。

从仓库根目录执行，使用 Conda `splendor`、`PYTHONPATH=src`。基线评测入口为 `splendor.evaluate`，冻结 checkpoint/search 的统一入口为 `splendor.evaluation`。配置种类包括 `greedy_checkpoint`、`search_checkpoint`（固定深度）和 `mcts_checkpoint`（PUCT）；两侧模型可不同。

## 配对与计分

每个 pair 是一个 game seed 的两局先后手交换；统计得分率为 `(wins + 0.5 * draws) / valid_games`。bootstrap 以 pair 为单位，不将两局当成独立样本。报告必须标明双方 checkpoint 内容身份、动作模式、规则/特征/后端、搜索预算、seed、完整 pairs 和无效局数。

`passed=true` 表示评测工程完成，不表示棋力提升。未完成、异常、工程截断结果不能用于完整面板排名。确定性 policy-cycle 判平属于评测裁决，不是引擎正式终局；只用于固定确定性策略，不自动用于有根噪声/温度采样的采集。DEV 与 FINAL 的区别由预留和选模使用方式决定，单独种子域不自动成为无偏 FINAL 验收。

## 可执行入口

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.evaluate \
  --player-a score --player-b development --seed-start 0 --seed-count 100
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.evaluation \
  --config configs/mcts_vs_greedy.json --output runtime/mcts_example/evaluation.json
# 相同配置、输出路径追加 --resume；保留相邻 .games/ sidecar
```

四组 sweep 的配置位于 `configs/mcts_50pair_sweep/`，输出位于 `artifacts/mcts_50pair_sweep/`；脚本失败会停止后续组。

```bash
# 仅检查配置、checkpoint 和命令，不开始对局
conda run -n splendor env PYTHONPATH=src python scripts/run_mcts_50pair_sweep.py --dry-run
# 真正继续历史 sweep 时使用 --resume；当前并非四组全部完成
```

实际 sweep 状态见 [结果报告](../reports/mcts_sweep_zh-CN.md)。AZ 扩展面板入口为 `scripts/run_az_generation_evals.py`；默认 θ1–θ4，θ1–θ5 要显式传 `--rounds 1 2 3 4 5`。summary 默认纳管，生成配置和 sidecar 忽略，见 [AZ 指南](alphazero_zh-CN.md)。

新实验使用新输出目录；恢复既有实验必须保持冻结配置和 checkpoint 身份。5C Quick/Formal/Promotion/FINAL 的种子域与晋升阈值见 [PPO 实验协议](../specs/ppo_training_protocol_zh-CN.md)。

单 worker 的有限 native 时间归因入口为 [`profile_speed_pipeline.py evaluation`](../../scripts/profile_speed_pipeline.py)，示例、计时边界及既有结果见 [分项报告](../reports/pipeline_profile_zh-CN.md)。它调用同一个固定配对 runner，保留 sidecar 和状态语义；不是新增跨局批量评测，也不能用 1 pair 宣称棋力或完整面板吞吐。

## 跨局 native 批量入口

新配置可增加 `execution`：`backend=native_batched`、`active_games`、`max_batch_size`、`candidate_buckets`、`inference_mode` 和 `actor_projection`；`workers=1` 指唯一推理 owner，`device` 可选 CPU/CUDA。示例为 `configs/speed_2026-10-03/evaluation_optimized.json`。固定θ5/θ4各50 pairs真实对照已完成：CPU eager8 workers为682.17秒，唯一CUDA owner active100/B64为207.21秒，100局终态/结果/长度一致；详见[完整性能专题](../reports/end_to_end_speed_optimization_zh-CN.md)。这是整条执行路径收益，非棋力提升证据。

此路径只接受双方均为 `native` 的 `mcts_checkpoint`。每局各自保留 policy seed，按 checkpoint 内容身份和推理配置隔离 evaluator/批次；原 checked runner 以协程推进真实规则、循环裁决、异常和截断。所有就绪局已经提供当前可用叶请求，因此 `max_batch_wait_ms` 必须为 0，尾批立即执行。恢复仍以完整 pair 为单位，半个 pair 使用原种子重跑；执行配置参与 manifest 身份，不允许与旧 CPU 路径拼接。

AZ 的 `evaluation.mcts_execution` 接通同一路径；需要 `mcts_backend=native`、`workers=1`，不影响旧配置的默认 CPU 多进程入口。两个配置应使用不同运行目录。
