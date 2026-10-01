# 固定配对评测与结果解释

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
