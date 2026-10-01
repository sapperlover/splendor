# AlphaZero seed1：五轮训练与逐代评测

证据核对日期：2026-10-01。`az_loop_seed1` 已完成 5 轮 × 500 局，不是仅微型 smoke。采集为 Python `batched_cpu`、8 workers × 4 active、batch 4；learner 使用 CUDA。教师为 PPO latest `local_005000000_lifetime_005512112.pt`，SHA-256 `009d17d2e9bc0d879142d7ca3ff1777a6f8caa9dda9243c6a2d5eb3538b5d5b0`，不是 PPO best-dev。

冻结配置：S=1024/N=1、根噪声 0.25、alpha=0.3；双方合计完成玩家回合数小于 10 时温度 1，之后 0；AdamW lr=1e-4、weight_decay=1e-4，batch 256，current/history 各 128（首轮无 history）。种子 collection=9500000、split=9501000、sampling=9502000。跨轮延续 optimizer，总计 2,982 步。

## 训练事实

| 代 | 新完整局 | value 样本 | 有效 policy 样本 | 优化步 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 500 | 43191 | 32049 | 601 |
| 2 | 500 | 42398 | 32001 | 598 |
| 3 | 500 | 42476 | 32064 | 591 |
| 4 | 500 | 42498 | 32407 | 595 |
| 5 | 500 | 42471 | 32491 | 597 |

五轮采集 summary 均记录 500/500，failed_games、exceptions、engineering_truncations 为零。最后 replay 覆盖 2,500 局，θ5 已提交；这里复核既有 summary 和身份记录，没有重新执行采集或全量 replay。

## 扩展配对评测

各代对同一初始网络；greedy 500 pairs，MCTS S=1024/N=1 50 pairs，均交换先后手。game seeds 从 9900000 起，policy base=9910000，CPU 8 workers、无根噪声、温度 0。区间为既有 pair bootstrap 95% CI。

| 代 | 面板 | 完整 pairs | 得分率 | CI |
| --- | --- | ---: | ---: | --- |
| θ1 | greedy | 500 | 49.20% | 46.50%–51.85% |
| θ1 | mcts | 50 | 61.00% | 51.00%–71.00% |
| θ2 | greedy | 500 | 50.50% | 47.35%–53.70% |
| θ2 | mcts | 50 | 62.50% | 53.50%–71.50% |
| θ3 | greedy | 500 | 51.15% | 47.95%–54.35% |
| θ3 | mcts | 50 | 52.00% | 42.00%–62.00% |
| θ4 | greedy | 500 | 50.70% | 47.40%–54.00% |
| θ4 | mcts | 50 | 66.50% | 58.50%–74.50% |
| θ5 | greedy | 500 | 53.65% | 50.40%–56.85% |
| θ5 | mcts | 50 | 56.50% | 47.00%–66.00% |

全部十个面板 passed=true、异常和工程截断为零。θ5 greedy 的点估计高于初始；MCTS θ4 点估计最高，但跨代并非单调。θ5 与 θ1 的 greedy 同 seed 差值为 +4.45 个百分点，既有区间 [+0.45,+8.55]；MCTS 跨代差值区间较宽。不能从单训练 seed、多个被查看面板推断稳定总体收益，也不能把本次扩展对照称为 5C 预留 FINAL 验收。

## 证据与限制

[训练摘要](../../artifacts/az_loop_seed1/training_summary.json)、[逐代汇总](../../artifacts/az_loop_seed1/generation_scores.json)、[原始评测副本](../../artifacts/az_loop_seed1/evaluations/) 和 [冻结 run 摘要](../../artifacts/az_loop_seed1/run_snapshot/) 随仓库保留；复制来源与 SHA 见 [保全清单](../../artifacts/retention_manifest_2026-10-01.json)。原 JSON 内的绝对路径/runtime 路径是测量时记录，没有为了整理而改写。

各轮采集 manifest 保存 `edcfb23…` / dirty 等测量时版本，具体值见原件；dirty 状态不能证明完整源码可恢复。checkpoint 和训练 NPZ 仍在忽略的 `runs/`，迁移需另行备份。没有 native 多轮长训、多训练 seed、扩容或超参 ablation 的验收结论。
