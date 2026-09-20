# 5B Stage 4 扩展成对评测与 Bootstrap CI

## 1. 结论

`step_000512112.pt` 对 `step_000403183.pt` 的直接评测已从 200 个扩大到 1,000 个 paired seeds，即 2,000 局且每个 seed 交换先后手。512112 的 score rate 为 **0.54725**，paired-seed percentile bootstrap 95% CI 为 **[0.52649, 0.56800]**。该区间不包含 0.5，因此在这一固定评测面板上，512112 对 403183 的直接优势得到了更大样本的支持。

但是，三个 heuristic panel 的成对差值给出了更细的结论：

- RandomPolicy 出现了可分辨的回落；
- ScorePolicy 和 DevelopmentPolicy 的点估计均上升，但当前 200 paired seeds 的 CI 仍包含 0，不能将它们称为已确认的提升。

## 2. 统计协议

- 直接对战：1,000 paired seeds / 2,000 games。
- heuristic 差值：原固定 DEV 面板的 200 paired seeds / 400 games per checkpoint per opponent。
- 一个统计单位是同一 game seed 下交换先后手两局的平均 score，bootstrap 不拆散这个单位。
- score 定义为胜 `1`、和 `0.5`、负 `0`。
- 95% percentile bootstrap，100,000 次重采样，固定 bootstrap seed `5120403183`。
- 直接对战沿用 Stage 4 的 DEV game-seed 和 policy-seed 协议；前 200 pairs 与原评测的 outcome、资源结果和 replay digest 逐局一致。
- 三个 heuristic CI 是各自的 95% 区间，未做 multiple-comparison/family-wise correction。

## 3. 512112 vs 403183

| 指标 | 结果 |
|---|---:|
| paired seeds | 1,000 |
| games | 2,000 |
| 512112 W/D/L | 1,080 / 29 / 891 |
| score rate | 0.54725 |
| paired bootstrap 95% CI | [0.52649, 0.56800] |
| 512112 先手 score | 0.61400 |
| 512112 后手 score | 0.48050 |
| engineering truncation | 0 |
| IllegalAction | 0 |
| invariant failure | 0 |
| exception | 0 |

先后手差异较大，但每个 seed 都完成了换位，总体估计没有因先手次数不平衡而偏移。

## 4. 同 seed heuristic 差值

差值定义为 `512112 score rate - 403183 score rate`。

| opponent | 403183 | 512112 | paired 差值 | 95% CI | 解读 |
|---|---:|---:|---:|---:|---|
| RandomPolicy | 0.99500 | 0.97125 | -0.02375 | [-0.04375, -0.00625] | 回落可分辨 |
| ScorePolicy | 0.41750 | 0.44625 | +0.02875 | [-0.03750, +0.09625] | CI 包含 0 |
| DevelopmentPolicy | 0.38625 | 0.39625 | +0.01000 | [-0.04625, +0.06500] | CI 包含 0 |

因此，扩展直接对战支持 512112 相对 403183 的总体优势，但这不等价于对每个固定对手都有稳定改善。当前证据更符合：**对 403183 的直接棋力提升，同时伴随小幅 Random 能力回落，而两个较强 heuristic 的提升尚未被 200-pair 面板确认。**

## 5. 可复现产物

- 完整 2,000 局结果：`runs/5b_stage4_selfplay_seed_004/extended_eval_512112_vs_403183/direct_512112_vs_403183.json`
- CI 摘要：`runs/5b_stage4_selfplay_seed_004/extended_eval_512112_vs_403183/paired_bootstrap_report.json`
- 可复运行入口：`python -m splendor.evaluate_stage4_ci`

新增的统计边界测试和全量回归均通过。本次只扩大评测并计算 CI，没有继续训练或启动 5C。
