# 工作包 5B Stage 4：短历史池 self-play 验证

## 1. 验收结论

Stage 4 完成了 150,947 个学习方决策的短 self-play，并完成了一次真实的 `train -> save -> 正常退出 -> reload -> continue`。frozen latest、historical PPO、ScorePolicy 和 DevelopmentPolicy 可在同一 batched collector 中稳定参与训练，没有 IllegalAction、invariant failure、schema/candidate mapping 错误、NaN/Inf 或 engineering truncation。

本次最佳 Stage 4 checkpoint 是 `step_000512112.pt`：固定 heuristic composite 为 `0.53125`，高于起点 `step_000403183.pt` 的 `0.52050`；对起点 checkpoint 的成对换先手 score rate 为 `0.6000`。这支持“本次短 self-play 在最佳检查点有益”，但最终 latest `step_000554130.pt` 的 composite 回落到 `0.5180`，说明训练仍有波动，latest 与 best-dev 必须继续分开管理。

Stage 4 没有启动 5C 或任何百万级长训练。

后续将 512112 vs 403183 直接对战扩大到 1,000 paired seeds 后，512112 score rate 为 `0.54725`，paired bootstrap 95% CI 为 `[0.52649, 0.56800]`。三个 heuristic 面板的同-seed 差值 CI 见 `docs/work_package_5b_stage4_paired_ci_zh-CN.md`。

## 2. 实验配置

| 项目 | 实际值 |
|---|---:|
| starting checkpoint | `step_000403183.pt` |
| starting global step | 403,183 |
| Stage 4 最终 global step | 554,130 |
| Stage 4 decisions | 150,947 |
| Stage 4 PPO updates | 20（累计 update count 51 -> 71） |
| learning rate | `3e-4 -> 1e-4` |
| gamma / GAE lambda | `1.0 / 0.95` |
| clip / value coef / entropy coef | `0.2 / 0.5 / 0.01` |
| grad norm / PPO epochs / minibatch | `0.5 / 4 / 512` |
| target KL | `0.02` |
| reward | 纯正式终局 `+1/-1/0`，`alpha=0` |
| active logical envs | 32 |
| rollout target | 8,192 |
| training seed | 4 |
| DEV panel | Stage 1 同一批 200 paired seeds / 400 局/对手 |
| neural training opponent mode | **sampled policy** |
| neural DEV opponent mode | **greedy / eval / inference_mode** |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |

`starting checkpoint`、`latest checkpoint` 和 `best-dev checkpoint` 分别管理：

- starting：原始且未覆盖的 `runs/5b_fixed_mix_seed_002_resume_206542/checkpoints/step_000403183.pt`；
- latest：`runs/5b_stage4_selfplay_seed_004/checkpoints/step_000554130.pt`；
- best-dev：`runs/5b_stage4_selfplay_seed_004/checkpoints/step_000512112.pt`。

checkpoint metadata 明确记录了 `3e-4 -> 1e-4`的阶段学习率切换。除学习率外，本轮没有同时修改其他 PPO 超参数。

## 3. 对手池与抽样

目标 episode mixture 为 latest/historical/Score/Development = `30%/20%/25%/25%`，不使用 RandomPolicy 作训练对手。每局的对手类型、checkpoint、参数和 RNG 一旦选定就保持到该局结束。

3,217 局训练对局的实际抽样为：

| opponent | games | actual | target |
|---|---:|---:|---:|
| frozen latest | 967 | 30.06% | 30% |
| historical PPO | 632 | 19.65% | 20% |
| ScorePolicy | 832 | 25.86% | 25% |
| DevelopmentPolicy | 786 | 24.43% | 25% |

历史池使用四个真实且时间跨度明显的 checkpoint，没有将 403183 重复放入 historical bucket：

| historical checkpoint | sampled games | historical fraction |
|---|---:|---:|
| 100005 | 165 | 26.11% |
| 157351 | 138 | 21.84% |
| 255701 | 173 | 27.37% |
| 304858 | 156 | 24.68% |

历史池使用简单均匀抽样，未实现 Elo、PFSP 或其他优先级算法。

## 4. frozen latest 和轨迹边界

每个 rollout collection cycle 开始前，当前 learner `theta_k` 被深拷贝为参数存储独立的 frozen latest `theta_k^opp`。旧对局即使跨过 rollout 边界，仍持有创建时的 opponent model；新对局才使用新的 latest snapshot。

每轮 update 都检查：

- learner 与 latest opponent 没有共享 parameter storage；
- latest opponent `requires_grad=False`；
- opponent parameter 不在 learner optimizer param groups 中；
- PPO update 前后 latest opponent 参数位级不变；
- `LearningEnvironment` 只保留固定 learning player 的 transition，opponent action/logprob/value 不写入 learner rollout。

sampled neural opponent 只接收 `DecisionRequest`，通过正式 3B encoding 和 4A Actor–Critic 推理，没有 `GameState` 入口。对手盲保留真实身份互换的 metamorphic test 证明同 seed 下的 neural opponent 输入和选择不变。

## 5. 固定 DEV 面板

| global step | Stage 4 delta | Random | Score | Development | composite | vs 403183 | vs historical 157351 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 403183 | 0 | 0.9950 | 0.4175 | 0.3863 | 0.5205 | 0.5000 | 0.5263 |
| 453974 | +50,791 | 0.9800 | 0.4225 | 0.3525 | 0.5060 | 0.4900 | 0.5150 |
| **512112** | **+108,929** | **0.9713** | **0.4463** | **0.3963** | **0.5313** | **0.6000** | **0.5263** |
| 554130 | +150,947 | 0.9650 | 0.4338 | 0.3788 | 0.5180 | 0.5625 | 0.5175 |

每个数值都来自相同的 200 paired seeds/400 局，交换先后手。最佳 512112 checkpoint 的 W/D/L 为：

- vs Random：`387/3/10`，执先/执后 `0.9700/0.9725`；
- vs Score：`177/3/220`，执先/执后 `0.4600/0.4325`；
- vs Development：`157/3/240`，执先/执后 `0.4700/0.3225`；
- vs 403183：`236/8/156`，score rate `0.6000`；
- vs historical 157351：`207/7/186`，score rate `0.52625`。

起点的 Random/Score/Development 面板精确复现 Stage 1 的 `0.9950/0.4175/0.38625`。开发中曾发现 Stage 4 给 RandomPolicy 使用了新 policy-seed 基址；这不影响训练轨迹，但会破坏历史面板的可比性。已先增加失败回归测试，再改为 Stage 1 完全相同的 game/policy seeds。旧现场保留在 `runs/5b_stage4_selfplay_seed_004_dev_seed_mismatch`，本报告只使用修正后的接受运行。

## 6. 策略漂移与遗忘

最佳 512112 相对起点：

- Random `-2.375` 个百分点，仍为 `97.125%`；
- Score `+2.875` 个百分点；
- Development `+1.000` 个百分点；
- heuristic composite `+1.075` 个百分点；
- vs 403183 为 `60.0%`。

因此没有出现“只学会克制 latest，同时固定 heuristic 系统性崩溃”，也没有灾难性遗忘。但 final latest 的 composite 回落到 `0.5180`，证明对 self-play 训练不能只保存 latest，必须继续使用固定 heuristic DEV 选模。

本次跟踪的三角关系中，512112 胜过 403183，403183 略胜 157351，512112 也略胜 157351，没有观察到明确 non-transitivity。这不能排除更大历史面板中的非传递性。

## 7. PPO 与行为诊断

Stage 4 的 20 次 update 中：

| 指标 | 首次 | 末次 | 范围/最大值 |
|---|---:|---:|---:|
| entropy | 1.5970 | 1.3140 | 1.3140–1.5970 |
| approx KL | 0.001214 | 0.000068 | max 0.001419 |
| clip fraction | 0.01157 | 0.00000 | max 0.01678 |
| value loss | 0.1754 | 0.2400 | 0.1351–0.2452 |
| explained variance | 0.7333 | 0.6468 | 0.6468–0.7937 |
| grad norm | 0.4839 | 1.5332 | max 1.5332 |

KL 始终远低于 `0.02`，没有 KL early stop。梯度范数记录的是 clip 前诊断，1,188 个 minibatch 中 667 个触发了既定 `0.5` 梯度裁剪。全部 loss 和诊断值为 finite。

Stage 4 累计 MAIN_ACTION 分布为：

| action | count | fraction | Stage 1 累计参考 |
|---|---:|---:|---:|
| TakeDistinct | 36,162 | 36.92% | 35.46% |
| TakePair | 702 | 0.72% | 2.24% |
| ReserveMarket | 7,632 | 7.79% | 10.53% |
| ReserveDeck | 3,926 | 4.01% | 1.81% |
| BuyMarket | 47,221 | 48.21% | 46.96% |
| BuyReserved | 2,300 | 2.35% | 2.99% |

末个小 rollout 的 BuyMarket 比例为 `58.72%`，高于首个 rollout 的 `45.63%`；由于尾 rollout 样本较小，此处只作漂移诊断，不单独解读为棋力提升。训练子集 score rate 仅作 sanity check：latest/historical/Score/Development 分别为 `0.4917/0.6274/0.3456/0.4154`，不用于代替固定 DEV。

## 8. 吞吐与资源

| 项目 | 实测 |
|---|---:|
| learner decisions/s | 303.56 |
| completed games/s | 6.47 |
| mean learner inference batch | 30.38 |
| rollout wall time | 497.26 s |
| environment + encoding | 292.00 s |
| learner batched inference | 25.17 s |
| neural opponent inference | 180.09 s / 75,730 decisions |
| GAE | 0.89 s |
| PPO update | 26.63 s |
| CUDA peak allocated | 166.11 MiB |
| CPU max RSS | 1,989.55 MiB |
| 总 wall-clock（含 4 次五对手 DEV） | 2,325.23 s |

加入未批处理的 sampled neural opponent 推理后，吞吐低于纯 heuristic Stage 2 的 513.69 decisions/s，但仍明显高于 serial 的 186.34 decisions/s，并且 32 环境的 learner 平均 batch 为 30.38。本阶段不为此引入新的 inference server、AMP 或分布式框架。

## 9. checkpoint/resume 实战验证

第一个进程从 403183 运行到 453974（+50,791），保存后正常退出。第二个独立进程从 `step_000453974.pt` 恢复，继续到 554130（累计 +150,947）。

恢复包括：

- learner model、Adam state、global step 和 PPO update count；
- Python、NumPy、Torch CPU/CUDA、PPO shuffle 和 sampler RNG；
- batched collector `next_episode_index`；
- opponent mixture RNG；
- historical pool manifest 和 pool RNG；
- historical checkpoint metadata/schema/model config 重新校验；
- `1e-4` 学习率和 Stage 4 不可变配置。

checkpoint 只在 32 个活动环境全部完成后保存，不丢弃或伪造局中状态。`resume_count=1`，恢复后对手抽样、训练和评测均正常继续。

## 10. 测试与正确性

新增 8 个 Stage 4 测试，覆盖：

- 奖励、mixture 和学习率配置冻结；
- frozen latest 与 learner 不共享参数，optimizer 只更新 learner；
- mixture RNG 和 historical pool RNG/manifest 恢复；
- neural opponent 盲保留隐藏信息边界；
- sampled neural opponent 在 batched collector 中完成 P0/P1 两种 learning seat；
- 已开局的 opponent checkpoint 不因 pool latest 刷新而改变；
- Stage 1/Stage 4 固定 DEV game/policy seed 协议精确一致；
- 小型真实 save/exit/reload/continue。

最终全量回归为 `194 passed, 2 skipped`；Python 编译检查通过。训练 3,217 局全部正常终局，forced skip 36 次，engineering truncation 0。四次 DEV 共 8,000 局，全部有效，IllegalAction、invariant failure、exception 和 truncation 均为 0。

## 11. 是否建议进入 5C

结论是 **有条件建议**：Stage 4 的正确性、恢复、吞吐和对手池基础设施已达到进入 5C 的技术条件，且最佳 checkpoint 在固定 heuristic 面板上有小幅实测改善。

5C 应从人工确认的 best-dev checkpoint 而不是自动从 latest 开始，继续保留 Random/Score/Development 绝对锚点，并用多个训练 seed 验证本次 `+1.075` 个百分点的 composite 提升是否稳定。本工作包不自动启动 5C。
