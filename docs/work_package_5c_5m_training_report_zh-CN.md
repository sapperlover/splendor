# 工作包 5C：首次 5M learning-decisions 主训练完成报告

## 1. 结论摘要

`5c_main_seed1` 已正常完成计划中的 `5,000,000` 个 run-local learning decisions。
训练过程共完成 `698` 次本阶段 PPO update、`110,344` 局完整训练对局，
没有 engineering truncation、NaN/Inf、IllegalAction 或 invariant failure。

最终应明确区分两个 checkpoint：

- **best-dev**：`local_004150000_lifetime_004662112.pt`，是当前应优先用于部署、人机对局和后续评测的模型；
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
`32.6875` 个百分点。这说明 5M 训练产生了明确、幅度较大的棋力提升。

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

## 3. 冻结的训练协议

### 3.1 PPO 与奖励

- reward：纯 terminal `+1 / 0 / -1`；
- reward shaping：`alpha = 0`；
- `gamma = 1.0`；
- `GAE lambda = 0.95`；
- rollout budget：`8192` learning decisions；
- PPO epochs：`4`；
- minibatch：`512`；
- clip coefficient：`0.2`；
- value coefficient：`0.5`；
- max gradient norm：`0.5`；
- target KL：`0.02`；
- optimizer：Adam，`weight_decay = 0`；
- advantage normalization：开；
- value clipping：关；
- learning rate：按 run-local step 从 `1e-4` 线性降到 `3e-5`；
- entropy coefficient：从 `0.01` 线性降到 `0.002`。

5M budget 只计入真正进入 PPO rollout/update 的 learner decisions。在 checkpoint、
Quick、Formal 或 phase 边界，已开始对局会被 drain 到完整结束，但 drain tail 中的
decisions 不计入 5M、LR/entropy schedule 或 PPO update。

实际产生了 `5,072,307` 个完整 episode 内 learner decisions，其中 `72,307`
个是上述 boundary drain tail，正好留下 `5,000,000` 个训练 decisions。

### 3.2 采样和动作模式

- `32` 个 logical environments；
- 正式运行使用 `single_process` batched collector；
- learner 在训练时使用 categorical stochastic sampling；
- latest/historical neural opponent 在训练时也使用 stochastic sampling；
- Quick/Formal/Promotion/Historical 评测使用 greedy argmax；
- frozen latest 在每个 rollout 开始前从当前 learner 深拷贝，新 episode 使用新快照，
  已开始 episode 不会在局中切换对手。

### 3.3 训练对手比例

Phase A，run-local `0..<1,000,000`：

- 30% frozen latest PPO；
- 20% historical PPO；
- 25% ScorePolicy；
- 25% DevelopmentPolicy。

Phase B，run-local `1,000,000..5,000,000`：

- 50% frozen latest PPO；
- 30% historical PPO；
- 10% ScorePolicy；
- 10% DevelopmentPolicy。

Phase B 的 historical 30% 内部拆成约 10% fixed anchors 和 20% dynamic history。
全程实际完成对局分布与加权目标一致：

| 对手类型 | 对局数 | 占比 | 训练期 learner score rate |
|---|---:|---:|---:|
| frozen latest | 50,674 | 45.92% | 49.72% |
| historical | 30,810 | 27.92% | 67.50% |
| ScorePolicy | 14,546 | 13.18% | 55.67% |
| DevelopmentPolicy | 14,314 | 12.97% | 59.81% |

训练期 score rate 混合了不同时间的 learner 和对手版本，只作 sanity check，不用于
checkpoint selection。

## 4. 评测与模型选择协议

### 4.1 Quick DEV

- 对手：ScorePolicy + DevelopmentPolicy；
- 规模：每个对手 50 paired seeds = 100 games；
- 用途：迅速诊断与 Formal candidate 预选；
- 不直接更新 best-dev。

实际共完成 `78` 次 Quick panel。运行初期有一段使用约 100k 节奏，从日志的
300k 之后切到约 50k；Formal 边界不重复执行 Quick。Quick 最高综合分是
4.60M 的 `74.25%`，最后一次 4.95M 为 `67.75%`。Quick 与 Formal 使用不同
seed domain，绝对数不应直接对齐。

### 4.2 Formal DEV

- 固定对手：ScorePolicy、DevelopmentPolicy、起点 `step_000512112.pt`；
- 每个对手 200 paired seeds = 400 games；
- 每 250k run-local decisions 一个 Formal cycle；
- 所有 candidate 使用同一套固定 Formal seeds；
- 主排序指标是 Score/Development 各 50% 的 `heuristic_score`；
- `vs_start_512112` 只作长期锚点，不进入 heuristic score。

共执行 `21` 个 Formal cycles（含 step 0 baseline），评测 `61` 个 Formal candidate。

### 4.3 Promotion

Formal candidates 按 heuristic score 排序。满足
`H_candidate >= H_incumbent - 0.01` 的 candidate 依次挑战 incumbent。

- 第一阶段 200 paired seeds；
- pair-level bootstrap，`100,000` 次重采样；
- `LCB90` / `UCB90` 分别是 bootstrap distribution 的 10% / 90% 分位数；
- `LCB90 > promotion_threshold` 则晋升；
- `UCB90 <= threshold` 则明确不晋升；
- 其余情况扩大到总计 500 paired seeds，仍只在 `LCB90 > threshold` 时晋升；
- 不继续追加样本。

共进行 `29` 次 initial promotion matches，其中 `19` 次扩展到 500 pairs。
所有 bootstrap seed 按预先冻结公式确定。

### 4.4 Historical diagnostic

- 每个活跃 historical member 使用 50 paired seeds = 100 games；
- 仅用于诊断遗忘、cycling 和 non-transitivity；
- 不参与 Quick/Formal 排名、promotion threshold 或 best-dev 选择。

共执行 `21` 个 Historical panels，覆盖 `160` 个 checkpoint-opponent evaluations，
合计 `16,000` 局。

### 4.5 评测循环判平

确定性评测中，如果双方各自第二次遇到同一 canonical state signature，
则记为 `policy_cycle_draw`。signature 包含会影响后续合法动作和转移的状态，
排除 completed turns、RNG draw counter、replay/logging counter 等纯记录字段。

这是确定性 policy-vs-policy 评测的循环裁决，不是 engineering truncation，
也不会改写 GameEngine 的正式资源规则。记录到的 Formal/Historical 摘要中
共出现 `12` 次 policy cycle draw。

## 5. Formal 学习曲线

下表每行是该 Formal cycle 内的 Formal 第一名，不一定是当时或最终的
global best-dev。“晋升”列以直接 promotion match 的最终结果为准。

| Formal cycle | 第一名 checkpoint step | Score | Development | Heuristic | vs start | 晋升结果 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 / source | 41.25% | 42.38% | 41.81% | 50.00% | baseline |
| 0.25M | 0.20M | 49.13% | 47.38% | 48.25% | 59.00% | 晋升 0.20M |
| 0.50M | 0.50M | 52.50% | 44.75% | 48.63% | 56.75% | 否 |
| 0.75M | 0.70M | 53.88% | 43.00% | 48.44% | 56.38% | 晋升 0.70M |
| 1.00M | 1.00M | 53.50% | 45.00% | 49.25% | 55.75% | 否 |
| 1.25M | 1.25M | 52.38% | 44.63% | 48.50% | 59.88% | 第一名失败；第二名 1.15M 晋升 |
| 1.50M | 1.50M | 53.13% | 48.50% | 50.81% | 60.13% | 否 |
| 1.75M | 1.65M | 55.88% | 50.00% | 52.94% | 61.88% | 晋升 1.65M |
| 2.00M | 2.00M | 54.63% | 54.88% | 54.75% | 58.00% | 否 |
| 2.25M | 2.25M | 57.38% | 62.63% | 60.00% | 66.25% | 晋升 2.25M |
| 2.50M | 2.40M | 61.75% | 66.88% | 64.31% | 66.75% | 晋升 2.40M |
| 2.75M | 2.75M | 64.25% | 63.63% | 63.94% | 74.13% | 晋升 2.75M |
| 3.00M | 3.00M | 66.75% | 68.88% | 67.81% | 70.50% | 否 |
| 3.25M | 3.10M | 67.00% | 72.13% | 69.56% | 70.00% | 否 |
| 3.50M | 3.50M | 69.00% | 74.38% | 71.69% | 74.38% | 晋升 3.50M |
| 3.75M | 3.75M | 73.38% | 69.88% | 71.63% | 76.50% | 否 |
| 4.00M | 3.95M | 69.13% | 73.00% | 71.06% | 75.63% | 否 |
| 4.25M | 4.15M | 70.38% | 78.63% | **74.50%** | 75.38% | **晋升，最终 best-dev** |
| 4.50M | 4.45M | 72.75% | 74.63% | 73.69% | 77.25% | 否 |
| 4.75M | 4.60M | 72.75% | 76.50% | **74.63%** | 76.88% | 否，直接对局证据不足 |
| 5.00M | 5.00M | 69.25% | 75.00% | 72.13% | **78.75%** | 否，不满足晋升资格线 |

曲线可分为四段：

1. **0–1.5M：平台与适应期。** Heuristic 综合分长时间在 48%–51% 附近。
2. **1.75M–2.5M：第一段明显上升。** 从 52.94% 提高到 64.31%，Development 提升尤其明显。
3. **2.75M–4.25M：继续上升但伴随非单调和 non-transitivity。** Formal 第一名最终到达 74.50%。
4. **4.25M–5M：高位平台/回落。** 4.60M 的 Formal heuristic 数值略高，但直接挑战不足以晋升；
   5M latest 下降到 72.13%，因此保留 4.15M best-dev。

## 6. Global best-dev 晋升链

除 step 0 baseline 外，共有 `9` 次成功晋升。

| Formal cycle | 晋升 checkpoint | H_new | 对 incumbent 得分率 | LCB90 | UCB90 | pairs |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25M | 0.20M | 48.25% | 52.75% | 50.70% | 54.75% | 500 |
| 0.75M | 0.70M | 48.44% | 57.63% | 54.63% | 60.63% | 200 |
| 1.25M | 1.15M | 48.44% | 53.00% | 50.13% | 55.88% | 200 |
| 1.75M | 1.65M | 52.94% | 56.88% | 53.88% | 59.88% | 200 |
| 2.25M | 2.25M | 60.00% | 57.00% | 53.75% | 60.25% | 200 |
| 2.50M | 2.40M | 64.31% | 56.50% | 53.50% | 59.50% | 200 |
| 2.75M | 2.75M | 63.94% | 54.65% | 52.65% | 56.70% | 500 |
| 3.50M | 3.50M | 71.69% | 52.50% | 50.55% | 54.50% | 500 |
| 4.25M | 4.15M | 74.50% | 54.38% | 51.25% | 57.38% | 200 |

2.75M 的 H_new 比当时 incumbent 低 `0.375` 个百分点，但仍在允许的 1 个百分点
eligibility 容差内，并在独立 promotion seeds 上明确战胜 incumbent，因此合法晋升。
这也说明固定启发式面板与模型间直接棋力不是完全可代替的单一排序。

4.60M 是最后期最有竞争力的挑战者：

- Formal heuristic：`74.625%`，略高于 incumbent 的 `74.500%`；
- 对 incumbent 直接对局：`51.10%`；
- 500-pair bootstrap：`LCB90=49.10%`，`UCB90=53.05%`；
- 因 LCB90 没有超过 50% threshold，按冻结规则不晋升。

5M cycle 的 Formal 三个 candidate 均低于 incumbent H 减 1 个百分点的资格线
`73.50%`，因此没有发起新的 promotion match。

## 7. Source、best-dev 和 latest 对比

| checkpoint | run-local step | Score | Development | Heuristic | vs 512112 |
|---|---:|---:|---:|---:|---:|
| source `512112` | 0 | 41.25% | 42.38% | 41.81% | 50.00% |
| **best-dev `4.15M`** | 4.15M | **70.38%** | **78.63%** | **74.50%** | 75.38% |
| latest `5.00M` | 5.00M | 69.25% | 75.00% | 72.13% | **78.75%** |

best-dev 的优势是启发式面板更强且先后手更均衡。latest 对起点的得分率更高，
但对 ScorePolicy 出现了明显的先后手差：`76.0% / 62.5%`，而 best-dev 为
`69.0% / 71.75%`。

对 latest 不应做两种过度解读：

- 不能说 latest “崩溃”：它仍对起点达到 78.75%，对两个 heuristic 也都明显强于起点；
- 也不能说 latest 与 best-dev 已证明等强：5M 时 latest 没有 promotion eligibility，
  因此没有一场最终 best-vs-latest 直接对局。

## 8. Historical pool 与遗忘诊断

最终 active pool 容量为 `12`：

- fixed anchors：`157351`、`512112`；
- dynamic history：`712112`、`1212112`、`1762112`、`2162112`、`2762112`、
  `2912112`、`3262112`、`4012112`、`4662112`、`5512112`。

5M Formal winner 作为 time-spaced representative 加入 dynamic pool，按确定性时间距离规则淘汰了
`1662112`。

5M latest 在加入自己前的最后 Historical diagnostic 中：

- active opponents：12；
- mean / median / minimum score：`65.21% / 66.50% / 42.50%`；
- vs fixed `157351`：81.0%；
- vs fixed `512112`：78.5%；
- vs `3262112`：50.5%；
- vs `4012112`：42.5%；
- vs best-dev `4662112`：50.0%。

best-dev 在 4.25M 时的当期 Historical diagnostic 为：

- mean / median / minimum：`68.95% / 71.00% / 47.50%`；
- vs `4012112`：49.0%。

这些数据表明：

- 对早期模型没有灾难性遗忘；
- 对中后期模型存在明显的近似对手、循环克制或 seed-sensitive 现象；
- historical pool 成员会变化，每个对手只有 50 pairs，因此不能把不同时点的
  pool mean 当作严格可比的单调学习曲线。

## 9. 训练对局与行为统计

### 9.1 对局结果

| 指标 | 数值 |
|---|---:|
| 完整对局 | 110,344 |
| learner 胜 / 平 / 负 | 62,333 / 633 / 47,378 |
| learner 胜 / 平 / 负率 | 56.49% / 0.57% / 42.94% |
| learning player 为 P0 / P1 | 55,229 / 55,115 |
| P0 / P1 占比 | 50.05% / 49.95% |
| 平均完整玩家回合数/局 | 58.14 |
| 平均 learner decisions/局 | 45.97 |
| 平均 learner 终局分数 | 14.17 |
| 平均购买发展卡/局 | 16.10 |
| 平均获得贵族/局 | 0.607 |
| forced-skip events | 619 |
| engineering truncations | 0 |

### 9.2 决策阶段

5M 个训练 decisions 恰好分解为：

| phase | 次数 | 占比 |
|---|---:|---:|
| MAIN_ACTION | 3,165,248 | 63.30% |
| PAYMENT | 1,748,054 | 34.96% |
| RETURN_TOKENS | 82,124 | 1.64% |
| CHOOSE_NOBLE | 4,574 | 0.091% |

### 9.3 主行动分布

| 主行动 | 次数 | MAIN_ACTION 内占比 |
|---|---:|---:|
| BuyMarket | 1,674,042 | 52.89% |
| BuyReserved | 75,022 | 2.37% |
| ReserveDeck | 89,287 | 2.82% |
| ReserveMarket | 182,521 | 5.77% |
| TakeDistinct | 1,141,653 | 36.07% |
| TakePair | 2,723 | 0.086% |

TakePair 在训练分布中非常少见，但仍有 `2,723` 次实际覆盖，并非完全未探索分支。

## 10. PPO 与系统健康度

### 10.1 优化指标

| 指标 | 数值 |
|---|---:|
| update 数 | 698 |
| 处理 minibatches | 39,200 |
| 发生 gradient clipping 的 minibatches | 37,509（95.69%） |
| KL early stop | 0 |
| 最大 observed approx KL | 0.00370 |
| target KL | 0.02 |
| 最大 clip fraction | 4.10% |
| 平均 explained variance | 0.682 |
| 首/末 explained variance | 0.732 / 0.657 |
| 首/末 value loss | 0.159 / 0.126 |
| 首/末 policy entropy | 1.498 / 0.953 |

没有出现 KL 失控、NaN/Inf 或 early-stop 风暴。但 gradient clipping 在绝大多数
minibatch 上都被触发，说明 `max_grad_norm=0.5` 是一个持续活跃的约束。
本次训练在该配置下仍保持稳定，但这是后续超参数对照实验值得单独记录的点。

### 10.2 性能与资源

| 指标 | 数值 |
|---|---:|
| rollout collection 总时间 | 18,493.71 s |
| PPO update 总时间 | 856.29 s |
| environment + encoding | 9,553.96 s |
| learner GPU inference | 776.37 s |
| neural opponent inference | 8,163.37 s |
| update 记录的平均 decisions/s | 258.18 |
| update 记录的中位 decisions/s | 270.34 |
| 按总 rollout 时间加权的 decisions/s | 270.36 |
| 包含评测的 active-wall decisions/s | 103.94 |
| 平均 learner inference batch size | 30.18 / 32 |
| 峰值 CUDA allocated memory | 175.90 MiB |
| 峰值 CPU RSS | 2,009.63 MiB |

rollout 时间中，environment/encoding 约占 51.7%，neural opponent inference 约占 44.1%，
learner batched inference 约占 4.2%。因此本次运行仍验证了早先的判断：主要吞吐成本
在 CPU 环境/编码和逐局 neural-opponent inference，而不是 PPO update 本身。

## 11. 健康性与失败统计

| 项目 | 结果 |
|---|---:|
| 训练 engineering truncation | 0 |
| 评测 engineering truncation | 0 |
| IllegalAction | 0 |
| invariant failure | 0 |
| 评测 exception | 0 |
| NaN/Inf | 0 |
| 未解释的无合法动作状态 | 0 |
| 中途 resume | 1 |

运行期间曾因确定性 policy 可进入重复状态循环而完善评测裁决，之后执行了一次
resume。修改的是确定性评测的 canonical-state cycle adjudication 和 Quick 频率，
没有改变 PPO 奖励、Formal 频率、训练对手比例或 GameEngine 资源/购买规则。

运行元数据记录工作树为 dirty，且 resume 后的 runtime Git commit 记录与准备阶段不同。
但 checkpoint 内的 rules/schema/config、seed domains、pool manifest、optimizer/RNG 恢复信息都被保存，
最终运行也通过了所有在线 invariant 和评测失败检查。后续正式论文级实验应从
干净 commit 启动新 run。

## 12. 对本次 5M 训练的整体判断

### 已经得到验证的结论

1. 纯 terminal reward 在完整双人游戏中可以把 Formal heuristic score 从 41.81% 提高到 74.50%。
2. Phase B 以 80% neural opponents 为主后，模型仍能继续提高对固定 heuristic 的棋力，
   没有灾难性遗忘。
3. historical/latest/fixed opponent 混合、动态历史池、checkpoint/resume 和成对换先手评测
   能够在 5M 规模下稳定运行。
4. Formal heuristic 排名和模型直接对局存在显著 non-transitivity；
   使用 Formal eligibility + promotion match 比单纯取 heuristic 最大值更稳妥。
5. 4.15M 后进入高位平台，继续训到 5M 没有产生足够证据取代 4.15M best-dev。

### 当前仍然不能宣称的结论

1. 不能根据同一 Formal DEV 面板宣称 4.15M 是真正的全局最优模型。
2. 不能用单一 training seed 估计训练方差和结论稳定性。
3. 不能把 changing historical-pool mean 解读为固定量尺上的绝对棋力。
4. 没有数据支持 reward shaping、`lambda=1.0`、网络扩容或其他超参数的优劣。

## 13. 建议的后续验收顺序

在不继续训练的前提下，建议先做一次独立 FINAL 验收：

1. 冻结 candidate 集：至少包含 source 512112、best-dev 4.15M、late challenger 4.60M、latest 5.00M；
2. 只使用训练期间从未使用的 `FINAL` seed domain；
3. 对 ScorePolicy、DevelopmentPolicy、source anchor 和冻结 historical representatives 做至少
   500 paired seeds；
4. 单独进行 best-dev vs 4.60M、best-dev vs latest 的成对直接对局与 bootstrap CI；
5. FINAL 结果产生后再决定正式发布 checkpoint，不用 DEV 面板反向修改本次训练。

如果进入后续实验，再使用独立训练 seeds 重复完整 run，并对 reward shaping、
GAE lambda 或 gradient clipping 做每次只改一个变量的对照。

## 14. 最终交付定位

当前阶段可以总结为：

> 首次 5M self-play/historical-pool PPO 主训练已成功完成；引擎与训练链路在
> 110,344 局训练对局中无未解释异常；当前 DEV 最优模型为 4.15M checkpoint，
> 对固定 heuristic 面板的综合得分率为 74.5%。独立 FINAL 评测和多训练 seed
> 复现仍是将“best-dev”升格为“最终模型”之前的必要工作。

## 15. 可复用的细粒度训练数据导出

为避免只保留汇总结论，已将 `metrics.jsonl` 和各评测 JSON 重新整理为
可直接使用 pandas、LibreOffice 或 R 读取的 CSV。导出工具为
`python -m splendor.analyze_5c`，原始运行目录仍为唯一数据来源。

| 需求 | 产物 | 行数/粒度 |
|---|---|---:|
| 每个 PPO update 的优化曲线 | [`ppo_updates.csv`](../artifacts/work_package_5c/ppo_updates.csv) | 698 个 update |
| 全部 Formal candidates | [`formal_candidates.csv`](../artifacts/work_package_5c/formal_candidates.csv) | 61 个 candidate |
| 四类训练对手的分段得分率 | [`opponent_training_100k.csv`](../artifacts/work_package_5c/opponent_training_100k.csv) / [`250k`](../artifacts/work_package_5c/opponent_training_250k.csv) | 50 / 20 个窗口 |
| 动作分布随时间变化 | [`action_distribution_100k.csv`](../artifacts/work_package_5c/action_distribution_100k.csv) / [`250k`](../artifacts/work_package_5c/action_distribution_250k.csv) | 50 / 20 个窗口 |
| 对局长度与终局质量 | [`game_quality_100k.csv`](../artifacts/work_package_5c/game_quality_100k.csv) / [`250k`](../artifacts/work_package_5c/game_quality_250k.csv) | 50 / 20 个窗口 |
| 训练当时已有的 historical diagnostics | [`historical_observed_matchups_long.csv`](../artifacts/work_package_5c/historical_observed_matchups_long.csv) / [`wide`](../artifacts/work_package_5c/historical_observed_matchups_wide.csv) | 160 个 matchup |
| 最终 active historical pool 全矩阵 | [`historical_final_pool_matrix.csv`](../artifacts/work_package_5c/historical_final_pool_matrix.csv) / [`pair details`](../artifacts/work_package_5c/historical_final_pool_pair_results.csv) | 12 × 12 |
| 导出口径和数量校验 | [`export_manifest.json`](../artifacts/work_package_5c/export_manifest.json) | 1 份 manifest |

两种分箱都使用右闭区间：例如 `0 < step <= 250000`。对手得分率按该类对手
实际完成的 games 加权；动作分布使用真实 action counts 重算；对局均值按每个
update 完成的 games 加权。因此没有把一个只完成 51 局的尾 update 和完整 update
给予相同权重。导出后已反向校验：对手局数、总局数和六类主动作均与
`summary.json` 完全一致。

### 15.1 逐 PPO update 优化曲线

`ppo_updates.csv` 每行对应一次 PPO update，保留了原 update record 中的全部标量字段，
包括 policy/value/total loss、entropy、explained variance、KL、clip fraction、
gradient norm、LR、entropy coefficient、采样吞吐和资源指标。

| 指标 | 首 update | 末 update | 698 updates 均值 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|
| policy loss | -0.00138 | -0.00175 | -0.00219 | -0.00411 | 0.00009 |
| value loss | 0.1592 | 0.1263 | 0.1658 | 0.0788 | 0.2268 |
| entropy | 1.4978 | 0.9532 | 1.1531 | 0.9329 | 1.4978 |
| explained variance | 0.7318 | 0.6567 | 0.6821 | 0.4904 | 0.7998 |
| approx KL | 0.000680 | 0.000438 | 0.001308 | 0.000035 | 0.003697 |
| clip fraction | 0.00391 | 0.00501 | 0.01568 | 0 | 0.04098 |
| gradient norm | 0.5997 | 1.2753 | 1.0052 | 0.5239 | 1.8519 |
| learning rate | 1.00e-4 | 3.00e-5 | 6.49e-5 | 3.00e-5 | 1.00e-4 |

最后一行是从 `4,999,152` 补到 `5,000,000` 的部分 rollout，只有 848 个训练 decisions，
不应与 8192-decision 的完整 update 做单点等权比较。曲线整体没有 KL 或
value 发散；entropy 随冻结 schedule 明显下降，gradient clipping 仍是持续活跃的约束。

### 15.2 全部 Formal candidates

`formal_candidates.csv` 不再只保留每周期第一名，而是收录 `21` 个 Formal cycles 中的
全部 `61` 个 candidates。每行包含：

- Formal cycle step、candidate step、rank、checkpoint 和原始 eval path；
- Score / Development / start anchor 的 W/D/L、先后手得分率和平均回合；
- heuristic score、是否 cycle winner、是否 promotion eligible、是否最终晋升；
- cycle draw 以及全部异常计数。

这份表可直接用来分析 Quick 预选是否错过 Formal 更强的 candidate，以及同一周期内
rank 和直接 promotion 结果之间的 non-transitivity。

### 15.3 四类训练对手的分段成绩

下表是 250k 分箱。这些是训练中 stochastic learner 对当时冻结对手的结果，
不是 greedy Formal DEV；其中 latest 和 dynamic historical 会随训练更新，因此只能解读为
当时训练分布上的成绩。

| 区间结束 | latest | historical | Score | Development | 局数 |
|---:|---:|---:|---:|---:|---:|
| 0.25M | 48.9% | 60.7% | 41.8% | 45.0% | 5,412 |
| 0.50M | 51.2% | 55.2% | 43.6% | 48.1% | 5,464 |
| 0.75M | 49.7% | 60.7% | 44.8% | 50.9% | 5,452 |
| 1.00M | 49.7% | 55.7% | 45.9% | 55.0% | 5,517 |
| 1.25M | 50.2% | 61.1% | 49.6% | 55.7% | 5,527 |
| 1.50M | 49.9% | 62.0% | 55.0% | 55.5% | 5,492 |
| 1.75M | 49.6% | 62.3% | 51.3% | 59.5% | 5,465 |
| 2.00M | 50.1% | 61.6% | 48.6% | 54.9% | 5,406 |
| 2.25M | 48.6% | 65.0% | 52.7% | 57.6% | 5,398 |
| 2.50M | 48.0% | 66.8% | 59.3% | 63.2% | 5,442 |
| 2.75M | 49.8% | 69.3% | 66.0% | 66.1% | 5,432 |
| 3.00M | 49.2% | 70.1% | 66.9% | 68.3% | 5,504 |
| 3.25M | 50.1% | 71.7% | 68.0% | 72.0% | 5,568 |
| 3.50M | 50.5% | 71.8% | 68.1% | 70.4% | 5,602 |
| 3.75M | 50.0% | 73.0% | 67.2% | 71.8% | 5,594 |
| 4.00M | 48.3% | 73.7% | 68.8% | 70.7% | 5,583 |
| 4.25M | 51.3% | 75.3% | 72.3% | 71.7% | 5,615 |
| 4.50M | 49.4% | 71.8% | 72.8% | 71.9% | 5,604 |
| 4.75M | 50.7% | 73.2% | 64.7% | 70.5% | 5,629 |
| 5.00M | 49.4% | 75.1% | 70.2% | 76.2% | 5,638 |

主要信号是：

- latest 长期维持在 48%–51%，符合同期 self-play 对手的预期；
- historical 从约 56%–61% 上升到 72%–75%，表明对旧版本的总体优势增大；
- Score / Development 都呈明显上升趋势，但单个 250k 窗口每类只有约 500–1,400 局，
  例如 4.75M Score 的回落不应单独当作模型棋力崩溃的证据。

### 15.4 动作分布随训练时间的变化

下表抽取若干 250k 窗口；完整 20 个窗口和 50 个 100k 窗口均已在 CSV 中。

| 窗口结束 | TakeDistinct | TakePair | ReserveMarket | ReserveDeck | BuyMarket | BuyReserved |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25M | 37.25% | 0.593% | 8.26% | 3.26% | 48.63% | 2.00% |
| 1.00M | 39.68% | 0.153% | 7.29% | 4.11% | 46.89% | 1.87% |
| 2.00M | 32.30% | 0.031% | 3.02% | 8.47% | 54.24% | 1.94% |
| 3.00M | 35.70% | 0.001% | 4.96% | 0.64% | 56.17% | 2.52% |
| 4.00M | 36.30% | 0.006% | 5.24% | 0.48% | 55.39% | 2.59% |
| 5.00M | 37.40% | 0.001% | 5.25% | 0.33% | 54.14% | 2.87% |

可见的策略迁移包括：BuyMarket 占比从约 49% 上升到 54%–56%；
BuyReserved 缓慢上升；TakePair 几乎消失；ReserveDeck 在约 2M 曾明显增多，
随后下降到 1% 以下。这些是行为分布变化，它们本身不能单独证明某个动作是否最优。

### 15.5 对局长度和终局质量随时间的变化

| 窗口结束 | 平均完整玩家回合 | learner 平均终局分 | 平均已购卡 | 平均贵族 | 训练 score rate |
|---:|---:|---:|---:|---:|---:|
| 0.25M | 60.50 | 13.56 | 15.34 | 0.439 | 48.36% |
| 1.00M | 59.94 | 13.76 | 14.64 | 0.349 | 51.28% |
| 2.00M | 59.17 | 13.91 | 16.67 | 0.699 | 53.83% |
| 3.00M | 57.01 | 14.36 | 16.77 | 0.724 | 59.63% |
| 4.00M | 56.71 | 14.38 | 16.48 | 0.663 | 60.16% |
| 5.00M | 56.43 | 14.55 | 16.12 | 0.617 | 61.61% |

在训练混合对手分布同时变化的前提下，可观察到对局逐渐缩短、learner 终局分数提高，
且训练 score rate 提升。已购卡数和贵族数非单调：约 2M–3M 较高，后期略下降，
同时终局分数仍上升，这更像是得分效率和游戏节奏的改变，而不是单纯“买更多卡”。

### 15.6 最终 historical pool matchup matrix

最终 active pool 有 12 个 checkpoint。每一对都使用统一 Historical seed domain、
50 paired seeds = 100 局、交换先后手、greedy argmax 和确定性循环判平。
总计 66 个无序 matchup，对应 6,600 局证据。其中 50 个 matchup 直接复用训练期使用
完全相同协议保存的 Historical diagnostics，其余 16 个缺口补跑；补跑中管理 shell
不可见 CUDA，因此使用 CPU greedy，共耗时 397.3 s。所有 matchup 合计为
0 engineering truncation、0 IllegalAction、0 invariant failure、0 exception，出现 2 局正式
policy-cycle draw。完整 seed、checkpoint SHA-256 和来源见
[`historical_final_pool_matrix_manifest.json`](../artifacts/work_package_5c/historical_final_pool_matrix_manifest.json)。

下表的行为待评模型，列为对手，数字是行模型的 score rate（%）。
标签使用 lifetime learning step；`mean*` 不包含 50% 的自对弈对角线。
因为每个 matchup 已经成对交换先后手，反向格由 `100% - score` 精确填充，
没有重复运行同一无序模型对。

| row \\ col | 157351 | 512112 | 712112 | 1212112 | 1762112 | 2162112 | 2762112 | 2912112 | 3262112 | 4012112 | 4662112 | 5512112 | mean* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 157351 | 50.0 | 40.0 | 43.0 | 38.0 | 31.5 | 23.5 | 31.5 | 18.0 | 28.5 | 20.0 | 17.0 | 19.0 | 28.2 |
| 512112 | 60.0 | 50.0 | 45.5 | 50.5 | 42.0 | 34.0 | 38.0 | 25.0 | 23.5 | 24.0 | 18.0 | 21.5 | 34.7 |
| 712112 | 57.0 | 54.5 | 50.0 | 45.5 | 47.0 | 38.5 | 40.0 | 40.0 | 32.0 | 33.0 | 29.5 | 24.0 | 40.1 |
| 1212112 | 62.0 | 49.5 | 54.5 | 50.0 | 56.5 | 37.5 | 41.0 | 46.0 | 37.5 | 38.0 | 29.0 | 33.0 | 44.0 |
| 1762112 | 68.5 | 58.0 | 53.0 | 43.5 | 50.0 | 50.5 | 38.5 | 39.5 | 40.0 | 27.5 | 22.5 | 27.5 | 42.6 |
| 2162112 | 76.5 | 66.0 | 61.5 | 62.5 | 49.5 | 50.0 | 53.5 | 35.5 | 39.0 | 30.0 | 28.0 | 25.0 | 47.9 |
| 2762112 | 68.5 | 62.0 | 60.0 | 59.0 | 61.5 | 46.5 | 50.0 | 39.0 | 28.0 | 34.0 | 32.5 | 40.5 | 48.3 |
| 2912112 | 82.0 | 75.0 | 60.0 | 54.0 | 60.5 | 64.5 | 61.0 | 50.0 | 37.0 | 38.0 | 42.5 | 36.0 | 55.5 |
| 3262112 | 71.5 | 76.5 | 68.0 | 62.5 | 60.0 | 61.0 | 72.0 | 63.0 | 50.0 | 59.0 | 52.5 | 49.5 | 63.2 |
| 4012112 | 80.0 | 76.0 | 67.0 | 62.0 | 72.5 | 70.0 | 66.0 | 62.0 | 41.0 | 50.0 | 51.0 | 57.5 | 64.1 |
| 4662112 | 83.0 | 82.0 | 70.5 | 71.0 | 77.5 | 72.0 | 67.5 | 57.5 | 47.5 | 49.0 | 50.0 | 50.0 | 66.1 |
| 5512112 | 81.0 | 78.5 | 76.0 | 67.0 | 72.5 | 75.0 | 59.5 | 64.0 | 50.5 | 42.5 | 50.0 | 50.0 | 65.1 |

矩阵呈现的不是严格单调序列：

- 总体上越后期的模型对早期模型越强；4.15M best-dev 对其他 11 个成员平均 66.1%，
  5M latest 为 65.1%；
- best-dev 与 latest 在这 50 pairs 上恰好 50.0%，但该样本量不足以证明两者等强；
- 明显存在 non-transitivity：`4012112` 对 `5512112` 为 57.5%，但对 `3262112`
  只有 41.0%；`3262112` 对 `4012112` 则是 59.0%；
- 因此矩阵支持使用固定 heuristics + anchor + direct promotion 的多信号选择方式，
  不支持将所有 checkpoint 强行压成一个严格 Elo 序列。
