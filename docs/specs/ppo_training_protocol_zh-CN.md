# PPO 主训练、评测、晋升与历史池协议

本文记录 `train_5c` run-format v4 的实验协议。`learning decision`指学习方实际作出的一次且被收入 PPO rollout 的决策，不是整局、玩家回合或 PPO update。

> 适用范围：本文的具体执行配置记录已完成的 `5c_main_seed1`；该 run 实际使用 `single_process` collector。2026-09-21 完成的 P0–P3 优化没有改变本文的训练、paired-seed、promotion、历史池或 RNG 协议；它们只改变新运行的执行后端和热路径。已完成 run 的结果见 [PPO 5M 报告](../reports/ppo_5m_zh-CN.md)；新建配置默认使用 worker-local CPU 神经对手，这不表示新后端已完成 GPU 长训收益验收。

## 1. 起点和通用训练配置

- learner 和 initial global best-dev 都是 `step_000512112.pt`；
- run-local 预算为 5,000,000 learning decisions；
- 32 个 logical environments；已完成的 `5c_main_seed1` 正式 collector 为单进程 batched collector；
- rollout 上限 8,192 decisions，minibatch 512，PPO epochs 4；
- reward 为纯 terminal `+1/0/-1`，`gamma=1`，`lambda=0.95`；
- LR 从 `1e-4` 线性降至 `3e-5`，entropy coefficient 从 `0.01` 线性降至 `0.002`；
- 常规 checkpoint interval 保持 100k；Quick 的 50k 边界和 Formal 的 250k 边界也保存当前候选 checkpoint；
- 到达 checkpoint/评测边界时会先完成已开始的 episode；边界 drain tail 中 learner 实际执行的决策只用于完成这些对局，不进入 PPO update，不计入 5M 预算，不推进 LR/entropy schedule，也不推进 50k/250k 计数器；它们单独记录为 `boundary_drain_decisions_not_trained`。

神经网络动作模式固定为：

- Quick、Formal、Promotion、Historical 和 FINAL 评测中，candidate 与 neural opponent 都使用 greedy argmax，不做随机采样；
- 训练 rollout 中 learner 与 latest/historical neural opponent 都按各自独立、可恢复的 RNG 做 categorical sampling；
- ScorePolicy 和 DevelopmentPolicy 本身是确定性策略。

新执行 `train_5c prepare` 时，代码默认生成 `multiprocessing_local_opponents`、4 workers × 8 envs 和 8 evaluation workers。已有冻结 config 不会被该默认值改写，resume 仍要求配置严格匹配。

## 2. Quick DEV

Quick 约每 50k run-local decisions 执行，但与 Formal 同一 step 时只执行 Formal。

固定对手：

1. ScorePolicy；
2. DevelopmentPolicy。

每个对手使用 50 paired seeds，每个 seed 交换先后手，因此每个对手 100 局，整个 Quick 共 200 局。

Quick 记录：

`quick_heuristic_score = 0.5 * ScorePolicy score_rate + 0.5 * DevelopmentPolicy score_rate`

Quick 只用于 TensorBoard 曲线、健康检查和下一个 Formal 周期的候选预筛。Quick 不能直接更新 global best-dev。RandomPolicy 和 neural checkpoint 不进入 Quick。

## 2.1 确定性评测的策略循环判平

Quick、Formal、Promotion、Historical 和 FINAL 的双方策略在单局内都是固定的确定性策略。在这些评测中，每个决策边界计算完整 `canonical_state_signature`：

- 包含当前玩家、phase、双方 token、已购买卡/保留卡/贵族及由其重算的 score/bonus、公开市场、公共贵族、完整有序牌堆、pending context、forced-skip/final-round 状态、会影响后续转移的 RNG 实际状态，以及其他权威 `GameState` 字段；
- 排除 `completed_turns`、`rng.draws`、replay index 和 logging counters 等纯记录字段；
- 按 `current_player` 分别记录 signature；某名玩家至少第二次遇到其某个完整 signature 时，该玩家标记为已重复；只有双方都已重复时，评测才记为 `policy_cycle_draw` 并按平局计分。因为 signature 本身包含 `current_player`，双方的 signature 不需要字面相同；
- `policy_cycle_draw` 不是 `GameEngine` 的 terminal result，不改变正式规则、forced skip 或训练 reward；普通 engineering truncation 仍然是评测失败；
- 每次判平都单独记录状态、replay prefix、首次/重复决策位置和 signature digest。

该判定不会在随机策略评测、训练 rollout、压力测试或正式引擎中隐式启用。

## 3. Formal 候选生成与固定主评测

Formal 约每 250k run-local decisions 执行。每个 Formal 只考虑自上一次 Formal 后出现的 checkpoint：

1. 按 `quick_heuristic_score` 降序取 Quick Top 2；
2. 加入本周期最后一个 checkpoint，即 Formal 边界 checkpoint；
3. 按解析后的 checkpoint 路径去重；
4. 候选重复时不顺延补足，所以实际候选数是 1～3。

Quick 排名的确定性 tie-break 为：

1. `quick_heuristic_score` 更高；
2. run-local step 更晚；
3. checkpoint 绝对路径字典序更小。

每个候选都使用完全相同的固定 Formal seeds 评测：

1. ScorePolicy；
2. DevelopmentPolicy；
3. `step_000512112.pt`，记为 `vs_start_512112`。

每个对手 200 paired seeds，即 400 局。主排名指标为：

`formal_heuristic_score = 0.5 * ScorePolicy score_rate + 0.5 * DevelopmentPolicy score_rate`

`vs_start_512112` 只记录，不进入 heuristic score。不使用 RandomPolicy，也不把任何 neural 对局结果混入 heuristic score。

完成固定主评测后，所有 Formal candidates 按 `formal_heuristic_score` 从高到低排列。平分时依次按 run-local step 更晚、checkpoint 路径字典序更大者在前。排名第一的候选称为 Formal winner，但所有达到资格门槛的候选都可按排名依次发起 promotion challenge。

## 4. Step-0 基线

5C 开始前，使用正式固定 Formal seeds 对 `step_000512112.pt` 完整评测 ScorePolicy、DevelopmentPolicy 和它自身。由此保存 initial `formal_heuristic_score`。

initial global best-dev 必须指向原始 `step_000512112.pt`，而不是 step-0 运行目录内的副本。以后每次晋升也保存当时同一套 Formal seeds 得到的 score，所以 `H_old` 永远可直接比较。

## 5. Promotion match

设某个候选的分数为 `H_candidate`，当前 global best-dev 的固定 Formal 分数为 `H_old`。所有满足：

`H_candidate >= H_old - 0.01`

的候选都具有晋升资格。候选按 Formal 排名从高到低挑战同一个 current best：第一个成功的候选立即晋升并结束本周期；失败则由下一个 eligible candidate 继续挑战；全部失败则 incumbent 保持不变。未达门槛者不发起对局。

`promotion_threshold = 0.5 + max(0, H_old - H_candidate)`

对每个 seed 的两场换边比赛先求平均 score，再对 pair-level score 做 100,000 次 bootstrap。定义：

- `LCB90` = bootstrap score distribution 的 10th percentile；
- `UCB90` = bootstrap score distribution 的 90th percentile。

第一阶段使用 200 paired seeds：

- `LCB90 > promotion_threshold`：立即晋升；
- `UCB90 <= promotion_threshold`：明确不晋升，保留 incumbent；
- `LCB90 <= promotion_threshold < UCB90`：结果 inconclusive，进入第二阶段。

第二阶段只执行预先保留的后 300 paired seeds，然后用前 200 与后 300 合并成的全部 500 pairs 重新 bootstrap：

- 新 `LCB90 > promotion_threshold`：晋升；
- 否则保留 old best；
- 500 pairs 后不再增加样本。

Promotion 结果不使用 Formal 或 Historical seeds，也不改写 `H_candidate/H_old`。

bootstrap 也是完全确定的。100,000 次重采样使用 `numpy.default_rng(PCG64)`，每个周期、候选和阶段的 seed 由下式唯一派生，并写入 promotion 结果：

- initial 200：`5_200_000_000_000_000 + cycle_step * 5_000_001 + candidate_step`；
- expanded 500：`5_300_000_000_000_000 + cycle_step * 5_000_001 + candidate_step`。

其中 `cycle_step` 和 `candidate_step` 都是 run-local learning decisions。比赛 seed 与 bootstrap seed 分开：固定前者不能替代固定后者。

## 6. Secondary Dynamic Historical Evaluation

每个 Formal 周期排名完成后，使用独立 Historical seed domain，对 Formal winner 与评测开始时的整个 active historical pool 执行诊断评测。本周期的动态池加入/淘汰只在该评测之后发生。

每个 historical opponent 使用 50 paired seeds。记录：

- `historical_pool_mean_score`；
- `historical_pool_median_score`；
- `historical_pool_min_score`；
- 每个 active historical checkpoint 的单独 score rate。

该结果不参与 Quick 排名、Formal 排名、heuristic score、promotion threshold 或 best-dev 决策。它只用于检查遗忘、cycling 和 non-transitivity。即使同一 checkpoint 同时是 Formal anchor 和 historical member，也会按两套独立协议重新对局，不复用结果。

## 7. 训练对手

每个 episode 只抽样一次对手；整局的对手类型、checkpoint、参数和 RNG 固定。Neural opponent 是冻结模型，不共享 learner 参数，不参与梯度，其 transition 不进入 learner trajectory。

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

RandomPolicy 不进入训练池。

Historical bucket 内部：

- dynamic pool 非空时，1/3 从 fixed anchors 均匀抽样，2/3 从 dynamic history 均匀抽样；
- dynamic pool 为空时，historical 全部从 fixed anchors 均匀抽样；
- 因此 Phase B 的总体比例是 50% latest、10% fixed history、20% dynamic history、10% Score、10% Development。

bucket 抽样和 bucket 内抽样都使用可保存的 `DeterministicRng`。

## 8. Active historical pool 和确定性保留规则

active capacity 为 12：

- 2 个 fixed anchors：`157351`、`512112`；
- 10 个 dynamic slots。

旧 checkpoint 仍保留在 archive/run 目录，不删除，但不自动占用 active pool。动态加入规则是：

1. 任何新 global best 一定加入；
2. 另外，若本周期 Formal winner 与所有现存 dynamic checkpoint 的 run-local step 距离都 `>= 500,000`，则也加入；空 dynamic pool 对该条件视为满足；
3. 已在 fixed/dynamic active pool 中的同一 checkpoint 不重复加入；
4. 不是每 250k 无条件加入。

当 dynamic slots 已满又有新成员符合加入条件时：

1. 新加入的成员必须保留；
2. 对每个旧 dynamic entry，计算它与其他旧 entry 及新 entry 的最小 training-step 距离；
3. 淘汰最小距离最小的 entry，以消除时间上最重复的版本；
4. 距离平手时淘汰 training step 更早的 entry；仍平手时按 `opponent_id` 字典序更小者。

该规则不使用人工判断、Elo、PFSP 或临场调参。

## 9. Seed domains

TRAIN、Quick、Formal、Promotion、Historical、FINAL 和 preflight 使用互不相交的 game/policy seed domains。

- 同一 Quick 中的 checkpoint 使用固定 Quick seeds；
- 同一 Formal 周期的所有候选使用完全相同的 Formal seeds；
- Promotion 前 200 与后 300 是同一个预先固定的 500-pair 序列的前缀与后缀；
- Historical 所有 pool member 使用相同 Historical seeds；
- FINAL seeds 在训练期间不访问。

## 10. Resume 语义

checkpoint 只在 episode boundary 保存，包含 model、Adam、sampler RNG、mixture RNG、historical pool RNG/manifest、dynamic member ids、Quick 候选列表、评测完成 step 和 collector next episode index。

既有 `5c_main_seed1` 在 250k 中断后将 Quick 间隔从 100k 修订为 50k。对应恢复兼容只允许迁移该字段，其他冻结配置仍必须逐项相同；不回填修订前已经越过的 50k/150k Quick 边界。新 run 从冻结配置开始，不把历史迁移过程当成常规调参入口。

如果在 step-0 Formal、Quick 或 Formal/Promotion/Historical 评测中断，边界 checkpoint 中没有对应的“已完成”标记。`resume` 会重新执行该边界的完整评测，成功后再原子性覆写同一边界 checkpoint。它不会跳过未完成评测，也不会重复将同一 checkpoint 加入 checkpoint manifest。

resume 还会校验 fixed bucket 数量和 dynamic ids 是否与 pool manifest 一致；不一致时立即失败，不静默修复。

## 11. 输出

- `eval/quick/`：Quick panel；
- `eval/formal/local_*/candidate_*.json`：各 Formal candidate 的固定主评测；
- `eval/formal/local_*/cycle_summary.json`：当期候选排名、eligibility、按序 promotion attempts 和结果；
- `eval/promotion/`：每个 eligible candidate 的 200-pair 首阶段及可选 300-pair extension，文件名同时包含 cycle step 和 candidate step；
- `eval/historical/`：动态历史诊断；
- `checkpoints/best_dev.json`：当前 global best 及对应 Formal score/protocol；
- `checkpoints/best_dev_history.jsonl`：每个 Formal 周期的晋升决策；
- `historical_pool/events.jsonl`：动态加入与淘汰记录；
- TensorBoard：Quick/Formal heuristic、`vs_start_512112`、promotion 进度、historical mean/median/min/逐对手分数、fixed/dynamic 训练份额。

Standalone FINAL 使用独立 FINAL domain，对手为 ScorePolicy、DevelopmentPolicy 和 `step_000512112.pt`，每个对手最少 500 paired seeds，不参与训练期间的任何选择或 pool 更新。

## 运行入口

`train_5c prepare` 在新目录生成冻结配置，依赖现有 5B 起点与固定 anchors。先核对这些 checkpoint 和目标设备；已有 `runs/5c_main_seed1` 已完成，不重新 prepare 或 train 到该目录。

```bash
# 查看参数，不启动训练
conda run -n splendor env PYTHONPATH=src python -m splendor.train_5c prepare --help
# 新实验示例：prepare 后先读生成的 config，确认预算、种子和设备
conda run -n splendor env PYTHONPATH=src python -m splendor.train_5c prepare --run-root runs/5c_new_seed1
conda run -n splendor env PYTHONPATH=src python -m splendor.train_5c preflight --config runs/5c_new_seed1/config.json --device cuda
# 明确启动长训练时运行；中断后用 resume 替代 train
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.train_5c train --config runs/5c_new_seed1/config.json
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.train_5c resume --config runs/5c_new_seed1/config.json
```
