# AlphaZero 式 policy/value 训练循环

> 状态：已实现并完成微型端到端 smoke（2026-09-28）；同日完成第二轮修复与复验。
>
> 本文说明循环的语义、数据、损失、checkpoint、恢复、评测口径与命令。生产力配置已冻结，但**默认 5 轮 × 500 局长训练尚未执行**。

## 1. 范围与定位

循环拓扑：

```text
冻结 theta_r → 双方 MCTS 采集 D_r → 更新跨代 replay buffer →
从 theta_r 继续训练到 theta_(r+1) → 简短固定评测 → 下一轮用 theta_(r+1) 采集
```

这是适配**不完全信息** Splendor 的 AlphaZero 式循环：每方用独立 determinization 的 PUCT 搜索（生产配置 S=1024、N=1），保持现有完全相同的搜索语义与特征结构。它不宣称是原论文完全信息搜索的原样实现，也不声称已经取得棋力提升。

本轮明确不做：PPO/GAE/KL penalty/entropy bonus/AMP、模型结构修改、搜索参数实验、模型扩容、引擎重写、历史 300 局数据自动导入。

## 2. 代码结构

| 文件 | 职责 |
|---|---|
| `src/splendor/alphazero_loop.py` | 单入口循环：配置、轮次编排、状态与恢复、两组固定评测、TensorBoard |
| `src/splendor/alphazero_train.py` | 联合损失、AdamW learner、轮训练循环、验证损失、指标 JSONL |
| `src/splendor/alphazero_replay.py` | 跨 run 的对局索引、冻结 split、样本抽样、整局 LRU 缓存 |
| `src/splendor/alphazero_checkpoint.py` | AZ checkpoint 格式、原子保存/加载、冻结 loader 归一化 |
| `src/splendor/opponent_pool.py` | `load_frozen_actor_critic_checkpoint()` 小范围扩展，同时接受 PPO 与 AZ checkpoint |

采集仍由 `selfplay_fast` 负责，评测仍由 `evaluation` 负责；训练逻辑没有进入 MCTS、规则引擎或旧 PPO trainer。

## 3. 冻结数据与 replay 索引

### 3.1 对局身份

每轮采集从 `game_id=0` 重新开始，因此对局身份是 `(round_index, game_id)`，局面身份再加 `decision_index`。训练指标与错误现场都记录完整引用：

```json
{"round_index": 1, "run_id": "az_loop_seed1-round-001", "game_id": 7, "decision_index": 42}
```

现有 `SelfPlayBatch.references` 只有 `game_id`/`decision_index`/`player_id`，训练层通过 `PositionPool.sample()` 返回的 `ReplayReference` 补足 run 身份，不修改已有归档格式。

### 3.2 索引规则

- 只纳入成功提交的 `natural_terminal` 归档；异常、截断、临时文件不进入索引，也不伪装成平局。
- buffer 按 `(round_index, game_id)` 稳定排序，保留最近 `replay_max_games=3000` 个完整对局；容量包含 train/validation，采样只用 train。
- 淘汰只移出活动索引，`games/game-*.npz` 归档保留在磁盘用于诊断。
- 一轮训练开始时构建一次索引并冻结；训练循环内不扫描目录、不重建 manifest。
- split 沿用采集器的整局划分（`validation_fraction=0.1`），同一对局不会一半训练、一半验证。

### 3.3 校验与缓存

- 每个对局在**首次进入索引**时做一次完整 `validate_game_arrays`。
- 校验结果持久化到输出根目录的 `replay_index.json`（键为 `(round_index, game_id)`，附带 path/size/mtime）。后续轮次对未变化的归档跳过重校验；文件被改写则重新校验。损坏的 sidecar 只会触发重新校验，不会中断。
- 随机访问经 256 MiB（`replay_cache_bytes`）整局 LRU 缓存，避免每取一个局面就解压整局 NPZ，也不引入数据库。

### 3.4 current / history 池

| 池 | 内容 | 用途 |
|---|---|---|
| current | 本轮 train split 的全部局面 | 每 batch 前 128 个（无历史池时 256 全部） |
| history | 本轮之前、仍在 3000 局窗口内的 train 局面 | 每 batch 后 128 个 |

两池都在轮开始时冻结。有历史池时 batch 组成固定为 128 current + 128 history；无历史池时 256 全部来自 current。各池内按局面均匀**有放回**抽样。抽样 RNG 是持久化的 `splitmix64-v1` 状态：轮内恢复从 checkpoint 快照精确续接；跨轮时每轮用该轮派生种子重新开始，因此一轮的 batch 序列只取决于轮种子，与上一轮实际跑了多少步无关。

## 4. 损失与训练语义

直接调用可训练 `ActorCritic.forward`（不经 `FrozenSearchInference`/`inference_mode`/`no_grad`）。learner 用独立实例：`requires_grad_(True)`、`train()`，与采集教师（只读快照）不共享可写参数。

```text
L_pi = sum_i(policy_weight_i * CE(policy_target_i, masked_softmax(logits_i))) / sum_i(policy_weight_i)
L_v  = sum_i(value_weight_i * (value_i - value_target_i)^2) / sum_i(value_weight_i)
L    = policy_loss_weight * L_pi + value_loss_weight * L_v
```

- `policy_target` 使用归档的根访问分布，不替换成 behavior policy、one-hot 或根 prior。
- padding 不参与 softmax 与分母；乘 target 前把 padding log-prob 置零，避免 `0 * -inf = NaN`。
- `policy_weight=0` 的单候选样本不参与策略分母。全单候选 batch 的策略损失是可组合的零值，critic 照常训练。
- `value_target` 是真实行动者视角的终局 +1/0/-1，直接使用归档标签。
- critic 仍是无界标量，训练用原始输出 MSE，不在训练路径增加 clamp/tanh；搜索端裁剪不变。
- 正则由 AdamW `weight_decay=1e-4` 承担，不重复加 L2。
- 每轮训练步数 `ceil(4 * new_train_positions / 256)`，`new_train_positions` 只统计本轮 train split 的全部真实局面（含单候选 value 样本）；验证样本不计入。`max_steps_per_round` 仅用于 smoke 覆盖。
- loss/logits/values/梯度范数非有限时立即抛出 `NonFiniteTrainingError`，不静默跳过；现场写入 `failures/round-XXX-<phase>.json`（含轮次、训练步、采样引用）。

训练超参：AdamW、`lr=1e-4`、`weight_decay=1e-4`、两种 loss 权重 1.0、`max_grad_norm=1.0`、batch 256。首轮只载入旧模型权重并创建新 AdamW；后续轮次经 `AlphazeroLearner.continue_from()` 延续同一 AdamW state 与全局训练步 `total_train_step`（跨轮单调递增，TensorBoard `train/*` 不会在轮边界重叠），只把轮内计数归零。

## 5. checkpoint 格式与兼容加载

### 5.1 AZ checkpoint

`algorithm="alphazero-policy-value"`、`checkpoint_schema="splendor-alphazero-checkpoint-v1"`，包含：

- 模型配置与 `state_dict`、参数内容哈希 `model_sha256`；
- `rules_version`、`feature_schema_version`；
- `parent_checkpoint`（kind/path/sha256，形成可回退的父子链）；
- `round_index`、轮内/全局训练步、轮步数目标；
- AdamW 配置与 optimizer state；
- 冻结 replay 索引 `replay_identity`（含 digest）；
- `sampling_rng_state`、Torch CPU/CUDA RNG state；
- 训练配置快照与可选 metadata。

全部字段都是基本类型或 tensor，继续走 `torch.load(..., weights_only=True)` 安全路径。保存使用临时文件 + 原子替换；模型哈希只在保存/轮边界计算一次。

### 5.2 冻结加载入口

`opponent_pool._checkpoint_payload()` 现在识别两种格式：PPO payload 保持原逻辑；AZ payload 经 `normalize_actor_critic_payload()` 归一化为同样的只读子集。因此下列路径都能直接加载新产物：

- 下一轮 `selfplay_fast` 的 CPU/GPU worker；
- `evaluation` 的 `greedy_checkpoint` 与 `mcts_checkpoint`（`SerializablePolicySpec`）；
- `OpponentPool.add_historical` 与历史池 manifest 校验。

兼容方向是单向的：AZ checkpoint 不携带 PPO 的 `ppo_config`/rollout 字段，旧 PPO 恢复器不会误接受 AZ 产物；AZ learner 也只从自己的 checkpoint 恢复 optimizer/RNG。

### 5.3 轮末 checkpoint 的字节稳定性

每轮 `final.pt` 是**轮末 step checkpoint 的字节复制**（不再是独立再次保存）。这保证"训练完成但状态文件未推进"的恢复会重新提交一个与评测 header 中记录哈希一致的文件，从而复用已完成的评测，而不是因为文件哈希变化被拒绝。

## 6. 全循环、状态与恢复

### 6.1 每轮阶段

```text
collect → train → evaluate(mcts_vs_parent) → evaluate(greedy_vs_initial) → complete
```

`loop_state.json`（原子写入）记录 `completed_rounds`、每轮 phase、theta_in/final checkpoint 身份、latest 指针。`config.json` 快照与配置指纹（含初始 checkpoint 的 sha256）在启动时校验，配置或教师身份不匹配时明确拒绝混合续跑。

### 6.2 恢复判定（以磁盘事实为准）

| 现场 | 恢复行为 |
|---|---|
| 采集已完整（`summary.json` 完整） | 跳过采集，不重跑任何对局 |
| 采集部分完成 | `selfplay_fast.generate(resume=True)` 只补缺失 `game_id` |
| `final.pt` 与轮次/父模型身份/冻结索引 digest/步数目标全部匹配 | 训练视为已完成，直接进入评测 |
| 只有部分 step checkpoint | 从最大步数的 checkpoint 恢复参数、optimizer、RNG、步数与索引身份，只跑剩余步 |
| final 已提交但状态未推进 | 不重复训练、不提前推进轮次（final 为事实来源） |
| 评测部分完成 | `run_fixed_evaluation(resume=True)` 只补缺失 pair，完成训练后评测中断不触发重新训练 |
| `final.pt` 已提交但评测未完成，且 `loop_state.json` 丢失 | 恢复扫描要求 final **与**两个完整评测 summary 同时存在才视为完成；该轮按“只补评测”恢复 |
| 评测面板 `passed=false`（截断/异常/非法动作/invariant 失败） | 抛出 `EvaluationEngineeringFailure`、写 `failures/round-XXX-evaluate.json`，本轮不标记完成、下一轮不启动 |
| 循环全部完成 | `--resume` 返回 `loop_no_work`，不追加轮数 |

判定不匹配（例如 replay digest 或 learner 配置变了）时明确报错，不做静默重启。

### 6.3 统计口径

- `session_phases` 记录该次调用实际执行的阶段，例如 `["collect", "train", "evaluate"]` 或 `["collect(skipped)", "train(skipped)", "evaluate"]`。
- `round_summary.json` 由最后一次会话改写，因此 `wall_seconds_this_session` 只覆盖该会话，不是整轮累计墙钟。
- 采集统计沿用 `selfplay_fast` 既有口径：`wall_seconds`/`inference`/延迟是 `collection_session` 覆盖的会话值，吞吐分子只含该会话对局的增量；完整逻辑样本数可从归档累计。多 worker 下逐局搜索时间之和大于墙钟，不能当作 run 耗时。
- 训练耗时按会话记录；`steps_completed` 是轮内累计优化步数，`steps_this_session` 是本次会话新增步数，恢复不会重复计入已提交步骤。
- `training_metrics.jsonl` 在每次进入训练时按最后一个 checkpoint 的轮内步数重写：丢弃未提交步骤的行与上一次验证行，再继续追加，因此每个步号只有一条记录、每轮只有一条 `validation` 记录。

### 6.4 latest-iteration 更新

训练完成的合法 theta_(r+1) 直接用于下一轮采集；不因小样本胜率低于某个阈值而拒绝更新。评测用于观察趋势，异常则停止并保留各代 checkpoint 便于回退。

## 7. 轻量评测

每轮结束后在 CPU、8 workers 上执行两组固定配对（每 pair 交换先后手）：

| 面板 | 对局 | 配置 |
|---|---|---|
| `mcts_vs_parent` | 新网络 MCTS vs 父网络 MCTS，10 paired seeds / 20 局 | 双方 S=1024、N=1、`c_puct=1.5`、`fold_forced_phases=true`、无根噪声、温度 0 |
| `greedy_vs_initial` | 新网络 greedy vs 初始 5.5M 网络 greedy，20 paired seeds / 40 局 | 观察原网络是否吸收搜索能力 |

- 两组 seed 面板相互独立，并在跨轮复用同一固定 DEV 面板以读趋势；不占用 FINAL seeds。
- 沿用 evaluator 的规则与 `adjudicate_deterministic_policy_cycles=true` 循环裁决选项；该选项只用于评测，不带入带探索采集。
- 结果保留双方先后手、胜平负、得分率、异常/截断与耗时。`passed` 只是工程完成状态（无异常/截断、对局完整），**不等于棋力提升**。
- 每个面板结束后立即检查 `passed`：失败即终止循环（写 failure 现场、不进入下一轮采集），因为用工程上已经损坏的评测继续训练属于静默污染。修复原因后 `--resume` 复用未受影响的对局，未完成 pair 按原种子重跑；但已写成完整记录、`status=error/engineering_truncation` 的 pair 不会被自动重放，需要人工清理该 pair 的日志，否则 resume 会再次如实报同样的失败。
- 评测使用既有 paired-seed 多进程路径；生产采集的 8 workers、active/batch=4/4 不受评测影响。

## 8. 生产配置

[`configs/alphazero_loop.json`](../configs/alphazero_loop.json)：

| 项目 | 值 |
|---|---|
| `num_rounds` / `new_games_per_round` | 5 / 500 个独立完整对局 |
| `replay_max_games` / `replay_cache_bytes` | 3000 局 / 256 MiB |
| 自博弈 | `mcts-selfplay-v2`、S=1024、N=1、`c_puct=1.5`、`fold_forced_phases=true`、根噪声 0.25/α=0.3 |
| 执行 | `batched_cpu`、`fast`、8 workers、active=4、batch=4、wait=2ms、torch 1 线程、FP32 |
| 温度 | 前 10 个双方合计已完成玩家回合为 1.0，之后 0.0（后期仍按配置加根噪声） |
| learner | `cuda` FP32、batch 256（128 current + 128 history）、AdamW 1e-4/1e-4、grad clip 1.0、每 100 步保存 |
| 评测 | 10 pairs MCTS vs parent + 20 pairs greedy vs initial，CPU 8 workers |
| 种子域 | collection / split / sampling 分别从固定整数基准按 round 派生；评测面板有独立 `seed_start` 与 `policy_seed_base` |

`configs/alphazero_loop_smoke.json` 是独立覆盖配置：2 轮 × 4 局、S=8/N=1、2 CPU workers、active/batch=2/2、learner CPU、batch=16、每轮最多 4 步、两组评测各 1 pair、输出到独立目录 `runtime/az_loop_smoke`，不写生产数据池。

## 9. 命令

正式启动（5 轮 × 500 局；本任务未执行）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -u -m splendor.alphazero_loop \
  --config configs/alphazero_loop.json \
  --output runs/az_loop_seed1

# 中断后原样恢复；已完成阶段不会重复执行
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -u -m splendor.alphazero_loop \
  --config configs/alphazero_loop.json \
  --output runs/az_loop_seed1 --resume
```

微型 smoke：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -u -m splendor.alphazero_loop \
  --config configs/alphazero_loop_smoke.json \
  --output runtime/az_loop_smoke

conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -u -m splendor.alphazero_loop \
  --config configs/alphazero_loop_smoke.json \
  --output runtime/az_loop_smoke --resume
```

TensorBoard：

```bash
conda run -n splendor tensorboard --logdir runs/az_loop_seed1/tensorboard --port 6006
```

生产产物布局：

```text
runs/az_loop_seed1/
├── config.json                    # 配置快照 + 指纹
├── loop_state.json                # 原子状态与 latest 指针
├── replay_index.json              # 派生校验缓存（可删除后重建）
├── tensorboard/
├── failures/                      # 异常现场（如有）
└── rounds/round-000/
    ├── selfplay/                  # mcts-selfplay-v2 归档（games/、manifest、summary）
    ├── checkpoints/               # step-XXXXXXXXX.pt（每 100 步）、final.pt、final.json
    ├── training_metrics.jsonl     # 每步 loss/熵/grad/抽样数 + 验证
    ├── training.json
    ├── evaluation/                # mcts_vs_parent(.json/.games)、greedy_vs_initial(.json/.games)
    └── round_summary.json
```

### 9.1 逐代对照评测脚本

[`scripts/run_az_generation_evals.py`](../scripts/run_az_generation_evals.py) 把「每一代 θ_g 对初始 5M 的两组固定评测」批量跑完：

- `greedy`：θ_g 裸网 vs 初始裸网，500 pairs（1000 局）；
- `mcts`：θ_g + PUCT S=1024/N=1 vs 初始 + 同配置，50 pairs（100 局）。

各代共用同一 seed 块（默认 9,900,000，和训练/DEV/smoke/sweep 的种子域不重叠），因此可以逐 seed 配对比较；默认跑 θ1–θ4，`--rounds 1 2 3 4 5` 可把最后一代一起纳入，`--panels greedy` 只跑便宜的面板，`--digest-only` 不重跑、只从已有输出重建汇总。

产物分工遵循 `artifacts/` 与 `runtime/` 的既有约定：

- 重量级逐局数据（每代 summary JSON 与可续跑的 `*.games/`）写入 `runtime/az_loop_seed1_evals/`，整体被 Git 忽略；
- 生成的评测配置（仓库相对路径，可移植）写入 `configs/az_loop_seed1_evals/`，同样被 Git 忽略；
- 只把紧凑汇总 `artifacts/az_loop_seed1/generation_scores.json` 纳入 Git，其中含每代 score / bootstrap CI / W-D-L / 先后手拆分，以及"同 seed 配对差 + 合并均值"的统计。

## 10. 指标记录

- 每步 JSONL：`policy_loss`、`value_loss`、`total_loss`、合法动作策略熵、`grad_norm`、`learning_rate`、`total_train_step`、current/history 抽样数。
- 每轮：验证 split 的局数、样本数与 policy CE / value MSE；无验证局时记为 `available=false`，不从训练池冒充。
- 终端周期进度（默认每 30 秒一条，`DEFAULT_PROGRESS_INTERVAL_SECONDS`，首个事件立即输出）：`collect_start`/`collect_progress`（已完成局数、本会话新增、局/分钟）、`train_start`/`train_progress`（轮内与全局步数、loss、步/秒、ETA）、`evaluate_progress`（面板、已完成 pairs、pairs/分钟）；阶段结束仍输出 `collect_complete`/`train_complete`/`evaluate_complete`。
- TensorBoard：`train/*`（x 轴为全局步）；`validation/policy_ce`、`validation/value_mse`、`validation/policy_entropy`、`validation/games`、`validation/positions`（轮末全局步）；`collect/round-XXX/games_completed`、`collect/round-XXX/games_per_minute`（轮内相对秒，按轮分标签）；`collect/completed_games`、`collect/samples`、`collect/policy_samples`、`collect/simulations`、`collect/wall_seconds`、`collect/games_per_second`、`collect/policy_samples_per_second`、`collect/simulations_per_second`（以该轮入口全局步为 x 轴）；`eval/mcts_vs_parent_score_rate`、`eval/greedy_vs_initial_score_rate`、`system/eval_*_seconds_this_session`。
- 采集吞吐沿用 selfplay 既有会话口径：`*_per_second` 的分子只含本次会话实际采集的对局，跨会话恢复不会伪造整轮速率。
- 跨轮 target 熵会变化，CE 单独下降不构成棋力证据；不要求单个 minibatch loss 单调下降。

## 11. 验证记录（2026-09-28）

### 11.1 新增测试

| 测试文件 | 覆盖 |
|---|---|
| `tests/test_alphazero_training.py` | 手算 CE/MSE 与熵、padding 不影响合法候选、单候选不进入策略分母、全零 policy weight 的零策略损失、步数公式、一次更新改变参数、checkpoint 恢复后抽样与更新逐位一致、跨轮 `continue_from` 延续 optimizer/全局步且抽样跟随轮种子、崩溃窗口后指标文件去重、AZ checkpoint 可被冻结 loader 与评测路径加载并做一次 MCTS 推理、生产配置解析检查 |
| `tests/test_alphazero_replay.py` | 同 game_id 跨 run 不冲突、split 隔离、淘汰只移索引且归档保留、current/history 抽样约定、校验 sidecar 避免重复校验且文件变动触发重校验、整局 LRU 缓存重复取样不重复解压、LRU 淘汰与超大对局不缓存、采集进度回调（含恢复会话的绝对局数） |
| `tests/test_alphazero_loop.py` | 两轮闭环与 theta 链、跨轮 optimizer/全局步连续、完成态 resume 无工作、状态落后时训练不重复执行、配置不匹配拒绝续跑、评测工程失败阻止下一轮、状态文件丢失不跳过未完成评测、周期进度事件与 TensorBoard 标量、进度节流 |

### 11.2 已运行的既有测试

`tests/test_selfplay.py`、`tests/test_search_accel.py`、`tests/test_mcts_policy.py`、`tests/test_opponent_pool_5b.py`、`tests/test_actor_critic_4a.py` 全部通过（冻结加载入口、自博弈、搜索与对手池未被破坏）。未运行全量 pytest。

### 11.3 smoke 结果

真实 5.5M checkpoint、2 轮 × 4 局、S=8/N=1、2 CPU workers、active/batch=2/2、learner CPU、最多 4 步/轮：

| 轮 | 采集 | 训练 | 验证 | 评测得分率 | `session_phases` |
|---|---|---|---|---|---|
| 0 | 4 局 / 352 样本 / 226 policy / 1808 sims，会话墙钟 2.88s | 4 步，cur/hist=16/0 | 1 局 / 89 局面，CE 2.819 / MSE 0.397 | MCTS vs parent 1.00、greedy vs initial 0.50（各 1 pair） | collect/train/evaluate |
| 1 | 4 局 / 352 样本 / 229 policy / 1832 sims，会话墙钟 3.10s | 4 步，cur/hist=8/8 | 1 局 / 83 局面，CE 3.188 / MSE 0.243 | MCTS vs parent 0.50、greedy vs initial 0.50 | collect/train/evaluate |

闭环事实：theta0 = 初始 5.5M；每轮 `final.pt` 的 `parent_checkpoint` 指向上一轮产物；第二轮 worker 成功加载第一轮 AZ checkpoint 完成采集得到 theta2；统计、归档、加载、恢复接口全部可用。第二轮训练 batch 由 8 current + 8 history 组成。

恢复验证（同一 smoke）：

1. 完成态 `--resume` → `loop_no_work`，无任何对局重跑。
2. 模拟"状态文件落后 + 评测中断"（回退一轮状态并删除一个评测 pair）→ resume 输出 `collect(skipped)`、`train(skipped)`、只补缺失 pair，最终状态回到 `completed_rounds=[0,1]`。
3. 模拟"训练完成但 final 未提交"（删除 `final.pt`/`final.json` 并回退状态）→ resume 从轮末 step checkpoint 恢复（`steps_completed=4`，本次 0 新步），`final.pt` 重建后与旧文件 sha256 相同，评测逐局日志零重跑。

round 1 的 `round_summary.json` 来自恢复会话：删除一个 MCTS pair 后 resume 只补跑了缺失 pair，其余逐局日志未改动，因此该轮的评测得分率与耗时只对应恢复会话实际执行的部分。

这些数字只证明闭环与恢复可用，**不能**用于声明棋力提升：每轮评测只有 1 pair，训练只有 4 步，且初始与目标数据规模都不具代表性。

### 11.4 未执行项

- 默认 5 轮 × 500 局生产训练与评测。
- 全量 pytest（按任务要求只运行直接相关测试）。
- 多训练 seed、FINAL 协议、容量/搜索参数对照实验。
- 历史 `runs/az_selfplay_seed1_round000/` 300 局数据未导入（生产起点默认保持空白）。

### 11.5 第二轮修复后的复验（2026-09-28）

同一 smoke 配置在修复后重跑到全新目录（非 `runtime/az_loop_smoke`），确认：

- 终端按 30 秒节流输出 `collect_start` / `collect_progress` / `train_progress` / `evaluate_progress`，并在 step 1 与轮末各打印一次训练进度；
- 第 1 轮 `train_start` 显示 `continued=true, total_train_step=4`，两轮各 4 步后 `total_train_step=8`，两个 `final.pt` 的 AdamW `step` 分别为 4 与 8（第二轮不再重置优化器）；
- 每轮 `training_metrics.jsonl` 的步号为 1–4、全局步为 1–4 与 5–8，各含一条 `validation`；
- TensorBoard 出现 `validation/policy_ce`、`validation/value_mse`（全局步 4/8）、`collect/round-00X/games_completed`、`collect/games_per_second`、`collect/policy_samples_per_second`、`collect/simulations_per_second`；
- 删除 `loop_state.json` 与第 1 轮 `greedy_vs_initial.json` 后 `--resume`：第 0 轮直接复用，第 1 轮输出 `collect(skipped)`、`train(skipped)`、`evaluate`，两个评测面板都从已完成 pair 日志重建，全部 8 局归档的 sha256 摘要前后一致（零对局重跑）。

## 12. 首轮实现中发现并修复的问题

- **final checkpoint 重建导致的评测身份漂移（本轮新代码）**：最初轮末用 `learner.save(final.pt)` 独立再保存一次；一旦"训练完成、final 未提交"后恢复，文件内容（含 RNG state）与首次不同，`evaluation` 的 checkpoint sha 身份校验会拒绝复用已完成评测。现改为把轮末 step checkpoint 字节复制为 `final.pt`，使恢复后的哈希稳定。
- **轮内步数计数（本轮新代码）**：`steps_completed` 原先只统计本次会话步数，恢复场景会低报并可能与状态文件口径不一致。现改为轮内累计值，另记 `steps_this_session`。
- 冻结加载入口、自博弈、评测、搜索等既有模块未发现需要修复的既有 bug；本轮对 `opponent_pool._checkpoint_payload()` 的修改是纯扩展（PPO 行为不变，相关既有测试通过）。

## 13. 第二轮修复记录（2026-09-28）

本轮的 bug 全部来自"未提交实现的复审 + 真实 smoke 复验"，修复后 `tests/test_alphazero_replay.py`、`tests/test_alphazero_training.py`、`tests/test_alphazero_loop.py`（24 个测试）、以及受影响的 `tests/test_search_accel.py`、`tests/test_selfplay.py`、`tests/test_opponent_pool_5b.py` 全部通过。

| 问题 | 触发条件 | 根因与修复 |
|---|---|---|
| 跨轮不延续 optimizer、全局训练步重置 | 第 2 轮及以后 | `_resume_or_warm_start()` 对 r>0 走 `warm_start()`（只读权重、新建 AdamW、`total_train_step=0`），使 AdamW 动量每轮丢失、TensorBoard `train/*` 曲线按轮重叠。新增 `AlphazeroLearner.continue_from()`：延续模型、optimizer state 与全局步，轮内计数与新轮抽样种子重新开始 |
| 整局 LRU 缓存从不写入 | 任何训练取样 | `_GameCache.sample()` 组装了 `entry`/`size` 但没有写回 `self._entries`，每次取样都重新解压整局 NPZ，同时 `bytes` 虚增。现按预算插入、按 LRU 淘汰最旧条目；单局超过预算时只服务不缓存 |
| 恢复后训练指标日志重复 | 崩溃点在"指标已写、checkpoint 未落盘"的窗口内 | 指标行先于周期性 checkpoint 追加，恢复后同一步号被再写一次。现在 `run_training_round()` 进入时按最后一个 checkpoint 的轮内步数重写 JSONL，丢弃未提交步骤与旧验证行 |
| 评测工程失败不阻止下一轮 | 面板出现截断/异常/非法动作/循环裁决失败 | 旧实现只把 `passed=false` 写进状态继续采集。现在两个面板各自结束后立即 `_require_evaluation_passed()`，失败抛出 `EvaluationEngineeringFailure`、写 `failures/round-XXX-evaluate.json` 并终止循环 |
| 状态文件丢失可能跳过未完成评测 | `loop_state.json` 丢失且 `final.pt` 已提交但评测未完成 | `_recover_state()` 原来只看 `final.pt`，而 final 在评测之前提交。现在恢复要求两个评测 summary 存在、`completed_pairs == total_pairs` 且 `passed=true`，否则该轮按“只补评测”恢复 |
| 训练失败现场阶段标注错误 | 训练期异常（如 `NonFiniteTrainingError`） | `_run_round()` 在训练前未把 phase 从 `collect` 切到 `train`，现场写成 `round-XXX-collect.json`。现已在调用训练前切换 phase |

另外补齐（本轮新功能）：

- 终端周期进度：采集局数/本会话新增/局每分钟、轮内与全局训练步/步每秒/ETA、评测面板已完成 pairs/pairs 每分钟；节流默认 30 秒，首个事件立即输出，阶段完成事件保留。
- TensorBoard：验证 CE/MSE（含熵、局数与局面数）、轮内采集进度（按轮分标签，x 轴为轮内秒）与轮级采集吞吐（以该轮入口全局步为 x 轴）。
- `selfplay_fast.generate()` 新增父进程 `progress_callback(completed_games, total_games)`：单 worker 逐局回调，多 worker 按 `progress_interval_seconds=1.0` 轮询归档目录，恢复会话报告的是绝对完成局数。
