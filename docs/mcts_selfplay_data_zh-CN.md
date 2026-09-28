# MCTS 自我对弈数据：冻结网络的第一轮采集

## 实现范围

`src/splendor/selfplay.py` 用记录型 Policy 包装现有 `PUCTSearchPolicy`，双方共享同一冻结 checkpoint 的只读网络对象，但各自有独立 MCTS 策略、树、统计与随机种子。真实对局由 `run_match()` 执行；每个正式 `DecisionRequest` 只调用一次带探索的 `search()`，执行前记录当前可见数值编码、完整根策略分布和所选候选索引。引擎自动 forced skip 不产生样本，真实 PAYMENT、RETURN_TOKENS、CHOOSE_NOBLE 等请求均保留。搜索策略仍只接收 `DecisionRequest`。

仅 `NATURAL_TERMINAL` 且有正式 `TerminalResult` 的局可以提交。每个样本的 value target 是 `terminal_result.payoff(真实 player_id)`，不按决策序号交替翻号。正式平局为 0；工程截断、异常和非法动作明确失败并保留现场，不作为训练数据。runner 显式关闭确定性循环判平，使用 `raise_on_error=True`、`lean=True`。

单候选请求保留以供 value 学习，`policy_weight=0`、`value_weight=1`、target/behavior 为 `[1]`，搜索次数和世界数为 0。多候选请求两种权重均为 1。`policy_target` 是第一步 MCTS 给出的各世界等权根访问分布；`behavior_policy` 是温度变换后实际走子的分布；`chosen_candidate_index` 是执行索引。三者分别保存，不从汇总访问量重算等权 target。

## 冻结配置与随机性

配置示例：[`configs/mcts_selfplay_smoke.json`](../configs/mcts_selfplay_smoke.json)（4 局、2 CPU workers）、[`configs/mcts_selfplay_round000.json`](../configs/mcts_selfplay_round000.json)（300 局、4 CPU workers）。相对 checkpoint 路径以运行命令时的工作目录为基准；示例从仓库根目录运行。`num_games=300` 是 300 个独立 game_id，并非 300 pairs。`first_player=game_id%2`。

种子协议 `splendor-selfplay-seed-v1`：对 JSON 规范数组 `[协议, domain, base_seed, round_index, game_id, ...]` 求 SHA-256，取前 8 字节作无符号 64 位整数。`game` 域控制真实开局；`search_world` 域按座位生成策略 seed，并与配置的 MCTS seed 作 XOR；`search_sampling` 域额外纳入真实座位和全局 decision_index，控制根噪声与走子采样。全局 decision_index 从 0 起，双方所有真实请求共用一个序列，含单候选请求。世界采样 seed 与 sampling_seed 分开。train/validation 使用独立 split_seed 和 `split` 域，按整局哈希划分；workers 不参与任何内容种子。

温度使用当前 Observation 中双方已完成真实玩家回合之和。小于 `opening_completed_turns=10` 时为 1.0，此后为 0.0；同一真实回合中的多个选择阶段保持同一温度区间。只要 `add_root_noise=true`，后期温度 0 仍会对所有多候选根加噪声。

## 归档格式

数据 schema 是 `splendor-mcts-selfplay-v1`。每个完整对局一个 `games/game-XXXXXX.npz`，用 `allow_pickle=False` 读取。`metadata_json` 是 UTF-8 JSON bytes 的 `uint8` 向量；其余仅为数值或布尔数组。`T` 为真实决策数，`C` 为各样本候选数之和，`N` 为配置世界数：

| 字段 | 形状 / 类型 | 语义 |
|---|---|---|
| `global_features`、market/reserved/nobles/pending 各状态字段及其 mask | `[T, ...]`，float32/bool | `encode_decision()` 的完整状态编码，按当时行动者可见信息；形状与现有 `EncodedDecision` 一致 |
| `candidate_offsets` | `[T+1]` int64 | 从 0 到 C，严格递增；样本 i 的候选是 `[offsets[i]:offsets[i+1]]` |
| `candidate_features` | `[C, action_dim]` float32 | 原合法候选顺序的完整编码，不保存或伪造 Action 对象 |
| `policy_target`、`behavior_policy`、`root_network_prior`、`root_used_prior` | `[C]` float32 | 各样本按 offsets 切分；保留零访问候选 |
| `root_visit_counts` | `[C]` int32 | 各世界访问计数的和 |
| `simulation_budgets` | `[T,N]` int32 | 每样本各世界实际预算；单候选行全零 |
| `per_world_root_visit_counts` | `[N,C]` int32 | 每世界各根动作实际访问量，列与扁平候选对齐 |
| `valid_world_count`、`simulations`、`max_tree_depth`、`transition_cap_hits` | `[T]` int8/int32 | 单候选 `valid_world_count=0`，不是 N 个执行过的零预算世界 |
| `player_id`、`decision_index`、`phase`、`completed_player_turns`、`chosen_candidate_index` | `[T]` int8/int32 | 真实座位、全局决策序号、阶段枚举索引、真实回合数及样本内候选索引 |
| `temperature`、`sampling_seed`、`searchable` | `[T]` float32/uint64/bool | 走子设置与审计信息 |
| `value_target`、`policy_weight`、`value_weight` | `[T]` int8/float32 | 正式终局 z 与后续监督权重 |

`metadata_json` 包含 schema、run/game 身份、语义配置指纹、checkpoint SHA、真实开局和双方有效搜索 seed、split、正式终局、样本计数、MCTS 统计、最终 digest 和已有引擎 replay。metadata、seed、z 不进入网络输入。概率写入 float32，校验容差 `1e-6`。

`SelfPlayDataset(root, split="train" | "validation" | None)` 只索引归档，按需加载当前局；`collate_selfplay_samples(samples)` 用 offsets 构造 `EncodedDecisionBatch`、padding 后的 policy/behavior 目标、z、两种权重和独立引用 metadata。模型可以直接消费：

```python
from pathlib import Path
from splendor.selfplay_data import SelfPlayDataset, collate_selfplay_samples

dataset = SelfPlayDataset(Path("runs/az_selfplay_seed1_round000"), split="train")
batch = collate_selfplay_samples([dataset[0], dataset[1]])
logits, values = frozen_model(batch.decisions)
# 使用 batch.policy_targets、batch.value_targets 和两种 weights 构造下一步损失。
```

## 提交、恢复与校验

完整局先在内存收集，正式终局后填 z、做局内验证，再写同目录临时 NPZ、flush/fsync、原子 rename 并 fsync 目录。只有最终 `game-*.npz` 是提交事实；遗留 tmp 文件不被读取。manifest 冻结全部语义配置、checkpoint 内容 SHA、规则/特征/RNG/seed/data 版本，以及 Git revision、dirty 状态和关键行为文件 SHA。resume 先核对当前配置与实现指纹、checkpoint 内容，再验证全部已提交局，随后只派发缺失 game_id；workers 可改变，语义配置和设备不可混用。同一输出根使用进程退出自动释放的文件锁。

独立 `validate` 会检查所有已提交局的形状、dtype、有限值、候选索引、概率、按世界等权 target、预算、终局 z、split、身份及 replay 摘要；`inspect` 给出从归档重建的汇总。整个数据集即使缺失 status/summary，也能从已提交整局恢复。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.selfplay generate \
  --config configs/mcts_selfplay_smoke.json \
  --output runtime/mcts_step2_smoke

conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.selfplay generate \
  --config configs/mcts_selfplay_round000.json \
  --output runs/az_selfplay_seed1_round000

# 中断后，同一配置恢复；可用 --workers 覆盖执行 worker 数。
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.selfplay generate \
  --config configs/mcts_selfplay_round000.json \
  --output runs/az_selfplay_seed1_round000 --resume

conda run -n splendor env PYTHONPATH=src python -m splendor.selfplay validate \
  --input runs/az_selfplay_seed1_round000
conda run -n splendor env PYTHONPATH=src python -m splendor.selfplay inspect \
  --input runs/az_selfplay_seed1_round000
```

当前只完成同一冻结教师的数据采集；监督训练器、optimizer 更新和跨代 replay buffer 尚未实现。数据中的胜负、熵和吞吐只用于数据健康及成本分析，不表示训练后的棋力变化。

## 首次 4 局 smoke（2026-09-27）

真实 5.5M checkpoint、S=256/N=2、2 CPU workers，执行命令见上方 smoke 示例。输出 `runtime/mcts_step2_smoke/`；独立 `validate` 通过，读回 3 个样本经原 ActorCritic 前向，合法 logits 和 value 均 finite。4 局全部正式终局，0 失败，先手座位 2/2；335 个 value 样本、221 个有效 policy 标签，3 个 train 局/246 个样本、1 个 validation 局/89 个样本。实际 56,576 次 simulation，53,583 次模型 batch，0 次转移上限命中；归档共 192,915 bytes。累计局内采集时间 82.82 秒，真实墙钟约 44.13 秒（含进程开销）；并行局内时间之和不能当作墙钟。该 smoke 仅用于验证采集链路与成本。

## 首批 300 局采集结果（2026-09-27）

执行配置为 [`configs/mcts_selfplay_round000.json`](../configs/mcts_selfplay_round000.json)，实际运行命令在上述 `generate` 命令末尾添加 `--workers 8`；这是执行并行度覆盖，不改变内容种子或语义指纹。冻结 checkpoint SHA-256 为 `009d17d2e9bc0d879142d7ca3ff1777a6f8caa9dda9243c6a2d5eb3538b5d5b0`。输出为 `runs/az_selfplay_seed1_round000/`，其中 `manifest.json` 固定身份，`games/` 有 300 个完整 NPZ，`summary.json` 为归档重建汇总。

全量独立 `validate` 和 `inspect` 均通过：300/300 局完成，0 缺失、0 失败；先手座位各 150 局。共 26,109 个 value 样本，其中 17,617 个多候选有效 policy 标签，8,492 个单候选 value 样本。train 为 268 局／23,329 样本，validation 为 32 局／2,780 样本，整局不跨 split。阶段计数：MAIN_ACTION 16,618、PAYMENT 9,267、RETURN_TOKENS 197、CHOOSE_NOBLE 27。正式终局中真实座位 P0 胜 132、P1 胜 165、平局 3；这些是数据健康统计，不是棋力对比。

实际执行 4,509,952 次 simulation，4,253,732 次模型 batch，0 次 transition cap 命中。NPZ 合计 14,776,150 bytes；采集状态记录墙钟约 1,338.55 秒（约 22.3 分钟），各局搜索时间之和约 10,489.98 秒，后者因 8 worker 并行大于墙钟。读回 train/validation 样本组成 ragged batch 后，原模型的合法 logits/value 均 finite。使用相同配置、改为 `--workers 4 --resume` 后，全部已提交局被正确跳过。
