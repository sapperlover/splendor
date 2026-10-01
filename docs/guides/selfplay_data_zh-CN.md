# MCTS 自对弈数据与恢复

`selfplay.py` 是历史 v1 采集入口；当前批量/native 采集使用 `selfplay_fast.py`，输出 `mcts-selfplay-v2`。`SelfPlayDataset` 可读取两种格式。双方使用同一冻结教师、独立树和种子；完整正式终局后才提交一局 NPZ。

## 冻结配置与随机性

历史 v1 配置示例：[`configs/mcts_selfplay_smoke.json`](../../configs/mcts_selfplay_smoke.json)（4 局、2 CPU workers）、[`configs/mcts_selfplay_round000.json`](../../configs/mcts_selfplay_round000.json)（300 局、4 CPU workers）。相对 checkpoint 路径以运行命令时的工作目录为基准；示例从仓库根目录运行。`num_games=300` 是 300 个独立 game_id，并非 300 pairs。`first_player=game_id%2`。

种子协议 `splendor-selfplay-seed-v1`：对 JSON 规范数组 `[协议, domain, base_seed, round_index, game_id, ...]` 求 SHA-256，取前 8 字节作无符号 64 位整数。`game` 域控制真实开局；`search_world` 域按座位生成策略 seed，并与配置的 MCTS seed 作 XOR；`search_sampling` 域额外纳入真实座位和全局 decision_index，控制根噪声与走子采样。全局 decision_index 从 0 起，双方所有真实请求共用一个序列，含单候选请求。世界采样 seed 与 sampling_seed 分开。train/validation 使用独立 split_seed 和 `split` 域，按整局哈希划分；workers 不参与任何内容种子。

温度使用当前 Observation 中双方已完成真实玩家回合之和。小于 `opening_completed_turns=10` 时为 1.0，此后为 0.0；同一真实回合中的多个选择阶段保持同一温度区间。只要 `add_root_noise=true`，后期温度 0 仍会对所有多候选根加噪声。

## 数值归档格式（v1/v2 共用）

数据 schema 分别为 `splendor-mcts-selfplay-v1` 和 `splendor-mcts-selfplay-v2`，以下数值字段共用。每个完整对局一个 `games/game-XXXXXX.npz`，用 `allow_pickle=False` 读取。`metadata_json` 是 UTF-8 JSON bytes 的 `uint8` 向量；其余仅为数值或布尔数组。`T` 为真实决策数，`C` 为各样本候选数之和，`N` 为配置世界数：

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

`metadata_json` 包含 schema、run/game 身份、语义配置指纹、checkpoint SHA、真实开局和双方有效搜索 seed、split、正式终局、样本计数、MCTS 统计、最终 digest 和已有引擎 replay。v1 包含上述完整 replay；v2 用下节 compact trace 替代逐步完整 replay。metadata、seed、z 不进入网络输入。概率写入 float32，校验容差 `1e-6`。

`SelfPlayDataset(root, split="train" | "validation" | None)` 只索引归档，按需加载当前局；`collate_selfplay_samples(samples)` 用 offsets 构造 `EncodedDecisionBatch`、padding 后的 policy/behavior 目标、z、两种权重和独立引用 metadata。模型可以直接消费：

```python
from pathlib import Path
from splendor.selfplay_data import SelfPlayDataset, collate_selfplay_samples

dataset = SelfPlayDataset(Path("runs/az_selfplay_seed1_round000"), split="train")
batch = collate_selfplay_samples([dataset[0], dataset[1]])
logits, values = frozen_model(batch.decisions)
# 使用 batch.policy_targets、batch.value_targets 和两种 weights 构造下一步损失。
```

## 提交、恢复与校验（区分 v1/v2）

完整局先在内存收集，正式终局后填 z、做局内验证，再写同目录临时 NPZ、flush/fsync、原子 rename 并 fsync 目录。只有最终 `game-*.npz` 是提交事实；遗留 tmp 文件不被读取。manifest 冻结语义配置、checkpoint 内容 SHA 和规则/特征/RNG/seed/data 版本，并记录测量时 Git revision/dirty。v1 还记录关键行为文件 SHA，执行 workers 可覆盖；v2 的 config（包括 execution 拓扑）进入语义指纹，不允许照搬 v1 的改 workers 恢复方式。resume 核对对应协议的指纹和已提交局，只派发缺失 game_id；不跨协议、模型或设备混用。同一输出根使用进程退出自动释放的文件锁。

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

监督训练与跨代 replay 已接入 [AlphaZero 循环](alphazero_zh-CN.md)。单次采集的胜负、熵和吞吐属于数据健康与成本统计，不代表训练后棋力。


## v2 执行路径与 compact trace

Python 引擎 `checked` 保留完整 replay、摘要与 invariant；`fast` 使用相同 Python 规则，但减少逐转移审计工作。native 配置要求 `engine_mode=fast`，初始洗牌由 Python 完成，后续真实对局由 `NativeMatchStepper` / C++ `NativeGame` 推进；详见 [native 指南](native_search_zh-CN.md)。

v2 `metadata_json` 的 `compact-action-trace-v2` 保存 game_id、seed、先手、规则/catalog/RNG 版本、真实动作、候选索引和终局，不保存每步 state digest。单候选样本 policy weight=0、value weight=1。普通 validate 检查数值与 trace 结构；`--replay` 在 Python checked 规则下重放全部真实动作。它复核已保存的动作轨迹，不保证重现 GPU 舍入或跨批搜索。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast generate \
  --config configs/native_search/selfplay_cpu_smoke.json --output runtime/native_search/cpu_smoke
conda run -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast validate \
  --input runtime/native_search/cpu_smoke --replay
# 中断后追加 --resume，保持冻结配置和 checkpoint 身份
```

故障保存 seed、真实动作前缀、请求，以及可用的搜索世界/分支现场。无需网络的重建入口：

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.reproduce_failure \
  --input runtime/your_run/failures/game-000000.json
```

训练 NPZ 和 checkpoint 保留在 `runs/`；值得长期复核的完成统计保存到 `artifacts/`。恢复、吞吐、并行局内累计时间与墙钟不是同一口径，不相互替代。
