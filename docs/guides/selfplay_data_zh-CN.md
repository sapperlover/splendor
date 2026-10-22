# MCTS 自对弈数据与恢复

2026-10-21新增[相同96根的精确推理输入复用诊断](../../artifacts/search_input_cache_rate_20261021/results.md)：每组两个seed，S1024N1／S256N4／S1024N4（总预算1024／1024／4096）及随机节点总1024、CPW关闭的单次搜索内潜在命中率为14.62%／19.20%／20.22%／2.69%。统计完整FP32输入与mask，排除padding、按根／seed／方法独立空缓存；N4总比例包含同批在途去重潜力，普通已完成结果缓存的命中率另列。没有实施缓存或测速，也未测跨走子复用，不等同花砖跨走子memo命中率。

旧θ5的[记录字段协议](../archive/next_az_record_fields_zh-CN.md)说明原NPZ／trace、逐局SHA、同集诊断引用及抽样覆盖计数；该源run后续十轮完成。额外全量checked replay的离线诊断成本单列；阶段划分依具体协议，不混用完成回合和公开分数阈值。

`selfplay.py` 是历史 v1 采集入口；当前批量/native 采集使用 `selfplay_fast.py`，输出 `mcts-selfplay-v2`。默认数据schema仍为v2；开启[预算随机化、Forced Playouts / Policy Target Pruning](search_training_zh-CN.md)时写v3，`SelfPlayDataset`和跨代replay均兼容v1/v2/v3。双方使用同一冻结教师、独立树和种子；完整正式终局后才提交一局 NPZ。

2026-10-21起，全部新采集统一采用[12张贵族规则池](../splendor_standard_rules_zh-CN.md)：经典10张加蓝绿黑333、红白44，双人局仍抽3张。Python、native CPU、单owner CUDA和多worker CUDA均从同一默认目录开局，不单独维护贵族表。两种采集入口的manifest都把`catalog_fingerprint`纳入semantic fingerprint；新进程不允许在旧10张池的未完成目录中`--resume`。旧NPZ仍可作为明确标注来源的历史监督读取，不能改称12张池采集数据；格式可读不等于规则相同。模型贵族输入仍为3×17，无需扩容，后续采集使用新目录；已经启动的旧进程需退出后用新代码重新启动。

下文固定预算、`policy_weight=searchable`及原始访问分布作为target的描述限定v1/v2。v3低预算步骤仅用于value，target可裁剪，字段与校验见[新增协议](search_training_zh-CN.md#数据与复核)。

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
| `simulation_budgets` | `[T,N]` int32 | 每样本各世界实际预算；单候选行全零；v3低预算可单独减少世界数，未用世界补零 |
| `per_world_root_visit_counts` | `[N,C]` int32 | 每世界各根动作实际访问量，列与扁平候选对齐 |
| `valid_world_count`、`simulations`、`max_tree_depth`、`transition_cap_hits` | `[T]` int8/int32 | 单候选 `valid_world_count=0`；v3可通过low_determinizations减少低预算世界数，实际计数小于N，均值只计有效世界 |
| `player_id`、`decision_index`、`phase`、`completed_player_turns`、`chosen_candidate_index` | `[T]` int8/int32 | 真实座位、全局决策序号、阶段枚举索引、真实回合数及样本内候选索引 |
| `temperature`、`sampling_seed`、`searchable` | `[T]` float32/uint64/bool | 走子设置与审计信息 |
| `value_target`、`policy_weight`、`value_weight` | `[T]` int8/float32 | 正式终局 z 与后续监督权重 |

`metadata_json` 包含 schema、run/game 身份、语义配置指纹、checkpoint SHA、真实开局和双方有效搜索 seed、split、正式终局、样本计数、MCTS 统计、最终 digest 和已有引擎 replay。v1 包含上述完整 replay；v2 用下节 compact trace 替代逐步完整 replay。metadata、seed、z 不进入网络输入。概率写入 float32，校验容差 `1e-6`。

2026-10-13新增可选[Policy Surprise Weighting](../../artifacts/entity_attention_r43_policy_surprise_s4096_20261013/README.md)：在训练时用`KL(policy_target || root_network_prior)`计算局内归一化的policy loss权重，明确使用加噪前教师策略。归档中的`policy_weight=searchable`、`value_weight=1`及数据schema不变；不加权的原留出指标保留。辅助分差与剩余玩家行动回合的标签从终局metadata和`completed_player_turns`生成；后续已接入[独立离线双辅助头入口](../../artifacts/entity_attention_r24_r43_offline_objectives_20261013/README.md)。这两项终局信息仅用作训练标签，原NPZ和输入特征schema不变。

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

2026-10-12归档核对：上面300局S256/N2实例与4局smoke均已有完成采集汇总，已迁入`archive/20261012/`并保留旧路径链接；仅说明原版本的采集与归档链路产生完整对局，不是独立棋力验证。完成数、用途和限制见[早期工程实验归档说明](../reports/archived_engineering_experiments_20261012_zh-CN.md)。


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

## 联合训练的两个在线窗口

`hybrid_train` 的可选 `policy_replay_max_games` 在 `online_only=true` 时缩小 policy 的最近完整对局窗口，value 继续使用 `replay_max_games`。先按完整局截取再排除整局 validation；两头均按 Train 局面均匀有放回抽样、每头保持原 batch。两头共享 encoder，合并损失后只执行一次全局梯度裁剪和 AdamW 更新。此配置与 `split_training` 互斥；未设置时原配置及采样行为保持不变。每轮快照记录 `joint_head_pools`，账本在 `usage/policy`、`usage/value`，`source_usage.json` 分头记录曝光及逐代实际抽样数。当前 [S4096、2k/6k 实验](../../artifacts/entity_attention_r43_s4096_p2k_v6k_20261013/README.md) 保存了命令、迁移期和结论边界。
