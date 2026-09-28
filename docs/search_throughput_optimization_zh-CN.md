# 冻结搜索与自博弈采集加速

## 执行路径

`GameEngine` 现在有 `checked` 和 `fast` 模式。前者继续保存完整 replay、逐转移状态摘要与全量 invariant，供旧存档、规则测试和故障重建使用。后者调用相同的 `rules.apply_legal_action` / `apply_forced_skip`，在每次模拟推进时不生成完整 replay event、不计算完整状态 SHA-256、不扫描全量 invariant。可信候选索引绑定引擎状态版本，外部 `apply(action)` 仍检查合法候选。强制阶段折叠在引擎内连续推进，只为返回的最终决策构造 Observation。旧网页和评测 runner 仍可使用 checked 引擎。

`FrozenSearchInference` 持有 eval 模式的 FP32 模型；搜索接入 policy/value、policy-only、state-only critic 三种入口。卡牌和 pending card 共用一次 card encoder 调用，单样本 collator 使用维度视图。CUDA 批量输入拼为连续 FP32 缓冲区后一次传输；输出整批取回，再按各请求合法候选数切分并检查有限值。模型参数名、形状、特征 schema 与训练用 `ActorCritic.forward()` 未变。传入训练态模型时推理 owner 先复制一份，避免改动 learner 的 train/eval 状态。

MCTS 的同步 `search()` 和自博弈批调度都驱动同一个 `MCTSSearchJob`。每棵 determinization 树最多挂起一个待评估叶；跨世界、跨真实对局聚合 policy/value 或 critic-only 请求，按唯一 request ID 回传。根预算、根噪声、等权 target、温度和 tie-break 与旧算法一致。已直接对照修改前实现：固定开局、S=257/N=2 时，根访问、target、先验和选招完全相同。旧逐叶 `_simulate` 实现已删除。

## v2 自博弈数据与故障重建

旧 `mcts-selfplay-v1` 的 300 局归档继续由 `SelfPlayDataset` 和原校验器读取。新采集使用 `mcts-selfplay-v2`，数值训练字段及权重语义相同；`metadata_json` 使用 `compact-action-trace-v2`，记录真实 game_id、seed、先手、规则/catalog/RNG 版本、每次真实请求的座位/阶段/结构化 Action/候选索引和正式终局，不包含逐转移 digest 或完整 replay。整局正式终局后验证并原子提交 NPZ；单候选样本仍为 policy weight 0、value weight 1。普通 `validate` 检查数值、标签及 trace 结构；`--replay` 用 checked 引擎按已记录的真实 Action 重放至终局。resume 核对语义配置及 checkpoint 内容，快速扫描已提交 NPZ 的 metadata，只补缺失 game_id。

异常时在输出目录的 `failures/game-XXXXXX.json` 写入已执行真实动作前缀、当前根请求、搜索配置、checkpoint 身份和批请求/输出。若异常来自搜索分支，还保留世界 seed、simulation 序号、已成功分支动作及尝试动作。重建命令不运行网络：

```bash
conda run -n splendor env PYTHONPATH=src \
  python -m splendor.reproduce_failure \
  --input runtime/search_accel/your_run/failures/game-000000.json
```

命令先在 checked 模式重放真实动作；有完整搜索分支信息时再从 JSON 根请求和 determinization seed 重建采样世界、执行分支动作，并核对待推理请求。它定位到已保存的错误位置，不声称重新产生 GPU 浮点或批调度异常。

## 命令与配置

四个 8 局短基准配置位于 [`configs/search_accel/`](../configs/search_accel/)：`serial_cpu_checked_8games.json`、`serial_cpu_fast_8games.json`、`batched_cpu_8games.json`、`batched_cuda_8games.json`。它们使用同一 checkpoint、8 个 game_id 和 S=256/N=2，总计最多 32 局。完整顺序执行：

```bash
conda run --no-capture-output -n splendor python -u \
  scripts/run_search_accel_short_benchmark.py \
  --output-dir runtime/search_accel/benchmark_8games_new
```

生产或自定义采集可直接用 `selfplay_fast`：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.selfplay_fast generate \
  --config configs/search_accel/batched_cpu_8games.json \
  --output runtime/search_accel/my_run

# 中断后保持同配置、同输出目录
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.selfplay_fast generate \
  --config configs/search_accel/batched_cpu_8games.json \
  --output runtime/search_accel/my_run --resume

conda run -n splendor env PYTHONPATH=src \
  python -m splendor.selfplay_fast validate --input runtime/search_accel/my_run --replay
```

`batched_cpu` 在每个 worker 内协作推进多个真实对局；`batched_cuda` 在单进程中持有一个 GPU 模型，配置要求 workers=1。GPU 不可用且需继续采集时会明确报错；已有 CUDA 归档仍可在 CPU 机器上读取与校验。`max_batch_size` 限制物理前向大小；协作调度在可运行任务均已到等待点时立即发出部分 batch，因此不会因尾批不足而卡住。`max_batch_wait_ms` 是预留的上界配置，在这一单进程协作实现中实际等待为 0。运行内容随机性不使用 worker、PID 或批返回顺序。

固定开局、中盘、后段搜索面板及 batch=1/8/32/64 推理微基准：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.benchmark_search \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --output runtime/search_accel/fixed_panel_cpu.json --device cpu \
  --repeats 3 --warmup 1
```

固定面板 JSON 保存真实动作前缀、版本/设备/线程数、每次耗时、选招及实际搜索工作量。单步基准的模型加载和请求生成在计时外；自博弈 summary 的 `wall_seconds` 包含本轮初始化和归档提交，另列 worker 初始化、实际采集、模型编码/前向、规则、写入、决策延迟和峰值内存。各局 `search_seconds_sum` 是重叠任务时间之和，不应当作 run 墙钟。不同 batch 形状和 CUDA 允许正常 FP32 舍入差异，故障重建依赖真实 Action trace。

## 验证与测量

修改前先保存了 `runtime/search_accel/baseline_before.json`：同一 checkpoint、CPU 单线程、seed=42 开局，MCTS S=256/N=2 单次 0.395 秒，折叠 alpha-beta D6/top5 单次 3.49 秒。修改前没有三次重复，因此不能把该单次读数当作严格 speedup 分母。

最终固定局面面板见 `runtime/search_accel/fixed_panel_cpu_final.json` 和 `runtime/search_accel/fixed_panel_cuda_final.json`。预热一次后每项重复 3 次，下表为 CPU 中位墙钟秒数；JSON 还保存范围、每次选招、simulation/扩展/模型位置数和可 checked 重建的真实动作前缀。三组局面分别是同一固定轨迹的决策 0、20、40。D6 checked 与 fast 的搜索深度/top-k 完全相同。

| 请求 | D6 checked | D6 fast | MCTS 1024/1 fast | MCTS 256/2 fast | MCTS 1024/2 fast |
| --- | ---: | ---: | ---: | ---: | ---: |
| 开局 | 3.122 | 2.452 | 1.193 | 0.248 | 0.956 |
| 中盘 | 0.901 | 0.682 | 1.165 | 0.210 | 0.908 |
| 后段 | 1.663 | 1.300 | 1.306 | 0.225 | 0.967 |

真实 checkpoint 的 CPU 单线程推理微基准包含编码、collate、前向和输出取回，64 个不同请求中有 MAIN_ACTION 43、PAYMENT 15、RETURN_TOKENS 6；batch=1/8/32/64 的中位每位置耗时分别为 0.887/0.359/0.315/0.335 毫秒。RTX 4050 Laptop GPU 的对应值为 1.661/0.396/0.262/0.234 毫秒。GPU batch=1 更慢；较大 batch 才显出优势。32 个真实请求的 CPU/CUDA FP32 最大 logit 绝对差为 3.82e-6，value 为 3.58e-7，原始 argmax 改变 0 个。

四档真实 8 局短采集结果见 `runtime/search_accel/benchmark_8games/comparison.json`。每档相同 8 个 game_id、S=256/N=2、121,856 次 simulation、688 个 value 样本、476 个有效 policy 样本，且逐局 checked 重放全部通过，异常与截断均为 0。该组测量后又合并了根编码复用、推理输入检查及异常路径修正；32 局短基准额度已用完，所列墙钟对应测量时的实现快照。表中墙钟包括模型 owner 初始化、编码、规则、推理、构造样本与文件提交；CPU 各档均为单 worker，以分离 batch 的收益。GPU 为一个进程持有一个 GPU 模型。

| 档位 | 墙钟秒 | games/s | 有效 policy 样本/s | 实际平均 batch | 物理前向次数 | CPU RSS 峰值 MiB | GPU 分配峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| checked 串行 CPU | 184.8 | 0.043 | 2.58 | 1.00 | 114,214 | 521 | 0 |
| fast 串行 CPU | 143.3 | 0.056 | 3.32 | 1.00 | 114,214 | 525 | 0 |
| fast 批量 CPU | 78.3 | 0.102 | 6.08 | 13.23 | 8,633 | 548 | 0 |
| fast 批量 CUDA | 79.6 | 0.101 | 5.98 | 13.23 | 8,633 | 857 | 12.2 |

相对本次 checked 串行档，fast 串行墙钟快约 1.29 倍，fast 批量 CPU 快约 2.36 倍。批量 CPU 的实际前向次数约为串行的 7.6%，没有减少 simulation 或训练样本。GPU 在这组 8 局、平均 batch 13.23 的采集里与 CPU 接近；其前向时间更短，但特征/传输等开销抵消了收益。RTX 4050 上当前默认推荐 `batched_cpu`、fast 引擎；更多并行局或更大 batch 是否改变结论，需要另做同工作量测量。

运行中的物理批次只由共享 evaluator 统计一次；每局 `mcts_stats.model_positions` 是逻辑模型位置，`search_seconds_sum` 是重叠搜索任务时间之和，都不能冒充 run 墙钟。最终代码另增规则、真实根编码、写入、决策延迟和 worker 初始化统计；2 局多 worker smoke 的中位/95 分位决策延迟为 9.32/19.15 毫秒，根编码 0.058 秒、搜索规则 0.283 秒、真实规则 0.035 秒、推理编码 0.063 秒、物理前向 0.464 秒、归档写入 0.039 秒，队列等待 0。该 2 局 smoke 与上表 8 局不同，不据此计算上表提速。

原 300 局 v1 数据已重新读取，共 26,109 个样本；v2 两局串行 CPU、两局批量 CPU、两局 CUDA、两局多 worker CPU 的 smoke 及 checked 重放通过。最后一次改动后又运行两局 CUDA 批量 smoke，得到 173 个 value 样本、115 个有效 policy 样本、920 次 simulation、平均 batch 3.38，checked 重放通过。测试覆盖 fast/checked 规则等价、陈旧候选拒绝、job 交错及等权 target、部分提交 resume、v2 Dataset、真实/分支故障重建及 CPU/CUDA 数值容差。没有降低 S/N、精度或样本定义来换取加速；本任务不包含训练器或新的搜索算法。

## 第二轮：审计、大批量 GPU 对比与搜索记忆化（2026-09-28）

本节记录对上一轮实现的审计结果与第二轮加速。上一轮的数字保留在上面，作为历史基线；本轮结论以本节的固定面板为准。

### 审计：发现并修复的问题

- **多 determinization 搜索没有走 fast 引擎。** `PolicyValueSearchPolicy._choose_combined_determinizations` 调用 `determinize_request` 时遗漏了 `engine_mode=self.engine_mode`，因此 N>1 的 D6 搜索始终使用 checked 引擎（逐转移 SHA-256、全量 invariant、完整 replay event），与"搜索默认 fast"的设计不符。已修复为与 N=1 相同，并加入回归测试 `test_combined_determinizations_use_requested_engine_mode`。
- 全量测试通过，未发现 TODO/占位实现、被跳过的正确性断言、异常冒充平局或半局入库的路径；`apply_candidate` 的 `state_version` 校验、外部动作合法性校验仍在。`search_inference`、`selfplay_fast`、`match_stepper`、`compact_trace`、`reproduce_failure` 链路保持可用。
- 上一轮文档中"`max_batch_wait_ms` 为预留上界、单进程协作实现实际等待为 0"的描述与代码一致；未发现未接通的配置项。

### 大批量 CPU 与 GPU 对比

新脚本 `scripts/benchmark_large_batch_inference.py` 用种子随机对局收集 1024 个**互不相同**的真实请求（main_action 644、payment 287、return_tokens 92、choose_noble 1），逐档测量包含特征编码、collate、传输与取回的每位置时间（中位数）。结果写入 `runtime/search_accel/large_batch_cpu_t1.json` 与 `runtime/search_accel/large_batch_cuda.json`：

| batch | CPU 1 线程 policy_value | CPU value_only | CUDA policy_value | CUDA value_only |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.831 | 0.497 | 1.469 | 1.023 |
| 8 | 0.346 | 0.180 | 0.384 | 0.248 |
| 32 | 0.264 | 0.138 | 0.233 | 0.145 |
| 64 | 0.260 | 0.129 | 0.213 | 0.131 |
| 128 | 0.269 | 0.131 | 0.208 | 0.124 |
| 256 | 0.338 | 0.129 | 0.203 | 0.121 |
| 512 | 0.324 | 0.130 | 0.201 | 0.119 |

单位毫秒/位置。结论：**batch=1/2/4 时 GPU 明显更慢**（batch=1 为 CPU 的 1.8 倍），**batch≥32 后 GPU 反超**并稳定在约 0.20 毫秒/位置；CPU 最佳区间是 32–128（约 0.26），更大 batch 因候选 padding 反而略慢。因此"D6 单次搜索改用 GPU"仍然不成立：搜索是逐叶顺序的 batch=1 调用，而把不同请求合并成 batch 会改变张量形状与浮点结果（实测最大 logit 差约 3.8e-6，近平分选招会翻转），与"优化前后结果完全一致"冲突。GPU 的优势只在真正的跨请求批量路径（`batched_cuda` 采集、`policy_value` 大 batch）中可用。

### 新优化一：搜索内记忆化（transposition 去重）

实测 D6/top-5 搜索内部有大量重复请求：开局局面的策略调用 54%、叶 critic 调用 65% 是重复内容（不同路径到达同一状态）；中盘约 7%/11%，后段约 16%/26%。

`PolicyValueSearchPolicy` 现在在每次 `choose()` 内维护三个内容寻址缓存（`SearchConfig.memoize`，默认开启）：

- 策略 logits：键为 `DecisionRequest`（冻结契约，可哈希，哈希约 6µs）；
- 叶 critic 值：键为 `Observation`（critic 只看状态编码，与候选无关）；
- 子扩展：键为 `(父 GameState, action_index)`，复用 fork 后的引擎与响应，跳过重复的规则推进。

命中时返回的是与原 batch=1 调用**完全相同输入、相同形状**的结果，因此逐位相同；`choose()` 开始时清空缓存，内存随搜索树有界。统计口径保持可比：`model_batches` / `model_positions` 仍是逻辑计数（与优化前逐项相等），新增 `physical_model_batches` / `physical_model_positions` / `cache_hits` / `cached_policy_calls` / `cached_value_calls` / `cached_expansions` 反映真实执行量。

### 新优化二：规则支付可行性快速判定

`legal_actions` 列主行动时原本对每张可见卡都完整枚举 `legal_payments_for_card` 只为判断"是否买得起"，且每张卡都重算一次 `player_bonuses`。现在：

- 每次列主行动只计算一次红利向量；
- 用 `rules.payment_exists`（金 token 每个恰好补 1 点缺口，因此"可支付 ⟺ 五色普通 token 缺口之和 ≤ 持有金"）做 O(5) 判定；完整枚举仍是 PAYMENT 阶段候选的唯一来源。

随机 4000 组（token 0–5、金 0–3、随机已购卡与目标卡）与 `bool(legal_payments_for_card(...))` 完全一致，已加入测试。

### 性能结果（固定面板，CPU 单线程，中位数，3 次重复）

优化前取 `runtime/search_accel/fixed_panel_cpu_final.json`，优化后取 `runtime/search_accel/fixed_panel_cpu_after_memo.json`（同一 checkpoint、同一固定局面与动作前缀、同一 `seed=9500000`）：

| 局面 / 搜索 | 前 | 后 | 加速 | 选招一致 | 逻辑位置 | 实际位置 | 缓存命中 |
| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| 开局 D6 检查模式 | 3.122 | 1.326 | 2.35× | 是 | 2802 | 1107 | 1695 |
| 开局 D6 fast | 2.452 | 1.014 | 2.42× | 是 | 2802 | 1107 | 1695 |
| 中盘 D6 fast | 0.682 | 0.579 | 1.18× | 是 | 778 | 701 | 77 |
| 后段 D6 fast | 1.300 | 0.971 | 1.34× | 是 | 1355 | 1060 | 295 |
| 开局 MCTS 1024/1 | 1.193 | 1.096 | 1.09× | 是 | 1025 | — | — |
| 开局 MCTS 256/2 | 0.248 | 0.195 | 1.27× | 是 | 257 | — | — |

MCTS 各档只有规则快速判定的收益（D6 记忆化不适用于 MCTS job），选招全部一致。全部 15 组（3 局面 × 5 搜索）的选招与逻辑 model positions 均与优化前一致。

### 正确性验证

- 固定面板与另取 10 个随机中盘局面（共 13 个）：`memoize=True` 与 `memoize=False` 的选招、`last_root_values`、`last_root_bound_values`、`expanded_nodes`、`model_positions`、`folded_decisions` 完全一致，0 处差异；
- 固定面板 15 组前后选招全部一致，逻辑搜索量不变（没有通过少搜换速度）；
- 新测试 `tests/test_search_memoization.py`：记忆化等价、统计关系、支付判定等价、N>1 引擎模式回归；
- 全量 `pytest -q` 通过（349 项）；另跑两局 fast 自博弈 smoke（mcts-selfplay-v2、S=8/N=2、2 局）并用 `validate --replay` 通过 checked 重放。
- 审计补注：`execution.max_batch_wait_ms` 仍是上一轮文档说明的预留上界（单进程协作调度不等待），不是未接通的功能。

### 已评估、本轮未采用的方案

- **搜索内跨请求批推理**（CPU 或 CUDA 都试了）：batch 形状会改变 FP32 结果，近平分选招会翻转，与"结果完全一致"冲突，故不用于 D6 搜索；只保留在真正的批采集路径。
- **`torch.compile`**：本环境（torch 2.14 CPU）在 `encode_state` 上触发 Dynamo guard 断言错误，且无法保证逐位一致，未启用。
- **根节点并行**（5 个根子节点同时搜索，进程或线程）：为并行必须给每个子节点全窗口。实测开局全窗口总代价 4.54s（顺序窗口 2.34s），并行墙钟由最慢子节点决定约 1.08s，与记忆化后的 1.07s 相当，没有收益；中盘全窗口 2.81s、最慢子节点 0.75s，也慢于记忆化后的 0.58s。放弃。
- **跨决策持久缓存**：同一局 30 次决策的实测中，仅搜索内缓存已省 26.4% 的模型调用；把策略/值缓存跨决策保留只再省 0.7 个百分点（2048 条上限时 +40MiB 峰值内存），不值得增加复杂度和内存。因此缓存仍在每次 `choose()` 内清空。

### 复现命令

```bash
# 大批量 CPU / GPU 推理对比（写入 runtime/search_accel/）
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u \
  scripts/benchmark_large_batch_inference.py \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --requests 1024 --max-batch 512 --repeats 3 \
  --output runtime/search_accel/large_batch_cpu_t1.json --device cpu --threads 1

conda run --no-capture-output -n splendor env PYTHONPATH=src python -u \
  scripts/benchmark_large_batch_inference.py \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --requests 1024 --max-batch 512 --repeats 3 \
  --output runtime/search_accel/large_batch_cuda.json --device cuda --threads 1

# 固定面板（开关记忆化对比：把 SearchConfig.memoize 改成 False 即优化前口径）
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u \
  -m splendor.benchmark_search \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --output runtime/search_accel/fixed_panel_cpu_after_memo.json \
  --device cpu --repeats 3 --warmup 1

# 全量测试
conda run -n splendor env PYTHONPATH=src python -m pytest -q
```

推荐配置不变：CPU 批量采集用 `batched_cpu` + fast 引擎；只有真正跨局批量推理（batch≥32）才用 CUDA。D6 单步搜索保持 CPU 单线程 + `memoize=True`。


## 故障重建与恢复统计修正（2026-09-28）

- MCTS 边保存实际执行的完整动作序列，包括折叠的支付等动作；重访已有边时将完整序列加入故障路径，支持不依赖网络的 checked 重放。
- `submit` 在校验 value/logits 后才移除 pending 请求。无效输出保留原请求的 world、determinization seed 和分支；消费请求后推进新分支发生异常时，仍记录正在推进的新分支。
- summary 的 `logical_model_positions` 从全部已归档对局汇总。`wall_seconds`、`inference`、延迟和资源统计采用本次成功采集会话口径（`statistics_scope=collection_session`），由 `statistics_game_ids` 明确覆盖范围；三个吞吐指标也只使用这些对局作为分子。首次完整采集时覆盖全 run，部分完成后恢复时只覆盖恢复阶段。
- `statistics_complete` 表示统计是否覆盖当前全部已归档对局，`statistics_missing_game_ids` 列出未覆盖对局。没有 status 时 scope 为 `unavailable`；旧 status 无覆盖信息时为 `unknown`，吞吐返回 null。历史未落盘的物理推理量和墙钟不能从并行对局耗时还原，不伪造累计值，也不增加逐步审计写入。

## 第三轮：FP32 合批搜索实验（D6 与 MCTS 的选招反转率与提速）

目标：把多条独立搜索的推理请求合并成 FP32 大 batch，量化选招反转率与真实提速。

### 实现

- D6 搜索重构为可挂起的生成器：`PolicyValueSearchPolicy._search` 每次产出 `SearchEvaluation`，同步 `choose()` 用本地求值器驱动同一生成器（遍历、剪枝、平局规则、统计口径完全不变，legacy 等价测试与固定面板继续通过），`PolicyValueSearchSession` 供合批运行器驱动。
- MCTS 直接复用现有 `MCTSSearchJob` 的挂起/回填协议，不需要改动搜索实现。
- 新增 `src/splendor/batched_search.py`：`BatchedSearchRunner` 把多条独立搜索的待评估请求按 kind（policy / value）合并为一次 FP32 forward；`max_batch_size` 限制物理 batch，D6 的并发会话池默认 `2 × max_batch_size`（流式补充，避免快搜索先结束时 batch 变小），MCTS 全部 job 并发。
- 新增 `scripts/benchmark_batched_search.py`：固定 seed 从随机对局收集局面，逐条顺序搜索 vs 全池合批搜索，报告选招反转、根值/目标差异、batch 分布与墙钟。

合批只改变张量形状，不改变搜索语义；逻辑位置数（搜索真正需要的评估次数）与顺序路径保持可对照，物理 batch/位置单独统计。

### 结果（CPU 1 线程 / CUDA，全部 FP32；D6 depth=6 top_k=5，MCTS S=256/N=2）

| 算法 | 设备 | 局面数 | batch 上限 | 均值 batch | 选招反转 | 结果差异 | 顺序耗时 | 合批耗时 | 提速 |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| D6 | CPU | 64 | 32 | 8.4 | 0/64 | max\|Δ根值\|=6.3e-7 | 114.2s | 75.6s | 1.51× |
| D6 | CPU | 128 | 32 | 10.7 | 0/128 | 同上量级 | 243.5s | 150.6s | 1.62× |
| D6 | CPU | 64 | 64 | 8.6 | 0/64 | max\|Δ根值\|=6.3e-7，逻辑位置 −0.03% | 107.4s | 67.3s | 1.59× |
| D6 | CUDA | 24 | 32 | 5.1 | 0/24 | max\|Δ根值\|=2.5e-7 | 61.3s | 24.3s | 2.52× |
| MCTS | CPU | 64 | 32 | 28.7 | 0/64 | 目标逐位相同 | 13.0s | 9.6s | 1.36× |
| MCTS | CPU | 128 | 32 | 29.8 | 0/128 | 目标逐位相同 | 32.6s | 18.7s | 1.74× |
| MCTS | CPU | 128 | 64 | 57.8 | 0/128 | 目标逐位相同，逻辑位置完全相同 | 25.2s | 17.6s | 1.44× |
| MCTS | CUDA | 64 | 32 | 28.7 | 0/64 | 目标逐位相同 | 21.2s | 11.4s | 1.86× |

结论：

- **选招反转率**：0 次反转 / 全部合批运行（D6 296 次决策、MCTS 384 次决策，共 680 次；覆盖同一批 128 个局面的多种 batch/设备组合）。按二项分布，0/680 的 95% 置信上界约 0.44%（按 128 个唯一局面计约 2.3%）。反转率没有随 batch 上限 32→64 明显变化。
- **数值差异**：D6 根值最大差 6.3e-7（batch 形状噪声量级），MCTS 的根访问分布完全没变（目标逐位相同）。MCTS 的逻辑位置数逐位一致；D6 有一次配置相差 0.03%（值噪声改变了剪枝路径），最终选招仍然一致。
- **提速**：MCTS 1.4–1.9×（job 结构同步，均值 batch ~29–58，利用率高）；D6 1.5–1.6×（CPU）/ 2.5×（CUDA），均值 batch 只有 5–11——不同搜索的深度进展不同步，且 policy/value 两类请求要分开合批，这是 D6 合批效率的天然上限。作为对照：D6 用 CUDA 合批（1.01s/决策）比现在默认的 CPU 单条（~1.9s/决策）快约 1.9×。
- **风险画像**：D6 batch=64 的 64 个局面里有 20 个根值前二名差距 < 1e-6（多数是完全平分，由 logits 决胜），属于理论上有翻转风险的位置，但本次没有出现反转；MCTS 128 个局面里有 5 个前二 visit 差 ≤1。换句话说，合批不是"逐位一致"，而是"在本次 1024 次决策里没有改变任何选择"。

### 使用方式

```bash
# D6：64 个局面一起合批，batch 上限 32
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u \
  scripts/benchmark_batched_search.py \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --output runtime/search_accel/batched_d6_cpu.json \
  --algorithm d6 --positions 64 --batch-size 32 --repeats 2

# MCTS：S=256/N=2，batch 上限 64，CUDA
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u \
  scripts/benchmark_batched_search.py \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --output runtime/search_accel/batched_mcts_cuda.json \
  --algorithm mcts --positions 128 --batch-size 64 --device cuda \
  --simulations 256 --mcts-determinizations 2
```

库内用法：

```python
from splendor.batched_search import BatchedSearchRunner

runner = BatchedSearchRunner(policy, max_batch_size=32)      # D6 或 MCTS policy 均可
choices = runner.choose_many(policy, requests)               # D6：多条独立决策一起搜
results = runner.search_many(mcts_policy, requests)          # MCTS
print(runner.stats.to_dict())                                # 物理 batch/位置统计
```

当需要"逐位完全一致"时，继续使用默认的同步路径（`policy.choose` / `policy.search`），合批路径只在可接受 ~1e-6 级 FP 噪声、且要批量吞吐时开启。

## 第四轮：GPU 合批的适用范围（哪些搜索能吃到 GPU 红利）

问题：现在跑的实际测试、以及后续 AlphaZero 式迭代学习（搜索产出训练数据），能否都用 GPU 合批加速搜索？

### 每位置推理成本：合批是主收益，GPU 只在 batch≥8–32 后略优

同一 checkpoint、1024 个真实请求、`policy_value`（含编码/传输/取回）中位每位置毫秒：

| batch | CPU 1 线程 | CPU 8 线程 | CUDA |
| ---: | ---: | ---: | ---: |
| 1 | 0.831 | 0.771 | 1.469 |
| 8 | 0.346 | 0.340 | 0.384 |
| 32 | 0.264 | 0.278 | 0.233 |
| 128 | 0.269 | 0.249 | 0.208 |
| 512 | 0.324 | 0.251 | 0.201 |

结论：batch=1 → 32 的合批收益约 3 倍（CPU 与 CUDA 都有）；同一 batch 下 GPU 只比 CPU 快约 1.1–1.25 倍（这个网络很小，GPU 用不满）。CPU 多线程只在大 batch 下提供约 1.3 倍。因此"要不要上 GPU"取决于能否把 batch 堆到 32 以上，而不是取决于设备本身。

### 端到端自博弈采集（AlphaZero 式数据生产）：GPU 已接通，但当前收益有限

16 局、S=256/N=2、fast 引擎、batch 上限 32（新配置 `configs/search_accel/batched_cpu_16games_b32.json` / `batched_cuda_16games_b32.json`，均为单进程协作调度）：

| 后端 | 墙钟 | 仿真/s | 平均 batch | 前向总时长 | 规则总时长 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `batched_cpu` | 141.0s | 1719 | 26.4 | 21.5s | 47.4s | — |
| `batched_cuda` | 132.8s | 1826 | 26.4 | 10.8s | 46.9s | 15.3 MiB |

同一份 16 局、批次分布完全相同（8,633 个物理 batch，22.8 万个逻辑位置）。GPU 把前向时间砍半，但前向只占总墙钟约 15%，规则/调度占大头，端到端只快约 6%。显存不是约束（batch 512 峰值也只有约 12 MiB）。

### 结论（按工作负载）

- **MCTS 自博弈数据生产（AlphaZero 迭代里的搜索阶段）**：可以端到端用 GPU 合批，已经接通（`execution.backend = "batched_cuda"`，`workers=1`）。收益 = (前向占比) × (GPU/CPU 前向比)，当前小网络、batch≈26 时约 1.06×；把并发局数提到 32–64、batch 上限提到 32–64 会更接近 1.1–1.2×。训练与搜索在迭代里交替使用 GPU，不冲突；显存充裕。
- **D6 alpha-beta（现有评测/阶梯/教师）**：合批核心已就绪（`BatchedSearchRunner`），但实测 batch 只有 5–11，恰好落在 GPU 不占优的区间（batch<8 GPU 更慢，8–32 基本持平）。应使用 CPU 合批（1.5–1.6×，见第三轮）或多进程，不要为 D6 单开 GPU 服务。
- **评测/阶梯（多进程 CPU）**：目前跨进程并行已经吃满 CPU；只有把很多对局的决策汇总到同一个 batcher（MCTS 评测能堆到 30+ 请求/轮）时 GPU 才有意义，收益量级同上（约 1.1×，端到端更小）。
- **更大的网络/更大的 GPU**：本机 6GB 卡 + 64/256 宽的小网络是 GPU 收益偏低的主因；若以后扩网络，前向占比和 GPU 优势都会上升，届时 `batched_cuda` 的优先级应提高。

命令：

```bash
# GPU 自博弈采集（16 局、batch 32）
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.selfplay_fast generate \
  --config configs/search_accel/batched_cuda_16games_b32.json \
  --output runtime/search_accel/my_cuda_run

# 对照：同一工作量 CPU 合批
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.selfplay_fast generate \
  --config configs/search_accel/batched_cpu_16games_b32.json \
  --output runtime/search_accel/my_cpu_run

# 每位置成本对照（CPU 1/8 线程、CUDA）
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u \
  scripts/benchmark_large_batch_inference.py \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --requests 1024 --max-batch 512 --repeats 3 \
  --output runtime/search_accel/large_batch_cpu_t8.json --device cpu --threads 8
```

## 第五轮：合批在 16 核机器上的最优形态（多进程 vs 单进程大 batch）

背景：目标配置是搜索端 S=1024/N=1、每轮约 500 局新数据，评测端频繁用 D6 与 S=2048/N=2。

### 单进程合批没有跑赢多进程（16 局、S=256/N=2 同一份数据，1,388 样本 / 242,432 simulations）

| 采集配置 | 墙钟 | simulations/s | 均值 batch |
| --- | ---: | ---: | ---: |
| `batched_cpu` 单进程（active 16、batch 32） | 141.0s | 1,719 | 26.4 |
| `batched_cuda` 单进程（active 16、batch 32） | 132.8s | 1,826 | 26.4 |
| `serial_cpu` 16 进程 × 1 局（batch=1） | 58.2s | 4,164 | 1.0 |
| `batched_cpu` 8 进程 × 每进程 2 局（batch≤4） | **48.1s** | **5,040** | 3.8 |

结论：这个网络很小（前向约占 27% 墙钟），单进程合批把引擎/Python 串行化了；16 核机器上「多进程 + 每进程小批次」比「单进程大 batch」快约 2.9 倍，GPU 单进程合批反而是最慢的一档。四者产出的对局数据一致（同样 5:11 胜负、947 个 policy 样本）。

### 搜索端 S=1024/N=1 试运行（8 进程 × 2 局、batch≤4）

16 局：墙钟 263.3s（16.5s/局），4,030 simulations/s，95.7 万逻辑位置，1,389 样本（1,036 policy 样本），0 异常。按此吞吐外推：

- 500 局 ≈ 500 × 66,304 simulations ÷ 4,030 ≈ **8,200s ≈ 2.3 小时/轮**；把 workers 提到 16、active 提高到 2–4 后还有 1.5–2 倍的余量。
- 对照：单进程合批在同样配置下约 0.50–0.52s/决策 × 约 87 决策/局 ≈ 44s/局 ≈ 6.1 小时/轮。

### 按你的配置实测合批收益（32 个并发局面，固定 seed）

| 配置 | 设备 | 逐条 | 合批 | 提速 | 均值 batch | 选招反转 | 目标差异 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| S=1024/N=1 | CPU | 31.0s | 16.5s | 1.88× | 26.9 | 0/32 | 逐位相同 |
| S=1024/N=1 | CUDA | 49.7s | 15.9s | 3.12× | 26.9 | 0/32 | 逐位相同 |
| S=2048/N=2 | CPU | 47.4s | 33.3s | 1.43× | 53.7 | 0/32 | 逐位相同 |
| S=2048/N=2 | CUDA | 65.9s | 29.6s | 2.23× | 53.7 | 0/32 | 逐位相同 |

注意 CUDA 的 3.12×/2.23× 是「同一块 GPU 上合批 vs 逐条」，不是「GPU vs CPU」：CUDA 逐条本身比 CPU 逐条慢（batch=1 每位置 1.47ms vs 0.83ms），跨设备同 batch 时 CUDA 只比 CPU 快 1.04×（S=1024/N=1）与 1.13×（S=2048/N=2）。

### 建议配置

- **500 局/轮的数据生产（S=1024/N=1）**：`backend=batched_cpu`，`workers≈8–16`，`active_games_per_worker=2–4`，`max_batch_size=4–8`；这是本次实测最快形态，不需要 GPU。示例配置：`configs/search_accel/batched_cpu_16games_s1024_n1.json`。
- **D6 评测**：保持多进程逐条（batch 只有 5–11，GPU 不占优）；若要用合批，仅在同一进程内有多局时才划算。
- **S=2048/N=2 评测**：同样优先多进程 + 小批次；单进程合批只比逐条快 1.43×（CPU）/2.23×（CUDA），而多进程并行在 16 核上有数倍空间。
- **GPU 何时值得上**：CPU 核被别的任务占满；或以后把网络做大（前向占比上升）；或做「多进程 producer → 单个 GPU batcher」的中央推理服务（需要新工程，收益取决于前向占比）。当前小网络 + 16 核 CPU 下，GPU 合批不是最优解。
- 显存不是约束：batch 512 的峰值也只有约 12 MiB。
