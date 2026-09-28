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
