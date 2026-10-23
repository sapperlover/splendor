# 采集吞吐与精确输入缓存可行性（2026-10-22）

状态：**小规模早中期窗口筛选完成，缓存未建立结论；完整 4000 局采集未运行，命令留给用户手动执行**。本目录只回答采集吞吐和输入缓存问题，不评估棋力；训练、搜索预算、根噪声、温度和规则均未改变。

## 1. 固定条件

- 每轮 **4000 局**；本轮小实验使用仓库当前 12 贵族规则。
- 混合预算：25% S256N4（每世界 256 次 × 4 世界 = 总 1024），75% S128N1（总 128）。
- FP32；`engine_mode=fast`。
- attention 教师：R20 raw，128×2／4 头／FFN512，583,682 参数。
- MLP 教师：Cross64 第 54,112 步，906,434 参数。
- 只改执行变量：后端、worker/活跃局数、batch、pipeline、microbatch、等待、shared 布局、编译模式。

## 2. 方法

使用现有 [benchmark_collect_throughput.py](../../scripts/benchmark_collect_throughput.py) 的真实采集循环：

- 先预热，再统计固定时间窗内的 simulations/s、网络前向位置/s、平均 batch、owner gather/dispatch、worker CPU；
- 冷启动（进程/编译）和预热后的固定窗口分开记录；
- 60 秒窗口用于主要比较，20 秒窗口只用于初筛；所有候选使用相同搜索预算和随机种子；
- 每个候选记录 `process_seconds`、`window_seconds` 和完整执行配置；窗口内的 owner 计时统一从 `clock.owner_timings` 读取，逻辑位置数从 `clock.totals` 读取，不使用 `gpu_owner.inference` 的累计值。

这里所谓“60 秒稳态”只是**预热完成后的固定负载窗口**，不是完整 round 的稳态：窗口内 `completed_games=0`，没有完整对局、没有尾段。按 R20 完整轮的 353.24 simulations/decision 和 66.79 decisions/game 估算，60 秒窗口覆盖约：attention base 9.2 decisions/game（约 14%）、attention active2000 5.1（约 8%）、MLP base 15.3（约 23%）、MLP active2000 10.7（约 16%）。因此这些数字只代表**早中期吞吐**，不能替代完整 4000 局。

性能实验期间 GPU 曾达到 **89°C、SM 1470/3105 MHz**，出现热节流；60 秒重复值存在约 5–10% 波动。因此小于约 5% 的 attention 差异不应当作确定结论，最终以用户手动跑的完整 4000 局为准。

两个最终配置都用 4 局 smoke 验证过 `python -m splendor.selfplay_fast generate` 调用链：各完成 4/4 局，`exceptions=0`、`engineering_truncations=0`。4 局时 active 被压到 4，batch 很小，只证明命令和产物流转可用，不代表吞吐。

## 3. attention 128×2 采集

### 60 秒早中期窗口

| 配置 | 12 worker / active | B | pipeline | microbatch | sim/s | 相对基线 |
|---|---:|---:|---:|---:|---:|---:|
| 基线（R20 执行配置） | 1000 | 500 | 2 | 64 | **53,889** | — |
| microbatch 128 | 1000 | 500 | 2 | 128 | 54,650 | +1.4% |
| active 2000 + mb128 | 2000 | 500 | 2 | 128 | 60,174 / 54,109 | 均值 **57,142**，约 +6% |
| active 2000 + mb256 | 2000 | 500 | 2 | 256 | 56,243 / 52,565 | 均值 54,404，约 +1% |

20 秒初筛也支持 active=2000、microbatch=128/256 的方向（约 63k sim/s），但短窗口明显高估早中期吞吐；60 秒值更保守，仍然不是完整轮。

### 结论

建议优先验证：

```text
12 search workers / total_active_games=2000 / B500 / pipeline_slots=2
local_microbatch=128 / max_batch_wait_ms=5 / shared_slots=2 / shared_candidate_width=256
compiled + native-cudagraph / fail_fast / FP32
```

相对基线的 60 秒窗口提升估计约 **3–6%**，但受热节流影响，必须先跑完整 4000 局 A/B 才能确认。`pipeline_slots=1`（约 50.8k）、`shared_slots=1`（约 38.6k）、`microbatch=32`（约 38.3k）均明显更差；`B256`（约 58.2k）也不如 B500。

参考的完整 R20 第 30 轮实际采集（4000 局）：**1704.4 秒、55,366 sim/s、2.347 局/秒、140.8 局/分钟**，平均 batch 480.8；host forward 48.75 秒（仅主机提交时间，不是 GPU 计算时间）、H2D 146.78 秒、wait 767.15 秒。这是当前 attention 基线的真实全量尺度。

## 4. 最新 Cross64 MLP 采集

### 60 秒早中期窗口

| 配置 | 12 worker / active | B | pipeline | microbatch | sim/s | 相对基线 |
|---|---:|---:|---:|---:|---:|---:|
| 基线（R20 执行配置） | 1000 | 500 | 2 | 64 | **90,074** | — |
| active 2000 + mb128 | 2000 | 500 | 2 | 128 | 118,270 | +31.3% |
| active 2000 + mb256 | 2000 | 500 | 2 | 256 | **126,353** | **+40.3%** |
| 16 worker + active 2000 + mb256 | 2000 | 500 | 2 | 256 | 124,032 | +37.7% |

20 秒初筛同样显示 mb256/active2000 最高（约 140.7k），但 60 秒值更保守，且仍只覆盖早中期。active 3000／4000 反而降到约 122.9k／114.9k；active 500 只有约 71.7k。`pipeline_slots=1`（82.5k）、`shared_slots=1`（46.2k）、`microbatch=32`（45.2k）均明显更差。

### 结论

建议 MLP 优先验证：

```text
12 search workers / total_active_games=2000 / B500 / pipeline_slots=2
local_microbatch=256 / max_batch_wait_ms=5 / shared_slots=2 / shared_candidate_width=256
compiled + native-cudagraph / fail_fast / FP32
```

按 4 局 smoke 的 21,184 次模拟/局估算，4000 局约 84.7M 次模拟；以 126.4k sim/s 计算约 **671 秒（11.2 分钟）采集**，加冷编译和尾段后预计约 **11–13 分钟**。这是估算值，需完整 4000 局确认。

## 5. CPU 与 GPU

| 模型 / 后端 | 配置 | 窗口 sim/s |
|---|---|---:|
| attention / `native_cpu` | 8 worker × 4 active，eager，B4 | 5,059 |
| attention / `native_cpu` | 12 worker × 4 active，eager，B4 | 4,663 |
| attention / GPU pool | 12 / 1000，compiled | 53,889–58,952 |
| MLP / `native_cpu` | 8 worker × 4 active，eager，B4 | 6,799 |
| MLP / `native_cpu` | 12 worker × 4 active，eager，B4 | 6,196 |
| MLP / `native_cpu` | 16 worker × 4 active，eager，B4 | 6,885 |
| MLP / GPU pool | 12 / 2000，mb256，compiled | 126,353 |

说明：CPU 行为 10 秒窗口，GPU 行为 60 秒窗口，不能当作严格同窗口排名，只作数量级比较。

结论：**当前 MLP 在采集路径上并不比 attention 慢；GPU 也明显快于 CPU**。CPU 增加 worker 没有带来有效扩展，说明它受单线程搜索/进程调度限制，而不是 GPU 搬运反向拖累。此前 [2026-10-01 推理吞吐报告](../../docs/reports/inference_throughput_zh-CN.md) 的 CPU eager 约 16.4k sim/s 是更小模型、不同搜索预算下的历史结果，不能直接和本组绝对值比较。

## 6. 缓存：已按正确口径重测，仍无实际 A/B

详细结果见 [cache_recheck_20261022](../cache_recheck_20261022/README.md)。

- 当前混合预算命中率：S128N1 ready-hit **4.905%**，S256N4 **16.291%**；按完整轮模拟量加权（高预算 72.87%）得到 **13.20% ready / 15.33% potential**。旧 15% 未包含 S128N1，已作废；这里仍是每次搜索空缓存、不含跨局。
- 生产 CUDAGraph B500 计时（`NativePackedTransfer._direct_graph`，已重测）：attention graph **13.77 µs/行**、MLP **3.97 µs/行**；attention 完整轮 graph 占墙钟约 **72.4%**，60 秒 tuned 约 **81.4%**，MLP 60 秒 tuned 约 **50.3%**。
- 若每条 ready-hit 完全跳过生产 CUDAGraph replay 且无任何 key/合并成本：attention 完整轮约 **9.6%**、tuned 约 **10.8%**，MLP 约 **6.6%** 墙钟。这是乐观上界，不是实际加速。
- 生产 native 路径仍没有缓存实现，也没有 cache off/on；Python key 约 8.83 µs/行（约 0.46 CPU 核当量，不能换算墙钟），实际 C++ key、合并、部分命中和 H2D/gather 影响未测。

上一版“forward 2.86% / 最多省 0.43%”的推算已撤回：`forward_seconds` 是主机提交计时，不是 GPU 计算；汇总混用累计值和窗口值的问题也已修正为 `clock.owner_timings` + `clock.totals`。目前不能判断缓存一定提速，也不能判断它一定不值；需要生产路径 cache off/on 60 秒窗口和完整 4000 局 A/B。

## 7. 编译启动与长轮次

- attention 首次冷进程约 **387 秒**（含编译），同一配置热进程约 **59–60 秒**；MLP 冷进程约 **360 秒**，热进程约 **58–60 秒**。
- eager 没有编译，但窗口内明显更慢：attention 约 41.9k、MLP 约 78.4k sim/s；4000 局的长轮次里 compiled 会摊薄冷编译并胜出。
- 因此不要把“极小批 + 极短窗口”当成最终配置排名；本目录的 60 秒值只覆盖早中期，仍需完整 4000 局复核。

## 8. 完整 4000 局手动验证命令

以下命令只采集，不训练、不评测棋力；输出目录必须是新的。已经存在的目录用 `--resume` 恢复。

### attention 基线（A/B 对照）

```bash
cd /home/lc/下载/splendor
/usr/bin/time -f 'external_wall_s=%e' \
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u -m splendor.selfplay_fast generate \
  --config artifacts/collection_tuning_20261022/configs/full_attention_base_4000.json \
  --output runtime/collection_tuning_20261022/full_attention_base_4000
```

### attention 候选（active2000 + mb128）

```bash
cd /home/lc/下载/splendor
/usr/bin/time -f 'external_wall_s=%e' \
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u -m splendor.selfplay_fast generate \
  --config artifacts/collection_tuning_20261022/configs/full_attention_best_4000.json \
  --output runtime/collection_tuning_20261022/full_attention_best_4000
```

### MLP 基线

```bash
cd /home/lc/下载/splendor
/usr/bin/time -f 'external_wall_s=%e' \
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u -m splendor.selfplay_fast generate \
  --config artifacts/collection_tuning_20261022/configs/full_mlp_base_4000.json \
  --output runtime/collection_tuning_20261022/full_mlp_base_4000
```

### MLP 候选（active2000 + mb256）

```bash
cd /home/lc/下载/splendor
/usr/bin/time -f 'external_wall_s=%e' \
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u -m splendor.selfplay_fast generate \
  --config artifacts/collection_tuning_20261022/configs/full_mlp_best_4000.json \
  --output runtime/collection_tuning_20261022/full_mlp_best_4000
```

运行后看输出目录的 `status.json` / `summary.json`：`wall_seconds` 包含本进程采集/编译；`/usr/bin/time` 的 `external_wall_s` 还包含启动。完整 4000 局约产生 230–240 MiB NPZ，本机当前约 7 GiB 可用空间，建议每组跑完核对后删除临时 NPZ，只保留 `status.json`、`summary.json` 和日志。

## 9. 产物

- 全部原始窗口结果：[measurements/](measurements/)
- 机器可读汇总：[summary.json](results/summary.json)
- 环境与模型身份：[environment.json](environment.json)
- 缓存 key 成本：[key_cost.json](results/key_cost.json)
- 仅 Python 路径的缓存微基准（非生产收益）：[cache_overhead_attention.json](results/cache_overhead_attention.json)、[cache_overhead_mlp.json](results/cache_overhead_mlp.json)
- 完整 4000 局配置：[configs/](configs/)
- 复现脚本：[scripts/](scripts/)

本次小实验没有修改生产训练/采集代码，没有启动训练或正式棋力评测；完整 4000 局由用户手动执行。

## 10. 参数覆盖：不是全参数扫描

本次是可解释的**单变量/少量组合筛选**，不是全因子扫描，也不是全局最优搜索。

已实际变化的执行参数：

- 后端：`native_cuda_pool`、`native_cpu`；
- `search_workers`：8／12／16（CPU 行为 `workers`）；
- `total_active_games`：500／1000／2000（MLP 另测 3000／4000）；
- `max_batch_size`：256／500（attention 两者，MLP 只测 500）；
- `batch_buckets`：R20 八档、`[64,128,256]`（仅 attention）；
- `pipeline_slots`：1／2；
- `shared_slots`：1／2；
- `shared_candidate_width`：64／256；
- `local_microbatch`：32／64／128／256；
- `max_batch_wait_ms`：2／5／10；
- `inference_mode`：compiled + native-cudagraph、eager；
- CPU worker 数：8／12／16，batch4、active4/worker、eager。

未测或只测了单一值的项目：

- `candidate_buckets` 只用了 `[16,32,48,64]`，未做宽度档位消融；
- `compile_mode` 只测 native-cudagraph 与 eager，未测 `default`／`reduce-overhead` 等；
- `native_cuda` 单 owner、`batched_cuda`、`serial_cpu` 等后端未系统比较；
- MLP 的 B256、batch buckets、shared 布局交互未测；
- `shared_slots=4`、`shared_candidate_width=128`、`local_microbatch=384/500` 未测；
- 搜索预算、根噪声、温度、forced playouts、policy target pruning 按任务要求固定，未作为变量；
- 只用了 R20 raw 与 MLP 54,112 两个 checkpoint，未测其他 seed／模型；
- 大多数配置只跑了一次 60 秒窗口，没有随机化顺序、重复和置信区间；GPU 热节流使 5–10% 的差异不可靠；
- **没有完整 4000 局**，没有终局/尾段/完整对局的 games/min。

因此“较优配置”只能看作待完整 4000 局验证的候选，不是全局最优。

## 11. 为什么 MLP 只快约 2 倍：算力与瓶颈

用 `torch.utils.flop_counter` 在相同真实请求上数一次 policy/value 前向：

| 模型 | FLOPs/行 | 平均候选数 |
|---|---:|---:|
| attention 128×2 | **25.25 M** | 9.45 |
| Cross64 MLP | **4.34 M** | 9.45 |

MLP 的 FLOPs 约为 attention 的 **1/5.82**，接近你说的 1/6。

但实测：

| 指标 | attention | MLP | 比值 |
|---|---:|---:|---:|
| B500 graph 宽度加权时间（生产 CUDAGraph） | 13.77 µs/行 | 3.97 µs/行 | 3.47× |
| 60 秒窗口 sim/s（各自最佳） | 60.4k | 126.8k | 2.10× |
| GPU graph 占墙钟（生产 CUDAGraph，tuned） | 81.4% | 50.3% | — |
| CPU 核数 / 16 | 2.95（18%） | 6.75（42%） | — |

**重要口径**：上表的 graph 时间和 GPU 占比已改用生产 `NativePackedTransfer._direct_graph` 的 CUDAGraph replay 计时（B500）；不是直接 `compiled._dispatch`。FLOPs、60 秒 sim/s、CPU 秒和 owner gather/dispatch 是直接测量。

所以 MLP **目前既没有跑满 GPU，也没有跑满 CPU**：

- MLP 的生产 CUDAGraph graph 只占约 **50.3%** 墙钟；attention tuned 约 **81.4%**，更接近 GPU 受限；
- MLP owner 在 60 秒窗口内：gather **31.8 秒（53%）**、GPU graph **30.2 秒（50.3%）**、dispatch 2.4 秒、H2D 6.85 秒；worker CPU 约 5.7 核；
- 单看生产 CUDAGraph，MLP 只比 attention 快 3.47 倍，而不是 5.8 倍：MLP 的矩阵更小、算术强度更低，kernel/launch/内存开销占比更高；
- 端到端只快约 2.1 倍，因为 GPU forward 只是整条流水线的一段。MLP 的 gather/dispatch、worker 搜索供给、共享槽和合批等待没有按 FLOPs 同比例缩小，GPU 经常等待数据或没有被填满。

因此当前 MLP 的瓶颈更像 **owner 侧 gather/dispatch 与 worker 供给/流水线协调**，不是模型 FLOPs，也不是 GPU 或 CPU 总算力。继续把 active 从 2000 提到 3000/4000，或 worker 从 12 提到 16，实测反而变慢，说明增加并发主要在加重 owner/内存/调度竞争，而不是填满 GPU。

要真正接近 GPU 饱和，需要降低 owner 每行的复制/调度成本（例如更少的共享布局转换、批处理下放、C++/融合 gather），或让 worker 侧缓存/复用结果；单纯缩小模型不会把端到端加速从 2× 推到 6×。

限制：GPU 占比是 graph dispatch 时间的宽度加权投影，不是 `nvidia-smi` 的逐时刻 SM 占用；CPU 核数来自进程 CPU 秒；60 秒窗口只覆盖早中期，完整 4000 局可能改变比例。

FLOPs 原始记录：[flops_attention.json](results/flops_attention.json)、[flops_mlp.json](results/flops_mlp.json)。

### 11.1 易调参数的实测结果

在 MLP 最佳配置附近做了几组低成本的 20 秒筛选；有希望的再用 60 秒复核：

| 配置 | 窗口 | sim/s | 结论 |
|---|---:|---:|---|
| 12 worker / active2000 / mb256（基线） | 60 s | 126,353 | 当前最好 |
| 14 worker / active2000 / mb256 | 60 s | 107,275 | 20 秒曾到 142k，60 秒没有复现，反而更差；不能推荐 |
| 16 worker / active2000 / mb256 | 60 s | 124,032 | 略差 |
| 20 秒基线 | 20 s | 132,439 | 仅筛选参考 |
| shared_slots=4 | 20 s | 114,052 | 更差 |
| shared_candidate_width=64 | 20 s | 118,094 | 更差 |
| local_microbatch=384 | 20 s | 113,235 | 更差 |
| active=1500 | 20 s | 126,156 | 更差 |
| active=2500 | 20 s | 132,244 | 与基线同量级，无增益 |
| wait=2 ms | 20 s | 134,502 | 约 +1.6%，但只有 20 秒，未在 60 秒确认 |
| owner 按 candidate width 分组（临时小改） | 60 s | 121,384 | 没有收益，已还原源码 |

结论：**目前没有已确认的、只改现有配置就能明显提高 GPU/CPU 利用率的选项**。12 worker / active2000 / mb256 仍是已测局部最优；继续加 worker、active、shared slots 或 microbatch 都变差。

比较值得再试的低成本项只剩：

- `max_batch_wait_ms=2` 的 60 秒复核（20 秒约 +1.6%，噪声级）；
- OS 层把 owner/worker 做 CPU affinity 绑定，减少 12 worker + owner 的迁移/竞争（本次未测）；
- 更细的 `candidate_buckets` 或变长候选计算（本次没有证明宽度分组有效，需先 profile 候选 padding 的真实成本）。

真正可能明显提高 GPU 利用率的是 **owner 侧 gather/dispatch 的实现**（融合复制、减少共享布局转换、C++ 化）或更深的流水线；这些不是改一个现有开关就能完成，且 `pipeline_slots` 当前代码上限为 2。
