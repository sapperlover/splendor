# 原生推理吞吐优化（有效合批、投影复用、编译推理、整批输入输出；含 07/08 修正与 08 全量重测）

本文对应 `prompt/06_native_inference_throughput_optimization_zh-CN.md`。改动集中在冻结推理路径与 native 自对弈调度；
游戏规则、MCTS 公式、S/N 语义、训练损失、网络结构与权重、精度（FP32）都没有改变。

## 当前结论（2026-09-29 08 全量重测；本节是唯一可用于决策的结论）

**方法**：单 GPU owner、同一 checkpoint（`local_005000000_lifetime_005512112.pt`，SHA-256 前缀 `009d17d2…`）、
S=1024/N=1、FP32、`native` 整批 + 状态投影 + compiled + `native_transport=reused`、`pipeline_slots=1`。
每档先 2 s 真实预热并显式编译候选宽度档位与 value-only 图，之后由父进程统一开窗 35 s；窗口边界在各 worker 内
采样 `resource.getrusage` 与计数器增量。GPU 档每档 1–3 个窗口，原始 JSON 见
`artifacts/inference_throughput/08/`（含 `summary_08_2026-09-29.json`）。

### 0. 传输分项复测（2026-09-30）

修正后的 `profile_inference_pipeline.py` 分开测 CPU 打包、pageable packed 输入的 H2D 完成时间、forward、设备内融合输出复制与 D2H 完成时间；replay 脱离搜索，`full_seconds` 是另一次 packed NumPy 路径的完整调用，分项独立测量且不应简单相加。捕获来自同一冻结 checkpoint、FP32、S=1024/N=1：旧形状 64/64 在 40 s 内取 96 个真实 64 行 batch，生产形状 500/500 在 30 s 内取 24 个真实 500 行 batch；两者候选宽度均为 32，生产捕获实际达到 500 活跃局。replay 每组重复 5 次；合成传输基准每档重复 20 次。均未完成整局，故不报告 games/min。

| compiled replay 形状 | CPU 打包 | H2D 完成 | forward | 设备输出复制 | D2H 完成 | packed 完整调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 行（64 档捕获） | 0.113 ms | 0.168 ms | 0.393 ms | 0.041 ms | 0.018 ms | 0.861 ms |
| 256 行（64 档重组） | 0.387 ms | 0.536 ms | 0.473 ms | 0.068 ms | 0.026 ms | 1.833 ms |
| 500 行（500 档捕获） | 0.647 ms | 0.853 ms | 0.748 ms | 0.174 ms | 0.031 ms | 2.849 ms |
| 1000 行（500 档重组） | 1.244 ms | 1.676 ms | 1.326 ms | 0.311 ms | 0.042 ms | 4.800 ms |

数据：`artifacts/inference_throughput/09/profile_cuda64_transfer_2026-09-30.json` 和 `profile_cuda500_transfer_2026-09-30.json`；未列出的 eager 与更大重组档也在原始 JSON。独立合成传输基准 `09/transfer_modes_cuda_2026-09-30.json` 对 64/256/500 行直接测得 pinned H2D 完成约 0.090/0.362/0.663 ms，融合 D2H 完成约 0.008/0.033/0.013 ms，三档回传内容核对均通过。合成路径与真实 replay 的 pageable packed 路径、生产 reused 的两块 pinned 输入不同，不将三者混为同一耗时。旧 06 报告的“H2D”实际混入打包，原始文件保留作历史记录。

新数据说明打包、H2D 和 forward 在 500 行档为同一量级，D2H 明显较小；仅凭旧的“再往上 H2D 主导”结论不成立。这是推理微基准，不能单独推断完整采集的 CPU/GPU 资源占比。

### 1. 端到端档位（单 owner，正确输入；稳态窗口形状）

> 这些窗口给每档喂了 400–700 局，512 档确实能长期维持 512 活跃；生产每轮只有 500 局，
> 形状不同，见 1.1 的完整对局对照。

| active/batch | 窗口样本（sim/s） | 均值 | CPU µs/模拟 |
| --- | ---: | ---: | ---: |
| 128/128 | 43,822 | 43,822 | 23.0 |
| 256/256 | 56,443 / 52,434 / 53,811 | 54,229 | 17.8–19.2 |
| 512/512 | 58,873 / 58,062 | **58,468** | **17.1–17.3** |

**512/512 是当前最佳已测配置**：两次窗口均高于 256 的三次窗口，样本区间不重叠；128 最低（43.8k）。
256 档单次波动达 7%（52.4k–56.4k），因此"256 与 512 同档"的说法只在错误输入下成立，已作废。

### 1.1 生产形状：500 局/轮整轮对照（完整对局，2026-09-30）

生产 AlphaZero 每轮 500 局，**活跃局数与 batch 都被 500 封顶**（每个 world 同时最多一个未返回叶节点），
所以 `active/batch=512` 永远填不满——512/512 的窗口优势不能直接搬到生产轮次。按完整对局（走生产路径：
自对弈 + 每局校验 + 写档）实测三轮：

| 配置 | 整轮墙钟 | 局/分钟 | 平均 batch | 整轮 sim/s | 峰值在途 | GPU 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256/256（原生产配置） | 926 s | 32.4 | 210 | 35,710 | 256 | 47 MiB |
| **500/500** | **802 s** | **37.4** | 391 | 41,198 | 500 | 82 MiB |
| 500/256 | 822 s | 36.5 | 220 | 40,202 | 256 | 49 MiB |

- active 从 256 提到 500 让整轮快 **13.3%**（926 s → 802 s）：500 局跑一波而不是两波，平均 batch 210 → 391。
- 500/500 与 500/256 只差 2.5%（单轮波动内）：想省 buffer 就选 500/256，收益基本相同。
- 整轮 sim/s（35.7k–41.2k）明显低于稳态窗口（54.2k–58.5k）：尾段 + 逐局校验/写档，窗口数字会高估生产效率；
  训练轮次请以完整对局的 games/min 为准。
- 结论：500 局/轮请用 **active=500**，`max_batch_size` 取 500（或 256）；不要配置 512 以上。
  原始数据见 `artifacts/inference_throughput/08/round500_compare.json`。

这三轮的旧 `status.json` 未保存逐 worker 编译统计，无法事后确认是否触发 fallback；墙钟和 batch 观测仍是原始记录。
若需把整轮排名作为“均保持 compiled 模式”的严格对照，须在新版本下重跑三档并检查 `worker_compile`。

### 2. 缓冲复用对照（256/256，slots=1）

| 传输 | sim/s | CPU µs/模拟 |
| --- | ---: | ---: |
| `packed`（改前：concatenate + 每次新分配） | 40,403 | 25.0 |
| `reused`（修复后，均值） | 54,229 | 17.8–19.2 |

同窗口单次对比为 56,443 vs 40,403（**+40%**），均值口径 **+34%**；CPU µs/模拟下降约 27%。
每轮实际 H2D 是 **2 次**（float blob + mask blob），输出是设备内 2 次复制 + 1 次融合 D2H；文档不再声称"一次 H2D"。

### 3. CPU 采集（native_cpu，8 workers × 4 活跃局，batch 4，35 s）

| 推理 | sim/s | CPU µs/模拟 | 平均占用核数 |
| --- | ---: | ---: | ---: |
| eager | 16,433 | 469.5 | 7.71 / 8 |
| compiled | 14,423 | 533.0 | 7.69 / 8 |

→ CPU 多 worker 下 compiled 明显亏损（吞吐 -12%，CPU µs/模拟 **+13.5%**）。06 的方向性结论在新口径下复现，
CPU 采集推荐 eager。

### 4. 双 slot（256/128，reused）

| slots | sim/s（两次窗口） | 均值 |
| --- | ---: | ---: |
| 1 | 48,317 / 49,809 | 49,063 |
| 2 | 45,302 / 48,936 | 47,119 |

→ **无可重复收益**（-4%）。重叠本身是真实的（D2H 等待 0.29 s→0.07 s，峰值在途 256 vs 128 行），但当前实现每轮排空，
且主机 CPU 是限制资源；默认保持 `pipeline_slots=1`，持续流水线形态留作条件项。

### 5. 参考量

- 人工 evaluator（瞬间固定 policy/value，仅 CPU 路径）：256/256 为 70,019 sim/s、14.3 CPU µs/模拟；
  该窗口有 1,531 个超出候选档位的批次，工作负载与真实网络不同，只能作参考，不是"真实搜索上限"。
- 传输微基准（合成 payload，CUDA event/完成同步，内容核对通过）：pinned 比 pageable 的 H2D 完成延迟快
  1.19×–1.65×（pinned 9.1–10.4 GB/s，pageable 5.6–8.7 GB/s；64 行档波动最大），融合 D2H 约 10–15 µs。
  旧的"5–9 倍 / 47–70 GB/s"来自把 CPU 缓冲当 CUDA 目标，作废。

### 6. 瓶颈与下一步

生产轮次形状（500 局/轮）下，建议 active=500（见 1.1）：它把整轮从 926 s 压到 802–822 s，收益来自避免"两波"
结构，而不是把 batch 做大到 512。以下主机侧分解仍来自稳态窗口：

256/256 窗口内主机 CPU 约占满 1 个核（35.16 CPU 秒 / 35.0 s）：原生编码 fill 6.3 s（18%）、submit + 搜索推进
12.4 s（35%）、compiled forward 主机侧 5.2 s（15%）、H2D+D2H 0.8 s（2%）、GPU 等待 0.2 s，其余约 10.2 s（29%）
为 pending 扫描、批组装与统计等调度工作。

→ 限制是**主机 CPU 供给**，不是 forward 内核或搬运（GPU 有明显空闲）。最值得做的下一步是减少逐批主机工作
（合并 pending 扫描与批组装、减少逐批 Python 对象、必要时把一轮推进合并成原生批量调用）；
多核 CPU 供给一个 GPU owner、以及不排空的持续双 slot 流水线仍是条件项，暂不实施。

### 7. 仍未验证（不作结论）

长时间稳定性与热漂移；其他 checkpoint / 网络规模；>512 并发；官方训练负载下的端到端；CPU 其他 worker 配置。
双 slot 持续流水线与多核供给本轮未实施，方案与启动条件见第 10 节。

### 8. 主机优化尝试（08 §8.1；结论：无可测收益，停止微调）

只动 08 授权的三类低风险热点（逐行对象、逐 job 往返、视图缓存），先做 15 s live 画像再改代码：

改前热点（256/256，CPU 15.2 s / 15 s 窗口）：compiled forward 主机侧 3.6 s（24%）、C++ 提交+搜索推进约 4.3 s（28%）、
C++ 原生编码 2.8 s（18%）、Python fill 胶水 2.3 s（15%）、plan 与逐 job 扫描 0.8 s（5%）、
H2D/D2H 主机侧 0.4 s（2%）、`start_job`+`encode_decision` 0.6 s（4%），GPU 等待 0.3 s。

本轮实施并保留：

- `NativeBatch.row_ids_array()` + 失败时才展开可读行：正常路径不再为每 batch 构造 256 个 Python 元组/dataclass，
  fill 侧胶水从 2.3 s/15 s 降到约 0.2 s/15 s；行级/整批故障定位仍走同一身份数组。
- reused transfer 缓存 packed 布局与按 `rows` 的设备视图：`_device_batch` 从 0.28 s/15 s 降到约 0.01 s/15 s，
  并且 dynamo 看到稳定的张量对象。
- 试行后回退：把每 job 的 `pending_rows`/`pending_width`/`root_pending` 合并成"冻结 plan"实测反而
  +0.12 s/15 s，且引入跨 phase 状态不变式，已恢复原来的实时扫描调度（对应测试一并移除）。

35 s 窗口改前/改后（同协议、同配置、同 checkpoint）：

| 配置 | 改前（08 基线窗口） | 改后（本轮窗口） | CPU µs/模拟 |
| --- | --- | --- | --- |
| 256/256 | 56,443 / 52,434 / 53,811（均值 54,229） | 54,451 / 53,827 / 52,615 | 17.8–19.2 → 18.6–19.3 |
| 512/512 | 58,873 / 58,062（均值 58,468） | 58,584 / 58,028 | 17.1–17.3 → 17.3–17.5 |

**结论：无可测收益**——差异全部落在本机 ±3% 重复波动内，每模拟 CPU 持平。主机侧 Python 胶水不是限制杠杆；
主要成本仍是 C++ 搜索推进与原生编码（约 46%）和 compiled forward 主机路径（约 24%）。按 08 "没有明显收益就
如实停止"，本轮就此停止微调；保留的两项只减少分配与对象 churn，不改变语义，且已通过全量测试与真实 GPU
逐字节复核（多行/尾批/宽度切换/value-only/双 slot，reused vs packed 逐位一致，compiled ≤3.8e-6）。

### 9. 下一步方案（本轮只交付方案，不实施）

1. **持续流水线（不排空）**：当前**不满足启动条件**。正确输入下 GPU 等待只有 0.6%（256/256）～3.4%（512/512），
   而 host CPU 全程约 100% 忙；再增加重叠无法提高吞吐。启动条件应由 `_live_snapshot` 判定为
   "wait 占窗口比例明显（例如 >10%）或 GPU 实际成为限制"。条件满足后的做法是：完成一个 slot 立即提交、
   推进对应游戏并补下一批，保持有界队列与"每 world 最多一个未返回叶节点"，验收点是 wait 占比下降、sim/s 提升、
   且同 seed 结果与同步路径逐位一致。
2. **多核 CPU 供给一个 GPU owner**（下一步最值得做的一项）：证据是 C++ 搜索推进（28%）+ 原生编码（18%）+ Python
   调度都串行在单核上，单核饱和而 GPU 空闲约 75%–90%。方案：N 个独立搜索执行单元（进程优先）各自持有游戏集合与
   请求映射，经有界队列把 batch 交给唯一 GPU owner，保留 per-game 所有权、model_version 与一次性消费检查。
   预期瓶颈：批组装/队列同步、GIL（纯 C++ 计算可释放）、模型版本切换安全。验收点：同 seed 每局结果与单 worker
   一致、峰值 batch 不下降、sim/s 随 worker 数近似线性直到 GPU 或内存上限。
3. **若限制转移到真实 forward**（GPU 利用率接近饱和）：再评估更大 batch、CUDA graph 等 torch 内选项；
   本轮不引入量化、TensorRT 或模型结构改动。

### 10. 失效结论清单（全仓）

| 旧结论 | 处置 | 当前正确说法 |
| --- | --- | --- |
| "active/batch=256 最优、512 回落"（06） | 作废 | 512/512 是当前最佳已测（58.5k vs 54.2k） |
| 07 全部性能数字与结论 | 数值作废（输入布局错误） | 本节的 08 重测；方向（reused 更快、双 slot 无收益、主机受限）在新数据下重新成立 |
| "pinned H2D 快 5–9 倍 / 47–70 GB/s"（06） | 作废（测错设备） | 1.19–1.65×，9–10 GB/s |
| "compiled 使 CPU 时间 +13%–35%"（06/07 曾降级为待验证） | 复测确认 | CPU 上 compiled 吞吐 -12%、CPU µs/模拟 +13.5% |
| "fake NN 是真实搜索严格上限"（06/07） | 措辞作废 | 人工 evaluator 参考值 70.0k sim/s，候选分布不同 |
| "35% 搜索 + 42% GPU + 23% live"（06） | 作废（不同负载相加） | 真实窗口分解见第 6 节 |
| "8 局 / 8 workers 墙钟 1.91×"（native_search_backend 文档） | 历史有效但不代表当前版本 | 该表是 06 之前的 8 局批=1 测量，仅存档 |


## 08 正确性修复记录与数据来源复查（2026-09-29）

07 的 CUDA `native_transport=reused` 在多行 batch 中按连续字段读取按行存放的特征和 mask；第二行起可能读到错误局面。
现已改为按 `float_stride` / `mask_stride` 的行视图读取，真实 CUDA 多行逐字段和 mask、冻结模型 logits/value、value-only、满批→尾批→满批、宽度切换及双 slot 小搜索均有回归覆盖。
07 的吞吐、`reused` 比 `packed` 收益、256/512 排名、双 slot 收益与主机瓶颈归因都产自错误输入，全部作废；旧 JSON 原样保留。packed 的旧分项 `forward_seconds` 计时点也有偏差。08 已补跑最小重测基线（见 08.1）：`reused` 在正确输入下仍明显快于 `packed`，256/512 排名与 CPU 供给结论需要用新窗口重新表述。

当前 CUDA 配置中的 256/256、单 owner、`pipeline_slots=1` 是 08.1 重测中的最佳已测配置（128/512 未在新布局下复测）。每个 batch 的 float 与 mask 分别执行一次 H2D；融合输出仍含设备内复制和一次 D2H。`feature_seconds` 在 packed 路径包含 CPU 打包与同步输入准备，`forward_seconds` 为主机墙钟调用耗时，`native_consume_seconds` 可能包含 GPU 等待，不能将它们作为互斥 CPU 时间相加，也不能直接视作 GPU kernel 时间。

逐 manifest 复查：仅 `runtime/native_search/07_batch_smoke`（2 局 S=8 冒烟）带有 `native_transport=reused`，确实是错误布局产物；`runtime/native_search/selfplay_cuda_native_truth_smoke`、`runtime/inference_throughput/cuda_batch_smoke`、`runtime/inference_throughput/e2e_cuda_prod`、`runtime/inference_throughput/e2e_cuda_docshape` 的 manifest 既无 `native_transport` 也无 `versions.native_gpu_layout`，属于 07 之前的 packed 传输，不受行 stride 缺陷影响（仍属旧代码版本，不作为本次基线）。未发现由任何 CUDA 采样目录派生的 learner checkpoint；历史文件一律保留。新 CUDA reused manifest 的 `versions.native_gpu_layout=row-stride-v2` 与旧产物区分，避免同配置续跑混合。

（完整重测见上文“当前结论”一节；本节只保留正确性修复与数据来源复查。）

---

> **以下为 06/07 历史记录。所有性能数字、排名与瓶颈归因均产自错误输入布局或旧计时口径，一律不得引用；
> 当前有效结论见顶部"当前结论"一节。**

## 附录 A：07 修正与复测（历史）

本文原为 06 工作包的记录；`prompt/07_inference_correctness_measurement_and_pipeline_zh-CN.md` 检查出的五类问题
已在代码中修复并复测。06 的原始 JSON 全部保留在 `artifacts/inference_throughput/`，但其中受影响的结论已作废或降级（见 0.6）。

### 0.1 修复的五个问题（代码 + 测试）

1. **C++ 批量输出生命周期**：`NativeBatch::submit` 曾把 `forcecast` 产生的临时 logits 数组放在 `if` 作用域内，
   指针在数组析构后仍被逐行消费；现在 owning 数组覆盖整个提交过程，新增 `submit_fused` 一次提交 logits+value，
   `owners_` 改为 `shared_ptr<SearchJob>` 显式持有 job。行数、候选宽度、有效位置 finite、generation 与可消费
   状态改为“全部先行校验、再统一应用”，坏批量不再留下半提交现场（`tests/test_native_batch.py`）。
2. **实时模拟计数**：新增统一快照 `_live_snapshot`（`selfplay_fast.py`）：完成对局累计 + 活跃 policy 的
   `live_simulations()`；后者把尚未回收的根搜索中已完成模拟（`job.core.stats()["simulations"]`）计入，
   根完成回收时只并入一次。网络位置按实际评估行计数并区分 root / policy / value-only 行
   （`tests/test_search_accel.py::test_live_simulation_count_includes_unfinished_roots`）。
3. **传输基准**：`benchmark_transfer_modes.py` 现在显式创建 CUDA 目标并断言 device/dtype/连续性/pinned，
   分别测 CPU pack、pageable/pinned H2D（提交延迟、完成延迟与 CUDA event 设备时间）、forward 与融合 D2H，
   D2H 做内容核对，字节按实际 float32 packed 布局计算。
4. **批量故障归属**：批量导出保留 batch/row/world/kind/generation 与 game 映射；行级异常定位真实游戏并写
   `failures/game-*.json`（含 `batch_world`/`simulation`/`job_id`/`model_version`）；无法归属单行的形状、设备或
   编译错误写 `failures/batch-*.json` 并列出该 batch 涉及的全部游戏，不再记到第一局。`validate_run` 单独统计
   `batch_failures`。
5. **编译失败回退**：`compile_failure_policy`（`fallback` 默认 / `fail_fast`）只捕获可识别的编译器与后端异常
   （`TorchDynamoException`、`InductorError`），覆盖构造与首次执行两个阶段；失败图进入禁用状态且不再重试，
   记录失败阶段、原因、图/宽度、`compile_failure_fallbacks`、`disabled_graphs` 与 `effective_mode`。
   OOM、非法数据与模型错误不会被吞掉。

### 0.2 历史窗口（07；输入布局错误，数字全部作废）

方法：单 GPU owner，S=1024/N=1，FP32，`native` 整批 + 投影复用 + compiled + 缓冲复用，同一 checkpoint；
每个 worker 先 2 s 真实预热，再显式编译配置中的候选宽度档位与 value-only 图；全部 worker 报到后由父进程统一
开窗 40 s，窗口结束时在各 worker 内采样 CPU 与计数器增量（`resource.getrusage(RUSAGE_SELF)`，不含初始化与预热）。

| active/batch | 窗口 1 | 窗口 2 | 均值 | CPU µs/模拟 |
| --- | ---: | ---: | ---: | ---: |
| 128/128 | 43,861 | — | 43,861 | 23.0 |
| 256/256 | 49,894 | 49,030 | **49,462** | 20.1–22.1 |
| 512/512 | 48,668 | 47,138 | 47,447 | 20.7–21.5 |

上述 07 表格与“256/512 不可区分”的判读均产自错误输入布局，**全部作废**；08 重测见顶部"当前结论"
（512/512 58.5k 为最佳已测，256/256 54.2k，128/128 43.8k）。07 的 CPU 16,329 sim/s 在 08 重测为 16,433 sim/s。

### 0.3 历史复用对照（收益数字已失效）

按 `(slot, state_only, 宽度档位)` 复用：native 编码直接写入 pinned host staging（C++ 以外部数组为存储，
owner 由 `NativeBatch` 持有），一次 H2D 搬进复用 device buffer；logits 与 value 融合成一次 D2H
（行内最后一列为 value），再由 `submit_fused` 一次提交。与“改前（逐批 `np.concatenate` + 新分配 host/device）”对照
（同 256/256、同窗口口径）：

| 传输 | sim/s | CPU µs/模拟 |
| --- | ---: | ---: |
| packed（改前：concatenate + 每次新分配） | 39,266 | 25.6 |
| reused（本轮：pinned staging + 复用 device/output） | 49,462（均值） | 20.1–22.1 |

旧窗口曾报告 **+26% sim/s**，现因 reused 输入错位而作废。实际输入为 float 与 mask 两次 H2D，输出为一次融合 D2H；
native 编码不再额外拼一份 host buffer。

### 0.4 历史双批流水线窗口（07；已由 08 重测）

`pipeline_slots=2`（active=256、max_batch=128）07 窗口为 slots=1 42.1k vs slots=2 41.7k sim/s；该批输入布局错误，数字作废。
08 在正确输入下重测：slots=1 49,063（48,317/49,809）vs slots=2 47,119（45,302/48,936）→ 仍无可重复收益（-4%），
重叠真实（D2H 等待 0.29 s→0.07 s，峰值在途 256 vs 128 行）。**保持 `pipeline_slots=1` 为默认**，双 slot 作为验证过的
可选实现保留（正确性：`tests/test_search_accel.py::test_two_slot_pipeline_matches_sync_with_controlled_evaluator`）。

40 s 窗口的主机侧分解（256/256，单 worker）：

| 项目 | 秒 | 说明 |
| --- | ---: | --- |
| CPU 总计 | 40.15 | = 1.00 核全程忙 |
| native 编码 fill | 7.50 | 搜索叶写入 staging |
| submit + 推进（含 C++ 搜索） | 13.88 | `buffer.submit` + `advance` + `initialize_worlds` |
| compiled forward 主机侧 | 7.20 | dynamo guard 与 kernel 提交，不含 GPU 执行 |
| H2D + D2H 主机侧 | 0.86 | 异步提交 |
| 等待 GPU | 0.83 | 墙钟（非 CPU）；512 档为 3.1 s/40 s |
| 其余（调度、快照、组装） | ~10.7 | pending 规划、统计采样等 |

→ 当前限制是**主机 CPU 供给**（搜索推进 + 原生编码 + 逐批 forward 主机开销），既不是 forward 内核也不是搬运；
GPU 窗口内有明显空闲。

### 0.5 传输微基准（07 重写；08 复跑）

07 重写后的方法仍是当前方法。08 复跑结果：pinned 比 pageable 的 H2D 完成延迟快 1.19×–1.65×
（pinned 9.1–10.4 GB/s，pageable 5.6–8.7 GB/s；64 行档波动最大），融合 D2H 约 10–15 µs，内容核对通过。
旧的“5～9 倍 / 47–70 GB/s”不成立；pinned 的价值主要在异步与可复用的 staging，而非数量级带宽优势。

### 0.6 旧结论处置

| 06 结论 | 处置 |
| --- | --- |
| “active/batch=256 已是最优、512 回落” | 07 判读作废（错误输入）；08 判定 512/512 为最佳已测 |
| “pinned H2D 快 5～9 倍” | 作废：原脚本把 CPU 缓冲当成了 CUDA 目标 |
| “compiled 使 CPU 时间 +13%～35%” | 方向复测确认：08 在正确口径下测得 CPU 吞吐 -12%、CPU µs/模拟 +13.5%；CPU 档保留 eager 为默认 |
| “fake NN 是真实搜索严格上限” | 降级为人工 evaluator 的吞吐参考：固定 policy/value 会改变树形与局面分布 |
| “35% 搜索 + 42% GPU + 23% live” | 作废：来自不同工作负载相加，不是同一条真实时间线；见顶部"当前结论"第 6 节 |
| `_tensor_batch_native()` 全耗时当作 H2D | 已改：该函数含 CPU 打包；新基准分别报告 pack / H2D / forward / D2H |

### 0.7 07 复现命令

```bash
# 三档 active 扫描（单 owner，40 s 公共窗口，2 s 搜索预热 + 显式编译预热）
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_collect_throughput.py \
  --config configs/native_search/throughput_cuda_128.json --games-per-worker 400 --seconds 40 --warmup-seconds 2 \
  --output runtime/inference_throughput/07/sync_cuda_128.json --scratch /tmp/bench07
# 256/512 同上（分别用 throughput_cuda_256.json / throughput_cuda_512.json，512 需要 --games-per-worker 700）

# 缓冲复用对照（packed = 改前传输）
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_collect_throughput.py \
  --config configs/native_search/throughput_cuda_256_packed.json --games-per-worker 400 --seconds 40 \
  --warmup-seconds 2 --output runtime/inference_throughput/07/packed_cuda_256.json --scratch /tmp/bench07

# 双批流水线：slots=1 与 slots=2（同 active=256、batch=128）
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_collect_throughput.py \
  --config configs/native_search/throughput_cuda_256_slots1_128.json --games-per-worker 400 --seconds 40 \
  --warmup-seconds 2 --output runtime/inference_throughput/07/slots1_cuda_256_b128.json --scratch /tmp/bench07
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_collect_throughput.py \
  --config configs/native_search/throughput_cuda_256_pipe2.json --games-per-worker 400 --seconds 40 \
  --warmup-seconds 2 --output runtime/inference_throughput/07/slots2_cuda_256_b128.json --scratch /tmp/bench07

# 传输微基准（真 CUDA 目标 + 内容核对）
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_transfer_modes.py \
  --rows 64,256,512 --repeats 20 --output runtime/inference_throughput/07/transfer_modes_cuda.json
```

原始数据见 `runtime/inference_throughput/07/`（含 `summary_07.json`）。该目录产自错误输入布局，
**只作本地审计保留、不入 Git**；08 的有效基线在 `artifacts/inference_throughput/08/`，其汇总为
`summary_08_2026-09-29.json`。06 交付物中仍入库的只有 `summary_2026-09-29.json`、`end_to_end_2026-09-29.json`、
`pure_forward_{cpu,cuda}.json` 与 `profile_2026-09-29.json`，其余 06 中间分片同样移到
`runtime/inference_throughput/06_stage_logs/`。

## 附录 B1：06 结论摘要（历史，性能数字全部不可引用）

- 之前"8 局 / 8 workers"的主对照实际只有 batch=1。本次把并发度作为测量前提：CPU 8 workers × 4 活跃局 = batch 4、
  GPU 单 owner × 32/64 活跃局 = batch 32/64，全部实测达到目标 batch。
- 逐叶节点的 Python 字段对象与 float/tuple 往返已从生产采集路径移除：搜索叶节点直接由 C++ 写入连续 FP32 buffer，
  推理结果整批回填。2 局同 seed 冒烟对局与逐叶参考路径**动作 trace 完全相同**。
- 策略头状态投影复用（同一局面只算一次状态投影）在 CPU 上降低约 5.7% CPU 时间/模拟，在 CUDA 上中性。
- ~~`torch.compile` 在 CPU 多 worker 采集下亏损（CPU 时间/模拟 +13%~+35%）~~ **08 复测确认**：CPU 8×4 下
  compiled 吞吐 -12%、CPU µs/模拟 +13.5%（eager 16,433 sim/s / 469.5 µs，compiled 14,423 sim/s / 533.0 µs）；
  CPU 档默认 eager，CUDA 档默认 compiled。
- 端到端（S=1024/N=1，45 s 稳定窗口，同一 checkpoint）：

| 采集配置 | 基线 sim/s（eager/逐叶） | 推荐配置 sim/s | 加速 |
| --- | ---: | ---: | ---: |
| CPU 8 workers × 4 活跃局 | 13,130 | 15,829（整批 + 投影 + eager） | 1.21× |
| GPU 1 owner × 32 活跃局 | 12,479 | 18,984（整批 + 投影 + compiled） | 1.52× |
| GPU 1 owner × 64 活跃局 | 14,742 | 31,555（整批 + 投影 + compiled） | 2.14× |

原始数据见 `artifacts/inference_throughput/summary_2026-09-29.json` 及各阶段 JSON。

## 2. 改动内容

### 2.1 策略头状态投影复用（`actor_critic.py`、`search_inference.py`）

- `ActorCritic.build_actor_projection()` 把 actor 首层权重按 `state_hidden | action_representation` 切成
  `W_state / W_action` 两个连续副本（尺寸取自 `self.config`，未硬编码 256/132/388）。
- `_actor_logits_projected()` 计算 `ReLU((W_state·state + b) + W_action·action_i)`：先相加再 ReLU，末层与 bias 不变。
- 投影缓存绑定在 `FrozenSearchInference` 实例上（每个冻结模型一份），换 checkpoint 会重建，不写回 `state_dict`。
- `InferenceConfig.actor_projection` 是显式开关；关闭时走原实现，作为数值参考。

### 2.2 编译纯张量 forward（`search_inference.py`）

- `CompiledForward` 只接收张量，返回 logits/value；policy/value 与 value-only 分开建图。
- 按候选宽度档位建图（默认 `(16, 32, 48)`，batch 维标记为 dynamic），宽度超过最宽档位时按受控回退走 eager，
  不无限建图；padding 行由 `candidate_mask` 正确屏蔽，不进入搜索统计。
- 记录每个图的首次编译耗时、调用次数、回退次数与 dynamo 计数（`compile_seconds / compiled_calls / eager_fallbacks / dynamo_delta`）。
- `InferenceConfig.mode="compiled"` 显式启用，失败或更慢时保持 eager 并如实报告。

### 2.3 原生整批输入输出（`native_search.cpp`、`native_policy.py`、`selfplay_fast.py`）

- 新增 `NativeBatch`：按字段预分配连续 buffer（global/market×3/reserved×2/nobles/pending/candidate + mask），
  `fill(job)` 直接把某个搜索 job 的待评估叶节点编码写进 batch 槽位，`arrays()` 返回零拷贝 NumPy 视图，
  `submit(logits, values)` 整批回填并推进各世界。
- 每个叶节点行在导出时被一次性认领（`exported` 标记），提交时校验 `(world, simulation)` generation：
  重复导出返回 0 行，stale/重复结果直接报错而不是写进新叶节点；容量满时 `fill` 安全停止。
- `SearchJob` 新增 `pending_rows/pending_width/root_pending_now` 与 `submit_exported`；value-only（transition cap）行走独立的
  state-only buffer。
- `NativeBatchRunner` 按（policy 行、value 行）收集各 job 的句柄，按 `max_batch_size` 分块，保持与逐叶路径相同的行顺序；
  根行提交后由 Python 调用一次 `initialize_worlds`。
- `FrozenSearchInference.policy_value_native/value_only_native` 直接把 buffer 变成张量（CPU 零拷贝视图；CUDA 单次打包 H2D），
  不再为每个叶节点构造 `EncodedDecision`，也不再逐元素转 Python float。

### 2.4 配置与元数据

`execution` 新增字段（旧配置不写这些字段时行为与之前完全一致）：

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `batch_interface` | `legacy` \| `native` | 逐叶适配（参考）或原生整批接口（生产） |
| `inference_mode` | `eager` \| `compiled` | 是否使用编译图 |
| `actor_projection` | bool | 是否复用状态投影 |
| `candidate_buckets` | `[16,32,48]` | 编译档位（同时是 native buffer 的宽度档位） |
| `compile_mode` | string \| null | 传给 `torch.compile` 的 mode |

- `native` 只允许与 native 后端组合；`selfplay_fast` 的 run manifest 新增 `inference` 与 `batch_interface` 版本标签，
  配置整体也在语义指纹里，恢复不会静默切换执行模式。
- 固定评测的 policy spec 支持可选 `inference` 字段（同样默认关闭，保持历史评测口径）。

## 3. 正确性验证

- 原生整批行与 Python 参考编码逐字段一致（含 candidate padding 与 mask 零值），见 `tests/test_native_batch.py`。
- 同一 job、同一组受控网络输出下，整批提交与逐叶提交的选招、根访问计数、policy target 完全一致；
  覆盖 N=2 与 transition cap（value-only）两种形态。
- 端到端：同一 seed、同一 S/N 的两局自对弈，`batch_interface=native` 与 `legacy` 的真实动作 trace 完全相同。
- 投影/编译：固定真实局面上 eager / 投影 / compiled 的 logits 最大差约 3.8e-6、value 差 ≤6e-8、argmax 一致
  （`tests/test_inference_modes.py`）；宽度超出档位时走 eager 回退且结果一致。
- 失败路径：非有限 value 被拒绝；批量行重复导出返回 0；已消费 buffer 再提交报错。
- 冒烟：native 整批 2 局自对弈在 CPU（eager + 投影）与 CUDA（compiled + 投影）两条配置下都通过 `validate --replay`；
  微型 AlphaZero 循环（采集 + Python learner + native 评测 + greedy 评测 + 断点）跑通；固定评测（policy spec 带
  compiled 推理）1 pair 通过。CUDA 冒烟首次决策含约 6 s 图编译，稳定窗口不计入该成本。

## 附录 B4：06 性能测量（历史；输入、口径与代码版本均已失效）

三层口径：纯 forward（`scripts/benchmark_inference_modes.py`）、真实采集稳定窗口（`scripts/benchmark_collect_throughput.py`）、
完整对局冒烟。所有采集数字来自 45 s 稳定窗口的实际增量计数（未完成对局不计入训练样本）。

### 4.1 纯 forward（固定真实局面，不含编码）

CPU（1 线程，中位数，相对 eager）：

| batch | projected | compiled | compiled+projected |
| ---: | ---: | ---: | ---: |
| 1 | 1.06× | 1.18× | 1.29× |
| 8 | 1.19× | 1.15× | 1.47× |
| 32 | 1.29× | 1.02× | 1.72× |
| 64 | 1.47× | 1.12× | 1.82× |

CUDA（含 H2D/D2H，相对 eager）：

| batch | projected | compiled | compiled+projected |
| ---: | ---: | ---: | ---: |
| 1 | 0.98× | 1.55× | 1.57× |
| 32 | 0.98× | 1.34× | 1.30× |
| 64 | 0.98× | 1.27× | 1.29× |

### 4.2 真实采集（S=1024/N=1，45 s 窗口）

CPU 8 workers × 4 活跃局（batch 4，数字为两次测量之一）：

| 阶段 | sim/s | CPU µs/模拟 | forward s/s |
| --- | ---: | ---: | ---: |
| 基线 eager 逐叶 | 13,059 | 656.6 | 5.37 |
| 投影复用 | 13,675 | 619.3 | 5.11 |
| compiled + 投影 | 13,021 | 768.0 | 4.86 |
| 整批 + 投影 + eager | **15,829** | **545.9** | 5.91 |
| 整批 + 投影 + compiled | 13,399 | 739.4 | 5.26 |

GPU 32 / 64 活跃局：

| 阶段 | GPU32 sim/s | GPU64 sim/s | GPU64 CPU µs/模拟 |
| --- | ---: | ---: | ---: |
| 基线 eager 逐叶 | 12,479 | 14,742 | – |
| 投影复用 | 11,013 | 13,226 | 78.6 |
| compiled + 投影 | 9,335 | 12,973 | 89.7 |
| 整批 + 投影 + eager | 17,625 | 27,898 | 36.9 |
| 整批 + 投影 + compiled | **18,984** | **31,555** | **35.7** |

要点：

- 纯 forward 的 compiled 优势在 CPU 采集下**不成立**：8 个单线程 worker 并发时，inductor 产物比 ATen 内核更慢，
  CPU 时间/模拟反而上升；CPU 推荐 eager。
- GPU 上投影单用中性甚至略负（多一次小 GEMM 的调度开销），但整批接口把每批固定开销摊掉后 compiled 稳定获胜。
- 主要收益来自整批接口：删掉逐叶 Python 对象、逐叶 submit、重复的 collate/pack；在同样记录了 CPU 口径的阶段里，
  GPU64 的 CPU 时间/模拟从（投影 + 逐叶）78.6 µs 降到（整批 + compiled）35.7 µs。基线 CUDA 两次窗口早于 CPU 口径计数，
  未记录该列。
- 45 s 窗口内完整对局数为 0，因此未从窗口外推"局/分钟"；完整对局速度以 2 局冒烟（CPU，S=8）与微型循环为准。

### 4.3 与 `native_search_backend_zh-CN.md` 的对照（完整对局）

该文档的性能表测得较早，其中 "8 局 / 8 workers" 一档实际只有 batch=1（同样的问题在本次 prompt 中被点名）。
为了可比，这里按**完全相同的形状**（同 8 局、同 workers/active/batch、同 S=1024/N=1、同 checkpoint）重跑，并额外跑生产形状的完整对局：

| 对比档 | 文档（改前） | 现在 | 墙钟加速 |
| --- | --- | --- | ---: |
| native CPU，8 workers，8 局 | 97.62 s，4.92 局/分，5,633 模拟/s，batch 1.00 | 90.61 s，5.30 局/分，6,069 模拟/s，batch 1.00 | 1.08× |
| native CUDA，1 owner，8 局 | 114.05 s，4.21 局/分，4,821 模拟/s，batch 6.76 | 70.29 s，6.83 局/分，7,459 模拟/s，batch 6.33 | 1.62× |
| native CPU，8 workers × 4 活跃局，32 局 | 该档文档未测（同并发下改前代码窗口 = 13,130 模拟/s） | 129.24 s，**14.86 局/分**，16,385 模拟/s，batch 3.57 | 3.02×（对文档 4.92 局/分） |
| native CUDA，1 owner × 64 活跃局，64 局 | 该档文档未测（同并发下改前代码窗口 = 14,742 模拟/s） | 137.40 s，**27.95 局/分**，30,660 模拟/s，batch 52.22 | 6.64×（对文档 4.21 局/分） |

拆解（避免把"改并发"算成"改代码"）：

- 同形状档的收益只来自本次改动（逐叶对象消失 + 投影复用；GPU 还含 compiled）：CPU 1.08×、CUDA 1.62×。
- 生产 CPU 档的 3.02× 里，约 2.3× 来自把每 worker 1 局提升到 4 局活跃（batch 1 → 3.57），约 1.25× 来自本次推理/整批改动。
- 生产 GPU 档的 6.64× 里，约 3.06× 来自 8 → 64 活跃局，约 2.08× 来自本次改动。
- 端到端完整对局数据见 `artifacts/inference_throughput/end_to_end_2026-09-29.json`。

### 4.4 完整 profile：搜索侧上限、推理 replay、batch 上限

三个额外基准（`artifacts/inference_throughput/profile_2026-09-29.json`）：

1. **fake NN（瞬间返回固定 policy/value）**：人工 evaluator 的吞吐参考（非真实搜索严格上限：固定 policy/value 会改变
   树形与局面分布，且跳过真实推理与搬运）。
2. **真实 batch replay**：从真实采集窗口里抓取真实 batch buffer，脱离 MCTS 单独回放，测 GPU 推理吞吐与 batch 曲线。
3. **active/batch 扫描**：64 → 128 → 256 → 512。

搜索侧上限（fake NN，S=1024/N=1，45 s 窗口）：

| 配置 | fake NN sim/s | 真实 sim/s | 真实/上限 |
| --- | ---: | ---: | ---: |
| CPU 8 workers × 4 活跃局 | 90,403 | 15,829 | 17.5% |
| GPU owner × 64 活跃局 | 93,455 | 31,555 | 33.8% |
| GPU owner × 128 活跃局 | 97,123 | 38,499 | 39.6% |

→ 搜索/传输侧本身能跑 ~9–9.7 万 sim/s，真实运行只用到三分之一；**继续优化 C++ 搜索收益有限，瓶颈在 GPU 推理往返**。

GPU 推理 replay（真实 batch，脱离 MCTS，compiled 路径；**历史计时口径待重测**）：

下面旧表的“H2D”实际包含 CPU 字段打包，D2H 占比曾用残差推算，均不能作传输分解。
采集脚本还固定只喂 64 局，因此这组 replay 不代表 256/500 活跃局的生产形状；修复后的脚本改为使用配置中的 `num_games`，并直接计量 pack、H2D、设备输出复制和 D2H。

| replay batch（行） | 输入准备（含打包） | forward | 整条往返 | forward 位置/s | 往返位置/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.41 ms | 0.37 ms | 0.84 ms | 174k | 77k |
| 256 | 0.95 ms | 0.39 ms | 1.39 ms | 655k | 184k |
| 1024 | 3.26 ms | 1.24 ms | 4.30 ms | 828k | 238k |
| 3072 | 8.06 ms | 4.94 ms | 12.7 ms | 635k | 241k |

→ 此历史 replay 的 forward 与往返数字只对应 64 局采集的记录；“再往上 H2D 主导”和“77k 行/s 为生产上限”均不能由旧分项数据推出，需按修复后的阶段计时重测。

batch 上限与扫描（真实网络，compiled + 整批，单 owner）：

| active = batch | sim/s | 轮/秒 | µs/轮 | RSS | 相对 64 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 31,555 | 504 | 1,985 | 1.36 GB | 1.00× |
| 128 | 38,499 | 307 | 3,262 | 1.59 GB | 1.22× |
| 256 | **45,610** | 185 | 5,394 | 1.94 GB | **1.45×** |
| 512 | 39,159 | 99 | 10,069 | 2.88 GB | 1.24× |

- `batch=64` 不是硬件或算法常数：在 N=1、且"每个世界同时只有一个待评估叶节点"的契约下，
  单轮 batch 上限 = 该 owner 的活跃局数。之前 64 只是 prompt 里指定的测量点，batch 直方图在 64 处堆积（49,001 个 batch 恰好 64 行）证明它在撞上限。
- ~~提高并发局数确实继续受益，直到 256 左右为本机最优点~~；**08 全量重测（正确输入）**：128/128 43,822、
  256/256 54,229（三次均值）、**512/512 58,468（最佳已测）** sim/s；“512 回落”确认不成立，旧判读作废。
- ~~轮时间分解（batch=64）… 35% 搜索/调度 + 42% GPU 往返 + 23% live 适配开销~~ **作废**：该分解把 fake NN、replay 与
  live 三个不同工作负载相加；正确输入下的真实窗口分解见顶部"当前结论"第 6 节。
- ~~因此下一步的真实收益在搜索与推理重叠（双缓冲/在途 batch）与提高活跃局数~~ **08 复测并作废该推断**：
  256/128 下 slots=1 49.1k vs slots=2 47.1k sim/s（无可重复收益），GPU 有明显空闲，瓶颈是主机 CPU 供给；
  详见顶部"当前结论"第 4/6 节。

## 5. 配置状态

- CPU：`configs/native_search/alphazero_loop_cpu.json`（native CPU、8 workers × 4 活跃局、batch 4、eager + 投影）
  08 重测 16,433 sim/s、469.5 CPU µs/模拟；compiled 在该形状下亏损，不作为 CPU 默认。
- CUDA：`configs/native_search/alphazero_loop_cuda.json` 已按生产轮次形状（500 局/轮）改为
  单 owner、**active/batch=500/500**、`pipeline_slots=1`、`native_transport=reused`；完整对局实测 802 s / 37.4 局/分钟，
  比原 256/256 的 926 s / 32.4 局/分钟快 13.3%。若想保留更小的 batch buffer，500/256 在噪声内等价（822 s / 36.5 局/分钟）。
  稳态窗口下 512/512 仍是最高窗口吞吐（58.5k sim/s），但 500 局/轮时 batch 封顶 500，512 配置没有意义。
- 双 slot（`pipeline_slots=2`）无可重复收益（47.1k vs 49.1k），不作为默认；持续流水线形态仍是后续候选。
- 旧 06/07 的性能数字与瓶颈归因全部作废，见顶部"当前结论"与"失效结论清单"。

## 6. 可复制命令

```bash
conda run -n splendor python -m pip install -e . --no-deps

# 纯 forward 对照（CPU / CUDA）
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_inference_modes.py \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --output artifacts/inference_throughput/pure_forward_cpu.json --device cpu
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_inference_modes.py \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --output artifacts/inference_throughput/pure_forward_cuda.json --device cuda

# 真实采集限时窗口（CPU 8x4 / GPU 64），可选阶段开关
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_collect_throughput.py \
  --config configs/native_search/throughput_cpu_8x4.json --output /tmp/cpu.json --seconds 45 \
  --scratch runtime/inference_throughput/scratch_cpu
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_collect_throughput.py \
  --config configs/native_search/throughput_cuda_64.json --output /tmp/cuda.json --seconds 45 \
  --games-per-worker 96 --scratch runtime/inference_throughput/scratch_cuda

# 冒烟与恢复（native 整批路径）
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast generate \
  --config configs/native_search/selfplay_cpu_smoke.json --output runtime/native_search/batch_smoke
conda run -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast validate \
  --input runtime/native_search/batch_smoke --replay
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.alphazero_loop \
  --config configs/native_search/alphazero_loop_smoke.json --output runtime/native_search/batch_loop_smoke
```

## 7. 未做与限制

- 未实现同一搜索树的多叶并行、virtual loss、异步在途 batch 或双缓冲（prompt 明确排除）。批量路径不保留 cancel 钩子：
  worker 停止或异常时直接丢弃会话，归档只写完整对局，未完成对局不进入训练样本。
- CPU 未使用 compiled：多 worker 实测亏损，保留为可选项并记录回退统计。
- GPU 档全部在一台笔记本 RTX 4050 Laptop 6 GB 上测得，存在显示占用与热降频；
  `cuda_32` 的 compiled 逐叶阶段比 eager 慢的绝对值可能受此影响，但整批 + compiled 的排序在 32/64 两档一致。
- 逐位复现不承诺：不同执行模式会带来 1e-6 量级的 FP 差异，接近平票时可能改变选招；manifest 已记录执行路径身份。
