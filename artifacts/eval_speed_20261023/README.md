# 500pair 搜索评测提速与三组正式结果（2026-10-23）

## 当前采用A与测速口径修订（2026-10-23）

用户已明确采用A作为后续日常及在线新搜索评测的可用默认，无需寻找最优配置。正式模板见[搜索A](../../configs/entity_attention_v1/evaluation_search_a_r43.json)，采集／greedy／日常及在线搜索的完整对照见[评测指南](../../docs/guides/evaluation_zh-CN.md)；原始三组结果与配置不改写。在线变更记录见[独立配置修订](../entity_attention_r31_10k_p60k_iteration_20261023/evaluation_a_update_20261023.json)。

下文保留最初完成汇总，其中速度归因按本段收窄：

- 留存探针为50 pair／100局完整评测，实际并发上限100；结果中`benchmark=null`。尚无1000活跃负载下预热后60秒窗口的配置矩阵，不能称已完成稳态最优筛选。
- A与C对应不同候选和热状态；中止的单槽arm也未给出完整墙钟或同时间窗模拟量。撤回“p2必要／p1不进入可用吞吐”的强判断；A仅是已完整运行的可用配置。
- 原始`execution_stats.models[SHA].inference.compile.compile_seconds`确有逐图编译计时：三组累计约16.73／32.91／32.62秒；旧R37面板约526.39秒。下文“没有记录逐图compile秒数”不正确，不能把42分钟到33分钟的墙钟差归因于流水线或桶设置。
- `root_start_seconds`是逐局、逐根启动搜索的累计时间，不是固定冷启动；双槽下部分计时可重叠，coalesce占墙钟比例不等于GPU空闲比例。高温／低频已记录，缺少throttle reason时间序列，不使用“全程热节流”作为已确认事实。
- 两组44k中止单槽arm已有455局完成，其终局摘要与对应双槽结果逐局一致；该核对支持已有执行结果的一致性，不替代完整速度A/B或所有桶形状的证明。

## 最初完成汇总（保留历史表述；速度解释以修订为准）

## 范围

三组正式面板均为：

- 500 pair / 1000 局，同一开局 seed 交换先后手；
- 双方均为 **S1024N1**（总预算 1024）；
- 候选 vs 古老 **R43**（`runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt`，SHA `b11f12c2…84f14`）；
- 12 贵族当前目录，无根噪声、temperature 0、FP/PTP 关闭；
- seed `37100000–37100499`，policy seed base `37110000`；
- 每组全新输出目录，不复用 pair/结果缓存。

## 三组正式结果

| 候选 | checkpoint | 得分 | 95% 整 pair 区间 | 胜/和/负 | 墙钟 |
|---|---|---:|---:|---:|---:|
| 80k 池，94695 步 | `runs/entity_attention_128x2_offline_80k_20261021/checkpoints/step-000094695.pt` | **74.050%** | [71.200%,76.750%] | 736/9/255 | **2001.5 s（33.4 min）** |
| 44k 池，59453 步 | `runs/entity_attention_128x2_offline_r21_r31_20261022/checkpoints/step-000059453.pt` | **70.700%** | [67.800%,73.600%] | 703/8/289 | **2049.0 s（34.2 min）** |
| 44k 池，降 LR 累计 63453 步 | `runs/entity_attention_128x2_r21_r31_lr2e5_20261022/checkpoints/step-000004000.pt` | **72.200%** | [69.350%,74.950%] | 717/10/273 | **2068.0 s（34.5 min）** |

三组均 `passed=true`、1000 有效局，异常/非法动作/工程截断/不变量失败均为 0。80k 面板有 1 次 forced skip 和 1 局确定性循环判和，不影响 `passed`。

原始结果：[80k](results/evaluation_80k_94695.json)、[59453](results/evaluation_44k_59453.json)、[63453](results/evaluation_44k_63453.json)。

## 速度结论

### 已完成配置

| 配置 | 模型 | pipeline | batch buckets | local microbatch | 墙钟 | mean batch | batches | captures | coalesce | H2D |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **A** | 80k/94695 | 2 | [64,128,256,500] | 128 | **2001.5 s** | 303.0 | 213,231 | **48** | 701.2 s | 174.2 s |
| **C** | 44k/59453 | 2 | 原 8 档 | 64 | 2049.0 s | 303.4 | 212,613 | 100 | 664.8 s | 187.3 s |
| **C** | 44k/63453 | 2 | 原 8 档 | 64 | 2068.0 s | 294.3 | 219,164 | 100 | 675.3 s | 189.8 s |

**本 session 可复现的较优配置是 A**：p2、精简 batch buckets `[64,128,256,500]`、candidate `[32,48,64]`、`local_microbatch=128`。它的墙钟最低、capture 最少（48），H2D 最低；但三组候选不同，不能把 2001 vs 2049 s 当成严格的配置因果差。

### p1 未能复现

历史 p1/B8/mb64 的 500pair 面板曾约 31 min。但本 session 的 p1 arm 都显著更慢：

- p1 + slim + mb64：30 min 仅完成 39/500 pair，已停止；
- p1 + 原 B8 + mb64：34 min 仅完成 12/500 pair，已停止。

因此我没有把 p1 当作“较优配置”来交付，也没有用这两个未完成 arm 的结果做棋力结论。p1 为何在当前双模型 eval 下明显退化，尚未隔离；可能是无重叠与当前双模型/热节流的交互，不能仅凭历史 31 min 宣称 p1 更优。

### 固定开销与稳态

- 固定部分：`root_start` 约 46–49 s，模型创建约 1 s，进程/conda 约 5 s；CUDAGraph capture 次数 48–100，但 evaluator 没有记录逐图 compile 秒数，不能把 capture 次数直接换成编译时间。
- 变量部分主导：coalesce 约 665–701 s，gather 274–282 s，dispatch 119–123 s；mean batch 只有 294–303，约 212k–219k 个 batch。
- GPU 5 秒监控：中位利用率 **69–73%**，中位温度 **85–86°C**，中位 SM **1665 MHz**，峰值 **88–89°C / 2505 MHz**；**全程存在热节流**。
- 整体行吞吐约 **31.2k–32.3k rows/s**；历史 p1 面板约 34k rows/s，但不可直接比较（模型、热状态、未复现）。

## 结论

1. 本次最可复现的提速方向是 **双槽 + 精简 B/candidate buckets + mb128**，但收益相对历史 p1 没有确认；p2 相比本 session 的 p1 arm 是必要的，因为 p1 arm 不进入可用吞吐。
2. 双模型分流仍是主要结构瓶颈：mean batch 只有约 300，coalesce 占 33–35% 墙钟；p2 重叠了约 179 s，但被 batch 碎片化和热节流抵消。
3. 三组正式搜索结果均已保留；80k/94695 对 R43 得分最高，但这是对 R43 的搜索面板，不是与另两个候选的直接互打，也不是此前 greedy 对 R20/R31 的分数。

## 产物

- 机器可读汇总：[results/summary.json](results/summary.json)
- 三组原始结果：[results/](results/)
- 配置：[configs/](configs/)
- GPU 温度/时钟：[results/full_80k_94695_optA.gpu.csv](results/full_80k_94695_optA.gpu.csv)、[59453](results/full_44k_59453_optC.gpu.csv)、[63453](results/full_44k_63453_lr2e5_optC.gpu.csv)
- 运行 sidecar 保留在 `runtime/eval_speed_20261023/results/`；未复制到 artifact。
