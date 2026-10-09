# 采集、learner 与固定评测：分项时间统计

测量起始日期：2026-10-02，整理日期：2026-10-03。任务范围是统计和归因，未实施速度优化，未启动正式长训练或完整 500/1,000 局复测。代码基点 `f213e6378559322d38ac5a3845118de1816950e2`；新增诊断脚本未提交，各 JSON 保存执行命令、脚本 SHA、模型身份、种子、配置及环境。入口为 [统计产物](../../artifacts/speed_profile_2026-10-02) 和 [诊断脚本](../../scripts/profile_speed_pipeline.py)。

2026-10-03 后续说明：本报告的“未实施/未执行”限于以上历史诊断阶段。生产等待/gather/64档、大cache配置和跨局批量评测已经落地；两组各1000局与50 pairs完整对照及8+128步learner结果见[端到端验收报告](end_to_end_speed_optimization_zh-CN.md)。历史JSON及SHA保留，不用后续结果重写原始测量。

## 计时方法与适用范围

- 历史完整 AZ 五轮和完整 pool 500 局重新统计，保留来源 SHA，不用当前版本补填历史版本字段。
- learner：既有首轮/第五轮 replay，8 步预热＋24 步测量，batch=256。第五轮延续 θ4 的实际 AdamW 状态，使用本轮原采样种子；所有更新和诊断 checkpoint 都在新 runtime，旧训练记录不变。
- learner 以互斥 host wall 分项为主要占比；`train_step` 包含子项，必须先扣除才能相加。`load_game` 是读取、ZIP 解压、NPY 解码及数组复制的整段，不能把它全部称作纯 zlib 解压。
- CUDA event 记录搬运/计算对应的 stream 区间，另行同步获取完成时间。stream 区间含主机提交空隙，不等于纯 kernel 时间，也不与 host wall 相加。每步额外的完成同步单独列出。
- learner 仅在 `train_step` 内启用函数 profiler；采集窗口和正式评测诊断不启用全程 cProfile。所有计数/包装仍有成本，短测用于归因，不以它们建立无仪器的速度排名。
- native 评测：θ5 对 θ4，1 个完整 pair，S=1024/N=1、CPU eager、1 thread、1 worker，无根噪声、温度 0，保留循环裁决及 sidecar。它是一个 worker 的时间分解，不是 8 worker 面板吞吐。
- 采集：生产形状 35 秒开局窗口，另做 16 局有限完整对局；两者保持同一 PPO latest checkpoint 和 S=1024/N=1。小轮次的 active/B=16，不外推为生产 active/B=500 的整轮比例。

## 完整历史流程占比

AZ seed1 五轮总墙钟 34,307.27 秒（约 9.53 小时）：

| 环节 | 秒 | 总墙钟占比 |
| --- | ---: | ---: |
| Python batched_cpu 采集 | 30,838.97 | 89.89% |
| learner，含训练轮内 checkpoint/验证 | 1,535.67 | 4.48% |
| MCTS 固定评测 | 1,868.11 | 5.45% |
| greedy 固定评测 | 17.49 | 0.05% |
| 准备和其他未包入上述阶段的时间 | 47.04 | 0.14% |

来源：[history.json](../../artifacts/speed_profile_2026-10-02/history.json)，逐轮原件为 `artifacts/az_loop_seed1/run_snapshot/round-*/round_summary.json`。旧闭环的第一大成本确定是采集；不能把其中约 6,100 秒/轮的旧 CPU 采集比例移用于新 native CUDA 闭环。

## learner：整局读取是主要瓶颈

以下为预热之后相同规模的 24 步；checkpoint/验证另列，不混入每步分项。

| 分项 | 首轮，256 MiB | 第五轮，256 MiB | 第五轮，2 GiB |
| --- | ---: | ---: | ---: |
| 24 步墙钟 | 3.027 s | 15.398 s | 3.246 s |
| 抽样/缓存/整局读取 | 89.38% | **97.82%** | 89.10% |
| 其中 `load_game` 整段 | 2.638 s | **14.779 s（95.98%）** | 2.850 s |
| collate | 3.88% | 0.83% | 4.22% |
| target＋feature 搬运主机区间 | 0.81% | 0.17% | 0.96% |
| forward＋loss＋backward＋clip＋optimizer 主机区间 | 3.94% | 0.81% | 3.95% |
| train_step 其余部分，含检查/标量读取 | 1.50% | 0.27% | 1.35% |
| 测量区间 cache hit | 87.14% | **26.01%** | **86.10%** |
| 测量区间整局解码次数 | 790 | **4,546** | **854** |
| 测量区间解码数组字节 | 0.503 GiB | **2.769 GiB** | **0.523 GiB** |

有效原件：[首轮](../../artifacts/speed_profile_2026-10-02/learner_r0_256mib_v2.json)、[第五轮 baseline](../../artifacts/speed_profile_2026-10-02/learner_r4_256mib_v2.json)、[同输入 cache 对照](../../artifacts/speed_profile_2026-10-02/learner_r4_2048mib_v2.json)。第五轮两档包含预热的抽样引用序列 SHA 一致，24 步逐步 total_loss 差值均为 0；数据、模型、optimizer、随机顺序与损失没有变化。

这项有限对照使测量步墙钟缩短 78.92%（4.74×），足以确认 cache miss 导致重复整局读取是 learner 的主要瓶颈。2 GiB 档末尾缓存约 1.28 GiB，仍在升温，不能把 3.246 秒称为全热 cache 稳态，更不能视为完整训练轮已取得该收益。只改变了诊断进程容量，生产配置未变。

第五轮 baseline 的索引准备约 0.962 秒、验证样本准备 0.158 秒、learner 初始化 0.799 秒；一次 checkpoint 约 0.007 秒、全量当前轮 validation 约 0.086 秒。2 GiB 档对应 checkpoint/validation 约 0.008/0.098 秒。baseline/2 GiB 的进程峰值 RSS 约 1.78/2.86 GiB，PyTorch GPU 分配峰值均约 89.8 MiB。RSS 包括框架、缓存、仍被 sample 引用的数组等，不能当作 cache 的实际大小。

本负载不支持把梯度计算、标量同步或 checkpoint 排为 learner 的首要问题。标量读取确实存在，但正常步骤剩余时间仅约 0.27%；不要根据调用次数把它升级成最大瓶颈。

另做 [2 批 replay 函数画像](../../artifacts/speed_profile_2026-10-02/replay_r4_cprofile.json)，不混入上述轻量占比：388 次整局读取产生 15,520 次 NPY array/header 读取、21,612 次 zlib decompress。zlib 的 self wall 为 0.261 秒；NPY header 读取的 inclusive wall 为 0.577 秒，包含其内部调用。这说明 miss 路径还反复承担 ZIP 成员管理、NPY header/AST 解析、CRC 和数组构造。该 cProfile 有额外开销，各函数 inclusive 时间不可相加，不能将其比例当作生产中纯解压/磁盘 I/O 的精确拆分。

初版 `learner_*` 无 `_v2` 的原件保留在 `runtime/speed_profile_2026-10-02/superseded/`：相对 run 路径使派生 replay 验证索引不命中，准备阶段重验证全部游戏；本报告使用修正为生产绝对路径的 `_v2` 结果。artifact manifest 保存首版原/新位置、哈希和被替代原因。

## 固定评测：batch=1 推理主导

有效原件：[evaluation_native_cpu_v2.json](../../artifacts/speed_profile_2026-10-02/evaluation_native_cpu_v2.json)，冻结配置见相邻 `evaluation_native_config.json`。

1 pair 的两局完成，passed=true，异常和工程截断均为 0；墙钟 **65.569 秒**：

| 互斥环节 | 墙钟占比 |
| --- | ---: |
| 网络 forward | **75.33%** |
| 编码数组 collate | 5.95% |
| tensorize | 4.98% |
| 推理输出转换及其余推理胶水 | 6.33% |
| 搜索中除上述推理的部分 | **7.17%** |
| 真实规则 decision/apply | 0.14% |
| 模型策略创建、sidecar、summary、runner 其余部分 | 约 0.09% |

总推理路径约 **92.59%**；125,872 次 policy/value 推理，每次均为一行。Native 搜索树不能自动改变 `search()` 的逐叶推理调用方式。跨局批量评测有明确的优化对象；真实规则推进和统计写盘在本样本中不是主要成本。未测 GPU batch=1 与跨局 batch 的速度收益，也未实施新评测执行路径。

## 采集：主机复制与等待显著，候选档位需看完整阶段

历史完整 pool 500 局为 744.079 秒，下面均是 owner 主机区间占整轮墙钟，worker 累计 CPU/等待不加入此表：

| owner 区间 | 秒 | 占整轮墙钟 |
| --- | ---: | ---: |
| input wait | 119.39 | 16.05% |
| coalesce wait | 187.22 | 25.16% |
| gather | 149.80 | **20.13%** |
| forward 主机区间 | 133.19 | 17.90% |
| CUDA event 完成等待 | 72.64 | 9.76% |
| H2D 主机区间 | 15.60 | 2.10% |
| D2H 主机区间 | 8.55 | 1.15% |
| dispatch | 25.34 | 3.41% |

计时器没有覆盖完整 owner 时间线，也不是 GPU kernel 分解。不能把 41.21% 的两类输入/合批等待承诺为可消除收益；这些等待会与 worker 的真实搜索工作重叠。GPU event 等待也不能解释成单独的 D2H 成本。

历史平均 batch=381.45；256 档占物理行数 9.10%、占物理候选容量 36.49%。owner 新增 padding 为容量的 47.69%，还未计 worker 内部 padding。这证明宽度放大有实际容量浪费，尚不等于已测得相同百分比的时间收益。

当前 500 活跃局、B=500 开局窗口实测 35.008 秒：

| owner 区间 | 窗口墙钟占比 |
| --- | ---: |
| input wait | 27.61% |
| coalesce wait | 23.31% |
| gather | **18.65%** |
| forward 主机区间 | 17.53% |
| dispatch | 3.57% |
| H2D / D2H 主机区间 | 2.36% / 1.22% |
| 完成等待 | 0.81% |

有效原件：[collection_opening_b500.json](../../artifacts/speed_profile_2026-10-02/collection_opening_b500.json)。窗口推理 1,802,312 行/3,637 批，平均 batch=495.55；实际根对局阶段全部为开局，没有完成局，不报告 games/min。cold compile 预热及真实预热在窗口之前；本次没有 eager fallback 或 compile failure。整段采集墙钟约 51.99 秒，不等于测量窗口长度。按物理宽度等 session 汇总包含预热，共 1,882,812 行 / 3,798 批；不得与上述 35 秒窗口的分母混用。

开局已经接近满批，进一步增大 B 缺少依据。历史和新窗口都支持 gather 是显著的可检查工程成本；是否是 CPU 全局饱和、等待中有多少是人为 deadline、宽请求具体影响哪些阶段，须结合小轮次和后续完整生产时间线判断。

### 有限完整 16 局：尾段放大已观测

有效原件：[collection_full16_b16.json](../../artifacts/speed_profile_2026-10-02/collection_full16_b16.json)。W=8、active/B=16、microbatch=64、slots=1、wait=5ms，搜索与模型同上述采集；全部 16 局完成并通过 checked replay，异常/工程截断为 0，产生 1,401 个 value 样本、1,031 个有效 policy 样本。采集墙钟 **306.08 秒**；这是带统计包装的小批诊断，不能与生产 500 局直接比较 games/min。

| owner 区间 | 秒 | 小轮次墙钟占比 |
| --- | ---: | ---: |
| coalesce wait | **121.29** | **39.63%** |
| forward 主机区间 | **92.03** | **30.07%** |
| input wait | 40.89 | 13.36% |
| gather | 12.45 | 4.07% |
| H2D / D2H 主机区间 | 8.34 / 6.64 | 2.73% / 2.17% |
| dispatch | 5.44 | 1.78% |
| 完成等待 | 0.80 | 0.26% |

平均 batch=13.66，总计 70,054 批，其中 19,339 批被旧计数器标成 `deadline`；该计数器合并了多种未满批情况，不能解释为 19,339 次真实超时。前 8 局完成于约 223.53 秒；第 14 局完成于 270.77 秒，最后两局于 305.78 秒完成。最后两局阶段耗时 **35.02 秒（整轮 11.44%）**，只处理 **6,223 行（全部推理行的 0.65%）**，平均 batch 降至 **1.57**。尾段时间明显高于其工作量占比，且 batch 上限/等待保持原设置。这是有界等待值得处理的直接证据，尚不能把整段尾部时间都认作人为等待。

按真实根对局的双方合计 completed turns 分类，每一行取所在搜索 job 的根阶段：

| 根对局阶段 | 推理行数 | 真实候选均值 | 真实候选最大值 | >48 候选行 |
| --- | ---: | ---: | ---: | ---: |
| 开局，<10 turns | 172,200 | 20.90 | 47 | 0 |
| 中盘，10–49 turns | 734,849 | 19.29 | 51 | 80 |
| 后盘，≥50 turns | 50,233 | 18.53 | 47 | 0 |

真实行计数合计 **957,282**，与 owner logical_model_positions 完全一致。本小样本的 >48 行仅 80 个，却扩为 1,260 个物理 256 档行（80 个 eager 批次）；没有 compile failure。全部物理候选容量 33,346,384，其中实际候选 18,707,569，worker padding 3,717,490，owner 新增 padding 10,921,325；两层合计 padding 为 **43.90%**。

此处最多 51 候选，64 档能覆盖本样本，但超档行太少，宽档总推理墙钟仅约 0.25 秒，不支持凭这 16 局把增加 64/96/128 排成小批路径的最高优先级。历史 500 局的 256 档行占 9.1%，须继续保持“物理档位”与“实际叶宽”区别；完整生产实际叶宽分布仍未测全。

## 瓶颈判断与验证边界

1. **learner 已确定：重复整局 NPZ 读取。** 后期 256 MiB cache 测量步中约 96% 是 `load_game` 整段，cache 对照同输入同 loss；不是先优化梯度计算或日志同步。
2. **固定 native CPU 评测已确定：逐叶单行推理。** 总推理路径约 92.6%，规则/sidecar 占比很低；批量评测尚未实现和测量。
3. **采集已确定显著成本：owner 的 gather 与两类等待；小批尾段放大。** 生产形状开局已接近满批，完整历史 gather 约 20%；小轮次 coalesce 约 40%，最后两局仅 0.65% 推理行却占 11.4% 墙钟。
4. **待确定：完整 500 局中各等待原因的可消除比例、宽档具体时间贡献、GPU kernel 利用率和 native 全闭环阶段占比。** 现有证据不支持把这些未知项写成精确优化收益。

测量包装通过 2 局 S=8/N=2 smoke 和上述 16 局完整 checked replay；缓存对照引用 hash 相同、逐步 loss 相同；相关有限回归见 artifact 摘要。没有更改生产算法、cache 默认值、搜索预算或评测执行路径。未来可复测完整生产时间线，但该长验证未在本任务执行。

## 可复现的有限命令

从仓库根目录执行。每次使用新 output/scratch 路径；旧原件不可覆盖。CUDA 在沙箱不可见时需按权限机制在宿主机执行，不能静默换成 CPU。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/profile_speed_pipeline.py \
  --output artifacts/speed_profile_new/learner_r4_256mib.json --scratch runtime/speed_profile_new/learner_r4_256mib \
  learner --round 4 --cache-mib 256 --warmup-steps 8 --steps 24
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/profile_speed_pipeline.py \
  --output artifacts/speed_profile_new/learner_r4_2048mib.json --scratch runtime/speed_profile_new/learner_r4_2048mib \
  learner --round 4 --cache-mib 2048 --warmup-steps 8 --steps 24
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/profile_speed_pipeline.py \
  --output artifacts/speed_profile_new/evaluation.json --scratch runtime/speed_profile_new/evaluation \
  evaluation --config artifacts/speed_profile_2026-10-02/evaluation_native_config.json --pairs 1 --device cpu
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/profile_speed_pipeline.py \
  --output artifacts/speed_profile_new/opening.json --scratch runtime/speed_profile_new/opening \
  collection --seconds 35 --warmup-seconds 2
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/profile_speed_pipeline.py \
  --output artifacts/speed_profile_new/full16.json --scratch runtime/speed_profile_new/full16 \
  collection --games 16 --active 16 --batch 16
```

`--cprofile` 可显式用于 collection/evaluation 的详细函数诊断，额外开销不应混入轻量时间占比；完整 profiler 文件、游戏、trace、checkpoint 保存在 runtime，必要摘要与身份保存在 artifacts。后续脚本的learner也只有显式`--cprofile`时启用函数/CUDA event重仪器，正式吞吐用关闭仪器的独立运行；以上历史原件保留其原脚本SHA与计时口径。以上诊断阶段未执行正式1,000局对照，后续完整结果见端到端验收报告。
