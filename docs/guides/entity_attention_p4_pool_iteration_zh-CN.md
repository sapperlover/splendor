# P4大池起点的连续自迭代

2026-10-10：正式9轮、同预算纯旧池对照、完整旧池留出及全部对战已完成，见[完整报告](../reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)。终点S1024/N1对P4／旧池对照／P12＋V4的200pair得分分别58.750%／61.625%／58.125%，区间均高于50%，限定单训练seed与固定DEV面板。入口为 `splendor.pool_iteration`；配置见 [总流程](../../configs/entity_attention_v1/p4_pool_iteration_20261010.json) 和 [采集／训练配置](../../configs/entity_attention_v1/hybrid_p4_pool_20261010.json)。本轮参数是研究基线，未确定最优步数、学习率或采样配额。

## 流程与教师

从大池离线训练的 **P4V4＝34,398步**开始，恢复完整网络、AdamW的一阶／二阶矩、优化器计数、采样RNG和Torch RNG。仅把checkpoint保存间隔从1000步改成50步；学习率保持1e-4，不重启优化器，也不继承后来8次平均抽样处的降学习率日程。

共9轮，每轮依次执行：

1. 当前完整学生作为双方教师，S1024/N1自对弈1000个自然终局。采集沿用根噪声0.25、alpha=0.3，开局前10个完成回合温度1，之后0。保存原行为搜索的单份根访问分布，value按决策方终局胜／负／和取+1／−1／0，不追加M4标签。
2. 更新在线窗口，按整局固定train／validation，再训练500步。每batch按下表渐进混合旧大池与在线窗口，第7轮起只抽在线窗口；policy/value联合训练、损失权重1/1、batch256、AdamW lr=1e-4、weight_decay=1e-4、共同clip1。encoder正常更新。
3. 在0／100／250／500步记录同一组局面的拟合数据。0步只作一次必要的训练前测量，没有大量自对战或多轮0步测试。
4. 每轮末做200pair greedy对固定P4V4；第3／6／9轮末再做200pair双方S1024/N1对固定P4V4，无根噪声、温度0、互换先后手、和棋半分，区间以整pair bootstrap计算。
5. **本轮500步完整模型无条件成为下一轮教师。** KL、MSE、得分及置信区间都不触发接纳、拒绝、回滚、选点或教师冻结。评测串行完成后继续下一轮；工程异常仍明确停止，修复后同一命令恢复。

| 轮次 | 在线窗口完整局数（split前） | 每batch旧池／在线位置数 |
|---|---:|---:|
| 1 | 1,000 | 224／32 |
| 2 | 2,000 | 186／70 |
| 3 | 3,000 | 147／109 |
| 4 | 4,000 | 109／147 |
| 5 | 5,000 | 70／186 |
| 6 | 6,000 | 32／224 |
| 7～9 | 最近6,000 | 0／256 |

前6轮旧池配额为 `round(224 - (r - 1) * 192 / 5)`，每轮减少约38.4个位置并取整；第6轮窗口充满，第7轮起停止旧池训练采样。这里的“旧池”专指启动前29,000局大池，在线窗口中较早教师代的数据仍正常参与采样。该日程替代交付初版的固定64／192，不代表已确定最优配额。

9轮主线完成后，才运行便宜的纯离线对照：同样从P4及其AdamW/RNG出发，同batch、lr和总4500步，只在旧池Train内均匀抽样。主线和对照最终全局步数都是38,898。对照记录0／1500／3000／4500步的固定旧池Train／留出指标；两条终点另评全部旧池留出。最后做主线终点对离线对照、对P12V4各200pair搜索。该对照检验新增在线混合数据是否超过继续训练旧池的收益，不能单独隔离教师换代贡献。

## 旧池的来源与划分

直接复用[既有冻结清单](../../artifacts/entity_attention_recent_pool_20261009/pool_games.jsonl)，SHA-256为 `d5d49d8e019767b90e6b2eb5457f0a5f74dbab1f116b10ecbcce06fc3675fe6b`。保留所有原始整局split，不重新随机划分、不搬移NPZ。它包含经过前次筛选的标准attention、native S1024/N1有效自对弈；清单中的合成round编号仅用于避免不同来源局号冲突，原代际见 `source_round_index`。

| 来源 | 采集规模 | Train局数 | 留出局数 |
|---|---:|---:|---:|
| attention早期自迭代，`entity_attention_10x1000_seed1`前3轮 | 3×1000 | 2,701 | 299 |
| 原1000局混合自迭代，`entity_attention_hybrid_static_graph_seed1` | 9×1000 | 8,095 | 905 |
| 后续N1分支，`entity_attention_hybrid_s1024_n1_from_round9_seed1` | 50×200 | 8,981 | 1,019 |
| 分头训练分支，`entity_attention_hybrid_split_p4_v15_from_round50_seed1` | 20×200 | 3,592 | 408 |
| M4分支，`entity_attention_avg4_behavior_20261009/seed1_avg4` | 12×200 | 2,154 | 246 |
| 50步分支，`entity_attention_iteration_trend_seed1_20261008/seed1_steps50` | 2×200 | 365 | 35 |
| native教师探针，`entity_attention_chance_training_probe_20261007/native` | 200 | 177 | 23 |
| **合计** | **29,000** | **26,065** | **2,935** |

M4来源只读取原始对局归档里的**单份行为搜索标签**；不读取其后追加的三份标签或平均标签。更多原始路径、采集配置、教师SHA和筛选依据见[原大池来源记录](../../artifacts/entity_attention_recent_pool_20261009/sources.json)与[原报告](../../artifacts/entity_attention_recent_pool_20261009/README.md)。旧池不包括更早的原θ5十轮10,000局Deep-Sets语料。

旧池Train为 **2,201,472个局面，1,735,434个有效policy局面**；留出为 **247,453个局面，195,067个有效policy局面**。留出从不进入训练抽样，但已用于此前研究与P4选点，因此是DEV，不能称全新FINAL测试集。

旧池在前6轮按上述配额参与训练，第7轮起训练配额为0；原始归档、冻结清单及split继续保留，用于固定拟合诊断、末尾纯离线对照和最终完整留出评估，不删除数据。池内按Train局面均匀、有放回采样，因此各来源按训练局面数占比抽样，并非七个来源等额，也并非每局等权。

通用learner配置中继承的 `history_batch_size=128` 和 `steps_per_new_position=4` 不控制本入口的配额／预算；实际由显式渐进混合日程和每轮500步决定。

## 新在线数据、窗口与采样

新采集与旧池是两个独立索引和缓存，第一轮在线池为空起步，不把旧池尾部塞进在线窗口。新数据只来自本次每轮的最新完整教师。9轮最多新增9,000局原始归档；全程留存原件，活动训练窗口最多保留最近6,000个完整对局，**容量在train／validation划分前计算**。

每局使用独立split种子确定10%概率留出，整局只能属于一个split。1000局约900局Train／100局留出，实际数量可波动，不强行固定900/100；中断恢复保留既定split。第7轮开始淘汰第1轮，第9轮窗口为第4～9轮。移出窗口只影响采样，不删除归档。

每batch的在线配额在当前窗口全部Train局面中均匀抽取，不额外优先最新轮，也不按教师代际等额。若各轮长度和Train比例接近，最新一轮占整个batch约为：第1轮12.5%、第2轮13.7%、第3轮14.2%、第4轮14.4%、第5轮14.5%、第6轮14.6%，第7轮起约16.7%。第7轮纯在线仍同时抽六代数据；实际份额和曝光次数保存在各轮 `source_usage.json`，逐步引用账本可复核是否抽到留出。

采集基础种子为collection 30700000、split 30710000、search 30730000，实际每轮种子由现行 `LoopSeeds` 派生。训练采样流从P4的30110000流状态继续，不每轮重置。Greedy固定开局30800000～30800199，搜索固定开局30900000～30900199；同类各轮复用开局便于配对比较，属于DEV面板。不同对手的得分不直接当成模型互打结果。

## 拟合指标的口径

- 旧池：七个来源各冻结最多2048个Train和2048个留出位置；报告来源与阶段指标，合计按原池有效policy局面数／value局面数分别加权。它是固定子集估计，不能称旧池全量留出。旧的4096位置验证仅为兼容辅助记录，不替代此来源加权口径。
- 在线Train：活动窗口内每代最多2048个固定位置；按代际独立种子冻结，同轮0／100／250／500步及后续仍在窗口内的轮次引用不变。这是Train分片诊断子集，包含尚未被本轮抽中的位置，不能称全部都是“见过的样本”。
- 在线留出：逐代遍历活动窗口内**全部**validation位置，保留逐局统计；每轮0／100／250／500步集合与标签固定，支持本轮前后比较。第1轮另保存2048位置固定辅助留出，即使离开窗口也可观察保持情况。
- 指标包含policy KL/CE、目标与预测熵、教师top1一致率、value MSE、零预测基线、bias、相关系数；阶段沿用双方最高分≥10为后期，否则前中期。绝对KL不跨教师列判断好坏，主要看同列变化。
- 主线与离线对照最终都遍历旧池全部2,935局／247,453位置留出，保存来源、阶段与逐局数据。

## 执行、恢复与产物

采集保持W8／active1000／maxbatch500、local_microbatch64、shared_slots=2、owner单缓冲、FP32、native-cudagraph，固定batch档调整为 **[16,32,64,128,192,256,384,500]**。补充小批次和中间档位，避免例如65行直接补到256、257行直接补到500；满500行请求保持500档。档位按需使用，不能据此宣称整轮加速或最优。搜索评测沿用native_batched、active100／maxbatch64、compiled。Greedy在CPU使用8个worker，每个Torch线程数1。

两池采用已经验证过的无损float32字典压缩缓存，总缓存容量16 GiB，两池各8 GiB；这是LRU容量，不是进程RAM上限。不会一次性解压整个19 GiB旧池，缓存只在容量内保留，在线淘汰时释放旧条目，采集／评测边界释放推理图。第7轮起旧池无训练抽样，但固定拟合诊断仍按需读取其Train／留出子集。没有cgroup硬RAM限制、磁盘配额或额外磁盘空间门槛；主线、对照、评测顺序执行。短smoke不能保证任意机器上的长期峰值，运行日志保留实际缓存统计。

以下命令保留用于复现与中断恢复；当前正式目录已完成，重复执行不会追加训练。从仓库根目录执行，**首次启动和中断后恢复都用同一命令**：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u -m splendor.pool_iteration run
```

Ctrl+C停止；采集恢复已提交对局，训练恢复最近50步checkpoint，最多重做约50个未提交更新，评测恢复已提交pair。配置、源清单、checkpoint及split身份冻结；同目录并发启动会被锁拒绝。训练中断不从随机初始化重来；完成后重复执行不追加训练。SIGTERM同样进入中断清理流程。

```bash
# 只读核对身份和参数，不采集、不训练、不对战
conda run -n splendor env PYTHONPATH=src python -m splendor.pool_iteration check
# 进度：已完成轮数、本轮已提交对局、最近checkpoint步数
conda run -n splendor env PYTHONPATH=src python -m splendor.pool_iteration status
# 重建已有结果汇总，不执行实验
conda run -n splendor env PYTHONPATH=src python -m splendor.pool_iteration analyze
```

`--max-steps-this-session N`仅用于限制当前会话主线更新预算，不自动启动末尾离线对照／最终对战；再次执行不带该参数的标准命令继续全部流程。

重型归档、checkpoint、逐步日志、引用账本、轮内完整统计与评测sidecar放在 `runs/entity_attention_p4_pool_iteration_20261010/`；源码身份和命令快照由既有hybrid机制保存。轻量汇总自动写入 `artifacts/entity_attention_p4_pool_iteration_20261010/{summary.json,results.md,plan.json}`。框架有限验证见[交付记录](../../artifacts/entity_attention_p4_pool_iteration_20261010/README.md)；渐进配额／新batch档的10项回归及GPU40形态、50次数值检查见[修订验证](../../artifacts/entity_attention_p4_pool_iteration_20261010/progressive_revision.json)。该GPU检查不是完整自对弈吞吐或长负载实验。
