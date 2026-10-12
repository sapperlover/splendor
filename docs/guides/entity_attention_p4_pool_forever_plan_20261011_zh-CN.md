# 第9轮终点续训：lr=2e-5、每轮1000步的长期自迭代计划

2026-10-11。**纯在线长期入口及正式配置已实现，正式任务尚未启动。** 本文替代此前误写的“从P4重新起步、重走旧池渐进混合”版本；以用户最新指令为准：**从原第9轮结束继续，继承6000局在线窗口，不使用旧离线池，每轮1000步完整checkpoint无条件升级为教师。**

依据：[原九轮流程](entity_attention_p4_pool_iteration_zh-CN.md)、[完整报告](../reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)、[原总配置](../../configs/entity_attention_v1/p4_pool_iteration_20261010.json)、[原采集／训练配置](../../configs/entity_attention_v1/hybrid_p4_pool_20261010.json)，以及原run的已提交状态、回放快照和当前源码。

## 1. 起点、教师与轮号

父run为`runs/entity_attention_p4_pool_iteration_20261010/`。从其**第9轮终点、全局38,898步**创建独立续训run，父run及其已完成实验保持原件。

| 身份 | 固定来源 |
| --- | --- |
| 初始完整模型／第一轮采集教师 | `runs/entity_attention_p4_pool_iteration_20261010/rounds/round-008/checkpoints/final.pt` |
| 起点SHA-256 | `ff34660fece12590762da93d031523e4ae09db4fa6af5d2284c4af0dcd57237b` |
| 继承在线池 | 父run的`rounds/round-008/replay_snapshot.json`，原第4–9轮，共6000局 |
| 固定评测对手P4V4 | `runs/entity_attention_recent_pool_20261009/checkpoints/step-000034398.pt` |
| P4V4 SHA-256 | `94f4ce89d27f42ec0c39e72055b460405ae36b8f9931051cbe360ee145ace5ec` |

恢复原第9轮完整encoder、policy、value、AdamW moments及计数、采样RNG和Torch RNG。初次恢复按来源1e-4配置加载，随后将有效学习率及optimizer所有param group的lr统一改为2e-5；不重置优化器。保存续训run自己的初始状态，之后恢复始终使用有效2e-5配置。

**下一次采集和训练为原第10轮**，续训本地轮号为1；两种编号都记录，展示与教师代际使用原绝对轮号。起点是原第9轮已经完成的500步终点，不再补训该轮，也不换成最近单变量实验的2e-5／1000步候选。

此后第g轮（g≥10）训练恰好1000步，**该轮1000步的完整checkpoint**成为第g+1轮教师。500步模型只作过程观测；不选最低loss节点，不做policy／value拼接。KL、MSE、greedy和搜索成绩只用于记录，任何好坏结果都不冻结、拒绝或回滚教师。

新run完成k轮后全局更新数为`38898 + 1000 × k`。第10／11／12轮终点分别为39,898／40,898／41,898步。

采用新配方的直接对战依据见[最近200-pair结果](../../artifacts/entity_attention_r9_lr_step_arena_20261011/results.md)：共同第八轮终点下，2e-5×1000步对1e-4×500步得分53.250%，95% CI为[48.500%, 58.000%]。这是可继续检验的方向，尚未确认棋力优势；本计划从原第9轮终点另行延续，不能把该单轮结果当作续训收益保证。

## 2. 训练配方

| 项目 | 续训设置 |
| --- | --- |
| 每轮新数据 | 当前完整教师自对弈1000局 |
| 每轮optimizer更新 | **1000步** |
| 有效学习率 | **固定2e-5**，无衰减日程 |
| Batch | 256 |
| Weight decay | 1e-4 |
| Policy／value loss权重 | 1／1，联合更新共享encoder |
| Grad norm clip | 1 |
| 在线窗口 | 最近6000个完整对局，Train与Validation合计计容量 |
| 抽样 | 只从窗口内Train局面均匀、有放回抽样 |
| 旧离线池 | 不加载、不采样，不执行旧池对照或旧池拟合 |
| 完整checkpoint | 每50步，保留权重、优化器、计数与训练RNG |
| 拟合节点 | 每轮0／100／250／500／1000步 |
| 主线终点 | 无预设轮数，人工中断或工程异常时停止 |

每batch全部256条来自同一个在线窗口；每轮共256,000条抽样引用。沿用原单份行为搜索policy标签和终局value标签，不启用M4平均标签、追加重搜索、分头训练、新损失、最新代加权或优先采样。原`history_batch_size`、`steps_per_new_position`不控制这个显式在线采样／1000步入口。

## 3. 数据来源、划分与窗口滚动

### 3.1 原6000局完整继承

直接继承第9轮快照中第4–9轮的归档路径、run身份、绝对代际、局号、原单份标签和既定整局split，不重新抽取6000局、不重标记Train／Validation。

截至该快照，可复核数量如下；“局面”指decision位置，policy指标另按有效policy权重计算分母。

| 原代际 | Train局数 | Train局面 | Validation局数 | Validation局面 |
| --- | ---: | ---: | ---: | ---: |
| 第4轮 | 902 | 73,310 | 98 | 7,994 |
| 第5轮 | 910 | 74,556 | 90 | 7,369 |
| 第6轮 | 890 | 71,814 | 110 | 8,873 |
| 第7轮 | 870 | 70,637 | 130 | 10,633 |
| 第8轮 | 911 | 73,226 | 89 | 7,217 |
| 第9轮 | 900 | 72,915 | 100 | 8,174 |
| **合计** | **5,383** | **436,458** | **617** | **50,260** |

这六代均属于原在线窗口；不再为它们另设“旧池配额”。原29,000局离线池、最近OFAT分支及步数扩展分支均不加入本次训练。父run其他产物仅保留为已有证据。

### 3.2 新数据与淘汰顺序

第10轮先用原第9轮终点采1000局，再插入窗口并淘汰原第4轮，**第10轮训练使用第5–10轮6000局**。以后同样滚动：

| 阶段 | 活动窗口 | 训练／下一轮教师 |
| --- | --- | --- |
| 续训初始化 | 原第4–9轮 | 原第9轮终点，38,898步 |
| 第10轮采集后 | 第5–10轮 | 训练1000步，终点作为第11轮教师 |
| 第11轮采集后 | 第6–11轮 | 训练1000步，终点作为第12轮教师 |
| 第12轮采集后 | 第7–12轮 | 训练1000步，终点作为第13轮教师 |
| 第15轮采集后 | 第10–15轮 | 继承的第4–9轮已全部自然移出窗口 |

容量始终按完整对局计算。每轮采集结束冻结窗口，训练期间不继续追加。移出窗口只取消训练索引和释放相应缓存，不删除原始归档，也不改split或标签。

新局仍以独立split种子按10%概率整局留出，约900局Train／100局Validation，实际数量允许波动；所有Validation位置都不参与梯度更新。最新代抽样占比由其Train局面数除以窗口Train总局面数决定，通常约六分之一，**不强制按代际等额**。保留实际抽样份额、曝光次数和覆盖率账本。

## 4. 一轮流程与评测节奏

每个绝对轮次g≥10依次执行：

1. 冻结上一轮完整终点为教师，完成1000局自对弈和整局split，更新最近6000局窗口。
2. 恢复连续learner状态，冻结窗口和诊断引用，记录0步拟合；不额外增加起点自对战。
3. 更新1000步，每50步提交恢复checkpoint与抽样账本；100／250／500／1000步记录拟合。
4. 若g为3的倍数，按原调用顺序先完成200 pair S1024/N1对固定P4V4。
5. 每轮都完成200 pair greedy对固定P4V4；评测串行执行，隔离评测对训练RNG的影响。
6. 提交本轮完成状态和1000步完整教师，更新汇总，进入下一轮；得分低也照常升级。

| 记录／评测 | 频率与口径 |
| --- | --- |
| 训练日志、抽样引用 | 每步，沿用原字段和提交边界 |
| 完整恢复checkpoint | 每50步及轮始／轮末 |
| 固定拟合 | 每轮0／100／250／500／1000步 |
| Greedy棋力 | **第10轮起每轮200 pair**，对固定P4V4 |
| 搜索棋力 | **第12／15／18……轮200 pair**，双方S1024/N1，对固定P4V4 |

搜索每3轮的节奏继续原绝对轮号；第12轮即续训第3轮。原第9轮已经完成，不重做其评测或收尾任务。取消此前计划误加的旧池对照、旧池完整留出与P12＋V4一次性里程碑；这些不是本次在线续训的一部分。

棋力评测沿用交换先后手、和棋半分和整pair bootstrap 10,000次的95% CI，保留逐局sidecar、异常计数、实际搜索预算、checkpoint SHA和配置。工程失败明确报错；成绩不影响教师接纳。

原固定DEV面板不变：greedy开局30800000–30800199、policy seed base30810000、bootstrap seed30820000；搜索开局30900000–30900199、policy seed base30910000、search seed30930000、bootstrap seed30920000。最近独立对战的31100000面板不替换常规面板。反复查看这些固定开局用于比较趋势，不作为新的独立确认实验。

采集collection基础种子30700000、split 30710000、search 30730000及现行`LoopSeeds`派生规则不变；**按绝对代际派生**，第10轮使用零起编号9，不从0重新派生。训练继承原第9轮保存的采样流状态，不重置到30110000种子。拟合种子30740000、辅助在线留出种子14640000保持原值。

## 5. 拟合与记录口径

保留在线部分全部诊断，增加1000步轮末节点；不新增旧离线池依赖。

- **固定Train子集：**每代最多2048位置，继承原第9轮`fitting/train_references.json`中仍在窗口内各代的引用；新代按原`seed + 绝对零起代际`规则冻结。原快照六代各2048位置已经核验。同一代在不同节点和后续轮次使用相同引用；它们属于Train，不等于全部已在训练中抽中过。
- **完整在线留出：**每个拟合节点遍历窗口内各代全部Validation位置，保留逐局分子／分母、按代际和阶段的指标。继续输出`online_train_fixed`与`online_validation_full`，不以2048小子集代替完整留出。
- **固定辅助在线留出：**继承父run的`fixed_online_dataset.json`和`fixed_online_validation.json`，保留原第一代2048位置的辅助可比性。它仅用于诊断，不加入6000局训练窗口，也不因fork而换成新第10轮。它与原29,000局离线池无关。
- **指标：**沿用policy KL／CE、目标与预测熵、教师top1一致率、value MSE、零预测基线、bias、相关系数；双方最高分≥10为后期，否则前中期。
- **矩阵：**每轮五个节点、最多六个活动代际，分别形成Train／完整留出矩阵。列名继续`g005`、`g009`、`g010`等绝对代际；最新代索引不能误写为续训本地`g001`。比较同代同集合的前后变化，跨教师列的绝对KL不直接用于选最强模型。
- **可复核记录：**保留配置、实际命令、代码版本与源码快照、环境、种子、teacher和checkpoint身份、整局split、归档及标签来源、固定引用、逐步抽样账本、`source_usage.json`、实际各代抽样份额、拟合原始产物、棋力sidecar与时间／资源记录。

旧离线池的`offline_fixed`在本计划中不执行，输出明确标记不适用或在带范围说明的新记录中省略；不填0、不挪用在线指标代替。原历史报告保留，不把旧池诊断缺席解释为指标改善。在线固定留出和棋力面板仍是DEV。

## 6. 采集速度、长期运行与恢复

采集与评测参数保持原值：

- 自对弈S1024/N1、cPUCT1.5、root noise0.25／alpha0.3、max simulation transitions512、fold forced phases；温度前10个completed turns为1，此后为0，max completed turns400。
- 采集search workers8、total active1000、max batch500，local microbatch64、shared slots2、pipeline slots1，FP32、native-cudagraph；batch档`[16,32,64,128,192,256,384,500]`。
- 搜索评测native_batched、active100／max batch64、compiled，root noise0、温度0；greedy CPU workers8，每worker Torch线程1。

保持原在线缓存的无损float32压缩LRU、**8 GiB在线缓存容量**；不再分配旧离线池缓存。这是缓存容量，不是进程硬RAM上限。滚动窗口淘汰后释放无用缓存，固定辅助留出所需条目仍受同一LRU约束；采集和评测结束释放推理图。

不设人为终止轮数，不按9轮重启教师或窗口。原始对局、模型及必要记录逐轮保存，不自动删除文件、不增加磁盘配额或空间门槛；归档会持续增长，需要相应物理存储。写入或硬件异常明确停止，修复后恢复。

首次启动才从父run第9轮建立fork并继承窗口；以后同一命令从新run自身恢复，不再次fork、不重置教师或优化器。恢复已提交采集对局、最近50步checkpoint／RNG、已提交完整评测pair；未提交工作允许重做。训练到1000步而评测未完时只补评测，不额外训练1000步。该轮训练及常规评测完成后才提交下一轮教师，完成轮次不重复升级。

主循环采用按轮落盘、增量更新的轻量汇总，保留全部原始历史。现有`pool_iteration.analyze`每轮重读全部历史拟合和详细对战，应在长期模式中避免把所有历史逐局对象常驻内存；全历史分析可流式重建。活动训练数据始终受6000局窗口与缓存容量约束。

## 7. 已实现的调用链

复用[hybrid_train.py](../../src/splendor/hybrid_train.py)、[pool_iteration.py](../../src/splendor/pool_iteration.py)、[pool_fitting.py](../../src/splendor/pool_fitting.py)，已补齐：

1. **在线fork。** `online_only=true`冻结父run第9轮checkpoint与回放快照，继承6000局、原split和固定Train／辅助在线留出引用；不复制、读取或评估旧离线池。父run原件不修改。
2. **连续优化器。** `learning_rate_override=2e-5`初次fork后同时设置learner和optimizer所有param group，保留moments、计数和RNG；之后按新run实际状态resume。
3. **长期总入口。** 总配置`forever=true`与有限`rounds`互斥；`fork_from.round=9`采用人类轮号。传入底层`online_rounds=None, forever=True`，没有旧池对照或额外收尾任务；旧有限配置保持原语义。
4. **诊断、编号与缓存。** 五节点在线Train／完整留出矩阵独立于旧池；`offline_fixed=null`表示不适用。采用8 GiB在线压缩LRU。日志、矩阵和status沿用绝对轮号，采集种子不重置；修复fork抽样统计覆盖各代计数的问题，保留实际代际使用量。
5. **评测、提交与汇总。** 固定对手显式为P4，绝对第12／15……轮搜索，每轮greedy。评测结束后无条件提交1000步教师，再刷新汇总。每个已提交绝对轮次的完整指标摘要单独写入`artifacts/<run_id>/rounds/round-000010.json`等文件，`summary.json`／`results.md`展示最新完成轮；历史原始矩阵和逐局记录均保留。`analyze`逐轮补齐遗漏摘要，不将全历史面板同时加载。

配置中的`offline`字段保留来源learner、模型及种子等现有参数容器，`offline.learner.learning_rate=1e-4`表示原checkpoint恢复配置；在线模式不访问其中旧数据路径。有效训练lr为2e-5。模型、搜索算法、规则、标签和在线抽样分布不变。

## 8. 启动、恢复与查看进度

已创建独立身份`entity_attention_r9_online_lr2e5_1000_forever_20261011`的配置：

- 总配置：`configs/entity_attention_v1/r9_online_lr2e5_1000_forever_20261011.json`。
- 采集／训练配置：`configs/entity_attention_v1/hybrid_r9_online_lr2e5_1000_forever_20261011.json`。
- 原始训练产物：`runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/`。
- 轻量汇总：`artifacts/entity_attention_r9_online_lr2e5_1000_forever_20261011/`。

父run、OFAT分支和已有对战产物不覆盖；产物遵循[现行约定](../../artifacts/README.md)。

**从仓库根目录执行；首次启动与中断续跑使用同一条命令：**

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u -m splendor.pool_iteration run \
  --config configs/entity_attention_v1/r9_online_lr2e5_1000_forever_20261011.json
```

入口自动区分fork与resume；同目录并发运行由锁拒绝。Ctrl+C／SIGTERM后重新执行原命令恢复。将命令中的`run`换成`check`、`status`、`analyze`，分别检查参数、查看进度、逐轮重建已有汇总，不启动正式实验。`status`显示绝对轮号及采集／训练／评测待提交阶段。保留`--max-steps-this-session N`作为有限smoke预算，不改变长期日程；它限制更新数，仍会先采完整配置规定的新局数，正式配置不适合用它进行小采集smoke。

## 9. 有限验收

验收只做必要相关回归和短smoke，不提前启动正式1000局或完整棋力赛：

1. 核对起点SHA、全局38,898步、原6000局路径／split／标签及固定引用；确认没有旧离线池训练或验证调用。
2. 验证初次恢复的权重、AdamW moments／计数和RNG连续，所有param group实际lr均为2e-5；中断续跑不重置，已提交抽样序列一致。
3. 小规模或mock验证第9→10轮的第4–9轮窗口变为第5–10轮、第15轮继承数据自然清空；第10轮教师确为原第9轮终点，之后均用各轮1000步完整模型，低得分不阻止升级。
4. 验证五个拟合节点、每50步恢复、绝对代际矩阵、每轮greedy／第12、15……轮搜索、逐pair续跑和提交状态；在线缓存及汇总不累计全历史详细对象。旧有限配置仍可读取。

正式配置的只读`check`已通过，核对起点38,898步、6000局（5383 Train／617 Validation）、有效lr2e-5、每轮1000更新及在线batch256。实际设备为RTX 4050 Laptop GPU、6 GiB；完整长任务和长期资源曲线尚未运行验证。

本次相关回归`tests/test_pool_iteration.py tests/test_hybrid_training.py`共28项通过，覆盖旧有限流程、纯在线fork、原split／固定引用继承、窗口滚动、实际lr、优化器／RNG恢复、低得分照常升级和绝对轮号评测。中断恢复与连续运行的最终权重、AdamW状态及采样流一致；新增RNG断言补测亦通过。GPU短测使用真实第9轮完整模型，batch256／lr2e-5更新2步，随后native_cuda_pool完成2局S8/N1、无工程异常；短采集仅用2 active／batch2／workers2，不代表正式B500、S1024/N1的长时资源验收。验证原件保存在`runtime/entity_attention_r9_online_validation_20261011/`。
