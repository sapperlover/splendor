# 当前256×4实体MLP：8万局离线训练与R20对战

2026-10-25存储更新：本实验共用的原80000局NPZ已按授权外置至[归档20261025](../repository_archive_20261025/README.md)；模型、配置、索引及已完成结论保持。复训或全量留出重新推理前须先恢复该批NPZ；模型独立对战不依赖此局池。旧“原80k局留本机”说明为当时状态。

2026-10-23归档位置更新：本专题关联`runs/entity_set_mlp_256x4_offline_80k_20261021`中的5个抽样账本／逐步日志已按用户授权外置至[当日T7归档](../repository_archive_20261023/README.md)，原路径保留相对链接。复算这些原件前需解包；复训前将需写入的文件恢复为原位真实文件。历史指标、正式结果及来源身份不改写。

存储更新（2026-10-21）：按用户明确授权删除96个普通恢复checkpoint，保留initial、七个已完成5%留出节点及97000步，共9个模型，保留文件SHA核验未变。训练账本、数据、配置和原结果未删；[清理记录与其余模型分级](../checkpoint_storage_audit_20261021/README.md)记录逐文件SHA和删除范围。原停止记录描述的是当时状态，保留不改。

后续更新：停训后的独立新授权greedy1000pair已[完成](../entity_set_mlp_94695_vs_r20_greedy_20261021/README.md)，94695步MLP对R20得**37.075%**，95%配对区间[35.000%,39.175%]。MLP训练仍停止，未启动原自动面板或搜索；[attention128相同8万局训练对照](../entity_attention_128x2_offline_80k_20261021/README.md)随后已训练并停止，选中94695步；对R20 greedy54.40%，后续对R43搜索74.05%，详见其专题。下文停训时的“不启动对战”指当时取消的原自动流程，停止原始记录保留。

2026-10-21按用户要求提前停止16GiB缓存重训，父子进程均已退出；最新可恢复checkpoint为**97000步**，已在CPU成功加载，模型、AdamW、RNG、抽样账本及日志保留。原计划270557步未完成，**不启动对战**。七个已完成百分比节点中，完整留出KL+MSE最低为**94695步：0.530577+0.782832=1.313409**；value MSE最低为40584步的0.768390。81168→94695步的联合指标仅改善0.000942，policy仍缓慢下降而value继续恶化。这里只是已完成节点内的比较，不伪造完整20节点选点；无棋力结论。详见[停止记录](user_stop.json)、[原始汇总](summary.json)及[指标表](results.md)。

此前已按用户要求删除旧8GiB缓存运行的659步产物并从零重训；历史删除范围及性能摘要见[重置记录](cache_restart_16g.json)。入口为[run.py](run.py)，原计划见`plan.json`，配置为[offline_80k_lr1e4_20261021.json](../../configs/entity_set_mlp_v1/offline_80k_lr1e4_20261021.json)。`prepare`仅核对输入，`run`会恢复训练并在完成后对战；用户本次已要求停止，因此不执行恢复命令。

## 第20轮模型的相同留出对照

用户停训后要求只评估raw R20，不进行对战。复用本次训练冻结源码中的`offline_objectives.evaluate`、同一原归档标签、固定引用、batch256及policy有效步掩码，FP32推理；没有训练更新。原始分子／分母、checkpoint SHA和执行源码身份见[r20_same_holdout.json](r20_same_holdout.json)。

| 模型 | 完整留出KL | 完整留出MSE | 固定4096子集KL | 固定4096子集MSE |
| --- | ---: | ---: | ---: | ---: |
| 第20轮raw attention | 0.445110 | 0.755265 | 0.416234 | 0.763407 |
| 新MLP 94695步 | 0.530577 | 0.782832 | 0.490035 | 0.808758 |

完整留出为8011局、641723个value局面，其中134304条有效policy标签；固定子集为4096局面，其中929条有效policy标签。R20的完整KL比该MLP节点低0.085467，MSE低0.027567；MLP单看MSE最佳的40584步（0.768390）也未达到R20的0.755265。由此可确认当前新MLP在同一留出上的两项指标均未追平R20；从零离线训练与继承在线训练的经历不同，不能仅凭这一比较把差距全部归因于结构，也不据此替代棋力对战。

复核命令（只推理，不恢复训练）：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=runtime/entity_set_mlp_256x4_offline_80k_20261021/source/src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u artifacts/entity_set_mlp_256x4_offline_80k_20261021/evaluate_r20.py
```

## 模型和数据

从零初始化当前完整`entity_set_mlp_v1`：牌编码64、贵族32、状态256宽／4残差块、actor/critic128，**855,682参数**。购买关系、动作90维直连、90→32→128非线性旁路、逐通道状态×动作乘积均开启。没有attention、PSW或辅助头；不继承R20权重和AdamW。

使用`runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-000`至`round-019`每轮4000局，合计80000局。**不使用滚动池最后留下的24000局，也不纳入未完整的第21轮**。原归档绝对generation ID为43–62（配置中的1起始轮号44–63），与本分支第1–20轮的显示编号区分。保留原整局split、原搜索访问分布及终局结果，不重新搜索标签。

| split | 对局 | 全部决策 | 有效高预算policy步 |
| --- | ---: | ---: | ---: |
| train | 71,989 | 5,771,871 | 1,206,369 |
| validation | 8,011 | 641,723 | 134,304 |
| 合计 | 80,000 | 6,413,594 | 1,340,673 |

清单由20份已完成`collection_index.json`生成，保留原round/game身份、路径及size/mtime，保存在忽略的`runs/entity_set_mlp_256x4_offline_80k_20261021/inputs/pool_games.jsonl`，SHA写入配置。`plan.json`保存每轮原index／manifest／summary的路径和SHA及教师身份；恢复检查清单SHA、配置与所有源归档size/mtime，不重新解码／重放全部对局，也不替原来没有逐局SHA的index补造历史hash。

这些历史对局由**10贵族池**采集；数值特征可复用，但不含新两张贵族在场样本。训练后的两项对战使用冻结的**12贵族池**，`catalog/`保存其规范表及计划中的指纹。仅改变初始贵族池分布，仍每局选3张；不宣称历史数据来自12贵族。

## 日程、抽样及选点

- 沿用前次从零离线的AdamW配置：lr=1e-4、wd=1e-4、FP32、batch=256、policy/value权重各1、共享encoder和全局clip=1。
- 本次缓存改为`cache_mib=16384`（16GiB），policy/value和留出共用同一个`CompactGameCache`。旧8GiB只能驻留约4.9万局，最近100步取数平均1.414秒、更新0.016秒，发生持续淘汰／重复解压及字典编码；重置准备时估算全8万局compact工作集约13.1GiB、16GiB足够容纳，当时尚未实测新吞吐。容量是数组预算，进程另有对象／模型等开销。首次从冷缓存读取仍需一次解压与编码，后续命中复用。缓存变更不改变样本、训练目标或更新次数。
- 复用现有`PolicyPool`：每步从全部train高预算有效步均匀有放回抽**256条policy**；另从全部train决策抽**256条value**。两份loss相加，一次backward／裁剪／AdamW更新，没有detach或分开更新。低预算步不参与policy监督，value高低预算都用；不对旧数据再做25%下采样。
- 预算沿用12次**value决策有放回曝光**：`ceil(12×5771871/256)=270557`次更新。完整policy batch对应约57.41次policy标签曝光，value约12次；两者曝光率不同，不能把12称为两头各12个epoch。sampling seed43110000，初始化43100000，固定留出43120000。
- 初始评估固定4096子集；之后在5%、10%…100%的20个节点保存checkpoint，均评估同一4096子集及**完整641723条留出**。policy统计依原`policy_weight`只使用有效高预算步，value使用全部留出。记录policy CE、目标熵、KL、预测熵、value MSE、零预测基线、总loss及原始分子／分母。训练逐步记录loss、KL、MSE、梯度范数／裁剪及抽样来源。
- 每1000步另存恢复checkpoint，**不因此增加留出评估**。中断恢复模型、AdamW、Torch／抽样RNG；两头抽样账本分别保存。
- 全部训练结束后，在预先规定的20个百分比checkpoint中，按**完整留出Policy KL＋胜负Value MSE最小**选择整个模型，相等选较早节点。没有按对战结果选点，不拼接policy/value。

## 自动串行对战

对手为本分支**第20轮raw模型**（不是EMA，不是旧R43），checkpoint SHA：`fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd`。

1. Greedy对Greedy：1000pair／2000局，CPU、8 worker，开局seed43200000起。
2. 双方原生S1024N1：500pair／1000局，每世界1024×1=总1024，根噪声0、T0、c_puct1.5、FP/PTP关闭。复用采集执行配置：12搜索worker、1000活跃局、B500、local64、native CUDA Graph，owner **pipeline_slots=1**；开局seed43300000起。

每pair同种子交换先后手、和棋半分，整pair bootstrap10000次输出95%区间。逐局核对实际预算、模型SHA、交换先后手、牌池身份及合法性／异常／截断。种子与正式对战规模冻结在`plan.json`，具体双方配置在选点后生成到`evaluation_configs/`。

首次`run`保存Python源码、native扩展及12贵族data快照到`runtime/<本实验>/source/`；训练、选点、搜索均在这份源码下执行，身份在`source.json`。不会在训练结束后无意切换到工作区后续代码。日志、锁、活动状态位于`runtime/<本实验>/`，checkpoint、逐步数据和账本位于`runs/<本实验>/`；关键留出表、完整汇总和配对结果保存在本目录。

## 命令

以下为原训练／恢复入口，当前已按用户要求停止，不再执行；相同命令会从97000步恢复原270557步计划，并在训练完成后自动对战：

```bash
bash scripts/run_entity_set_mlp_offline_20261021.sh
```

只准备／查看状态，不启动训练或对战：

```bash
bash scripts/run_entity_set_mlp_offline_20261021.sh prepare
bash scripts/run_entity_set_mlp_offline_20261021.sh status
```

`train`动作只训练并完成留出选点；随后用默认命令继续对战。首次执行`run`需要实际CUDA及已构建native扩展；CPU有限测试不替代正式GPU吞吐或棋力结果。

## 有限验证

初版流程相关测试**74项通过**，见保留的`tests.xml`；本次16GiB重置后目标回归**7项通过**，见`cache_reset_tests.xml`。`tests/test_offline_mixed_budget.py`检查高预算policy完整batch、低预算value覆盖、整局split隔离、联合更新和百分比checkpoint、严格恢复后的参数／AdamW／RNG一致、实际855682参数预设两步更新及两项动作头梯度、选点、两个正式对战配置和串行调度。初版另回归原scratch、MLP及在线随机预算训练入口。

16GiB重置后的两项prepare曾核对完整80k清单、固定留出引用及数据摘要一致，确认270557步和20个百分比节点；当时尚未启动的状态记录保留在`validation.json`。随后实际训练已运行至97000步，冻结源码见`source.json`，执行中断见`execution.json`；当前状态以上方停止记录为准。此次停训核对了最新checkpoint可加载、855682参数及七个评估节点checkpoint的SHA，未改训练配置或补跑对战。
