# Attention128：同8万局离线训练对照

最新核验（2026-10-21）：按新授权完成[94695步对raw R20的1000pair greedy](../entity_attention_128x2_94695_vs_r20_greedy_20261021/README.md)，得分54.400%，95%整pair bootstrap区间[52.299%,56.525%]，1078胜／20和／902负。候选在对战前固定为截至108223步的八个节点内完整留出KL+MSE最小者；这不是完整训练后的最终选点，也不是搜索棋力结果。本次未修改或中断训练；核验时训练日志／执行记录已为`interrupted`，约135600步以`KeyboardInterrupt`退出，最后提交恢复点135279步（完整留出KL 0.427218、MSE 0.760894），没有仍在运行的训练进程，未自动恢复。下文清理期间“仍在运行”是当时的历史记录。

后续运行状态（2026-10-21）：交付后已由用户启动，本次[checkpoint存储审计](../checkpoint_storage_audit_20261021/README.md)时仍在运行；本实验全部checkpoint及其后新增模型均列为保留，清理未中断进程或改动配置。下文“正式训练未启动”及`validation.json`指首次准备交付时的历史状态；实时提交进度以`runs/<本实验>/progress.json`及runtime日志为准。

2026-10-21按用户要求准备，**正式训练未启动**。从零训练现有attention128预设，除模型结构及独立输出／manifest路径外，逐字段沿用已停止MLP运行的冻结配置。准备阶段核对两者完整dataset相等、固定4096引用完全相同、预算和20个百分比节点一致；不重新采集或搜索标签。本脚本只训练和留出选点，**不自动对战**。

模型为`entity_attention_v1`：hidden128、2层、4头、FFN512、dropout0，**583,682参数**。保留市场及双方明保留牌的购买关系特征，value读取全局／己方／对手／市场均值；无辅助头、PSW或MLP专属动作旁路。预设见[model_relations_128x2.json](../../configs/entity_attention_v1/model_relations_128x2.json)。

| 项目 | 与MLP相同的设置 |
| --- | --- |
| 数据 | 新在线分支第1–20轮共80000局；train71989局、5771871步、1206369条有效policy；留出8011局、641723步、134304条有效policy |
| 标签与抽样 | 原搜索标签；每次256条高预算有效policy＋256条全预算value，分别均匀有放回抽样，一次联合更新和全局裁剪 |
| 优化 | AdamW，lr=1e-4恒定、wd=1e-4、FP32、两头权重各1、clip=1；随机初始化及全新AdamW |
| 缓存 | 16GiB，共用CompactGameCache，不改变样本 |
| 种子 | 初始化43100000、抽样43110000、固定留出43120000 |
| 预算 | 12次value位置曝光，共270557步；对应约57.41次policy标签曝光；两者不是各12个epoch |
| 评估与保存 | 每5%保存并评估固定4096子集及完整留出；另每1000步仅保存恢复checkpoint，不额外评估 |
| 选点 | 20个预先规定的百分比节点中完整留出KL+胜负MSE最小，平局选较早；选择整个checkpoint，不拼接两头 |

完整配置见[offline_128x2_no_aux_80k_lr1e4_20261021.json](../../configs/entity_attention_v1/offline_128x2_no_aux_80k_lr1e4_20261021.json)，冻结计划和源20轮身份见[plan.json](plan.json)。所有源数据来自旧10贵族池，不含新两张贵族在场样本；本次只拟合已有数值特征，不能声称覆盖新12贵族分布。准备阶段没有checkpoint、训练progress或正式运行源码快照。

仓库根目录启动／中断后恢复（只训练＋留出选点）：

```bash
bash scripts/run_attention128_offline_20261021.sh
```

同一命令会继承本实验模型、AdamW、Torch／抽样RNG及两头账本，继续原预算。`prepare`仅核对数据并生成配置，`status`仅汇总，`select`仅在已完成训练后复核原20节点选点；不会在中断后提前把剩余节点视为完成。

```bash
bash scripts/run_attention128_offline_20261021.sh prepare
bash scripts/run_attention128_offline_20261021.sh status
```

首次正式启动时冻结Python／native源码、模型预设、训练配置和数据目录到`runtime/entity_attention_128x2_offline_80k_20261021/source/`，后续训练／选点使用这份源码；启动后修改工作区代码不会热切换训练。原始训练数据、checkpoint和两头抽样账本位于`runs/entity_attention_128x2_offline_80k_20261021/`；日志在相应`runtime/`；本目录`summary.json`汇总已生成的完整留出与选点，正式运行后`execution.json`记录命令、环境和结束状态。

有限CPU回归**10项通过**，见[tests.xml](tests.xml)：包含实际583682参数attention两步联合更新、精确中断恢复（参数／AdamW／抽样RNG相同）、高预算policy掩码／全部value、百分比留出选点、逐字段配置对照、默认入口只训练＋选点的调度；同时回归原MLP入口。准备与验证记录见[validation.json](validation.json)。这不构成正式GPU训练或棋力结论。
