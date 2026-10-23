# 第21–31轮attention128：最低KL+MSE节点降学习率续训

2026-10-22最新状态：本段正式训练已运行，新增10000步（累计69453步）因完整留出KL＋MSE连续三次严格上升而正常早停；最低节点为新增4000步、累计63453步，KL0.217213／MSE0.759487／联合0.976699。完整训练曲线见[results.md](results.md)，原训练入口仍只训练和选点。

用户追加授权对R31做1000pair greedy；本次使用已选中的完整4000步checkpoint，不按对战重新选点。复用父模型对R31面板的12贵族规则、冻结评测源码、44000000–44000999开局、CPU8worker、和棋半分及10000次整pair bootstrap，共2000局。配置与候选／对手SHA见[greedy_vs_R31_plan.json](greedy_vs_R31_plan.json)，实际命令与阶段见[greedy_vs_R31_execution.json](greedy_vs_R31_execution.json)。这是原R31已查看开局面板的复测，原训练summary/results继续仅报告训练；本次评测单独保存，不恢复训练或启动搜索。

评测已完成：992胜／22和／986负，得分**50.150% [48.025%,52.300%]**；异常、非法动作、工程截断、不变量失败均0。相对降lr前同开局对R31的49.525%，得分差+0.625pp，配对95%区间[-2.051,+3.326]pp，未确认greedy棋力提升或超过R31。完整报告与复现命令见[greedy_vs_R31_results.md](greedy_vs_R31_results.md)，原始逐局结果与核验见[greedy_vs_R31_summary.json](greedy_vs_R31_summary.json)；未按棋力重新选点。

原训练命令如下，本段已完成后重复执行直接返回，不启动任何对战：

```bash
bash scripts/run_attention128_r21_r31_lr2e5_20261022.sh
```

## 起点与不变项

从[上一离线实验](../entity_attention_128x2_offline_r21_r31_20261022/README.md)已完成的11个完整留出节点内最低点继续：**原59453步**，完整留出KL **0.2271240486100514**、胜负MSE **0.7680132969437563**，联合 **0.9951373455538077**。不是最后84000步恢复点。源checkpoint SHA为`3f43ec717fe3b2a548f5eb3e5e9878d3e402b15c9d848871e9934543b4862f5a`，位于原run的`checkpoints/step-000059453.pt`；候选、完整留出及配置身份见[plan.json](plan.json)。不按既有greedy结果另选起点。

保留该节点的完整权重、AdamW一二阶矩和更新计数、采样RNG、Torch CPU/CUDA RNG；**只把优化器学习率从1e-4改为2e-5**。不重新初始化网络或优化器，不重新从采样seed开头抽样。累计训练步继续从59453计，本段局部步从0计。

保持attention128×2／4头／FFN512、583682参数、购买关系及value读取、FP32、weight decay1e-4、全局clip1、policy/value损失权重1/1、每1000步恢复点、16GiB按需compact缓存。辅助目标和PSW关闭。仍使用第21–31轮44000局原始标签与整局split：39629训练局／3170796位置，4371留出局／349435位置，其中有效policy标签分别664043和72631条；固定4096留出引用逐条与父run一致。policy高预算正权重和value全部预算各B256、一次联合更新不变。原NPZ、父run与父模型均只读，不改写或恢复父训练。

## 评估、早停与预算

改为每**新增2000步**评估同一固定4096子集及完整349435个留出位置，分别记录policy KL与胜负MSE；完整留出的KL+MSE用于早停。本段2000步对应累计61453步，4000步对应63453步，以此类推。结束点不足下一2000步时补一次完整评估。

按“其他不变”保留连续三次严格上升早停：比较相邻已完成检查点，持平或下降重置；本段step0不参与比较，不继承父实验的上升计数。最早在本段8000步（2000→4000→6000→8000）停止。训练起点原完整指标保存在plan中；step0仍只做原固定4096子集检查。

沿用12次value曝光作为**本段新增训练上限**，最多新增148632步（累计208085步），不是恢复父run并扣除59453步。policy新增期望曝光约57.300次。早停通常可能提前结束，上限不构成必须跑满的要求。字段见[配置](../../configs/entity_attention_v1/offline_128x2_r21_r31_lr2e5_20261022.json)。

达到早停或上限后，在本段已到达的2000步检查点及结束点中按完整KL+MSE选最低的整个checkpoint，并列选较早步；父起点不混入本段候选集合。`selection.json`使用本段局部步，累计步为59453＋本段步。**选点后直接结束，不启动对战。**

## 恢复与产物

同一命令从本段最近提交checkpoint恢复模型、AdamW和RNG，冻结数据／配置必须相同；已完成时直接返回。执行源码、模型构建代码、native扩展及规则数据已冻结到同名runtime的`source/`，身份见`source.json`。`prepare`仅生成准备材料，`status`仅重建summary和results，不启动负载。正式运行日志为同名runtime的`train.log`，`live.json`记录监督与子进程。

- `runs/entity_attention_128x2_r21_r31_lr2e5_20261022/`：冻结数据／划分、checkpoint、逐步指标、两头抽样账本、固定与完整留出、早停轨迹、进度、选点。
- 本目录：[plan.json](plan.json)、[summary.json](summary.json)、[results.md](results.md)，以及实际执行后生成的命令与环境记录`execution.json`。
- 有限验证：[tests.xml](tests.xml)，包含权重／AdamW／全部CPU RNG继承、只改lr、固定间隔／非整倍数结束点评估、早停、精确中断恢复、选点与无对战CLI。这些准备时测试使用微型合成池；后续实际训练与追加授权greedy的执行记录分别保留，不将回归结果充作棋力证据。

复用通用`offline_objectives`和`AlphazeroLearner`，新增`offline_finetune`组合继承初始化、原混合预算抽样和固定间隔评估；原从零／百分比配置及其默认行为保留。
