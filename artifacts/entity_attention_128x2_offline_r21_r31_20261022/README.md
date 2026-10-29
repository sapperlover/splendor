# Attention128：第21–31轮固定池、连续上升早停与R20/R31对战

2026-10-23归档位置更新：本专题关联`runtime/entity_attention_128x2_offline_r21_r31_20261022`中的5个抽样账本／逐步日志已按用户授权外置至[当日T7归档](../repository_archive_20261023/README.md)，原路径保留相对链接。复算这些原件前需解包；复训前将需写入的文件恢复为原位真实文件。历史指标、正式结果及来源身份不改写。

2026-10-23归档位置更新：本专题关联`runs/entity_attention_128x2_offline_r21_r31_20261022`中的83个普通checkpoint、5个抽样账本／逐步日志已按用户授权外置至[当日T7归档](../repository_archive_20261023/README.md)，原路径保留相对链接。复算这些原件前需解包；复训前将需写入的文件恢复为原位真实文件。历史指标、正式结果及来源身份不改写。

2026-10-23后续独立补测：原选定59453步模型对原R43，双方S1024N1（总1024）、12贵族、500 pair搜索得70.70% [67.80%,73.60%]；[配置与原始结果](../eval_speed_20261023/README.md)。这是新输出目录下的R43面板，不补跑或改写下方已取消的原R20／R31搜索，训练仍停止。后续日常／在线新搜索采用[配置A](../../docs/guides/evaluation_zh-CN.md)。

2026-10-22最新指令：用户要求停止训练，直接执行两组greedy。已向原监督进程发送SIGINT并确认监督／训练子进程退出，保留84000步恢复点；原148632步预算未完成，也未触发自动早停。对战前按11个已完成百分比节点的完整留出KL＋MSE固定**59453步整个checkpoint**（KL0.227124、MSE0.768013、联合0.995137）。对raw R20／R31各1000pair greedy均已完成，原两组搜索取消；范围变更及候选身份见[manual_stop_greedy_plan.json](manual_stop_greedy_plan.json)、[selected_candidate.json](selected_candidate.json)，结果见[results.md](results.md)。原训练启动事实及中断记录继续保留在launch／execution中。

两组greedy均已完成，共4000局：

| 对手 | pair | 候选胜／和／负 | 候选得分 | 95%整pair区间 |
| --- | ---: | --- | ---: | --- |
| raw R20 | 1000 | 1059／18／923 | 53.400% | [51.250%,55.550%] |
| R31 final | 1000 | 976／29／995 | 49.525% | [47.375%,51.601%] |

支持该候选greedy强于raw R20，尚未确认超过R31，也不能以跨50%的区间宣称两者等效。未测搜索及value棋力；不同对手的两组得分差不是候选直接互打的胜率。异常、非法行动、工程截断和不变量失败均为0；R20面板含2局确定性策略循环判和。全部开局／交换先后手、双模型与规则SHA、逐局零搜索及10000次整pair bootstrap已复核，见[核验记录](greedy_after_stop_verification.json)。原始逐局配对结果见[对R20](panels/greedy_vs_R20.json)／[对R31](panels/greedy_vs_R31.json)；新入口与实际命令见[启动记录](greedy_after_stop_launch.json)、[执行记录](greedy_after_stop_execution.json)。未根据棋力重新选点，原训练不恢复，搜索未运行。

## 数据与训练

使用`entity_attention_r20_s256n4_noble12_iteration_20261022`分支第21–31轮全部**44000局**，包括已经退出在线活动池的对局。保留原始搜索标签和整局train/validation划分，不重新采集或标注。来源局面的`round_index=63–73`，离线配置的1基绝对轮号为64–74；它们与用户所说的第21–31轮是同一批数据，不是旧分支第1–20轮的8万局。

- Train39629局、3170796位置，其中664043条有效policy标签；Validation4371局、349435位置，其中72631条有效policy标签。
- 数据与评测均使用12贵族牌池；逐轮采集manifest、教师谱系、完整性、collection index身份及每局size/mtime已核对，见[plan.json](plan.json)。源NPZ只读复用，不复制或改写。
- 从零初始化attention128×2／4头／FFN512，共583682参数，保留购买关系与`global_players_market` value读取；全新AdamW，无辅助头、PSW、FP或PTP。
- 除数据范围和早停外，配置沿用此前[同8万局attention离线训练](../entity_attention_128x2_offline_80k_20261021/README.md)：lr1e-4、weight decay1e-4、全局clip1、FP32、16GiB按需compact缓存，初始化／抽样／验证seed分别43100000／43110000／43120000。
- policy从正权重高预算位置均匀抽256条，value从全部预算位置均匀抽256条；共享encoder、一次联合反传／全局裁剪／AdamW。12次曝光按value计，上限**148632步**，policy期望曝光约57.300次，不是两头各12次。

配置：[offline_128x2_r21_r31_early_stop_20261022.json](../../configs/entity_attention_v1/offline_128x2_r21_r31_early_stop_20261022.json)。训练内核复用`offline_mixed_budget`／`offline_objectives`／`AlphazeroLearner`，新增`offline_early_stopping`仅组合早停配置与现有流程，旧配置的序列化和默认行为不变。

## 早停和选点

每1000步保存恢复checkpoint；另外在原预算5%、10%……100%节点保存checkpoint，评估相同4096固定留出子集及**全部349435个留出位置**。5%首节点7432步，完整20个节点见plan。

早停只使用完整留出的`policy_kl + value_loss`（胜负MSE）：连续三次严格大于**前一个5%节点**就停止，持平或下降重置计数。step0只做原4096子集检查，不参与早停或选点。因此最早在20%／29727步停止，需要5%→10%→15%→20%连续三次上升；这不是“连续三次未超过历史最佳”。

达到早停或12次value曝光上限后，训练阶段记为complete，并在progress中区分`consecutive_validation_increases`和`exposure_budget`。`final.pt`表示实际停止点；用于对战的是全部已到达5%节点内完整KL+MSE最低的**整个checkpoint**，并列选较早步，不拼接policy/value。选择写入`selection.json`及长期`selected_candidate.json`，在任何对战之前冻结，不能按棋力重新选点。

恢复时从最近已提交checkpoint恢复模型、AdamW和所有RNG，核对冻结配置及数据身份；早停计数从已完成完整留出文件重建。若在checkpoint已保存、完整留出未写完时中断，先补完该节点评估再决定停止，不额外更新。

## 原固定评测队列（搜索已按最新指令取消）

| 顺序 | 候选与对手 | pair／局 | 执行 |
| --- | --- | ---: | --- |
| 1 | 选中checkpoint vs原raw R20 | 1000／2000 | greedy、CPU8worker |
| 2 | 同候选 vs第31轮final | 1000／2000 | greedy、CPU8worker |
| 3 | 同候选 vs原raw R20 | 500／1000 | 用户已取消，未运行 |
| 4 | 同候选 vs第31轮final | 500／1000 | 用户已取消，未运行 |

R20 SHA为`fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd`，R31 SHA为`c57a2a8a7721fe65de90dbc1336483f5d0358d10c4ff4d5c893986e88e588f6c`；使用完整原模型，不是EMA或旧R43。路径、身份和模板已冻结在plan。

每个pair同一开局交换先后手，和棋半分，整pair bootstrap10000次。greedy开局44000000–44000999，搜索44100000–44100499；同模式两组复用开局，但对手不同，不把两组得分差解释为候选直接互打。

搜索无根噪声、温度0、FP/PTP关闭，每世界1024次×1世界，总预算1024。执行配置复用当前采集家族的12 search worker／1000活跃局／B500／FP32／native-cudagraph，评测owner为单槽，checked runner负责真实对局与确定性循环裁决。结束后核对双方模型、规则、逐局实际模拟数、交换先后手、有效局数及配对区间。已完成阶段自动跳过，评测sidecar可恢复。

## 启动、恢复与产物

当前从仓库根目录启动或恢复**仅两组greedy**，复用原冻结评测源码、规则和种子；已完成面板跳过，不恢复训练、不启动搜索：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=runtime/entity_attention_128x2_offline_r21_r31_20261022/source/src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u artifacts/entity_attention_128x2_offline_r21_r31_20261022/run_greedy_after_stop.py run
```

新入口的`prepare`核对停止记录、冻结已到达节点候选及两组greedy配置；`status`只重建汇总，不启动工作负载。`runs/progress.json`继续保留训练`incomplete`，没有将用户中断伪装成完整训练或自动早停。已保存84000步恢复点与选中59453步用途不同。

原`bash scripts/run_attention128_r21_r31_offline_20261022.sh`仍为历史完整训练＋四组队列入口，**不用于本次仅greedy任务**。原配置、预算、代码快照和计划未改写；后续训练续跑或搜索须依据新的明确任务处理。

- `runs/entity_attention_128x2_offline_r21_r31_20261022/`：冻结数据manifest／split、checkpoint、完整留出、progress、早停轨迹、抽样引用账本、逐步训练指标。
- 同名`runtime/`：冻结执行源码／native扩展、阶段日志、进程锁和`live.json`；正式启动总日志为`launch.log`。
- 本目录：配置／计划身份、执行命令与环境、固定候选、`evaluation_configs/`、`panels/`逐局原始统计、[summary.json](summary.json)及[results.md](results.md)。结果汇总在阶段结束后更新，训练进度以runs为准。

72项有限回归全部通过，见[tests.xml](tests.xml)和[verification.json](verification.json)：覆盖严格上升／持平和下降重置、完整留出与子集隔离、停止点、最低节点与平局选早、配置冻结、精确恢复及中断评估恢复、四组队列顺序与完成跳过，以及原learner／混合训练回归。真实GPU短验证见[gpu_smoke.json](gpu_smoke.json)，实际20步训练、四组各1pair共8局（搜索为S2N1总2、eager小池），无异常／非法动作／工程截断，仅验证工程链路，不是正式棋力结果；正式搜索的S1024N1／compiled／12worker配置另由配置回归核对。
