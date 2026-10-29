# 37423步起点：混合预算、policy约6k／value24k持续迭代

2026-10-25存储更新：本分支原80000个采集NPZ已流式外置至[归档20261025](../repository_archive_20261025/README.md)，原路径保留相对链接；原始index／manifest／summary／状态和模型不迁移。旧分支续跑、80k离线复训及全量数据复核须先恢复该包；挂载T7本身不会自动展开归档。下方更早的数据保留说明按当时解释。

2026-10-23归档位置更新：本专题关联`runs/entity_attention_37423_random_budget_p6k_v24k_20261019`中的100个抽样账本／逐步日志已按用户授权外置至[当日T7归档](../repository_archive_20261023/README.md)，原路径保留相对链接。复算这些原件前需解包；复训前将需写入的文件恢复为原位真实文件。历史指标、正式结果及来源身份不改写。 原80000局仍在本机，本批仅外置本run账本／日志。

2026-10-22用户授权从已完成第20轮另设[继承续跑分支](../entity_attention_r20_s256n4_noble12_iteration_20261022/README.md)：继承raw R20的AdamW/RNG及第15–20轮24k活动池，高预算改S256N4总1024，低预算保持S128N1，新采集用12贵族。原第21轮未完成批次已授权删除（2591个NPZ）；原第20轮状态及历史池未动，本目录下文“第21轮已开始”保留为旧时点记录。后续运行与恢复用新分支命令。

存储更新（2026-10-21）：本分支439个历史checkpoint已进入[20261021外盘包](../repository_archive_20261021/README.md)，R20 raw的`round-019/checkpoints/final.pt`仍为本机实体；8万局原NPZ及账本未迁移。需要旧轮模型／EMA来源或恢复旧在线流程前，先按对应批次说明解压恢复。当前另行运行的attention128离线训练不受此迁移影响。

状态：**2026-10-20本次EMA启动前核对：已完成20轮，第21轮采集已开始。** 较早的“完成17轮／第18轮KeyboardInterrupt中断”记录描述当时时点；原训练已由用户恢复。[第17–19轮EMA回溯](../entity_attention_ema_17_19_20261020/README.md)独立运行，不改变本分支模型、AdamW或采集配置。初版及下述2026-10-19调参记录继续保留其当时范围。

从仓库根目录运行；同一命令启动或恢复，默认持续迭代：

```bash
bash scripts/run_random_budget_iteration_20261019.sh
```

默认前台打印采集和训练进度。只读检查可追加 `--check`，不会建立正式输出目录。可追加 `--rounds N` 将完成轮数限制为绝对N轮，恢复时不是再加N轮。Ctrl+C后重用原命令：采集跳过已提交完整局，训练从已提交checkpoint恢复，最多重做尚未保存的50步。不要对同一输出目录修改冻结算法配置。

## 调整采集执行参数及恢复（2026-10-19）

源配置现为 `max_batch_size=500`、`pipeline_slots=2`，batch buckets恢复为16/32/64/128/192/256/384/500；保留12个search worker、总1000个active game等其余配置。第1–3轮为500/1，第4轮为768/2；第5轮已以768/2冻结配置并开始采集，本次改回500后仍沿用768/2完成。**第6轮开始自动使用500/2，无需重启；中断恢复第5轮也仍使用其冻结的768/2。** 本次修改记录见 `batch500_update.json`。

采集配置恢复仅放开 `selfplay.execution` 的 `max_batch_size`、`batch_buckets`、`pipeline_slots` 三个字段；采集的搜索预算、worker数、种子、训练及数据池等其他字段仍须与初始冻结配置一致。搜索评测规模和执行参数的2026-10-20调整另见下文。模型权重、AdamW、抽样RNG、活动池和既有NPZ不因这三个参数变化而重置。不同GPU批次形状可能带来浮点差异，不保证新采集的搜索轨迹逐位相同。

新代码在每轮采集开始前重读启动命令指定的配置文件。每轮 `collection_config.json` 冻结本轮实际采集配置，恢复时优先沿用它；对于旧脚本已开始但没有该文件的轮次，从已有 `selfplay/manifest.json` 读取实际配置并补充记录。原始manifest和顶层初始 `config.json` 不改写；`source/<session>/config-iteration-NNN.json` 记录每轮当时请求的配置。

首次加入恢复及重读支持时，旧进程需退出并重新加载代码一次；该步骤现已完成，第4–5轮已经由新代码运行。之后这三个参数的配置修改在下一新轮次自动读取，当前轮次保持冻结参数。本次改回500没有停止或重启用户进程。

## 配方

- 起点：新版128×2、无辅助头的37423步，SHA `87a7e70a1b6bd4b3b9d8af23ca3775b2776cb779e6a254b56e5748a6743eb5ca`。继承网络、AdamW矩与累计步数、抽样RNG，将lr降到2e-5。不是从零或只加载权重。
- 每轮4000个新对局，按多候选真实决策独立抽样：25% S1024/N1、75% S128/N1。低预算不加根噪声，高预算保留25%根噪声；温度沿用前10个完成回合T1、之后T0。Forced Playouts k=0、Policy Target Pruning=false、PSW关闭、辅助头关闭。
- 每轮1000次更新，policy/value各抽batch256，CE和MSE各权重1；共享encoder、一次合并反传、全局clip1、一次AdamW，weight_decay1e-4。value从所有Train局面抽，包含高/低预算和单候选；policy只抽有效高预算监督。
- 原始历史：原R24–R43恰好20000局，保留整局Train/validation划分，其中17971局为Train。只读检查已核对20000个路径的size/mtime、源manifest SHA、起始模型身份及步数。

## 24k池与policy比例

首轮20k旧局＋4k新局正好24k，不淘汰；之后每轮超额4k，优先从**剩余原始旧局**中均匀无放回随机淘汰4k。原始旧局全部换出后，对新迭代对局按FIFO淘汰最老4k。淘汰仅移出活动池，不删除NPZ；被淘汰的旧局不能重新进入。

历史policy每局固定保留 `floor(有效policy步数×0.25)` 个步骤，由独立seed按整局身份洗牌选择，跨轮/恢复保持同一子集，原标签不改。新局仅使用 `policy_weight>0` 的高预算步，不先抽入低预算再以零权重消掉。

policy的**累计新局部分**配额按活动池中新局占比确定，避免历史标签密度稀释新局：

| 新迭代轮次 | 原始旧局 | 累计新局 | policy新局配额 | policy旧局配额 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20000 | 4000 | 1/6 | 5/6 |
| 2 | 16000 | 8000 | 1/3 | 2/3 |
| 3 | 12000 | 12000 | 1/2 | 1/2 |
| 4 | 8000 | 16000 | 2/3 | 1/3 |
| 5 | 4000 | 20000 | 5/6 | 1/6 |
| 6起 | 0 | 24000 | 1 | 0 |

quota以全部完整局的比例定义；两来源内部均只从Train的有效policy步均匀有放回抽样。value在全部Train局面均匀抽样，不另设quota。新局包含仍留在池里的全部新迭代代次，不只表示最新一轮。256不能整除6时，用相邻累计取整差分配每步配额；任意累计步数与目标误差少于1条样本。约6k等效规模是搜索步密度近似，不是恰好6000个完整对局。

## 记录与评测

输出 `runs/entity_attention_37423_random_budget_p6k_v24k_20261019/`。保存初始历史清单、每轮活动/淘汰清单、policy子集digest、实际新旧抽样数、分头账本、采集预算统计、源码身份、checkpoint及AdamW/RNG。新采集采用数据v3，历史数据只读复用。

每轮0/100/250/500/1000步，在同一份最多2048个整局留出来源的固定局面上记录CE、KL、value MSE等；checkpoint每50步保存。此留出policy指标按原档案有效权重归一化，不等同于训练的新旧quota目标，也不作为教师接纳门槛。每轮完整1000步无条件更新教师。

2026-10-20起，新搜索面板改为每3轮对原R43跑500pair S1024/N1；每轮greedy仍为1000pair。搜索执行配置复用当前采集的12 worker、1000活跃局、B500及native CUDA Graph；按最新要求，评测owner改为`pipeline_slots=1`，采集保持2槽，按模型SHA隔离请求/缓存，保留checked runner。双方全程S1024，不启用采集的低预算随机化、根噪声、温度采样或强制探索/裁剪。第3/6/9/12/15轮旧搜索结果仍为各200pair，不改写、不自动扩样。

同一脚本可继续恢复；搜索评测的pair数、execution及inference执行选项允许更新，搜索预算/种子/对手等仍冻结。已准备面板保留`*_config.json`，新面板使用500pair；第18轮继续沿用其已有采集配置。本次未恢复训练或启动正式面板；有限验证记录见`search_evaluation_update.json`，机制见[评测指南](../../docs/guides/evaluation_zh-CN.md)。

实现：[random_budget_iteration.py](../../src/splendor/random_budget_iteration.py)；[配置](../../configs/entity_attention_v1/random_budget_37423_p6k_v24k_20261019.json)。

## 验证

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q tests/test_random_budget_iteration.py
bash -n scripts/run_random_budget_iteration_20261019.sh
```

4项测试通过：累计quota精确性、随机整局淘汰及FIFO且不返池、固定子集/高预算过滤/value RNG连续性、继承AdamW后中断与连续结果逐参数一致。最后一项使用synthetic数组和**mock采集器**，没有启动真实对局采集；仅在临时目录做少量CPU单元测试更新。

以上为初版验证。本次增加旧轮次恢复后切换768/2、运行中在下一轮重读配置、禁止改变冻结搜索预算的回归；变更后的完整验证记录见 `execution_tuning_verification.json`。测试仍使用mock采集和少量CPU更新，不占用正在采集的GPU。

## 搜索评测执行器验证（2026-10-20）

[更新记录](search_evaluation_update.json)：相关回归74通过、2个CUDA条件测试在CPU沙箱中跳过；另做了真实RTX4050上的独立短验证。双模型普通MCTS/CPW均与checked串行逐局一致，覆盖pipeline1/2、分流、部分pair中断恢复、工程截断排除及已准备面板配置冻结。原脚本`--check`和shell语法检查通过。

[GPU短验证](search_evaluation_gpu_verification.json)只包含2pair／S3和双方模型各两条lane的B500／width64重复根前向；4局正常完成，worker均未创建CUDA上下文，13次图捕获、514次重放，最大logit误差6.68e-6。原始[配置](search_evaluation_gpu_config.json)、[summary](search_evaluation_gpu_summary.json)和[逐局记录](search_evaluation_gpu_games.json)保留；不作为棋力或正式S1024吞吐结论。完整源码快照、dirty patch和临时sidecar在`runtime/search_evaluation_pool_smoke_final_20261020/`。

复现有限GPU工程验证（新输出目录；不启动正式面板）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python \
  artifacts/entity_attention_random_budget_iteration_20261019/search_evaluation_smoke.py \
  --output runtime/search_evaluation_pool_smoke_recheck
```

## 单槽评测与17轮学习轨迹（2026-10-20后续）

搜索评测现为`pipeline_slots=1`；采集仍为2，worker共享slots仍为2，500pair与其他搜索参数不变。新面板采用单槽；已准备的历史面板保持原冻结配置。此改动不启动训练或对战，恢复仍用原脚本。初版两槽GPU短验证保留原始记录；更新依据见`search_single_slot_update.json`。

已汇总17轮×5节点共85份留出记录及17000步训练日志：[完整表格与分析](learning_curves_20261020/results.md)、[逐节点CSV](learning_curves_20261020/validation_by_step.csv)、[逐轮CSV](learning_curves_20261020/round_summary.csv)。原滚动小面板显示policy小幅吸收、后期value改善不稳定；这仅适用于各轮面板，不能解释成第12轮后value停止学习。单看早期KL不足以缩短训练：KL＋MSE在9/17轮及整体均值上以1000步最低。面板跨轮重抽，后期policy有效样本只有406–446，跨轮绝对指标变化混有数据分布差异；下述全量固定面板补充了不同观察。

## 六轮固定整局留出回补（2026-10-20）

[完整结果与轨迹](fixed_holdouts_20261020/results.md)使用第1–6轮2397局原始validation评估第6–11轮，第6–11轮2406局原始validation评估第12–17轮。各面板约4万有效policy步骤、19万value步骤，检查与17轮Train快照无交集；各评估30个节点，共60个。只做FP32前向，未修改训练配置或新增对战。

第6–11轮训练前0步→末轮1000步，KL 0.466969→0.466637，净变化区间跨0；MSE 0.763401→0.751138，Δ−0.012263，整局配对95%区间[−0.015837,−0.008925]。第12–17轮同口径KL 0.430570→0.437992，Δ+0.007422；MSE 0.768743→0.770732，Δ+0.001989，区间跨0。第二组第12–16轮各轮终点MSE仍下降，第17轮500步0.762603、1000步0.770732；末尾回升的大部分在描述性分解中来自平均预测偏差增大。支持优先检查更新／校准稳定性，尚不能归因于局数不足或确定某种稳定手段有效。固定旧teacher的KL恶化也不等于棋力退化；两组绝对数不能跨面板比较。

全部节点CE／KL／MSE／相关性／预测方差及按高低预算分层见[CSV](fixed_holdouts_20261020/checkpoints.csv)和[summary](fixed_holdouts_20261020/summary.json)。[原始清单](fixed_holdouts_20261020/raw_manifest.json)定位逐局充分统计量、数据／模型SHA及源码快照；整局配对bootstrap为4000次，按来源轮次分层。运行和验证见[记录](fixed_holdouts_20261020/verification.json)。

复算统计及绘图，不重新运行网络：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python \
  artifacts/entity_attention_random_budget_iteration_20261019/summarize_fixed_holdouts.py \
  --input artifacts/entity_attention_random_budget_iteration_20261019/fixed_holdouts_20261020/raw \
  --output runtime/fixed_holdouts_recheck
conda run --no-capture-output -n base env MPLCONFIGDIR=/tmp/splendor-fixed-holdouts-mpl python \
  artifacts/entity_attention_random_budget_iteration_20261019/summarize_fixed_holdouts.py \
  --output runtime/fixed_holdouts_recheck --plot-only
```

全量前向复现需要原始NPZ／checkpoint仍在清单位置：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python \
  artifacts/entity_attention_random_budget_iteration_20261019/evaluate_fixed_holdouts.py \
  --output runtime/fixed_holdouts_forward_recheck --device cuda --batch-size 512
```

绘图使用已有base环境Matplotlib，没有改动splendor依赖。两组首模型的优化前向均与现有learner联合forward损失核对，最大差2.08e-7。
