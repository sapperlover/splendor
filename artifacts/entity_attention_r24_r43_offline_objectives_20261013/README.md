# R24–43 固定数据：PSW 与 value 辅助目标离线训练

2026-10-19后续证据：原数值结果继续保留；[普通训练／仅剩余回合辅助消融](../entity_attention_r24_r43_aux_ablation_20261019/results.md)已完成。旧双辅助减plain的共同对手greedy差+0.050pp [−0.900,+1.025]pp，未确认辅助头额外收益；该后续对照不回填为原10/13实验的一部分。

2026-10-13。两组各22681步训练完成；final对原R43的greedy评测已从200pair扩至各1000pair，原记录保留、各新增800pair。PSW得分51.350% [49.225%,53.450%]，双辅助53.900% [51.675%,56.075%]；支持辅助组对R43的greedy提升，PSW尚未确认。未测搜索棋力，详见[训练与棋力结果](results.md)。两组原R43起点、同绝对R24–43的20000局、原split和相同逐步抽样；PSW只改policy权重，辅助组使用普通policy监督加两个辅助头，两项改进分别启用。

## 固定输入与训练设置

从原43000局清单 `artifacts/entity_attention_r24_history_pool_20261012/pool_games.jsonl` 中筛选 `round_index=23..42`，对应绝对 R24–43，恰好20000局；不包含R1–23或后续S4096数据，不重新搜索标签、不改变原整局Train／validation身份。

| Split | 对局 | 局面 | 有效policy局面 |
| --- | ---: | ---: | ---: |
| Train | 17971 | 1451525 | 1207170 |
| Validation | 2029 | 163595 | 136370 |

起点为 `runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt`（原R43，global72898），SHA256 `b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14`。沿用AdamW、lr2e-5、batch256、weight decay1e-4、policy/value权重1:1、全局clip1、FP32、8GiB无损字典缓存。保留旧参数的AdamW矩，新辅助头没有历史矩。池采样seed34410000、固定留出seed34420000，两组相同；辅助头初始化seed36100000，初始化后恢复R43的Torch随机状态。

全部Train局面均匀有放回抽样，平均追加曝光4次，`ceil(1451525*4/256)=22681`步。两个头使用同一个批次；`history_batch_size`不产生历史／当前分池抽样。固定4096个整局留出内的局面，按20代平衡分配，开始／每1000步／末尾验证；末尾流式评估全部163595个留出局面。逐步训练loss、普通留出policy CE/KL、value MSE、辅助误差、原始抽样引用和源码身份都保存。PSW训练CE/KL按surprise权重统计，普通留出不加权。固定留出由两组同一seed生成；训练曲线记录当前训练批次，未额外建立固定Train子集。

## 两种目标

PSW：沿用已经实现的50%局内均匀／surprise混合，surprise为 `KL(已存搜索分布 || 采集时未加噪prior)`。只调整policy权重，保持每局policy总权重，value权重不变。正式留出仍用普通、不加surprise权重的CE/KL，便于共同比较。历史局面仍使用归档教师标签；本命令没有用R43重标注历史局面。

辅助组：架构为 `entity_attention_aux_v1`，在原共享encoder的global token后增加两个独立MLP；旧policy与胜负value参数逐位继承，新的heads不改变初始policy/value输出。

- 分差标签：`(终局当前行动者得分 - 对手得分)/15`，线性输出，不裁剪；并列分数与胜负标签分别保留，不能用分差替代按规则决出的胜负。
- 剩余回合标签：`(终局双方completed_turns之和 - 当前completed_player_turns)/40`，softplus输出。一个玩家的完整行动算一回合，支付／弃牌／选贵族不另加回合；处在未完成回合中时仍计入该回合。
- 联合损失：`CE + 胜负MSE + 0.25*分差MSE + 0.25*剩余回合MSE`。两个辅助梯度进入共享encoder，单次全局裁剪和AdamW更新。0.25是本次准备的初始系数，未由棋力实验选定。

终局信息只生成监督标签，不进入模型输入。归档schema与原始NPZ不变。checkpoint严格记录新架构，frozen loader可直接读取；搜索／greedy仍使用原policy与胜负value输出，不把辅助预测混入搜索Q。当前只有本离线入口接入辅助损失，尚未给在线自迭代配置新增辅助开关。

## 启动与恢复

从仓库根目录依次执行，分别写独立目录，不自动执行评测：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m splendor.offline_objectives \
  --config configs/entity_attention_v1/offline_r24_r43_policy_surprise_20261013.json \
  --output runs/entity_attention_r24_r43_policy_surprise_offline_20261013

conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m splendor.offline_objectives \
  --config configs/entity_attention_v1/offline_r24_r43_value_auxiliary_20261013.json \
  --output runs/entity_attention_r24_r43_value_auxiliary_offline_20261013
```

重复同一命令自动从最近完整checkpoint恢复，校验配置、数据身份与源归档size/mtime；中途打断最多回退到最近1000步节点。`--max-steps-this-session N`可以限定本次调用的新增更新数，并在结束时保存。`--prepare-only`仅冻结输入和计划，不加载训练模型。两组已完成正式训练，终点和原始指标保存在下述输出目录；greedy评测由独立命令执行。

各输出目录的 `checkpoints/final.pt` 为训练终点；中间节点为 `checkpoints/step-XXXXXXXXX.pt`。`progress.json`记录训练完成状态，`validation_curve.jsonl`、`full_validation.json`、`training_metrics.jsonl`保留指标，`references/`保存逐步引用，`source/`保存每次调用的源码／命令身份，`dataset.json`保存全量冻结清单。重型产物在被忽略的runs目录；本目录保存准备摘要、有限验证、训练汇总及greedy逐pair原始结果。

## 已完成的1000pair greedy对原R43

原200pair记录保留在 `greedy_vs_r43/`。按用户新标准，两组各补800pair，最终各1000pair／2000局，共4000局；同一完整开局库、交换先后手，纯greedy、CPU八worker，和棋半分，完整pair bootstrap 10000次。

| 候选 | 胜／和／负 | 得分率 | 95%配对区间 |
| --- | ---: | ---: | --- |
| PSW | 1017／20／963 | 51.350% | [49.225%,53.450%] |
| 双辅助 | 1066／24／910 | 53.900% | [51.675%,56.075%] |

辅助组对R43的区间高于50%，PSW尚未确认提升。辅助减PSW对共同R43的得分差+2.550pp，95%区间[−0.050,+5.200]pp；不是候选直接互打，也尚未确认候选之间的差异。3999局自然终局、1局policy cycle draw按现行规则计半分，零工程失败；剔除循环所在完整pair的敏感性检查仍支持辅助组对R43的提升。扩样复用已查看的前200pair，不能称为新的独立确认，区间未作顺序检验校正。

[1000pair报告](greedy_vs_r43_1000pair/results.md)、[原200pair报告](greedy_vs_r43/results.md)、[扩样配置与原件身份](greedy_vs_r43_1000pair/manifest.json)、[合并与敏感性核对](greedy_vs_r43_1000pair/verification.json)。AGENTS.md已约定：以后纯greedy默认每组1000pair，含搜索策略的对战默认每组200pair，用户明确指定时覆盖默认值。

恢复同一扩样面板，已完成的新增部分直接跳过：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u artifacts/entity_attention_r24_r43_offline_objectives_20261013/greedy_vs_r43_1000pair/run.py run
```

纯CPU重算已有训练与最新棋力汇总，不执行训练、搜索或对战：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_r24_r43_offline_objectives_20261013/analyze.py
```

## 验证

[verification.json](verification.json)保存正式清单、两组同数据／同验证引用／同预算的核对，以及20代各一局的标签与PSW检查。67项相关CPU测试通过，见[tests.xml](tests.xml)：覆盖分差视角、未完成回合、旧参数和AdamW矩继承、辅助梯度进入encoder、严格checkpoint加载、两组相同抽样、完整离线调用和连续／断点恢复逐位一致。另用真实R43和原始Train样本，对两组各做1步CPU冒烟并复核frozen loader输出，临时checkpoint在runtime，不计入正式训练。正式训练与greedy结果现见[结果报告](results.md)；verification.json中的prepared状态是交付时有限验证的历史记录。

```bash
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_value_auxiliary.py tests/test_entity_attention.py \
  tests/test_alphazero_training.py tests/test_policy_surprise.py \
  tests/test_pool_iteration.py tests/test_hybrid_training.py
```

## 2026-10-19冷产物位置

非主要checkpoint、逐步指标及抽样账本已集中到`archive/20261019/runs/`下相同run结构；关键模型与验证／选点文件留在原位置，原文件路径保留相对链接。见[逐文件迁移清单、保留策略及恢复说明](../repository_archive_20261019/README.md)。只读诊断可在恢复挂载后访问；精确续训前须按清单撤回该run的迁移，恢复原位真实账本／日志和权重，避免续训改写冻结归档。本次没有启动或恢复训练。
