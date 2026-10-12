# 近期 attention S1024/N1 大池离线训练

2026-10-09。用户要求汇总近期有效 attention 自对弈，形成固定大离线池，并仿照此前成功的离线经验充分训练。**103,194 步训练已完成；终点对 θ5000 的 200-pair 搜索得分 55.50%。八个拟合节点中选出的 34,398 步得分 86.125%。最新组合使用最终网络的 policy 与 34,398 步网络的 value，于 23:27 完成 200 pair，得分 92.25%，配对 95% CI [89.625%,94.75%]；较原 34,398 步提高 6.125 个百分点，配对差 CI [+2.00,+10.25]。** 同一轨迹上 policy 的后续进步与 value 的后续退化可以同时存在，联合模型的停止点会掩盖前者。训练状态见 [progress.json](progress.json)，原训练曲线见 [training_summary.json](training_summary.json)，原三里程碑拟合复核见 [fit_review.json](fit_review.json)，选点与棋力见 [extended_selection.json](extended_selection.json)、[early_battle_summary.json](early_battle_summary.json)、[policy_swap_summary.json](policy_swap_summary.json)。

`training_summary.json` 和 `fit_review.json` 保留训练结束时的原始复核快照，其中“棋力未测／正在测”的状态说明已由下文两组完整搜索面板替代；原数值与历史文件保持可复核。

## 数据与标签

共 **29,000 局、2,448,925 个决策局面**；原整局划分为 Train 26,065 局／2,201,472 局面，validation 2,935 局／247,453 局面。有效 policy 局面分别为 1,735,434／195,067；单候选局面的 policy 权重为零，value 仍参与训练。

| 来源 | 总局数 | Train 局面 | 留出局面 |
| --- | ---: | ---: | ---: |
| 早期 attention 自迭代（3×1000） | 3,000 | 232,094 | 25,647 |
| 混合训练原 9 轮（9×1000） | 9,000 | 696,358 | 77,961 |
| S1024/N1 连续 50 轮（50×200） | 10,000 | 756,141 | 85,329 |
| 分头窗口训练（20×200） | 4,000 | 296,050 | 33,714 |
| M4 分支行为对局（12×200） | 2,400 | 176,558 | 20,062 |
| 50 步趋势分支（2×200） | 400 | 29,794 | 2,886 |
| native 教师探针 | 200 | 14,477 | 1,854 |

以上全部来自标准 determinized native S1024/N1、attention 对自身的搜索自对弈，采用原 NPZ 的 `policy_target` 和真实自然终局 `value_target`。M4 分支只纳入行为轨迹及行为搜索的原始单份标签，额外平均标签 sidecar 不混入。这里的池是冻结引用清单，原 NPZ 保持原位，不重复复制重型对局文件。

没有重复轨迹或重复 `game_seed`，保留原来的整局 train/validation 身份；给各采集轮重新分配全池唯一的 round ID，同时保存原 run/round/game 身份，避免不同分支的同名 game ID 冲突。完整归档路径、SHA、字节数、mtime、教师身份与划分见 [pool_games.jsonl](pool_games.jsonl)，来源配置、源码／规则／搜索版本及代表性数值核验见 [sources.json](sources.json)，统计见 [pool_summary.json](pool_summary.json)。

已核对全部来源 manifest、完成 summary、教师文件 SHA 与实际 attention 配置、每个归档 SHA 和终局元数据；常规每个采集轮抽查首／中／末三局的完整数值契约。50 轮的 `round-002` 留有一次历史 batch 中断标记，但最终 200 局均已完成且 failed/exception/truncation 为零；额外对该轮全部 200 局进行数值核验后纳入。没有重新做全量规则重放。S512/N2、chance/CPW、smoke、对战评测、仅重标注的数据副本、未完成／空采集轮和旧 DeepSets 教师数据不进入本池。

## 起点与日程

采用与此前成功离线蒸馏相同的 **532,738 参数 `entity_attention_v1` 随机初始化**，initialization seed=13100000；这使新学生的学习来源清晰，原始留出局从未参与该学生训练。已核对与旧成功实验 `initial.pt` 的模型参数 SHA 完全相同，见 [initialization_comparison.json](initialization_comparison.json)。此前成功结果本身也是随机初始化 attention 在固定池上离线训练，故此处先复现这一经验；没有预先确定哪一个旧 checkpoint 是最佳热启动。

固定原池，所有 Train 决策局面均匀有放回抽样，**无容量淘汰、无近期加权、无新数据采集或教师迭代**。不同来源的抽样占比由其 Train 局面总量决定。batch=256，AdamW，weight decay=1e-4，policy/value loss 权重 1:1，联合训练全部参数，整体梯度裁剪 1.0；没有沿用后期分头窗口与 detached value 的训练方法。

总预算为平均抽样 **12 次／Train 局面，103,194 步**。前 68,796 步（平均 8 次）lr=1e-4，之后 lr=2.5e-5；预算与降学习率点事先固定。相对旧成功实验的约 4 次平均抽样，本次给出更长拟合预算及后期细化。这里的“次”是平均有放回抽样覆盖量，不是每个样本恰好经过一轮，也不预先认定 12 次为最优。

每 1000 步及中断边界保存模型、AdamW、Torch RNG、采样 RNG 与进度。保留全部定期 checkpoint，另保存 `best_validation_kl.pt` 和完成时的 `final.pt`。模型选择使用同一固定 DEV 子集的来源加权 policy KL；不以最低 Train loss 或最后一步自动认定最优棋力。

## 完成后的拟合结论

| 平均抽样次数 | 步数 | 完整留出 KL | 完整留出 value MSE | Train 固定子集 value MSE |
| --- | ---: | ---: | ---: | ---: |
| 4 | 34,398 | 0.441453 | 0.742282 | 0.722482 |
| 8 | 68,796 | 0.425301 | 0.868594 | 0.512619 |
| 12 | 103,194 | 0.415934 | 0.929513 | 0.404004 |

policy 仍在改善，但后段边际收益较小；8→12 次完整留出 KL 再降约 2.2%。这一段 lr 也降低了 4 倍，不能据此确定架构容量／标签噪声的拟合上限。Train 与留出 policy KL 没有出现 value 那样的明显分叉。

**value 明显过拟合。** 三个完整检查点间 Train MSE 下降、留出 MSE 持续上升。4→12 次在全部 7 个来源上均恶化；按 2,935 局配对 bootstrap，ΔMSE=+0.187231，95% CI [+0.172091,+0.201809]，同时 ΔKL=−0.025520，95% CI [−0.026315,−0.024719]。区间反映这批留出对局的经验变异，不涵盖训练 seed 变异或反复选 checkpoint 的偏差。

前中期完整留出 MSE 从 0.802483 升至 0.996241，相关系数从 0.4358 降至 0.3215；终点 MSE 点估计接近、略高于该阶段零预测基线 0.988665。后期 MSE 从 0.391489 升至 0.540690，相关系数从 0.7779 降至 0.6917。因此退化涉及两个阶段，并非只有均值校准变化，也没有证明 value 已完全不含信息。

固定子集 value MSE 最低节点是 31,000 步（0.763455），policy KL 最低节点是 102,000 步（0.408611）；后者的完整留出 MSE 为 0.933319。**KL 最佳节点不能自动视为搜索棋力最佳。** 4 次抽样只是三个完整里程碑中 value 最好，当前未确定精确最优停止点。

总训练会话约 59.8 分钟（前 8 步 93 秒，长会话 3494 秒，不含会话间处理时间）；无损缓存最终容纳全部 29,000 局，长会话 RSS 峰值约 6.73 GiB，CUDA allocated 峰值约 223 MiB。学习上的主要问题是 value 后期泛化退化，不是仍在反复解压或内存增长。继续同样的 policy/value 联合长训缺少依据，应将后续价值目标的停止／正则化与 policy 日程区分，并用实际搜索对战选择模型；本次没有追加训练或扫描更多节点。

## 记录与验证含义

- 每步：policy CE／KL、value MSE、目标／预测熵、value 零预测基线、梯度范数与裁剪、实际 lr、各来源抽样数、精确局面引用与哈希、抽样／组批／GPU 更新耗时。
- 每 1000 步：每个来源各最多 2048 个固定 Train 和 2048 个固定 validation 局面，记录 KL、MSE、CE、熵、允许教师首选并列的 top1 一致率、value bias／相关系数；按双方最高分是否达到 10 分拆前中期／后期，与之前阶段搜索口径一致。
- 每个来源的固定子集便于比较；总 KL 用原池各来源有效 policy 局面数加权，MSE 用各来源局面数加权。原始 `__all_observed__` 子集聚合偏向来源平衡，不等于全池均值，不能拿它替代来源加权结果。
- 第 **34,398／68,796／103,194 步**（平均抽样 4／8／12 次）评估全部 **247,453** 个原 validation 局面；末尾额外评估子集 KL 最佳 checkpoint 的完整留出。完整结果保存各来源、阶段以及逐局指标，支持后续按对局配对复核变化。
- 额外保存已有 hybrid9500、latest17700 两个 checkpoint 在同一固定子集上的参考值，见 [baselines.json](baselines.json)。这些是现有模型的拟合参考，不是本次学生的初始权重。
- 精确抽样账本及最终覆盖率／未抽中比例／使用次数分布均保留。全量留出是已经用于多轮研究的 DEV 数据；没有新独立 FINAL 棋力证据，也不能由 stored KL／MSE 的下降推出学生强于搜索教师或 PPO。

原始数据解压后约 19.3 GiB，超出当前可用内存。采用 **8 GiB 上限的无损字典缓存**：仅对输入 float32 的位模式编码，抽样时还原需要的行／候选切片；标签保持原值。缓存不改变训练算法、采样流或输入精度，也不产生大磁盘副本。已验证全部组批字段与原始样本一致，4 步模型逐位复现原始缓存版本；连续 4 步与 2+2 续跑的模型、AdamW、RNG、引用、覆盖统计逐位一致，并验证丢弃未提交日志／账本尾部。流式完整留出 KL／MSE 与生产评估函数吻合。见 [verification.json](verification.json)。这是执行等价性与有限资源验证，不承诺任意机器的资源峰值。

正式运行预热后，已测缓存约 4.8 GiB、进程 RSS 约 6.7 GiB；最近 100 步平均抽样约 0.001 秒、组批约 0.010 秒、GPU 更新约 0.015 秒。原始缓存首批 8 步平均抽样约 0.678 秒，但两个阶段抽样根与缓存热度不同，不能把比值当成严密的对照加速倍率。实测快照见 [execution_profile.json](execution_profile.json)，后续完整会话耗时以 runs 下 sessions.jsonl 为准。

正式前 8 步已使用原始缓存提交，随后从同一 checkpoint／AdamW／RNG 续跑无损缓存，保持实验身份。首次会话训练已成功，结束后的绘图因 `splendor` 环境没有 matplotlib 失败；已修正为调用本机已核验的 `base` 环境绘图，训练 checkpoint 与数据未丢失。

## 命令与位置

训练现已全部结束，无需再续跑。“续跑”仅指中断后恢复已提交状态，不表示预算尚未完成。从仓库根目录，以下命令自动准备／核验冻结清单、恢复最近 checkpoint、训练并产出汇总和图；保留用于复核及恢复操作。**已有后台任务正在运行时不要再启动第二个**，训练目录锁会拒绝并发写入。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u artifacts/entity_attention_recent_pool_20261009/run.py run
```

Ctrl+C 或向 `train.py` 进程发送 SIGTERM 会在当前更新结束后保存 checkpoint。再次执行同一命令续跑，恢复模型、优化器和 RNG；崩溃时恢复最后已提交 checkpoint，丢弃之后的日志与引用。支持 `--max-steps-this-session N` 分段运行。后处理绘图调用 `conda run -n base` 的 matplotlib；训练始终使用 `splendor`。

```bash
# 查看提交进度及最近实时进度；live.json 的 time_unix 用于识别陈旧状态
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_recent_pool_20261009/run.py status
# 训练日志
tail -f runtime/entity_attention_recent_pool_20261009/train.log
# 训练期间或完成后重新生成轻量汇总／曲线
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_recent_pool_20261009/run.py analyze
```

配置见 [config.json](config.json)。轻量发布结果在本目录；NPZ 原件仍在各来源 run，checkpoint、逐步日志、精确抽样账本、源码／命令／环境快照在 `runs/entity_attention_recent_pool_20261009/`；后台启动身份、stdout、有限验证现场在 `runtime/entity_attention_recent_pool_20261009/`。首次保守排除历史中断轮的准备清单留在 runtime 的 `first_preparation/`，正式身份只以当前本目录冻结清单为准。

## 最终模型对 θ5000：200 pair

用户在训练完成后明确要求测一个对 θ5000 的 200-pair 带搜索面板。候选按预定终点选 `final.pt`（103,194 步），并非以此宣称它是棋力最佳节点；双方 S1024/N1、无根噪声、温度 0、互换先后手，共 400 局。采用此前四面板同一批 DEV 种子（game 14900000+、policy 14910000+、search 14930000+），以便计算与旧离线9500／hybrid第3、6、9轮的配对得分差及区间；不称独立 FINAL。

沿用既有 `native_batched`、100 active games、max batch=64 的实际评测配置。配置见 [battle_config.json](battle_config.json)，原始结果见 [battles/final103194_vs_theta5000.json](battles/final103194_vs_theta5000.json)，汇总见 [battle_summary.json](battle_summary.json)。19:49:40 开始，20:11:40 完成，实际评测会话约 21.9 分钟。

**终点模型 220 胜、4 和、176 负，得分 55.50%，按 200 个整 pair bootstrap 的 95% CI [50.75%,60.125%]。** 和棋计半分，纯胜局比例为 55.00%。候选先手 98 胜／2 和／100 负，得分 49.50%；后手 122 胜／2 和／76 负，得分 61.50%。400 局全部自然终局，异常、非法动作、不变量失败、工程截断和循环裁和均为零。实际配置、两模型文件 SHA、互换先后手及全部区间已独立调用生产统计函数复算，见 [battle_verification.json](battle_verification.json)。

| 同种子旧面板（均对 θ5000） | 旧得分 | 新终点减旧得分 | 配对差 95% CI |
| --- | ---: | ---: | ---: |
| 旧离线9500 | 67.375% | −11.875 个百分点 | [−18.375,−5.375] 个百分点 |
| hybrid 第3轮 | 58.875% | −3.375 个百分点 | [−10.125,+3.125] 个百分点 |
| hybrid 第6轮 | 66.750% | −11.250 个百分点 | [−17.625,−4.750] 个百分点 |
| hybrid 第9轮 | 74.375% | −18.875 个百分点 | [−24.750,−13.000] 个百分点 |

终点在本 DEV 面板上略胜 θ5000，但没有复现旧离线9500或 hybrid 后段的结果；与这三者的配对得分差区间均低于零，与第3轮尚不能区分。以上是同一对手上的得分比较，并非新旧学生直接对战；差值为探索性比较，未作多重比较校正。较大的池与更长训练并未自动带来更强的终点，不能从 policy KL 降低推出搜索棋力提高。

结合完整留出指标，value 后期过拟合是明确问题，但本面板没有隔离 critic 的因果贡献。终点评测时尚未测较早节点，随后按用户要求追加下节的早期选点与一个搜索面板；不能预先声称更早节点一定更强，也不能确定精确最优停止点。池的来源／目标差异、随机初始化与优化日程的作用也尚未被独立控制。

```bash
# 此面板已完成；保留命令用于复核或中断恢复，不会追加新 pair
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  python -u artifacts/entity_attention_recent_pool_20261009/battle.py run
tail -f runtime/entity_attention_recent_pool_20261009/battle.log
# 复算完成训练的配对拟合变化；不启动搜索
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_recent_pool_20261009/analyze_fit.py
```

对战的逐局可恢复 sidecar 为相邻的 `*.json.games/`，按仓库规则忽略；配置、汇总和结论纳管，源码／命令身份在 `runtime/entity_attention_recent_pool_20261009/battle/`。这一面板只评价终点模型相对 θ5000 的结果，不能单独确定最佳步数或把棋力变化归因于 critic。

## 早期三个取点与选点对战（补充）

用户要求在 0～34,398 步间均匀补充三个取点，评估 value MSE 和 policy KL，随后选一个“看起来最好”的点做 200 pair。理想内部点为 8,599.5／17,199／25,798.5 步；直接使用最近的已有 **9,000／17,000／26,000 步** checkpoint，不为补点重新训练。整局划分及完整留出保持不变：2,935 局、247,453 局面，其中有效 policy 局面 195,067。

在看到这三点的完整留出结果之前，冻结选点规则：把三点与已测 34,398 步共同比较，选择 **完整留出 KL＋MSE 最小**的节点，并列选更早者。这沿用原训练 1:1 权重；对固定标签而言，CE 与 KL 只差同一目标熵常数，因此该和与原监督目标的排序一致。它是一个明确的 DEV 拟合选点规则，不能预先保证搜索棋力最好；报告同时保留单项与分阶段指标。

流程于 20:19 启动。三个早期节点的完整留出已完成；用户随后要求先暂停搜索，补测平均抽样 5～8 次。按同样的“最近已有 checkpoint”规则，新增 43,000／52,000／60,000 步，实际平均抽样为 5.0003／6.0469／6.9771 次；8 次复用既有 68,796 步完整结果。没有为对齐精确步数重新训练。

| 步数 | 实际平均抽样次数 | 完整留出 policy KL | 完整留出 value MSE | KL＋MSE |
| --- | ---: | ---: | ---: | ---: |
| 9,000 | 1.0466 | 0.493718 | 0.792806 | 1.286523 |
| 17,000 | 1.9769 | 0.466717 | 0.763495 | 1.230212 |
| 26,000 | 3.0234 | 0.457680 | 0.747820 | 1.205500 |
| **34,398** | **4.0000** | **0.441453** | **0.742282** | **1.183736** |
| 43,000 | 5.0003 | 0.434271 | 0.757242 | 1.191513 |
| 52,000 | 6.0469 | 0.429051 | 0.802622 | 1.231674 |
| 60,000 | 6.9771 | 0.426915 | 0.831564 | 1.258479 |
| 68,796 | 8.0000 | 0.425301 | 0.868594 | 1.293895 |

**八个已测节点中 value MSE 和原 1:1 联合拟合目标最低的均为 34,398 步，policy KL 则继续下降。** 26,000→34,398 步 ΔMSE=−0.005537，整局配对 95% CI [−0.009140,−0.001966]；34,398→43,000 步 ΔMSE=+0.014959，CI [+0.009555,+0.020346]。这批留出数据支持约第 4 次平均抽样附近出现 value 的转折，后续优先加密区间为 26,000～43,000 步；尚未确定精确最低点，也未证明全局最优。区间不覆盖训练 seed 变异或自适应选点偏差。

4→5→6→7→8 次的前中期 MSE 为 0.802483／0.817772／0.861867／0.895088／0.929680，后期为 0.391489／0.404530／0.457403／0.461410／0.512643；两个阶段均有后期退化，相关系数也总体下降。这里的“后期退化”是训练日程的后段，并非只指棋局后期。policy 与 value 的最好停止点分离，单一联合目标的最小值也不是棋力保证。

对应的固定 Train 子集 value MSE 持续从 0.722482 降至 0.687216／0.649385／0.584718／0.512619，留出误差却上升，支持训练后段的过拟合判断。Train 这里是原固定子集，完整留出是全量，两种统计范围应明确区分；原曲线见 [training_summary.json](training_summary.json)。

评估复用 [train.py](train.py) 的完整留出实现，不恢复训练优化器、不进行参数更新。初始取点冻结见 [early_probe_config.json](early_probe_config.json)，初始选点见 [early_selection.json](early_selection.json)；5～8 次冻结见 [mid_probe_config.json](mid_probe_config.json)，完整八点与最终选择见 [extended_selection.json](extended_selection.json)，配对拟合变化见 [extended_fit_review.json](extended_fit_review.json)。所有新增节点均核对整局／阶段局面身份和权重、全量聚合口径及 checkpoint SHA。

补点后候选仍为 34,398 步，于 20:29 恢复原定的单个 200-pair 面板，**20:50:15 全部完成**；恢复后的评测会话约 20.7 分钟，不含首次暂停前的未完成搜索和补点时间。原冻结搜索配置与候选不变，补充选点身份单独记录；暂停／恢复命令与时间保存于 runtime。搜索配置见 [early_battle_config.json](early_battle_config.json)，原始结果见 [battles/early_step000034398_vs_theta5000.json](battles/early_step000034398_vs_theta5000.json)，最终统计见 [early_battle_summary.json](early_battle_summary.json)。

**34,398 步：344 胜、1 和、55 负；纯胜率 86.00%，得分率 86.125%，95% 配对 bootstrap CI [82.75%,89.375%]。** 先手 173 胜／0 和／27 负，得分 86.50%；后手 171 胜／1 和／28 负，得分 85.75%。400 局全部自然终局，异常、非法动作、不变量失败、工程截断和循环裁和均为零。

| 同种子面板（均对 θ5000） | 对方模型的得分 | 34,398 步减对方得分 | 配对差 95% CI |
| --- | ---: | ---: | ---: |
| 本条轨迹终点 103,194 步 | 55.500% | +30.625 个百分点 | [+25.000,+36.250] 个百分点 |
| 旧离线9500 | 67.375% | +18.750 个百分点 | [+12.875,+24.750] 个百分点 |
| hybrid 第9轮 | 74.375% | +11.750 个百分点 | [+6.500,+16.875] 个百分点 |

这是同一对手、同一搜索预算和同一批配对种子上的得分差，不是这些学生之间的直接对战。比较为探索性，未作多重比较校正；复用 DEV，不能称为新的独立 FINAL。已复算全部区间，核对模型 SHA、双方实际 S1024 预算、每个种子的互换先后手，以及八个完整留出文件的身份和聚合；200 个评测开局种子与整个 29,000 局池无重合。见 [early_battle_verification.json](early_battle_verification.json)。

这条训练轨迹在 34,398→103,194 步之间出现明确搜索得分退化，尽管 policy KL 继续下降。value 泛化同期恶化是明确的相关证据，但这两组完整模型的 policy、encoder 和 critic 均随联合训练改变，单靠它们尚未隔离 value 的贡献；随后追加的分支组合结果见下节。34,398 步是当前八个拟合节点中选出的候选，并非已扫描得到的精确棋力最优点；新池的规模、来源组成和优化日程各自的贡献也未单独控制。该选点补充只完成一个选中节点面板，没有追加其他训练节点的搜索面板或训练。

对战配置及 DEV 种子沿用终点面板，只替换候选模型与元数据。最终计算选中节点相对终点、旧离线9500和 hybrid 第9轮的同种子配对得分差；这些是单对手上的探索性比较，不是新旧模型直接互打，也不作多重比较校正。

```bash
# 此面板已完成；保留命令用于复核／中断恢复，不追加新 pair
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  python -u artifacts/entity_attention_recent_pool_20261009/early_probe.py run --extended-selection
tail -f runtime/entity_attention_recent_pool_20261009/early_probe.log
# 复核 5～8 次取点及八点选择；不启动搜索
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python artifacts/entity_attention_recent_pool_20261009/mid_probe.py
# 复算八点之间按整局配对的拟合变化；不启动搜索
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python artifacts/entity_attention_recent_pool_20261009/early_fit_review.py --extended
```

源码、环境与命令快照在 `runtime/entity_attention_recent_pool_20261009/early_probe/` 和 `mid_probe/`；各留出结果与选点文件原子保存，评估中断只需重做尚未完成的那个 checkpoint，对战按既有生产机制恢复已完成的 pair。暂停时生产评测器正常返回中断码 130，最初包装器将其打印为 `CalledProcessError`；这是用户要求的主动暂停，已保存原因，包装器现会原样返回 130。此补充没有新采集或额外训练。

## 最终 policy 与第4次 value 的组合

用户要求保留第4次平均抽样（34,398 步）的 value，把 policy 换成最后一次（103,194 步），再测 200 pair。两个 checkpoint 各自完整保留，分别使用自己的 encoder：MCTS 的先验取最终网络的 policy，叶节点评估取第4次网络的 value。没有把最终 policy head 接到早期 encoder 上，也没有重新训练。组合权重保存在 `runs/entity_attention_recent_pool_20261009/compositions/final_policy_early_value.pt`，来源文件及 SHA 记录在 [policy_swap_config.json](policy_swap_config.json)。双 encoder 增加推理计算量，本实验控制的是搜索模拟次数，不是等墙钟预算。

实现沿用生产冻结模型和搜索入口，普通／CUDA 编译的 policy+value、policy-only、value-only 都走相同分支。完整权重逐张量一致；32 个原留出对局的开局与最后决策共 64 个局面上，普通前向与来源精确一致，实际 CUDA 编译前向最大绝对差 3.22×10⁻⁶，编译失败／回退为零。见 [policy_swap_verification.json](policy_swap_verification.json)。相关组合、模型工厂及冻结加载回归通过（一个已有 CUDA 测试在沙箱环境跳过，实际 CUDA 已由上述核验覆盖）。

全部 247,453 个原留出局面复算得到 **policy KL 0.415934、value MSE 0.742282**，分别等于最终网络的 KL 和第4次网络的 MSE，且各来源／阶段的对应指标一致。见 [policy_swap_holdout.json](policy_swap_holdout.json)。这是接线正确性的验证，两个指标的组合本身不证明棋力提高。

搜索面板采用同一 θ5000、同一 200 个 DEV pair 种子、双方 native S1024/N1、互换先后手及原有 compiled/native_batched 配置；只变更候选模型与身份元数据。23:05:19 开始，**23:27:36 全部完成**，实际搜索评测约 22.3 分钟。原始结果见 [battles/final_policy_early_value_vs_theta5000.json](battles/final_policy_early_value_vs_theta5000.json)，统计见 [policy_swap_summary.json](policy_swap_summary.json)。

**组合模型 368 胜、2 和、30 负；纯胜率 92.00%，得分率 92.25%，按 200 个整 pair bootstrap 的 95% CI [89.625%,94.75%]。** 先手 183 胜／1 和／16 负，得分 91.75%；后手 185 胜／1 和／14 负，得分 92.75%。400 局全部自然终局；异常、非法动作、不变量失败、工程截断、循环裁和以及报告的编译失败／回退均为零，逐局实际搜索次数与 S1024/N1 一致。见 [policy_swap_battle_verification.json](policy_swap_battle_verification.json)。

| policy 来源 | value 来源 | 完整留出 KL | 完整留出 MSE | 对 θ5000 的得分 |
| --- | --- | ---: | ---: | ---: |
| 第4次，34,398 步 | 第4次，34,398 步 | 0.441453 | 0.742282 | 86.125% |
| 最终，103,194 步 | 最终，103,194 步 | 0.415934 | 0.929513 | 55.500% |
| **最终，103,194 步** | **第4次，34,398 步** | **0.415934** | **0.742282** | **92.250%** |

| 同种子比较 | 组合减原模型得分 | 配对差 95% CI |
| --- | ---: | ---: |
| 原第4次完整模型 | +6.125 个百分点 | [+2.000,+10.250] 个百分点 |
| 最终完整模型 | +36.750 个百分点 | [+32.000,+41.500] 个百分点 |

**固定第4次 value 网络时，最终 policy 的实战收益得到支持；固定最终 policy 时，恢复第4次 value 的收益更大。** 因而这条轨迹的后段并非所有学习都无效：policy 继续改善，而 value 网络退化压过了它的收益。这里控制的是各分支的完整函数，包括其 encoder，不能把差异只归为 critic head。结果支持分别选择 policy 与 value 的训练停止点，并为本轨迹的 value 留出 MSE 提供实际搜索表现的交叉证据。

以上比较均为相对 θ5000 的得分变化，使用复用 DEV 种子及单一搜索预算，属于未作多重比较校正的探索性结果。尚未测独立开局、其他对手和反向组合，因此不能确定普遍最强组合、精确最优停止点或全部 policy/value 交互。本次没有新增训练。

```bash
# 同一面板可中断恢复；完成后重跑不会新增 pair
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  python -u artifacts/entity_attention_recent_pool_20261009/policy_swap.py run
tail -f runtime/entity_attention_recent_pool_20261009/policy_swap/driver.log
# 完成后只复算配对统计，不启动搜索
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_recent_pool_20261009/policy_swap.py analyze
```

组合实验源码、冻结配置、环境与命令快照保存在 `runtime/entity_attention_recent_pool_20261009/policy_swap/`，长期身份与有限验证命令见 [policy_swap_execution.json](policy_swap_execution.json)。原始逐局可恢复 sidecar 仍位于对战结果相邻的 `*.json.games/`。实现和命令见 [policy_swap.py](policy_swap.py)。

2026-10-10 后续对战另存为[分支组合与greedy连续队列](../entity_attention_recent_pool_crossplay_20261010/README.md)：补充P12V4直接对P4V4、P4V12对θ5000、P8V4对θ5000，以及主要policy节点的greedy对战。新结果与状态见[自动结果表](../entity_attention_recent_pool_crossplay_20261010/results.md)，不改写本目录已经完成的三个历史面板。
