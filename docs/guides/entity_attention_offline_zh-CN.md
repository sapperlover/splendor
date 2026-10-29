# 实体注意力模型与离线搜索经验学习

2026-10-25存储更新：[原80000局池及四组已完成辅助实验的普通checkpoint／逐步日志已外置](../../artifacts/repository_archive_20261025/README.md)。原80k池的离线训练／全量复核需先恢复NPZ；旧普通节点／曲线复核需恢复对应模型／日志，续写前恢复为真实文件。R21–R37的104k池及当前在线初始64k／新采集数据、关键模型保持本机；本页训练执行默认不变。

2026-10-24，未来新建离线统一默认流式累计、4 worker共享只读供数及传输／同步优化，缓存按池大小配置；已覆盖普通、旧辅助、PSW、混合预算、续训、早停和continuation入口。[通用设置与恢复](offline_execution_zh-CN.md)。当前队列后两组均已另建18GiB／4 worker发布并接入，未启动正式训练；[实际发布与验证](../../artifacts/offline_training_defaults_20261024/README.md)。

2026-10-24新增[当前实验后的三个串行消融](../../artifacts/entity_attention_auxiliary_queue_20261024/README.md)，仅改变辅助开关和指定44k范围；主policy／终局value保持1/1。value辅助0时不注册头／生成标签／计算loss或诊断，policy-only与双辅助的对手头初始权重一致。旧模型缺`predict_shortterm_value`默认true，当前运行的冻结实现不改。各组完整训练／选点／R37 greedy1000pair／搜索A500pair成功后，阻塞队列再启动下一组。

2026-10-23正式流程新增[104k局R21–R37仅短期value实验](../../artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023/README.md)：从零attention128＋H2/H6/H14辅助0.25，对手policy辅助关闭，lr1e-4；沿用16GiB、两头B256和12次value曝光上限，每5%完整主KL+MSE三次严格上升早停、最低节点对R37 greedy1000pair后搜索A500pair。最新轻量记录不写大抽样账本。`expected_games_per_round`可为正整数或覆盖全部一基绝对轮号的JSON映射，如`{"74":4000,"75":10000}`，总数相同但分轮不符仍拒绝；未改旧整数冻结身份。

2026-10-23新增独立`offline_continuation`入口，支持2／6／14玩家回合短期value，以及对手下一主行动的类型／购买牌／保留牌辅助；[标签、覆盖率与命令](continuation_auxiliary_zh-CN.md)。两份新配置从零训练并以主KL＋MSE选点，未启动正式实验，旧短期进展任务仍关闭。

2026-10-23记录策略：当前`offline_objectives`／`offline_mixed_budget`及`hybrid_train`默认不再写两份抽样`.bin`、两份`.steps.jsonl`及恢复时完整指标副本；每步训练指标、batch摘要及小型身份JSON保留。原`offline_train`精确覆盖审计和已冻结历史源码沿用旧行为，旧实验不会自动升级；见[训练产物规则](training_artifacts_zh-CN.md)。

2026-10-23[新增购买结算方案已撤回](../../artifacts/entity_attention_first_round_20261022/README.md)：删除新增购买／贵族／终局派生特征及专用offline_first_round入口、配置和启动脚本。当前使用既有attention与offline_early_stopping／offline_finetune等入口；短期进展辅助保持关闭。原组合／纯购买准备与短验证保留历史身份，不用当前代码直接恢复这些历史实验模型。

2026-10-22的[同44k池最低节点lr2e-5续训](../../artifacts/entity_attention_128x2_r21_r31_lr2e5_20261022/README.md)已在新增10000步触发三次联合指标上升早停，选中新增4000步／累计63453步。原59453步继承AdamW/RNG、每2000步完整留出、两头B256和原标签／split不变；原训练命令仍只训练＋选点，已完成后返回。用户后续独立授权对R31做1000pair greedy，同原R31面板开局，评测命令、原始结果及核验见专题；不恢复训练或运行搜索。

2026-10-22的[第21–31轮44000局attention128离线实验](../../artifacts/entity_attention_128x2_offline_r21_r31_20261022/README.md)：用户后续主动停止训练，保存84000步，原148632步／12次value曝光预算未完成，也未触发自动早停。按已到达完整留出节点最低KL＋MSE固定59453步整个checkpoint，仅执行对R20／R31各1000pair greedy，搜索取消；使用专题的`run_greedy_after_stop.py run`命令，不以原shell恢复完整队列。原lr1e-4、种子、两头各B256、16GiB、数据及整局split和冻结源码不变，训练与评测均12贵族。

2026-10-21[attention128的同8万局对照](../../artifacts/entity_attention_128x2_offline_80k_20261021/README.md)已由用户启动，后续执行记录为KeyboardInterrupt退出，最近保存135279步，原270557步计划未完成；本次新实验不恢复或修改旧run。命令`bash scripts/run_attention128_offline_20261021.sh`会恢复旧训练：128×2／4头／FFN512／583682参数，保留购买关系和value读取；除模型和独立输出路径外逐字段沿用MLP配置，固定4096引用和完整dataset相同。只训练＋完整留出KL+MSE选点，不自动对战。

[当前实体MLP的8万局离线流程](../../artifacts/entity_set_mlp_256x4_offline_80k_20261021/README.md)已按用户要求停止，保留97000步，原270557步计划及自动面板未完成。七个节点内94695步完整留出KL+MSE最低（KL 0.530577、MSE 0.782832），value最佳在40584步；之后新授权的该候选对R20 greedy1000pair已完成，MLP得37.075%，95%配对区间[35.000%,39.175%]，无搜索对战。训练使用新分支第1–20轮80000局、lr1e-4、16GiB缓存，每5%固定4096子集和完整留出；`offline_mixed_budget`两头各256独立抽样、一次联合更新，policy只用高预算有效步，value全部预算，无辅助或PSW。历史训练来自10贵族池，本次greedy对战为12贵族。

2026-10-21新增[实体汇总＋残差MLP](../../artifacts/entity_set_mlp_256x4_20261021/README.md)，架构`entity_set_mlp_v1`；[当前256×4预设](../../configs/entity_set_mlp_v1/model_256x4.json)含非线性动作旁路及状态×动作乘积，为855,682参数，原848,418参数属于已冻结初版。`offline_scratch`仍支持其共享minibatch的none/turns-only/dual配置。集合汇总后进入共享残差主干，动作直接读取原82维属性及目标牌关系；原输入schema不变，attention权重不能直接恢复到该模型。已完成有限训练/恢复/搜索正确性验证，未启动MLP正式训练；下方attention历史结论不自动适用于MLP。

2026-10-19产物位置更新：7个离线run的非主要checkpoint、抽样账本和逐步指标共7.03GiB已集中到`archive/20261019/`，原路径保留相对链接；关键模型、验证曲线、选点和两份原始训练池留在原位。[清单、外置校验与恢复方法](../../artifacts/repository_archive_20261019/README.md)。只读复核可恢复挂载后访问；恢复训练前须按清单撤回该run的迁移，将账本／日志等恢复为原位真实文件，避免写入冻结归档。

2026-10-19新增[新版128×2从零训练与对战流程](../../artifacts/entity_attention_128x2_relations_no_aux_r24_r43_20261019/README.md)：购买关系和value多实体读取开启，两个辅助头及PSW关闭，lr1e-4；R24–43原20000局、68041步、每5%固定4096＋完整留出评估，按KL＋胜负MSE最小选完整模型，再测1000pair greedy／200pair S1024。其余配置与256实验相同，`offline_scratch`现支持`value_auxiliary=null`，已有辅助配置保持兼容；正式训练与两组对战均完成，选37423步，greedy53.775%、搜索56.625%；限制见专题结果。

2026-10-19实现[公开卡牌购买关系与value直接读取双方玩家/市场](../../artifacts/entity_attention_relations_20261019/README.md)：市场及双方明保留牌在模型内由旧输入派生8维公开关系，共享投影且不新增参数，盲保留不追加，历史NPZ可直接使用；主value与已启用辅助头读取全局、双方玩家及市场masked mean。新[128×2](../../configs/entity_attention_v1/model_relations_128x2.json)／[256×4](../../configs/entity_attention_v1/model_relations_256x4.json)预设显式启用，两项开关独立，可做结构消融；下文旧结构描述仍对应省略开关的历史默认。`offline_scratch.model_config`可使用新预设，旧checkpoint严格加载为原结构，不自动按新head续训；结构交付阶段未启动长训练，后续新版128训练见上方独立专题。

2026-10-19新增[原小模型配方的无辅助／仅剩余回合辅助对照](../../artifacts/entity_attention_r24_r43_aux_ablation_20261019/README.md)：`splendor.offline_objectives`允许普通CE＋胜负MSE（两个可选目标均null），以及score_loss_weight0、turns_loss_weight0.25的单回合辅助。单回合checkpoint为`entity_attention_aux_v1`、`predict_score=false`，state_dict实际不包含分差头；旧双头配置省略该字段仍兼容。两组同原R43／AdamW／2万局／22681步，final各对R43测greedy1000pair与搜索200pair，两组训练及四个面板已完成；plain／turns-only的greedy为53.850%／53.150%，旧双辅助减plain的+0.050pp区间跨0，未确认辅助头额外收益；具体命令、抽样一致性与结果见专题。

2026-10-15完成[R43小模型64根×16次独立续局的value信号诊断](../../artifacts/entity_attention_r43_value_signal_64_20261014/results.md)：单次MSE0.7555，结果方差估计0.6936，有限重复修正后的期望value误差0.0620。高方差监督与已学到的胜负信号并存，不能仅按单次终局MSE判定value没有进步，也不能把更平滑目标上的低MSE当成棋力提升；本次未修改离线训练标签或训练配置。采样器、历史教师和开局温度差异见专题README。

2026-10-14交付[256×4双辅助头固定池训练与自动选点对战入口](../../artifacts/entity_attention_256x4_aux_r24_r43_20261014/README.md)：原R24–R43的20000局，PSW关闭，随机初始化及全新AdamW，双辅助各0.25；保持原优化参数，曝光12次／68041步，每5%保存并评估固定4096与完整留出，每1000步只保存恢复点。按完整留出KL＋胜负MSE最小选整个checkpoint，对原R43测1000pair greedy和200pair双方S1024。用户任务已运行至4.44万步附近后中断，终点与搜索面板尚未完成；旧0步派生汇总已于2026-10-19按最近保存44227步和中断记录修正；实际状态与来源见专题。

2026-10-13完成[R24–43固定数据的PSW／value辅助目标离线训练](../../artifacts/entity_attention_r24_r43_offline_objectives_20261013/README.md)：两组原R43起点、同20000局各22681步；50% PSW与双辅助头分别训练。随后final对原R43各1000pair greedy，PSW得分51.350% [49.225%,53.450%]，双辅助53.900% [51.675%,56.075%]；详见[结果](../../artifacts/entity_attention_r24_r43_offline_objectives_20261013/results.md)。训练入口为 `splendor.offline_objectives`，greedy扩样使用专题中的独立命令；较早的 `splendor.offline_train` 不自动支持这些目标。

2026-10-04实现并完成有限CPU/GPU验证；2026-10-06状态更新：`runs/entity_attention_offline`已完成12045步，用户完成1000步、5000步及final对PPO 5.5M的搜索/greedy共六个200局面板。S1024/N1搜索score分别为11.75%、41.25%、59.5%；greedy分别为38.5%、60%、61.75%。现在混合训练使用的theta_5000与这里5000步节点全部模型参数相同，但它比完整离线final在共同PPO锚点上的表现更弱。身份核对及原第9轮/当前第50轮补测见[强锚点报告](../../artifacts/entity_attention_strong_anchor_20261006/README.md)。这些是重复DEV面板，不是独立FINAL验收。

下文“仅准备”“尚未启动”描述相应命令首次交付时的状态，不能据此否认上述已完成的原始结果。配置见[目录说明](../../configs/entity_attention_v1/README.md)，原有限验证见[验证产物](../../artifacts/entity_attention_v1_20261004/README.md)，加载瓶颈及优化见[缓存记录](../../artifacts/offline_training_speed_20261004/README.md)。

## 结构与兼容性

2026-10-14新增[256×4模型预设](../../configs/entity_attention_v1/model_256x4.json)，复用现有 `entity_attention_v1` 实现，尺寸由完整 `model_config` 保存；省略尺寸仍使用历史128×2默认。此前固定128×2的校验已开放为正整数尺寸、hidden须能被heads整除，dropout继续保持0。历史训练配置／checkpoint仍按其原尺寸解析；R43权重不能直接作为256×4的续训权重。

| 结构 | hidden | heads | 状态编码层数 | FFN | 参数 |
| --- | ---: | ---: | ---: | --- | ---: |
| 历史默认 | 128 | 4 | 2 | 128→512→128 | 532,738 |
| 256×4预设 | 256 | 8 | 4 | 256→1024→256 | 3,628,034 |
| 256×4＋双辅助头 | 256 | 8 | 4 | 同上 | 3,761,156 |

新主网络为原来的6.810倍参数，FP32参数本体13.840 MiB，不包含梯度、AdamW状态或激活。每头32维，四个 Pre-LayerNorm self-attention block 加一个策略 cross-attention；GELU、残差、attention与FFN dropout均为0。新增预设是 `model_config` 片段，使用时嵌入新训练配置的 `model_config`；它不是完整的离线训练CLI配置。例如通过现有工厂构建：

```python
import json
from splendor.actor_critic import build_actor_critic

with open("configs/entity_attention_v1/model_256x4.json") as source:
    model = build_actor_critic(json.load(source))
```

有限验证已完成：22项相关CPU测试、真实归档4样本的AZ训练／checkpoint恢复、CPU后续更新逐位一致、CUDA eager/compiled前向一致，以及开／中／后期三个S8/N1原生批量搜索根。[结构与验证报告](../../artifacts/entity_attention_256x4_20261014/README.md)保存参数分项、命令与原始产物；这不是正式训练、棋力或大batch吞吐结论。

状态共 25 个 token：1 个全局、2 个玩家、12 张市场牌、双方各 3 张保留牌、3 个贵族、1 张 pending card。已有 148 维全局特征拆为双方各 14 维玩家数值和其余 120 维上下文，包含全部 public_purchased 字段。卡牌、贵族与动作仍使用原 29/17/82 维输入；特征版本和规则版本不变，不读取牌堆或对手暗牌。

角色 embedding 表示类型、归属和市场层级，同角色集合没有槽位位置参数。无效实体在编码前及 block 后排除；对手盲保留是有效 unknown 实体。候选动作各自作为 query，经一次 cross-attention 读取共享状态，保留 query 残差，再输出 logit。候选互不做 attention。独立 critic 从交互后的全局 token 输出当前行动者视角的无界 value，不读取候选；一次完整前向只编码一次状态。

`build_actor_critic` 是 frozen loader、纯权重 loader 和 AZ checkpoint 的架构分派入口。缺少 architecture 的已知旧格式按 Deep-Sets 构建，未知架构报错；各模型均严格加载 state_dict。新 checkpoint 保存完整结构配置，沿用 AZ 协议。辅助头升级也保留指定的hidden／heads／layers／FFN。PPO训练恢复仍遵循原PPO契约；已有 `entity_attention_policy_value_v1` 双网络组合结构仍固定构造历史128×2分支，不会自动适配本次大模型。

新结构要求 `actor_projection=false`，启用时启动报错；旧结构继续允许 `true`。现有 eager、compiled 和 reused native 传输复用同一模型接口，FP32、候选档位 [16,32,48,64]、编译失败即报错。混合评测按模型和推理配置分开路由，新旧双方可使用不同 projection。

结构设计参考 [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) 的集合注意力思想；具体 Splendor 结构和项目收益仍须本项目实验验证。

## 离线数据与实验设置

数据源为已完成的 `runs/az_theta5_10x1000_seed1/rounds/round-000` 至 `round-009`。从每轮现有 replay snapshot 取该轮新局，并使用 `artifacts/az_theta5_10x1000_seed1/round-XXX/dataset.json` 的已记录身份；不会把最后一轮在线 snapshot 的 6,000 局上限当作离线数据范围。执行时检查所选轮完成状态，缺失则报错。原数据统计：

| split | 对局 | 局面 |
| --- | ---: | ---: |
| train | 9,051 | 770,870 |
| validation | 949 | 80,699 |

整局 split 保持原样。训练从全部所选 train positions 均匀有放回抽样，policy 标签是已存 MCTS 访问分布，value 标签是真实终局结果；mask、单候选权重和玩家视角沿用现有 collator/learner。配置中 `history_batch_size` 是复用 LearnerConfig 的字段，离线没有 history pool，因此不执行在线 128/128 抽样。

冻结固定验证子集 4,096 个局面：前六轮各 410、后四轮各 409。保存原留出局面的引用、组成、seed 和摘要；冻结仅依据计数抽引用，不需解码所有数组。训练前、每 1,000 步及末尾评估此子集，正式完成后流式评估一次全量留出。保留原始分子和权重分母、policy CE/目标熵/预测熵/KL、value MSE/零预测基线。

| 设置 | 新结构与随机旧结构对照 |
| --- | --- |
| 初始化 / 抽样 / 固定验证 seed | 13100000 / 13110000 / 13120000 |
| batch / 精度 | 256 / FP32 |
| optimizer | AdamW，lr=1e-4，wd=1e-4 |
| policy / value 权重 | 1 / 1 |
| grad clip | 1 |
| 更新预算 | ceil(4 × 770870 / 256) = **12,045** |
| 平均曝光 | 约 4.00005 次有放回抽样，不是无重复 epoch |
| LRU 容量 | **4 GiB**，记录实际驻留、峰值、hit/miss、decode 和加载耗时 |
| checkpoint | 每 1,000 步及末尾；initial/latest/final 与逐步节点 |

4 GiB 是初始冻结配置值。当前十轮数据全部数组的大小上界为5.918 GiB（ZIP目录中的NPY大小，含头部，不解码数据），4 GiB缓存造成反复淘汰/解压。32 GiB主存环境下建议运行时覆盖为8 GiB，仍按需加载、有界LRU，不预加载全量数组。新旧同预算对照应使用相同的运行时容量。新旧都从随机参数开始，不继承旧 θ5 的权重或 optimizer。相同 seed 固定各自初始化，但不同结构的随机消耗序列不会产生相同参数。

`best_validation_kl.json` 只引用固定子集 KL 最低节点，同时保留其 value 指标，不能称作最强棋力模型。中断恢复校验冻结配置、数据元信息和归档 size/mtime，复用原已记录 archive hash，不重新 checked replay/计算全量 archive hash。恢复模型、optimizer、采样 RNG、Torch RNG 和已提交步数；舍弃未提交日志/引用后续训，不重复累计步骤。

原`offline_train`链路的每步指标见 `training_metrics.jsonl`，验证曲线见 `validation_curve.jsonl`，精确引用曝光见 `usage.json` 和 `usage/`；当前近期离线链路按本页顶部轻量记录规则。每次调用保留 `source/<session>/` 源码快照、代码身份和命令，`sessions.jsonl` 记录设置/训练/固定验证耗时与显存峰值；全留出耗时在 `full_validation.json`，本次调用总耗时在 `training.json.session_total_seconds`。cache 统计为当前进程统计，resume 后重新计数；累计曝光来自已提交引用。

## 执行命令

从仓库根目录执行。离线拟合需要真实 CUDA和上述源归档，后续native搜索还需要已构建扩展。下列命令初次交付时未执行；随后用户已启动新结构正式拟合，本文保留独立旧结构对照和后续命令。

随机初始化新模型及公平旧结构对照分别写独立目录，建议顺序运行：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.offline_train \
  --config configs/entity_attention_v1/offline_attention.json --output runs/entity_attention_offline --replay-cache-mib 8192
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.offline_train \
  --config configs/entity_attention_v1/offline_deep_sets.json --output runs/deep_sets_offline --replay-cache-mib 8192
```

中断后原配置、原输出目录加 `--resume`：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.offline_train \
  --config configs/entity_attention_v1/offline_attention.json --output runs/entity_attention_offline --resume --replay-cache-mib 8192
```

`--replay-cache-mib` 是执行参数，可在resume时调整；不要为扩容编辑已冻结的配置。原batch、loss、seed、预算、数据/验证引用保持一致，缓存访问不消耗抽样RNG，实际容量写入每步cache、source identity和session记录。现有进程不会热更新；若切换，在checkpoint保存/验证完成后中断，再执行resume。当前每1000步保存，期间中断将退回最后已提交节点，最多重做999步；恢复后缓存重新预热。若已完成，resume不会增加训练步数。

末尾自动导出 `runs/entity_attention_offline/checkpoints/final.pt`；无须手工转换。固定面板默认比较它与旧 θ5 `runs/az_loop_seed1/rounds/round-004/checkpoints/final.pt`。前置条件是正式离线 checkpoint 已生成；若选择其他节点，应复制面板配置并明确替换 policy_a checkpoint。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.evaluation \
  --config configs/entity_attention_v1/mcts_vs_theta5.json \
  --output runs/entity_attention_offline/evaluation/mcts_vs_theta5_50pairs.json
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.evaluation \
  --config configs/entity_attention_v1/greedy_vs_theta5.json \
  --output runs/entity_attention_offline/evaluation/greedy_vs_theta5_50pairs.json
```

两个面板各 50 pairs。MCTS 双方 S1024/N1、无 root noise、native_batched，双方推理配置分别指定；greedy 面板双方均为裸网 greedy。旧 θ5 是固定棋力锚点，不替代随机旧结构的学习对照。评测中断恢复同样加 `--resume`。

按用户后续指定，另提供新离线final对PPO lifetime5,512,112（`local_005000000_lifetime_005512112.pt`）的两个100 pairs面板，各200局交换座位。两个面板使用相同game/policy seed输入；搜索面板双方均S1024/N1、root noise为0，使用各自架构对应的推理配置。配置仅准备，没有启动评测：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.evaluation \
  --config configs/entity_attention_v1/greedy_vs_ppo_5_5m_100pairs.json \
  --output runs/entity_attention_offline/evaluation/greedy_vs_ppo_5_5m_100pairs.json
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.evaluation \
  --config configs/entity_attention_v1/mcts_s1024n1_vs_ppo_5_5m_100pairs.json \
  --output runs/entity_attention_offline/evaluation/mcts_s1024n1_vs_ppo_5_5m_100pairs.json
```

2026-10-05按用户要求，增加离线训练1000步/5000步（`checkpoints/step-000001000.pt`、`step-000005000.pt`）对同一PPO5.5M的四组配置；每组100 pairs/200局，搜索时双方均S1024/N1，种子及执行设置沿用以上final面板。仅准备并检查配置，未启动评测。依次执行：

```bash
for step in 1000 5000; do
  for mode in greedy mcts_s1024n1; do
    conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.evaluation \
      --config "configs/entity_attention_v1/offline_step${step}_${mode}_vs_ppo_5_5m_100pairs.json" \
      --output "runs/entity_attention_offline/evaluation/offline_step${step}_${mode}_vs_ppo_5_5m_100pairs.json"
  done
done
```

以下3×200为此前建议，保留历史命令。2026-10-04用户改为新模型10×1000自迭代，同时调度1000局，每轮对固定新模型起点做搜索/greedy各50 pairs；当前启动命令及有限验证见 [自迭代专题](entity_attention_self_iteration_zh-CN.md)。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.alphazero_loop \
  --config configs/entity_attention_v1/online_3x200.json \
  --output runs/entity_attention_online_3x200 --progress-format human
```

此配置仅继承离线 final 权重，创建新 optimizer 和新的在线 replay，初始 total_train_step 为 0；不导入离线旧池。在线仍用原 current/history 比例及 6,000 局上限，后续轮间正常续接 optimizer。它与离线 `--resume` 的完整状态恢复语义不同，不增加胜率接纳门槛。

## 本次有限验证结果

相关测试一次：27 passed、1 skipped（CUDA 条件用例在沙箱中不可见；实际 GPU 另行验证）。覆盖 shape/mask、集合不变性、候选等变性、unknown/padding、value-only、架构分派及严格旧/新权重加载。

微型链路使用首轮 6 局真实归档，batch16、固定验证16：先更新4步保存，再恢复至8步，累计128次抽样、107个不同 train positions，无重复累计。固定验证 KL 0.80342→0.78194，value MSE 0.99072→0.98953，样本量和更新量仅供验证调用链。导出权重完成 native_cuda_pool 2局，以及混合新旧架构 native_batched 1 pair；均 S16/N1、compiled、reused、无编译失败 fallback。追加1步 CPU 小检查仅覆盖新增全留出记录和已完成 run 的零步 resume。

GPU：RTX 4050 Laptop 6 GiB，PyTorch 2.14.0+cu130。真实归档编码重复填充 B、按 C 正确 padding，旧 projection=true、新 false；每档首次调用后3次预热、10次测量：

| B / C | 架构 | 首次调用 | 预热后平均前向 | 相对旧模型 | allocated / reserved MiB | peak allocated MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 64 / 32 | Deep-Sets | 4.575 s | 0.426 ms | 1.00× | 25.19 / 80 | 51.20 |
| 64 / 32 | Entity attention | 7.147 s | 1.011 ms | 2.37× | 22.50 / 164 | 54.49 |
| 500 / 64 | Deep-Sets | 1.802 ms | 1.078 ms | 1.00× | 25.42 / 160 | 131.99 |
| 500 / 64 | Entity attention | 6.566 ms | 6.182 ms | 5.73× | 22.73 / 190 | 119.99 |

compiled 输出与 eager 在容差内一致，实际 compiled_calls 已记录，无 fallback。首次调用包含实际编译/缓存命中成本，没有清编译缓存，不等于纯编译时间。显存为同进程测量，包含驻留输入和输出，reserved 在模型间保留，不能由表中 allocated 反推参数显存或训练显存。前向倍率不能直接换算整轮采集倍率。

当前明确代价是新结构前向更慢，尤其大 batch；它是否能以更少更新吸收搜索目标、验证改善且不损害 value，需要同预算新旧正式拟合后按误差曲线和单位时间比较，再进行棋力面板。未收敛不能直接判结构失败，最低 KL 也不等同最佳棋力。

复现本次有限验证（首次运行需要独立输出目录；已有 smoke 不要直接覆盖）：

```bash
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_entity_attention.py tests/test_actor_critic_4a.py tests/test_alphazero_training.py
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/validate_entity_attention.py \
  --output runtime/entity_attention_v1_smoke --artifacts artifacts/entity_attention_v1_20261004
```

这些结构验证已完成，无须反复执行。新增缓存参数另以4步连续CPU更新与2+2步扩容resume验证，模型/optimizer/RNG逐位一致，引用序列相同；24局真实工作集的短测确认缓存容纳后miss归零。未测真实8 GiB设置下整轮GPU提速倍率，后续以会话记录为准。正式拟合已由用户启动，棋力和在线迭代仍是后续运行阶段。
