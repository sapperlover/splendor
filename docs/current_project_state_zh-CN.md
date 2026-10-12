# 当前项目状态

核验日期：2026-10-09；P4九轮完成结果与导航补充：2026-10-10；第9轮终点长期在线入口补充：2026-10-11。本文是导航快照，详细事实以当前实现、冻结配置和可复核原始产物为准。完成模型训练、完成评测、确认收益是三个不同状态；历史日期和测试数量不代表当前工作树重新验收。

## 当前研究结果

最新见[P4九轮完整报告](reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)：新增在线方案在已测S1024/N1面板胜过起点、同预算旧池对照及P12＋V4；旧池拟合退化与棋力提升并存，value改善不均衡，单训练seed和固定DEV尚不支持普遍最优结论。[M4 与固定数据诊断完整结论](reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md)限定较早分支：稳定fresh净泛化与持续棋力收益未确认；最新数据优先、删除历史、较小学习率仍非独立确认的最佳日程。早期5000→9500混合run有累计留出拟合与对θ5000的棋力收益，见[9代矩阵及基线复核](../artifacts/entity_attention_hybrid_kl_trend_20261009/README.md)，各阶段结论不能混用。

| 实验／分支 | 已完成与限制 | 依据 |
| --- | --- | --- |
| 第9轮终点长期在线续训（2026-10-11） | 入口及配置已实现，正式未启动：38898步、继承第4–9轮6000局原split；每轮1000局／1000步、lr2e-5，纯在线，不依赖旧离线池；每轮1000步完整模型无条件升级。每轮greedy及绝对第12／15……轮搜索各200pair；支持同命令续跑。新配方长期收益未验证 | [流程、验证与启动命令](guides/entity_attention_p4_pool_forever_plan_20261011_zh-CN.md)、[总配置](../configs/entity_attention_v1/r9_online_lr2e5_1000_forever_20261011.json) |
| P4大池起点九轮自迭代（2026-10-10） | 全部完成：9000局、4500主线更新及4500旧池对照更新、完整旧池留出、14个200pair面板。终点搜索对P4为58.750% [54.000,63.500]，对旧池对照61.625% [57.000,66.250]，对P12＋V4为58.125% [53.375,62.875]；greedy末三轮约71%。第4～7代累计留出KL明确下降，第9代MSE改善；旧池全量KL升至0.513325。支持本方案棋力收益，未隔离教师换代或确定最佳参数；单seed、固定DEV | [完整报告](reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)、[流程／数据／命令](guides/entity_attention_p4_pool_iteration_zh-CN.md)、[原件入口](../artifacts/entity_attention_p4_pool_iteration_20261010/README.md) |
| 大池分支组合与greedy连续对战（2026-10-10） | 固定三组200pair搜索：P12V4直接对P4V4、P4V12对θ5000、P8V4对θ5000；随后十组主要policy节点greedy，各200pair。新开局直接互打及greedy、复用DEV的θ5000搜索分别解释；运行状态与已完成组数由结果表自动更新，未完成组不作结论 | [流程与恢复命令](../artifacts/entity_attention_recent_pool_crossplay_20261010/README.md)、[自动结果表](../artifacts/entity_attention_recent_pool_crossplay_20261010/results.md) |
| 近期大池离线训练（2026-10-09） | 29,000局标准attention S1024/N1，103194步完成。八个完整留出节点中，MSE与原1:1联合目标最低均为34398步；后续policy KL继续降、value过拟合。34398步对θ5000的200pair得分86.125%，终点55.50%。最新双网络组合取最终policy＋34398步value，得分92.25% [89.625%,94.75%]，较原34398步提高6.125个百分点，配对差CI [+2.000,+10.250]；较最终完整模型高36.750个百分点 [32.000,41.500]。支持在固定早期value时后续policy有收益，而后期value退化掩盖收益；复用DEV、单对手及预算，未确定普遍最佳组合或全部分支交互 | [数据、日程与结果](../artifacts/entity_attention_recent_pool_20261009/README.md)、[八点选点依据](../artifacts/entity_attention_recent_pool_20261009/extended_selection.json)、[组合模型棋力](../artifacts/entity_attention_recent_pool_20261009/policy_swap_summary.json) |
| 行为复用 M4 | 12／16轮，2400新局、1200更新、17700步，当前停止；r2／4／8对冻结16500起点为48.5／56.25／53.0%，区间均跨50%；扩样后r4（300pair）50.83%[46.92,54.83]、r12（300pair）49.17%[45.25,53.25]，均未确认超起点，轮间配对差全跨0 | [运行记录](../artifacts/entity_attention_avg4_behavior_20261009/README.md)、[扩样](../artifacts/entity_attention_scaleup_20261009/README.md) |
| 固定历史 A／B／C | 三个起点、4500更新、36节点；A@100精确复现正式模型／AdamW；原stored留出未确认B优于A；后续fresh validation的B−A@500区间不跨0，但属边界探索性结果，B尚未确认优于训练前；C无稳定整局留出收益 | [原实验](../artifacts/entity_attention_policy_history_20261009/README.md)、[扩样复核](../artifacts/entity_attention_scaleup_20261009/README.md) |
| 独立重搜索 | 已修正 bootstrap 重复权重和遗漏补充集；未抽中 Train 的768根 fresh ΔKL +0.00415[−0.00154,+0.00980]；扩样到1024根/轮后为+0.00781[+0.00454,+0.01100]，确认平均净退化；账本组扩样后仍跨0 | [修订与原始证据](../artifacts/entity_attention_avg4_delta_kl_20261009/README.md)、[扩样](../artifacts/entity_attention_scaleup_20261009/README.md) |
| 不确定结果扩样（2026-10-09） | 原A@100未抽中组平均fresh KL退化得到支持；lr降低4倍后相对A的近期stored／fresh整局留出优势加强，但@500相对起点fresh validation ΔKL +0.00262 [−0.00219,+0.00749]，净收益未确认；B仍为候选，C训练／留出差距保留；已修正stored聚类键与范围表述 | [报告与复算](../artifacts/entity_attention_scaleup_20261009/README.md) |
| 步数／lr／clip | 旧16500单起点固定账本提供较小更新候选；50步初筛点估计较高，最优步数／学习率未确定；高裁剪频率不能证明 AdamW 的学习率作用很弱 | [剂量](../artifacts/entity_attention_step_dose_20261007/README.md)、[lr](../artifacts/entity_attention_formal_lr_step_20261007/README.md)、[clip](../artifacts/entity_attention_clip_sweep_20261009/README.md) |
| critic 适配 | 额外0／50／200／500步完整留出 MSE 0.808967／0.789053／0.787500／0.786814；已完成三个100pair DEV，51.75／46.0／45.75%，均未确认超50%；HEAD及阶段替换确认未完成 | [报告与对战记录](../artifacts/entity_attention_critic_adaptation_strength_20261008/README.md) |
| 冻结 encoder 拟合 | critic-only留出下降；两版16500／16600 encoder新head重拟合500步留出约0.79247／0.79242，差值区间跨0；未测出绝对拟合上限 | [只练critic](../artifacts/entity_attention_critic_only_replay_20261008/README.md)、[重新拟合](../artifacts/entity_attention_encoder_critic_refit_20261008/README.md) |
| 分阶段搜索 | 同16500权重，各500pair对greedy：只搜后期67.0%、只搜前中期69.35%、全程81.8%；阶段两组差值区间跨0，全程明确高于两组 | [报告](../artifacts/entity_attention_stage_search_20261008/README.md) |
| 固定256根诊断 | 独立真实对局、每局单根，前中期／后期各128；动作变化、JSD和排序只衡量行为变化，没有正确动作标签，不能换算棋力 | [指南](guides/decision_probe_zh-CN.md) |
| B→C→D趋势套件 | B仅完成2轮／16600步及100pair首测51.25% [44.5%,58.0%]；C／D未运行，整体未完成 | [流程](guides/entity_attention_iteration_suite_zh-CN.md)、[原件](../artifacts/entity_attention_iteration_trend_seed1_20261008/README.md) |

较早扩样已完成A／B／C与低lr的固定数据复评；统计修订区分了相对A的优势和相对训练前的净变化。对应分支的后续候选为低lr＋A/B，不把500步当作最佳日程；这些低lr候选的多轮自迭代、critic HEAD与分阶段搜索扩样仍未运行。P4九轮为另一已完成方案，结论分别解释。详见[扩样报告](../artifacts/entity_attention_scaleup_20261009/README.md)。

## 实现与较早训练

| 能力／历史阶段 | 证据范围 | 入口 |
| --- | --- | --- |
| 双人规则、Observation、数据与学习接口 | Python checked 规则参照；native 为另一个实现，依对应版本回归验证 | [标准规则](splendor_standard_rules_zh-CN.md)、[契约](specs/engine_contract_zh-CN.md)、[学习契约](specs/learning_contract_zh-CN.md) |
| 本地人机网页（2026-10-11） | 支持固定策略、greedy、固定深度搜索及原生 MCTS；MCTS 的 S/N 预算可开局设置及对局中应用，CPU单线程与训练并行。新增当前玩家视角的模型 value 评估条、逐步悔棋、顶部临时支付／选择栏；第24轮模型命令见指南 | [评测与网页指南](guides/evaluation_zh-CN.md#本地人机网页) |
| PPO seed1、AZ seed1 | PPO 5M本轮decisions；AZ5×500及θ1–θ5扩展面板完成，DEV best不等于独立FINAL best | [PPO](reports/ppo_5m_zh-CN.md)、[AZ](reports/alphazero_seed1_zh-CN.md) |
| 旧θ5的10×1000源数据 | 轮0–9完成，10000局原split用于离线学习；770870训练／80699留出位置；准备说明已归档，不在本次重验棋力 | [数据统计](../artifacts/entity_attention_v1_20261004/data_summary.json)、[历史准备](archive/next_az_training_zh-CN.md) |
| 实体注意力离线 | 532738参数、12045步及六个PPO面板完成；final搜索59.5%、greedy61.75%，重复DEV | [指南](guides/entity_attention_offline_zh-CN.md) |
| 新模型10×1000、混合训练 | 10×1000曾运行，已有轮0–2完成记录；混合run完成9轮／9500步；9代各2048留出局面，相对θ5000均改善，相对各代训练前为8降1升，本轮500步为5降4升；N1到14500对原九代均比各自训练前更差，当前窗口拟合未由此验证；离线重放9500对θ5000为67.38%，hybrid第3/6/9轮为58.88/66.75/74.38%，第6轮与离线9500不可区分、第9轮高+7.00pp[+0.62,+13.25]；不同分支不能互称完成 | [自迭代](guides/entity_attention_self_iteration_zh-CN.md)、[混合](guides/entity_attention_hybrid_zh-CN.md)、[KL趋势](../artifacts/entity_attention_hybrid_kl_trend_20261009/README.md)、[离线9500对照](../artifacts/entity_attention_hybrid_offline9500_20261009/README.md) |
| 第九轮N2与N1 fork | N2至少33新轮／12800步记录，曾OOM；N1完成50新轮／14500步后停止；原9／50轮共同PPO为63.75／61.75%，差值区间跨0 | [N2](guides/entity_attention_hybrid_fork_zh-CN.md)、[N1](guides/entity_attention_hybrid_s1024_n1_zh-CN.md)、[补测](../artifacts/entity_attention_strong_anchor_20261006/README.md) |
| 分头窗口、value detach | 20新轮／16500步完成；policy近4轮、value3000局近期50%；长期净收益未确认 | [指南](guides/entity_attention_split_training_zh-CN.md) |
| 搜索教师比较 | 目标稳定性不等于棋力；同S1024随机节点31.25%、CPW37%均弱于旧native；正式教师未切换 | [系列报告](reports/entity_attention_search_teachers_zh-CN.md) |
| 性能与进程管理 | 多worker、CUDA Graph、缓存与退出修复有有限验证；各报告限定形状／机器／版本，未确定全局最快配置 | [性能验收](reports/end_to_end_speed_optimization_zh-CN.md)、[新模型执行](reports/entity_attention_execution_speed_zh-CN.md)、[OOM修复](../artifacts/entity_attention_worker_cleanup_20261005/README.md) |

所有文档与配置见 [导航](index.md)。状态分类及原件身份见 [实验索引](../artifacts/experiment_index.json)；本次路径和结论维护见 [整理记录](../artifacts/repository_cleanup_20261009/README.md)。
