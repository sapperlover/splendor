# 文档导航

[近期辅助目标完成结果](reports/auxiliary_offline_results_20261027_zh-CN.md)：六组原训练／选点／对战已完成，新增五模型greedy及S128N1各10000 pair十组也已完成；含在线13轮和已完成的人类棋谱／支付诊断入口。

[到终局分数归属辅助](guides/score_ownership_zh-CN.md)：双方×三个牌等级和贵族的8路新增分，时点、损失、兼容性、执行配置及有限验证。

[Policy辅助140221步起点的60k在线自迭代](../artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/README.md)：R31–R37历史64k，继续policy辅助0.15，每轮10k／3000步／lr2e-5；固定raw R37、采集较优设置／搜索A。已提交13轮，第12轮greedy66.075%、完整搜索60.20%，第13轮greedy63.75%；当前入口训练结束释放CPU缓存。见[平台分析](../artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/plateau_review_20261025/results.md)。

[离线训练执行默认](guides/offline_execution_zh-CN.md)：2026-10-24起新任务默认流式累计、4 worker共享只读供数及传输／同步优化，缓存依池配置；后续三组已用优化发布完成训练和原两组对战；当时后两组各18GiB。[实际入口与183项回归／CUDA短验证](../artifacts/offline_training_defaults_20261024/README.md)。

[当前实验后的三个串行辅助消融](../artifacts/entity_attention_auxiliary_queue_20261024/README.md)：104k仅policy、104k双辅助、44k仅value；沿用主KL+MSE早停及R37两组对战，监督锁与成功状态决定接续。

[第21–37轮104k局attention128＋仅短期value离线训练](../artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023/README.md)：从零、lr1e-4，5%完整主KL+MSE早停选点；对raw R37 greedy1000pair后搜索A500pair，最新轻量记录；已完成训练和原两组对战，选中140221步；后续10000 pair面板也已完成。

[短期value与对手policy辅助目标](guides/continuation_auxiliary_zh-CN.md)：指数标签公式、玩家回合口径、两层软CE、可见牌／未知类、覆盖限制及独立离线运行命令。

[训练记录与checkpoint规则](guides/training_artifacts_zh-CN.md)：当前在线／近期离线停写双份抽样账本与双份逐步日志，保留主指标和身份／batch摘要；R31在线500步保存＋七个固定诊断点，恢复边界与旧冻结入口说明。

[在线全量固定留出](guides/evaluation_zh-CN.md#在线全量固定留出)：每六轮使用此前六轮的全部原始validation，段内固定；每轮七节点自动评测／恢复补齐；[R32–R37共42节点回补结果](../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/fixed_holdout_20261023/results.md)。

[当前采集与评测配置](guides/evaluation_zh-CN.md)：并列整理在线采集、日常greedy／搜索、在线greedy／搜索。当前采集active2000／原B、C桶；日常及在线新搜索统一A（12worker／active1000／B500／microbatch128／owner双槽／精简B、C桶），纯greedy默认CPU8 worker／1000 pair；搜索默认500 pair，双方S1024N1总1024。可执行日常模板和记录命令见指南。

[R31起点10k新局／60k池自迭代](../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/README.md)：R22–R31的40k历史、lr2e-5／每轮3000步、16GiB compact缓存；首轮50k、第二轮60k，已完成6轮至谱系R37。采集配置保持，后续搜索A在新轮界读入后用于新面板；原state、旧面板及准备记录不改写。

[新增购买结算方案已撤回](../artifacts/entity_attention_first_round_20261022/README.md)：2026-10-23删除全部新增派生特征及专用候选入口，原关系attention128恢复583682参数；短期进展辅助保持关闭。历史准备、工程验证、冗余审查与撤回快照见专题，未启动正式训练。

[最低节点attention128降lr续训](../artifacts/entity_attention_128x2_r21_r31_lr2e5_20261022/README.md)：新增10000步三次上升早停，完整留出选中新增4000步／累计63453步；用户追加R31 greedy1000pair的配置、同开局复测结果与核验入口。

[第21–31轮44000局attention128离线训练](../artifacts/entity_attention_128x2_offline_r21_r31_20261022/README.md)：用户已停止，保存84000步；按完整留出最低联合指标固定59453步，仅执行对R20／R31各1000pair greedy，搜索取消。仅greedy命令、范围变更与结果入口。

[第21–30轮及Cross64离线冷产物T7归档](../artifacts/repository_archive_20261022/README.md)：完成清单、逐文件校验及恢复方法；全部第21–31轮采集对局留本机，291个模型及43个账本外置，净释放4.3250GB。

[MLP角色初始化标定＋同8万局离线训练](../artifacts/entity_set_mlp_cross64_role005_offline_80k_20261022/README.md)：std0.05、906434参数，原数据／抽样／lr1e-4／16GiB及5%留出不变；用户后来启动，现中断、已保存81168步。六节点中KL＋MSE最低54112步[对R20 greedy1000pair](../artifacts/entity_set_mlp_cross64_54112_vs_r20_greedy_20261022/README.md)得43.225% [41.075%,45.325%]；原270557步计划未完成，未测搜索。

[第20轮继承续跑：S256N4＋12贵族](../artifacts/entity_attention_r20_s256n4_noble12_iteration_20261022/README.md)：继承状态与24k池，高预算256×4总1024，低预算128×1；每轮4k局／1000更新，默认无限自迭代，启动／恢复命令和验证见专题。

[Checkpoint去重与外置完成](../artifacts/repository_archive_20261021/README.md)：161个副本、1885个冷模型，T7包10.80GB，本阶段释放12.85GB，冷模型读取前需解压。[原审计及首次清理](../artifacts/checkpoint_storage_audit_20261021/README.md)保留历史记录。[Attention128同8万局离线对照](../artifacts/entity_attention_128x2_offline_80k_20261021/README.md)最近提交135279步，核验时日志KeyboardInterrupt退出，未自动恢复；[94695步对R20 greedy1000pair](../artifacts/entity_attention_128x2_94695_vs_r20_greedy_20261021/README.md)得54.400%，区间[52.299%,56.525%]。[MLP同节点对R20 greedy1000pair](../artifacts/entity_set_mlp_94695_vs_r20_greedy_20261021/README.md)得37.075%，区间[35.000%,39.175%]。

2026-10-19按已保存原始产物更新完成状态与冷产物位置。研究事实和限制集中在[当前状态](current_project_state_zh-CN.md)，实验身份与证据见[实验索引](../artifacts/experiment_index.json)。历史准备说明和被替代设计见[历史文档](archive/README.md)。

| 近期专题 | 当前结论与入口 |
| --- | --- |
| 实体汇总残差MLP | [256宽×4块实现与验证](../artifacts/entity_set_mlp_256x4_20261021/README.md)：848,418参数，目标牌/贵族属性直连动作head；旧schema/目标兼容，无正式训练或棋力结论 |
| 12张贵族规则池 | [规则修正与验证](../artifacts/noble_pool_bga12_20261021/README.md)：加入蓝绿黑333、红白44；双人仍抽3张，Python/native/BGA/采集/评测共用规范表，旧产物按原牌池保留 |
| 搜索训练新功能 | [预算随机化、Forced Playouts与Policy Target Pruning](guides/search_training_zh-CN.md)：默认关闭，有限工程验证，无正式训练与棋力结论 |
| 新版128×2无辅助从零训练 | [完成结果](../artifacts/entity_attention_128x2_relations_no_aux_r24_r43_20261019/results.md)：68041步，选37423；greedy53.775%、搜索56.625%，单训练seed与固定面板 |
| 原小模型辅助消融 | [完成结果](../artifacts/entity_attention_r24_r43_aux_ablation_20261019/results.md)：无辅助已提供普通训练对照，未确认辅助头额外收益 |
| R43历史池离线 | [完成结果](../artifacts/entity_attention_r43_history_pool_20261012/results.md)：独立500pair为50.50%，未确认超原R43 |
| 256×4双辅助从零训练 | [中断状态与恢复](../artifacts/entity_attention_256x4_aux_r24_r43_20261014/README.md)：最近保存44227步，终点和搜索未完成；[结构有限验证](../artifacts/entity_attention_256x4_20261014/README.md)单独解释 |
| R43搜索与value诊断 | [搜索预算／随机节点／CPW完成结果](../artifacts/entity_attention_r43_search_variants_arena_20261014/results.md)、[64根终局信号](../artifacts/entity_attention_r43_value_signal_64_20261014/results.md)、[Greedy动作排序完成结果](../artifacts/entity_attention_r43_action_ranking_64_20261015/results.md)；[S1024续局分支已停止](../artifacts/entity_attention_r43_action_ranking_64_s1024_20261015/README.md) |
| 冷产物与恢复 | [本次7.03GiB迁移清单](../artifacts/repository_archive_20261019/README.md)、[归档登记](../archive/README.md)；20261012旧归档已由用户外置，当前未挂载 |

## 规则与契约

- [双人标准规则](splendor_standard_rules_zh-CN.md)，卡牌与常量见 `data/`。
- [引擎与 Observation](specs/engine_contract_zh-CN.md)。
- [特征 schema](specs/feature_schema_zh-CN.md)。
- [学习数据与模型契约](specs/learning_contract_zh-CN.md)。
- [PPO 训练、历史池与评测协议](specs/ppo_training_protocol_zh-CN.md)。

## 操作入口

| 目的 | 文档 |
| --- | --- |
| 搜索算法、后端与推理 | [搜索](guides/search_zh-CN.md)、[native 构建](guides/native_search_zh-CN.md)、[多 worker GPU](guides/multiworker_gpu_zh-CN.md) |
| 数据格式、采集与恢复 | [自对弈数据](guides/selfplay_data_zh-CN.md) |
| 训练与恢复 | [离线执行默认](guides/offline_execution_zh-CN.md)、[AZ 循环](guides/alphazero_zh-CN.md)、[实体注意力离线学习](guides/entity_attention_offline_zh-CN.md) |
| R43 起点 S4096/N1、policy 2k／value 6k | [连续自迭代配置、时限与恢复](../artifacts/entity_attention_r43_s4096_p2k_v6k_20261013/README.md)：保持 lr2e-5、1000 局／1000 步和联合更新，单次 8 小时会话 |
| R43起点同历史池离线（完成） | [流程与命令](../artifacts/entity_attention_r43_history_pool_20261012/README.md)、[完成结果](../artifacts/entity_attention_r43_history_pool_20261012/results.md)：48776步、5组greedy／4组搜索及独立500pair均完成；选中＋24388步独立确认为50.50% [47.35%,53.70%]，未确认收益；前置R24离线候选对R43为51.25%，区间跨50% |
| R24起点历史池离线（完成） | [完整流程](../artifacts/entity_attention_r24_history_pool_20261012/README.md)、[结果](../artifacts/entity_attention_r24_history_pool_20261012/results.md)：选中＋24388步，对原R24独立500pair为54.00% [50.80%,57.10%]；尚未由此证明超过R43 |
| 近期大池离线训练 | [29,000局 attention S1024/N1固定池](../artifacts/entity_attention_recent_pool_20261009/README.md)，34398步对θ5000得分86.125%，最终完整模型55.50%；最终policy＋34398步value的双网络组合200pair得分92.25%，较34398步提高6.125个百分点，配对差CI [+2.000,+10.250] |
| P4起点九轮自迭代（完成） | [完整报告](reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)：终点搜索对P4／同预算旧池对照／P12＋V4得分58.750%／61.625%／58.125%，各200pair且区间高于50%；[流程与命令](guides/entity_attention_p4_pool_iteration_zh-CN.md)，旧池224／在线32渐变至第6轮32／224，第7轮起纯在线 |
| 第9轮终点长期在线续训（截至第43轮已完成34轮续训） | [已完成结果与限制](../artifacts/entity_attention_r9_online_lr2e5_1000_forever_20261011/README.md)、[流程与统一启动／恢复命令](guides/entity_attention_p4_pool_forever_plan_20261011_zh-CN.md)：从38,898步继承原第4–9轮6000局窗口，1000局／1000步／lr2e-5，不用旧离线池，各轮1000步完整模型升级教师；每轮greedy、绝对第12／15……轮搜索，各200pair |
| 大池模型组合与greedy对战队列 | [三组搜索＋十组greedy，各200pair](../artifacts/entity_attention_recent_pool_crossplay_20261010/README.md)，串行、可中断恢复；[进度与结果](../artifacts/entity_attention_recent_pool_crossplay_20261010/results.md)自动更新 |
| 最近的连续实验 | [行为复用 M4](guides/entity_attention_avg4_iteration_zh-CN.md)，12／16轮完成，当前停止；[分头窗口](guides/entity_attention_split_training_zh-CN.md)，20轮完成 |
| 较早训练分支的复现 | [10×1000](guides/entity_attention_self_iteration_zh-CN.md)、[5000步混合分支](guides/entity_attention_hybrid_zh-CN.md)、[第九轮 N2 fork](guides/entity_attention_hybrid_fork_zh-CN.md)、[N1 小步分支](guides/entity_attention_hybrid_s1024_n1_zh-CN.md) |
| 未完成的 B→C→D 趋势实验 | [套件流程](guides/entity_attention_iteration_suite_zh-CN.md)，B仅完成前两轮，C／D未运行 |
| 棋力与快速诊断 | [配对评测及网页](guides/evaluation_zh-CN.md)、[固定256真实局面](guides/decision_probe_zh-CN.md) |
| BGA 实时/离线只读推荐 | [本地权重推荐器](guides/bga_advisor_zh-CN.md)：已有 Chrome 自动识桌／任意轮次接入；双方行动及翻牌后模型估计；greedy／原生MCTS；用户自行执行 |
| 配置与运行工具 | [configs](../configs/README.md)、[scripts](../scripts/README.md) |

## 最近的实验结论与证据

较早已完成方案见[P4起点九轮自迭代完整报告](reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)：完整对照支持本次在线方案的棋力收益，但不单独证明教师换代贡献或最优参数。较早的 [M4 与固定数据诊断完整结论](reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md) 统合以下实验，结论限定对应分支，不能用本次新方案回填旧分支收益。

| 问题 | 专题与原始记录 |
| --- | --- |
| 冷产物对应什么结论 | [本次7个离线run归档](../artifacts/repository_archive_20261019/README.md)、[旧归档45来源／721入口结论索引](../artifacts/repository_archive_20261012/conclusions.md)；旧归档已外置，当前本地副本见[登记](../archive/README.md)，专题报告与必要原始统计保留 |
| 早期旧θ5续训十轮是否完成 | [完成报告](reports/az_theta5_10x1000_zh-CN.md)：10000局／12050更新及20个常规面板完成；终点对旧θ5搜索200pair未确认提升，对PPO追加面板只有100/200pair |
| 早期采集、smoke与临时校准器 | [工程试验说明](reports/archived_engineering_experiments_20261012_zh-CN.md)：说明保存事实及适用范围，没有正式棋力证据的明确标出 |
| P4新分支第九轮的步数／lr／池大小／batch | [10组单变量完成分析](../artifacts/entity_attention_r9_ofat_20261010/analysis.md)、[100–2000步全部55点矩阵](../artifacts/entity_attention_r9_ofat_20261010/steps_100_2000.md)、[完整曲线与配对区间](../artifacts/entity_attention_r9_ofat_20261010/results.md)：同一第八轮终点及AdamW，固定第九轮完整留出；较小lr对policy有正面信号，1k池加重value过拟合；单个候选的后续棋力补测见下一条 |
| 第九轮小学习率候选的直接搜索棋力 | [lr2e-5×1000步 vs lr1e-4×500步](../artifacts/entity_attention_r9_lr_step_arena_20261011/README.md)、[完成结果](../artifacts/entity_attention_r9_lr_step_arena_20261011/results.md)：新开局200 pair，双方S1024/N1；前者得分53.250%，95% CI [48.500%, 58.000%]，尚未确认棋力优势 |
| 平均标签与连续自迭代 | [早期固定数据对照](../artifacts/entity_attention_target_averaging_20261007/README.md)、[M4 12轮记录](../artifacts/entity_attention_avg4_behavior_20261009/README.md) |
| 历史配额、训练／留出、梯度与 AdamW | [固定 A／B／C 对照](../artifacts/entity_attention_policy_history_20261009/README.md) |
| 独立新标签及未抽中 Train 局面 | [fresh ΔKL 修订与补测](../artifacts/entity_attention_avg4_delta_kl_20261009/README.md)；撤回原“抽中显著改善／未抽中稳定退化”解释 |
| 步数、学习率与裁剪 | [单轮剂量](../artifacts/entity_attention_step_dose_20261007/README.md)、[正式账本 lr×步数](../artifacts/entity_attention_formal_lr_step_20261007/README.md)、[clip 探针](../artifacts/entity_attention_clip_sweep_20261009/README.md)；50步仅是初筛较高点估计 |
| value 能否拟合及能否提升棋力 | [policy／value 消融](../artifacts/entity_attention_policy_value_ablation_20261006/README.md)、[critic-only](../artifacts/entity_attention_critic_only_replay_20261008/README.md)、[两版 encoder 重拟合](../artifacts/entity_attention_encoder_critic_refit_20261008/README.md)、[critic 适配与已完成 DEV](../artifacts/entity_attention_critic_adaptation_strength_20261008/README.md) |
| 搜索在哪些阶段有收益 | [只搜前中期／后期／全程](../artifacts/entity_attention_stage_search_20261008/README.md) |
| 搜索噪声、随机节点与 CPW 教师 | [系列报告](reports/entity_attention_search_teachers_zh-CN.md)，从报告定位各原始实验 |

## 基线与性能报告

- [引擎历史验收](reports/engine_validation_zh-CN.md)、[PPO 5M seed1](reports/ppo_5m_zh-CN.md)、[AZ 五轮与扩展评测](reports/alphazero_seed1_zh-CN.md)、[MCTS sweep 完成情况](reports/mcts_sweep_zh-CN.md)。
- [推理吞吐有效基线及失效结果](reports/inference_throughput_zh-CN.md)、[多 worker 的窗口／整轮比较](reports/multiworker_gpu_zh-CN.md)。
- [流程时间口径](reports/pipeline_profile_zh-CN.md)、[端到端性能验收](reports/end_to_end_speed_optimization_zh-CN.md)、[实体注意力执行优化](reports/entity_attention_execution_speed_zh-CN.md)。性能结果限定机器、模型、形状与版本，没有确认所有负载下的最佳 batch／worker。

## 维护与历史资料

[产物约定](../artifacts/README.md) 说明哪些原件纳管、哪些保留本地。[2026-10-12 本地归档](../artifacts/repository_archive_20261012/README.md) 给出全量采样盘点、迁移、结论映射与恢复说明；[2026-10-09 整理记录](../artifacts/repository_cleanup_20261009/README.md) 保留此前的路径迁移与结论修订；更早的本地文档包在 `runtime/document_archive/2026-10-01/`。`prompt/` 为被忽略的历史任务资料，只有本次明确采用时才是需求。
