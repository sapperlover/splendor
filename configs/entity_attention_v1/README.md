# Entity attention v1 配置

2026-10-27新增[`offline_r37_104k_lr1e4_20261027.json`](offline_r37_104k_lr1e4_20261027.json)与[`offline_r37_104k_lr2e5_20261027.json`](offline_r37_104k_lr2e5_20261027.json)：相同104k无辅助离线预设，仅增加原R37权重身份并设对应lr；新AdamW／原离线种子，18GiB／4 worker、350552步上限、5%主KL＋MSE早停选点保持。[命令与验证](../../artifacts/entity_attention_r37_104k_offline_lr_20261027/README.md)，未启动正式训练。

2026-10-27完成状态：[六组原离线实验及五模型大面板](../../docs/reports/auxiliary_offline_results_20261027_zh-CN.md)均已完成，policy＋得分和无辅助不再是准备／等待状态；policy在线已提交13轮，第12轮搜索完成。下方按日期记录原配置发布过程；仅得分辅助的独立预设仍未执行正式训练。

2026-10-26新增[`offline_noaux_r21_r37_20261026.json`](offline_noaux_r21_r37_20261026.json)：原104k池，从零583682参数基础attention128，无任何辅助目标；与policy＋得分组相比仅移除continuation辅助及其梯度诊断字段，lr1e-4／18GiB／4共享worker／预算／验证／早停／选点保持。[串行命令与验证](../../artifacts/entity_attention_128x2_noaux_r21_r37_20261026/README.md)。

2026-10-26新增[`offline_policy_score_r21_r37_20261026.json`](offline_policy_score_r21_r37_20261026.json)：与policy104k实际prefetch配置相比仅新增得分归属辅助0.25，policy辅助0.15、短期value关闭；原104k池／split、从零128×2、新AdamW、lr1e-4、18GiB／4 worker、5%完整主KL＋MSE早停选点保持。[脚本、命令与验证](../../artifacts/entity_attention_128x2_policy_score_r21_r37_20261026/README.md)，未启动正式训练。

2026-10-26新增[`offline_score_ownership_r21_r37_20261026.json`](offline_score_ownership_r21_r37_20261026.json)：104k原池，仅启用双方×四类未来新增分的8路归属回归，权重0.25；沿用offline_continuation，4 worker共享供数、17GiB（按同池旧arena上界核算）、流式／state-only／批量回读。已准备身份与有限验证，未启动正式训练；[定义与命令](../../docs/guides/score_ownership_zh-CN.md)。

2026-10-25存储更新：使用原80000局池的配置运行前，须先从[归档20261025](../../artifacts/repository_archive_20261025/README.md)恢复原始NPZ；四组完成辅助实验的普通模型／主日志也在本批。配置身份、关键模型及当前在线初始64k／新采集输入保持，不改写冻结配置。

2026-10-25新增[random_budget_policy140221_r31_r37_p60k_20261025.json](random_budget_policy140221_r31_r37_p60k_20261025.json)：仅policy辅助最优离线140221步全量继承AdamW/RNG，在线lr2e-5、3000步／新10k／容量60k；历史R31–R37共64k使用逐轮4000／10000映射，`initial_history_eviction=true`首轮淘汰14k历史。继续policy辅助0.15，固定raw R37，采集较优设置／评测A，六轮全量留出沿用。[命令及验证](../../artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/README.md)；已训练12轮，第12轮搜索中断。缓存容量仍16GiB，完整轮训练后释放，实际脚本已换独立发布，不增加算法配置字段；见[缓存生命周期](../../artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/cache_release_20261025/README.md)。

2026-10-24发布记录：后两组已使用`offline_continuation_r21_r37_prefetch_20261024.json`和`offline_shortterm_value_r21_r31_vs_r37_prefetch_20261024.json`，本次均18GiB／4 worker，启用流式统计和传输优化。新建离线默认采用后三项，缓存按池大小配置；原16GiB配置与冻结计划保持。[实际发布与命令](../../artifacts/offline_training_defaults_20261024/README.md)，[通用默认](../../docs/guides/offline_execution_zh-CN.md)。下方后两组未切换等说明为此前任务记录。

2026-10-24发布记录：当前policy104k组与队列已停止在115000步，流式统计／传输优化发布继续沿用下方18GiB／4 worker配置，无算法配置变更。本次只准备命令，未重启；[实际冻结发布与恢复](../../artifacts/training_streaming_20261024/README.md)。

2026-10-24执行配置：`offline_opponent_policy_r21_r37_prefetch_20261024.json`为当前仅policy组的18GiB／4 worker恢复配置，算法字段沿用原预设；原16GiB配置保留为历史身份。[恢复与共享只读供数](../../artifacts/training_prefetch_20261024/README.md)。

2026-10-24新增`offline_opponent_policy_r21_r37_20261024.json`（104k，value辅助0／policy辅助0.15）、`offline_continuation_r21_r37_20261024.json`（104k，0.25／0.15）与`offline_shortterm_value_r21_r31_vs_r37_20261024.json`（原44k，0.25／0）。其余复制当前104k仅value预设；44k仍为12次value曝光，预算自动148632步，三组均对raw R37做greedy1000pair后搜索A500pair。[串行顺序、开关与冻结身份](../../artifacts/entity_attention_auxiliary_queue_20261024/README.md)。

2026-10-23新增[offline_shortterm_value_r21_r37_20261023.json](offline_shortterm_value_r21_r37_20261023.json)：R21–R37共104k原对局，一基绝对轮64–80，每轮局数映射4000／10000；从零attention128＋H2/H6/H14短期value权重0.25、对手policy辅助0，lr1e-4，16GiB，两头B256，350552步／5%主KL+MSE早停选点。随后对raw R37 greedy1000pair、最新A搜索500pair；[实现、命令与启动身份](../../artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023/README.md)。

两份continuation离线配置新增`encoder_grad_norm_every_steps=100`，记录首步及每100步的逐任务共享编码器梯度；1逐步记录、0关闭。对手覆盖率、未知牌质量比例和条件CE每步及各验证集合均记录，分母定义见[诊断字段](../../docs/guides/continuation_auxiliary_zh-CN.md#训练与验证诊断2026-10-23追加)。

2026-10-23新增[`offline_shortterm_value_r21_r31_20261023.json`](offline_shortterm_value_r21_r31_20261023.json)与[`offline_continuation_r21_r31_20261023.json`](offline_continuation_r21_r31_20261023.json)，均由`python -m splendor.offline_continuation`执行：前者仅2／6／14玩家回合指数value（权重0.25），后者另加两层对手policy（权重0.15）。44k原池、原split、B256双batch／lr1e-4／12次曝光／三次上升早停；只准备配置并完成[有限验证](../../artifacts/continuation_auxiliary_20261023/README.md)，未运行正式训练。覆盖限制、参数量、恢复及命令见[专题](../../docs/guides/continuation_auxiliary_zh-CN.md)。

2026-10-23记录策略更新：R31在线`checkpoint_every_steps`改为500，仍强制保存并评估0／100／250／500／1000／2000／3000七节点；当前在线／近期离线默认停写双份`.bin`和双份`.steps.jsonl`，每步主指标保留。详见[记录规则](../../docs/guides/training_artifacts_zh-CN.md)，历史配置与产物不回写。

当前在线配置使用`six_round_holdout`：第一段manifest路径＋SHA、batch512、block_rounds6。R32–R37使用R26–R31留出，R38–R43改用R32–R37留出，之后每六轮推进；每轮七节点全量评价、恢复自动补齐。初版永久固定面板设置`fixed_holdout`已由此替代；[定义与执行](../../docs/guides/evaluation_zh-CN.md#在线全量固定留出)。

2026-10-23当前配置：[random_budget_r31_10k_p60k_20261023.json](random_budget_r31_10k_p60k_20261023.json)从raw R31完整模型／AdamW／RNG起步，R22–R31历史40k、每轮10k新局、容量60k、lr2e-5／3000步、16GiB compact缓存。已完成6轮至谱系R37，第7轮已建立采集配置；首次重置后的“未重启”状态已过时。采集继续12worker／active2000／B500／microbatch128／owner双槽，历史固定25%policy子集／淘汰／配额不变。**后续新搜索面板改为配置A**：12worker／active1000／B500／microbatch128／owner双槽，B桶[64,128,256,500]、C桶[32,48,64]；greedy仍CPU8 worker／1000 pair，每3个本地轮次搜索500 pair。新轮界读取源配置，旧面板保持冻结参数；[运行与变更记录](../../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/README.md)。

日常评测新增可执行示例：[evaluation_greedy_r43.json](evaluation_greedy_r43.json)为CPU8 worker／1000 pair零搜索；[evaluation_search_a_r43.json](evaluation_search_a_r43.json)为双方S1024N1（总1024）／500 pair／GPU配置A。示例候选固定R37、对手固定原R43，实际比较前复制模板、明确更换模型及输出路径；不自动追踪最新模型。采集、日常greedy／搜索、在线greedy／搜索的完整对照及命令见[统一指南](../../docs/guides/evaluation_zh-CN.md)。

下方按日期保留的旧分支单槽、已取消搜索及首次准备状态属于历史配置／当时范围，不作为当前默认。2026-10-23三组离线候选已另对R43完成500 pair搜索，结果见[独立补测](../../artifacts/eval_speed_20261023/README.md)；它不补写原R20／R31搜索面板。

2026-10-23撤回新增购买结算关系方案，删除专用`offline_first_round_r21_r31_20261022.json`及入口／启动脚本。原583682参数关系attention128、现有离线与在线配置保持原配方，短期进展辅助关闭；旧组合／纯特征准备及删除前配置见[历史记录](../../artifacts/entity_attention_first_round_20261022/README.md)。

2026-10-22新增[offline_128x2_r21_r31_lr2e5_20261022.json](offline_128x2_r21_r31_lr2e5_20261022.json)：原59453步继承权重／AdamW／RNG，lr2e-5，每2000新增步固定子集／完整留出；44k原数据、两头B256及16GiB不变。已在新增10000步三次严格上升早停，选中新增4000步／累计63453步；原新增148632步上限保留，训练不恢复。用户后续追加R31 greedy1000pair独立面板，当前命令、原始结果及核验见[专题](../../artifacts/entity_attention_128x2_r21_r31_lr2e5_20261022/README.md)。

2026-10-22新增[offline_128x2_r21_r31_early_stop_20261022.json](offline_128x2_r21_r31_early_stop_20261022.json)：第21–31轮44000局，原始round_index63–73（配置1基64–74），从零attention128及全新AdamW，lr1e-4，其余继承原attention128同8万局离线配置。原12次value曝光上限148632步／连续三次完整留出严格上升早停配置保留；用户后续主动停止于保存84000步，按已完成节点固定59453步，仅执行对R20／R31各greedy1000pair，搜索取消。当前仅greedy命令及结果见[实验入口](../../artifacts/entity_attention_128x2_offline_r21_r31_20261022/README.md)，不使用原完整队列恢复训练。

2026-10-22新增[random_budget_r20_s256n4_noble12_20261022.json](random_budget_r20_s256n4_noble12_20261022.json)：继承raw R20状态、AdamW/RNG及24k活动池，从第21轮无限续跑；25%高预算S256N4（基础simulations仍为总1024）、75%低预算S128N1（low_simulations128、low_determinizations1），新采集用12贵族。其余有效在线配置沿用原分支，旧21轮残留已授权删除。[配置身份、验证与恢复](../../artifacts/entity_attention_r20_s256n4_noble12_iteration_20261022/README.md)。下文原在线分支“完成17轮”是历史时点，实际已完成20轮。

后续状态（2026-10-22核对）：下方attention128同8万局对照已由用户启动，后续以KeyboardInterrupt退出，最近保存135279步；“正式训练未启动”保留为首次配置交付时的状态。旧run不由本次新实验恢复或修改；原270557步计划未完成。

2026-10-21新增[同MLP的8万局attention128从零离线对照](../../artifacts/entity_attention_128x2_offline_80k_20261021/README.md)：`offline_128x2_no_aux_80k_lr1e4_20261021.json`使用现有128×2／4头／FFN512购买关系预设、583682参数，lr1e-4、16GiB缓存、policy高预算/value全预算各256、270557步，每5%同固定子集与完整留出。除模型及独立输出路径外与MLP配置相同，正式训练未启动；命令`bash scripts/run_attention128_offline_20261021.sh`只训练＋KL+MSE选点，不自动对战。

新增[random_budget_37423_p6k_v24k_20261019.json](random_budget_37423_p6k_v24k_20261019.json)：37423步继承AdamW、lr2e-5，每轮4k局／1000步，25% S1024＋75% S128，FP/PTP关闭；原R24–43共20k局接入24k池，旧局随机淘汰，旧policy固定25%子集并控制累计新局quota为1/6、1/3…。[启动／恢复及限制](../../artifacts/entity_attention_random_budget_iteration_20261019/README.md)。2026-10-20已完成17轮，第18轮中断未运行。采集为B500/pipeline2；后续搜索评测改为500pair并复用采集execution的其他参数（12 worker、1000活跃局、B500及native CUDA Graph），搜索owner单槽，双方仍全程S1024、无根噪声。原命令可恢复；旧评测保留200pair，已准备面板沿用冻结配置。

新增[collection_randomized_budget.json](collection_randomized_budget.json)：新版128×2的37423步教师，25% S1024＋75% S128，根Forced Playouts k=2＋Policy Target Pruning；仅高预算policy监督、全部步骤value监督。待运行采集示例；[用法与在线训练配置位置](../../docs/guides/search_training_zh-CN.md)。

2026-10-19状态核对：新版128×2、无辅助／仅回合消融均已完成，结果见各专题；256×4双辅助保持中断。7个离线run的中间checkpoint与详细账本已[集中归档](../../artifacts/repository_archive_20261019/README.md)，配置JSON与冻结路径不变；只读复核需恢复归档挂载；恢复训练前按清单撤回对应run的迁移，避免写入冻结归档。

2026-10-19新增[新版128×2、无辅助头、lr1e-4从零训练](../../artifacts/entity_attention_128x2_relations_no_aux_r24_r43_20261019/README.md)：`offline_128x2_relations_no_aux_r24_r43_20261019.json`保留新版购买关系与value读取，R24–43原20000局、曝光12次/68041步、每5%完整留出，选KL＋MSE最小完整模型后对R43测1000pair greedy／200pair S1024。其余配置逐字段沿用256实验；`offline_scratch.value_auxiliary=null`实际不注册两个辅助头。

2026-10-19新增[128×2](model_relations_128x2.json)和[256×4](model_relations_256x4.json)购买关系预设：`market_relations=true`、`value_readout="global_players_market"`，市场与双方明保留牌共用8维公开购买关系＋全局/双方玩家/市场均值的value读取；无辅助参数分别583,682／3,828,226。它们是`model_config`片段，旧配置不自动切换；辅助升级会保留两项开关。明保留扩展复用原开关和参数，盲保留不追加关系。[结构、兼容性和有限验证](../../artifacts/entity_attention_relations_20261019/README.md)。

操作和结果说明见 [专题指南](../../docs/guides/entity_attention_offline_zh-CN.md)。所有命令从仓库根目录执行。

| 配置 | 用途与前置条件 |
| --- | --- |
| `offline_256x4_aux_r24_r43_20261014.json` | 256×4随机初始化、双辅助各0.25、PSW关闭，原R24–43的20000局，曝光12次／68041步，每5%完整留出评估并按KL＋胜负MSE选完整checkpoint，随后对R43测1000pair greedy／200pair S1024；[运行与续跑](../../artifacts/entity_attention_256x4_aux_r24_r43_20261014/README.md)。本配置使用 `splendor.offline_scratch`，不能交给旧 `offline_objectives` CLI。 |
| `model_256x4.json` | 新模型的 `model_config` 片段：hidden256、heads8、layers4、FFN1024、dropout0，3,628,034参数；嵌入新训练配置或交给`build_actor_critic`，不是完整训练CLI配置。历史默认仍128×2，旧R43不能直接按大模型续训。[结构与有限验证](../../artifacts/entity_attention_256x4_20261014/README.md) |
| `offline_r24_r43_policy_surprise_20261013.json`、`offline_r24_r43_value_auxiliary_20261013.json` | 原R43起点，绝对R24–43的20000局固定池，各22681步；50% PSW与双辅助头（各权重0.25）分别训练。训练完成，随后独立执行final对原R43各1000pair greedy：51.350%／53.900%，辅助组区间高于50%。[命令](../../artifacts/entity_attention_r24_r43_offline_objectives_20261013/README.md)、[结果](../../artifacts/entity_attention_r24_r43_offline_objectives_20261013/results.md) |
| `r9_online_lr2e5_1000_forever_20261011.json`、`hybrid_r9_online_lr2e5_1000_forever_20261011.json` | 原第9轮／38898步续训，继承第4–9轮6000局及原split；纯在线、1000局／1000步／lr2e-5／batch256、B500采集。1000步完整模型无条件升级；每轮greedy、绝对第12／15……轮搜索各200pair；无限循环，同一命令启动／恢复。已完成截至绝对第43轮的34轮续训；历史命令保留，重新执行将恢复／继续运行。[流程与命令](../../docs/guides/entity_attention_p4_pool_forever_plan_20261011_zh-CN.md) |
| `p4_pool_iteration_20261010.json`、`hybrid_p4_pool_20261010.json` | P4/34398步及连续AdamW，9轮1000局/500步，旧池／在线224／32渐变至第6轮32／224，第7轮起仅在线近6000局；采集B500、Graph档16/32/64/128/192/256/384/500。教师无条件升级；主线、同预算对照、全量留出及14面板已全部完成，终点搜索胜过三个已测对手，限定单seed/DEV，未确定最优参数。[完整报告](../../docs/reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)、[数据与命令](../../docs/guides/entity_attention_p4_pool_iteration_zh-CN.md) |
| `offline_attention.json` | 新结构随机初始化，全部十轮原 split，batch256、12,045步、4 GiB LRU；写 `runs/entity_attention_offline` |
| `offline_deep_sets.json` | 旧结构也随机初始化，与新结构同数据、seed、优化及抽样预算；写 `runs/deep_sets_offline` |
| `offline_smoke.json` | 首轮前6局、batch16、8步、验证16、64 MiB LRU；由有限验证脚本使用 |
| `selfplay_smoke.json` | 新 smoke 导出模型、native_cuda_pool、2局、S16/N1、compiled/reused、projection=false |
| `evaluation_smoke.json` | 新 smoke 对旧θ5，native_batched、1 pair、S16/N1；分别配置新false/旧true projection |
| `mcts_vs_theta5.json` | 正式离线 final 对旧θ5，双方S1024/N1、50 pairs；需要先生成新 final |
| `greedy_vs_theta5.json` | 正式新旧裸网 greedy、50 pairs；需要先生成新 final |
| `greedy_vs_ppo_5_5m_100pairs.json` | 新离线final vs PPO lifetime5,512,112，双方greedy、100 pairs/200局 |
| `mcts_s1024n1_vs_ppo_5_5m_100pairs.json` | 同一双方checkpoint，双方MCTS S1024/N1、100 pairs/200局，compiled/reused、新false/旧true projection |
| `offline_step{1000,5000}_greedy_vs_ppo_5_5m_100pairs.json` | 离线训练1000/5000步对PPO lifetime5,512,112，双方greedy，各100 pairs |
| `offline_step{1000,5000}_mcts_s1024n1_vs_ppo_5_5m_100pairs.json` | 相同离线节点与PPO双方S1024/N1，各100 pairs；与final面板复用相同game/policy/search/bootstrap seed |
| `online_3x200.json` | 早期短阶段候选保留，不是近期M4配置 |
| `online_10x1000.json` | 历史自迭代：离线final起点，10轮×1000局、同时调度1000局、S1024/N1；每轮对固定起点搜索/greedy各50 pairs；[命令与验证](../../docs/guides/entity_attention_self_iteration_zh-CN.md) |
| `collection_static_graph.json` | 可选采集预设：W8/A1000/B500、shared_slots=2、固定B64/256/500、native-cudagraph、owner单缓冲；只验证短窗口收益，[报告与命令](../../docs/reports/entity_attention_execution_speed_zh-CN.md) |
| `hybrid_static_graph.json` | 从随机初始化离线5000步，逐渐混入自迭代；每轮1000局/500更新，第16轮起纯在线；共享槽2＋固定地址Graph、owner单缓冲；replay总上限16 GiB，混合时两池各8 GiB；[启动与恢复](../../docs/guides/entity_attention_hybrid_zh-CN.md) |
| `hybrid_cpu_smoke.json` | 独立有限CPU链路验证：离线4步、每轮2局/2更新、batch8、S2/N1；不用于正式训练或棋力结论 |
| `hybrid_fork_s512_n2.json` | 从第九轮9500步分支，继承AdamW/RNG；S512/N2、100局/100步、在线窗口2000局、active100/B128、共享槽2/固定地址Graph/owner单缓冲；[命令与数据替换](../../docs/guides/entity_attention_hybrid_fork_zh-CN.md) |
| `hybrid_fork_s1024_n2.json` | 同上，仅采集总模拟改为1024，每世界512；其余训练/采样/评测参数相同 |
| `hybrid_fork_s1024_n1.json` | 从原第九轮9500步恢复，S1024/N1、AdamW/RNG接续，200局/100步、3000局/15个教师；B200/Graph[16,32,64,128,200]，共享槽2/owner单缓冲，每5轮对theta_5000评测；已完成50新轮/14500步后停止。原resume仍全在线池均匀采样，不执行最新25%；[状态与命令](../../docs/guides/entity_attention_hybrid_s1024_n1_zh-CN.md) |
| `hybrid_split_policy_value.json` | 从当前N1第50新轮/14500步fork；policy近4轮，detach value保留3000局/近4轮50%，各batch256、独立clip1；200局/100步，固定PPO双方S1024/N1每5轮200局；[启动与恢复](../../docs/guides/entity_attention_split_training_zh-CN.md) |
| `iteration_avg4_behavior_20261009.json` | 16500步分支，行为＋3份完整S1024/N1平均policy标签；16轮200局／100步／lr1e-4，原policy4轮／value3000局；追加标签8个CPU worker／总256根／单GPU batch200；每轮256局面诊断，2/4/8/12/16轮对冻结起点各100pair，无硬RAM／磁盘门槛；[启动／恢复](../../docs/guides/entity_attention_avg4_iteration_zh-CN.md)，已完成12轮／17700步，原早期面板未确认稳定收益；后续r4、r12各300pair扩样完成，分别50.83%、49.17%，区间均跨50%；固定历史采样与梯度诊断已完成，暂不改正式配额 |
| `iteration_trend_20261008.json` | 未完成趋势方案：共同16500步起点，按B→C→D完成三组各seed1×16轮；2/4/8/12/16轮对冻结起点100pair，4GiB缓存/16GiB内存上限，无磁盘门槛；[同一启动/恢复命令](../../docs/guides/entity_attention_iteration_suite_zh-CN.md) |
| `online_smoke.json` | 本次两轮GPU短测：每轮4局、2步、S16/N1，搜索/greedy各1 pair；独立runtime输出 |

配置交付时没有启动正式任务；随后新结构离线已完成12045步及六个PPO搜索/greedy面板，原混合run完成9轮、小步N1分支完成50轮。本次[共同PPO补测配置与结果](../../artifacts/entity_attention_strong_anchor_20261006/README.md)独立归档，没有改写正式训练配置。旧[最新单轮份额](../../docs/archive/entity_attention_fresh_share_zh-CN.md)保留为历史设计，其JSON不能作为CLI配置；分头窗口/detach分支后续已完成20新轮／16500步；上述40项回归和短GPU验证是当时的工程检查，不代表长期收益确认。

两个离线配置保持原冻结4 GiB值，实际运行建议加`--replay-cache-mib 8192`；resume同样可加此执行参数，无须修改配置。`history_batch_size`在离线入口无作用；离线只有一个覆盖全选定train positions的池。新结构不支持actor projection，混合评测通过每方`inference`明确区分。

[旧四组配置](../archive/entity_attention_iteration_suite_20261008.json) 已移入 `configs/archive/`，是被替代的方案历史快照，仅供复核旧准备记录；当前入口不再支持其协议。
## 2026-10-19：小模型辅助目标对照

新增`offline_r24_r43_no_aux_20261019.json`与`offline_r24_r43_turns_only_20261019.json`，除`value_auxiliary`外逐字段沿用旧小模型双辅助配置：无辅助为null，仅剩余回合为score_loss_weight0／turns_loss_weight0.25。两组R43 warm start各22681步，串行训练及final对R43的1000pair greedy／200pair S1024见[专题入口](../../artifacts/entity_attention_r24_r43_aux_ablation_20261019/README.md)。默认旧双辅助行为保留；单回合模型配置`predict_score=false`且不注册分差头。
