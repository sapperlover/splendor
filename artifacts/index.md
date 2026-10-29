# 实验导航

2026-10-29：[R37初值lr2e-5固定140221步对raw R37](r37_lr2e5_step140221_vs_r37_10k_20261029/README.md)，先greedy10000pair，再双方S128N1（总128）10000pair，固定用户指定节点、不跟随最新；15项回归及CPU根验证通过，正式评测未启动。

2026-10-27：[R37权重起点的104k离线lr1e-4／2e-5两组](entity_attention_r37_104k_offline_lr_20261027/README.md)，新AdamW、原离线RNG、全部辅助关闭，其余原离线设置保持；有限CPU／恢复检查通过，仅给命令，未启动正式训练。

2026-10-27：[五个104k best的10组大面板](offline5_vs_r37_10k_20261027/README.md)全部完成，各greedy／S128N1各10000 pair，共20万局；原训练及原S1024N1评测也均已完成。[六组指标与结论](../docs/reports/auxiliary_offline_results_20261027_zh-CN.md)。

2026-10-27：[104k无辅助](entity_attention_128x2_noaux_r21_r37_20261026/README.md)及[后继队列](entity_attention_noaux104k_queue_20261026/README.md)已完成；210332步早停、选中122694步，原greedy／S1024N1为51.075%／57.50%。

2026-10-27：[104k policy＋得分](entity_attention_128x2_policy_score_r21_r37_20261026/README.md)已完成训练及原两组对战，选中140221步，原greedy／S1024N1为53.475%／56.80%；后续10000 pair面板也已完成。

[第12轮模型2000个多支付局面矩阵](payment_search_matrix_local12_2k_20261026/README.md)：同根greedy、S256N4、S1024N1、S4096N4各有／无噪声，12000次搜索／7372.8万次模拟全部完成；多金首选率无噪为0.25／1.40／2.35／2.50%，有噪为3.00／3.70／4.85／5.30%；分时期、剔除直接终局及前三份额／配对区间见专题，不能当作花金动作劣势或训练因果结论。

[分数归属8输出工程验证](score_ownership_20261026/README.md)：未来新增分标签、四类来源／双方视角、4 worker共享池恢复、逐任务梯度与主指标选点；已补[基线与分阶段诊断](score_ownership_20261026/diagnostics/README.md)，支付样本保留，61项回归及17局1356个决策的真实验证通过，训练／AdamW／RNG／原验证指标逐位保持；无正式棋力结论。

[第12轮final直接对原混合预算R20 greedy](entity_attention_local12_vs_mixed_r20_greedy_20261026/README.md)：1000 pair／2000局，第12轮得70.350%，95%整pair区间[68.350%,72.325%]，1398胜18和584负；当前12贵族池、CPU8、103.10秒，全部有效。与单盘人类棋谱吻合度分别解释，未测双方搜索对战。

[第12轮模型对原人类棋谱的完整搜索矩阵](bga_927500015_local12_analysis_20261026/README.md)：同原R20的8组×3种子，1824根／15876096模拟全部完成，16.03分钟，无失败或回退；56步主行动首选吻合7组下降、1组上升，前三覆盖6升1平1降。[逐配置比较](bga_927500015_local12_analysis_20261026/comparison.md)。

2026-10-25已完成[四组离线普通模型／逐步日志和原80k局池归档](repository_archive_20261025/README.md)：672个普通checkpoint、4份主日志、80000个NPZ，包8.84GiB，净释放11.06GiB。最优／完整留出／initial/final模型和当前在线输入保留；原80k离线复训或完整复核须先恢复本包。下方较早“原80k留本机”是当时记录，当前位置以本批清单为准。

[Policy辅助离线最优→60k在线分支](entity_attention_policy140221_r31_r37_p60k_iteration_20261025/README.md)：140221步继承AdamW/RNG，R31–R37历史64k，首轮移出14k历史；继续policy辅助0.15、每轮10k新局／3000步／lr2e-5、采集较优设置／评测A，对raw R37。已提交13轮；第12轮greedy66.075%、完整搜索60.20%，第13轮greedy63.75%；[近期平台分析](entity_attention_policy140221_r31_r37_p60k_iteration_20261025/plateau_review_20261025/results.md)及[训练后释放缓存发布](entity_attention_policy140221_r31_r37_p60k_iteration_20261025/cache_release_20261025/README.md)已完成，本次未续跑。

[后两组全部接入与离线默认](offline_training_defaults_20261024/README.md)：104k双辅助、44k仅value已另建18GiB／4 worker冻结发布并切换原脚本；未来离线默认流式累计、4 worker及传输优化，缓存按池大小设置。183项有限回归及两组真实CUDA逐位对照通过，749个历史原件哈希保持，本次未启动正式训练。下文仅policy适用范围为前一任务记录。

[流式统计与主进程传输优化](training_streaming_20261024/README.md)：当前104k仅policy组及队列已停止于115000步，未重启。固定大小会话累计、state-only value／辅助传输、标量合并回读；87项回归和真实CUDA逐位一致性通过，已给单组／队列恢复入口。下面prefetch运行描述为此前时点记录。

[18GiB与4 worker训练供数发布](training_prefetch_20261024/README.md)：当前104k仅policy组从102000步恢复，单份共享只读训练池、独立验证缓存、已消费RNG边界；63项回归及真实CUDA逐位恢复验证通过。热缓存小数据测速约1.16倍，原队列与后续两组配置保持。

[三组后续辅助消融的串行队列](entity_attention_auxiliary_queue_20261024/README.md)：当前104k仅value完整结束后，104k仅policy → 104k双辅助 → 44k仅value；90项回归、CPU真实更新／原生搜索通过，等待监督锁且失败不推进。

[R21–R37从零attention128＋仅短期value](entity_attention_128x2_shortterm_value_r21_r37_20261023/README.md)：104000局，H2/H6/H14权重0.25、对手policy辅助关闭，lr1e-4／350552步上限／5%主KL+MSE早停；最低点对raw R37 greedy1000pair后搜索A500pair，最新轻量记录；已按更正停止队列，initial.pt／0步保留，命令与停止身份见专题。

[辅助任务诊断追加](continuation_auxiliary_20261023/diagnostics/README.md)：逐任务共享编码器梯度、对手标签覆盖率、未知牌质量比例与条件CE已记录；156项回归及真实局短验证通过，诊断开关下训练更新逐位一致。默认首步及每100步取梯度，可在恢复时调整。

[短期value与两层对手policy辅助](continuation_auxiliary_20261023/README.md)：2／6／14玩家回合指数标签、类型及购买／保留牌软CE，已接离线训练／精确恢复／主指标选点；3局真实源的一步CPU训练及3个原生根只作工程验证，未做正式棋力实验。

[训练记录／checkpoint更新](entity_attention_r31_10k_p60k_iteration_20261023/recording_checkpoint_20261023/README.md)：停写当前双份抽样与双份逐步明细，在线500步周期点＋七个固定诊断点；86项工作区／37项冻结源码回归通过，实际入口预检及已完成目标恢复通过，历史产物保持。

2026-10-23已完成[五类冷产物T7迁移](repository_archive_20261023/README.md)：223个离线普通checkpoint、1011个抽样账本／逐步日志、76038个旧NPZ，流式包5.8104GB，完整SHA／纳秒mtime校验后净释放13.9211GB。原80k局、父分支46142局、当前60k局及所有未选模型保留；旧局池复训／完整诊断需先恢复归档。本次未启动训练或棋力评测。

2026-10-19核对。机器可读身份、适用范围及当前证据SHA见[experiment_index.json](experiment_index.json)。完成训练、完成评测、确认收益分别解释；历史报告中的准备日期不代表当前状态。

| 近期实验 | 已保存结果与限制 |
| --- | --- |
| 当前采集／日常与在线评测 | [统一配置与命令](../docs/guides/evaluation_zh-CN.md)：greedy CPU8／1000 pair，日常及在线新搜索统一A／500 pair；[三组离线搜索](eval_speed_20261023/README.md)已完成，80k／44k／低lr得74.05%／70.70%／72.20%，同R43面板，尚非最优测速结论 |
| R31起点10k新局／60k池／3000步自迭代 | [运行、配置与A变更](entity_attention_r31_10k_p60k_iteration_20261023/README.md)：[固定留出首段已回补42节点](entity_attention_r31_10k_p60k_iteration_20261023/fixed_holdout_20261023/results.md)；[六轮分段调度](entity_attention_r31_10k_p60k_iteration_20261023/six_round_holdout_20261023/README.md)已生效，R38–R43用R32–R37留出；已完成6轮至R37，greedy68.90%、搜索74.55%；采集active2000保持，新搜索A为active1000／双槽／精简B、C桶；新轮界读取，旧面板不改写 |
| 新增购买结算方案已撤回 | [历史记录与撤回验证](entity_attention_first_round_20261022/README.md)：2026-10-23删除新增购买／贵族／终局派生分支及候选入口，恢复原583682参数模型；短期进展辅助关闭。原准备／短验证／冗余分析及删除前源码保留，无正式训练或棋力结论 |
| 第21–31轮attention128最低节点降lr续训 | [训练与评测入口](entity_attention_128x2_r21_r31_lr2e5_20261022/README.md)：新增10000步三次上升早停，选中新增4000步／累计63453步；对R31 greedy1000pair为50.150% [48.025%,52.300%]。相对父模型同开局差+0.625pp、区间跨0，未确认greedy收益；训练不恢复、搜索未测 |
| 第21–31轮44000局attention128从零离线训练 | [脚本、配置与证据](entity_attention_128x2_offline_r21_r31_20261022/README.md)：用户停止训练并保存84000步，原148632步预算未完成；完整留出选中59453步，对R20／R31各greedy1000pair为53.400%／49.525%，支持超R20、未确认超R31。两组搜索取消；配对区间、原始结果及仅greedy命令见专题 |
| 第21–30轮及Cross64离线冷产物T7归档 | [清单、完成核验与恢复](repository_archive_20261022/README.md)：291模型／43唯一账本外置，12相同账本去重，外盘2.45GB、净释放4.3250GB；第21–31轮全部44000采集NPZ、未完成轮2142局及原80k池完整保留；各轮final／离线关键节点原位，未改模型、数据或训练配置 |
| Cross64 MLP 54112步对R20 greedy1000pair | [完成结果与核验](entity_set_mlp_cross64_54112_vs_r20_greedy_20261022/README.md)：完整留出KL＋MSE预选54112步，858胜／13和／1129负，得分43.225% [41.075%,45.325%]；2000局、12贵族池、无搜索，支持本候选greedy弱于R20 |
| Cross64 MLP角色std0.05＋8万局离线训练 | [配置、命令与验证](entity_set_mlp_cross64_role005_offline_80k_20261022/README.md)：906434参数，同旧MLP数据、种子、lr1e-4、16GiB及270557步／5%留出。用户后来启动，现已中断，保存81168步；六节点联合最低54112步0.472901／0.772611，value最低40584步0.769011，后续回升；原预算及自动对战流程未完成 |
| R20继承续跑：S256N4＋12贵族 | [配置、启动与验证](entity_attention_r20_s256n4_noble12_iteration_20261022/README.md)：继承raw R20状态与24k池，25%高预算总1024／4世界，75%低预算总128／1世界，每轮4k局及1000步，连续运行；旧第21轮2591局残留已删除 |
| Checkpoint去重与T7外置 | [完成清单与恢复](repository_archive_20261021/README.md)：161个副本去重、1885个冷模型已校验外置至`archive-20261021.tar.zst`，包10.80GB，本阶段释放12.85GB；当前训练及关键模型保留，冷模型使用前需解压 |
| Checkpoint占用分级与首次清理 | [冻结审计记录](checkpoint_storage_audit_20261021/README.md)：先删5组编译缓存及96个MLP普通恢复点，共3.4335GB；之后的新授权去重／外置见上一行，原审计不改写 |
| Attention128 94695步对R20 greedy1000pair | [结果与核验](entity_attention_128x2_94695_vs_r20_greedy_20261021/README.md)：54.400%，95%配对区间[52.299%,56.525%]，1078胜／20和／902负；当前12贵族池，未测搜索 |
| Attention128同8万局离线训练 | [配置与命令](entity_attention_128x2_offline_80k_20261021/README.md)：最近提交135279步，日志KeyboardInterrupt退出，未自动恢复；583682参数、128×2／4头／FFN512，其余数据／抽样／种子／lr1e-4／270557步／5%留出与MLP相同，本次未改配置或中断进程 |
| MLP 94695步对R20 greedy1000pair | [原始结果与核验](entity_set_mlp_94695_vs_r20_greedy_20261021/README.md)：已完成2000局，MLP得37.075%，95%配对区间[35.000%,39.175%]，736胜／11和／1253负，12贵族池 |
| 当前完整实体MLP的8万局离线训练 | [脚本、停止记录与指标](entity_set_mlp_256x4_offline_80k_20261021/README.md)：仍停止并保留97000步；七个已评估节点中94695步KL/MSE为0.530577/0.782832、联合最低；原自动面板未启动，新授权greedy独立保存，无搜索对战 |
| 实体汇总256宽×4残差块MLP | [64维／2头动作cross-attention](entity_set_mlp_256x4_20261021/action_cross_64x2_20261022/README.md)：当前906,434参数，读取25个未汇总实体，保留动作直连及非线性／乘积独立开关，value-only跳过新增分支；197项通过、1项CUDA跳过，CPU编译／原生短验证通过。原855,682／848,418参数记录及8万局训练保留原身份，无新正式训练／棋力结论 |
| 第20轮S1024N1对同模型Greedy | [同开局N1/N4对照](entity_attention_r20_s1024n1_vs_greedy_20261021/README.md)：1000pair、当前12贵族池，与S256N4面板共用全部开局及策略seed；12 worker/B500/单槽，完成后自动输出两组共同Greedy得分的配对差区间；不是N4对N1直接对战 |
| 四种搜索的输入缓存潜力 | [96相同基准根×2seed完成结果](search_input_cache_rate_20261021/results.md)：S1024N1／S256N4／S1024N4（总1024／1024／4096）及随机节点总1024为14.62%／19.20%／20.22%／2.69%；仅单次搜索内精确输入复用，不启用缓存、不测速、不含跨走子 |
| 第20轮S256N4对同模型Greedy | [配置及12 worker入口](entity_attention_r20_s256n4_vs_greedy_20261021/README.md)：1000 pair／2000局，256×4=总1024模拟，当前12贵族规则；12 worker、1000活跃/B500、owner单槽，greedy批量根前向且零模拟；运行状态见runtime/live.json，结束自动保存配对区间与逐局预算核验 |
| 第20轮128固定局面搜索稳定性 | [脚本与口径](entity_attention_r20_search_stability_128_20261021/README.md)：96个分阶段基础根＋32个独立预筛分歧根，7种候选及两个高预算参考各8 seed；S×N为总预算，比较JS、漏搜、Top-k、重要性与参考一致/分歧分组。后台状态见该实验runtime/live.json，完成后自动汇总，不是棋力或真实最优性结论 |
| 12张贵族规则池修正 | [规则与验证](noble_pool_bga12_20261021/README.md)：新增蓝绿黑333、红白44，双人从12张抽3张；Python/native/BGA/采集/评测统一，manifest冻结牌池指纹；180项通过、3项CUDA条件跳过，另4局真实模型CPU/GPU采集及checked重放通过 |
| 第20轮纯随机节点S128对同模型Greedy | [配置与运行入口](entity_attention_r20_chance_s128_vs_greedy_20261021/README.md)：1000 pair／2000局，native_chance逐模拟重采、CPW关闭；双方为BGA分析所用R20原模型。完成后自动输出配对区间和逐局预算核验，运行状态见该实验runtime/live.json；不修改在线教师 |
| BGA #934713322第18步提前计划诊断 | [恢复与条件节点](bga_934713322_move18_plan_20261021/README.md)：第18步V逐位恢复；原N1/seed0盲保留645/1024，假想下一手买023，买061为0访问；S16384同样。24根均未展开真实第20步完整信息集，不能把未覆盖当作提前拒绝061 |
| BGA #934713322买061单步分析 | [完整结果](bga_934713322_move20_061_20261021/README.md)：精确复现S1024/N1的607／255／52访问量；同一根3档预算×4种方式×3种子，061并非始终首选。S16384随机节点3/3选买，native N1/N4及CPW均0/3选买；单局面描述，不代表真实行动损失或方法棋力 |
| BGA #927500015逐步模型分析 | [全部统计](bga_927500015_r20_analysis_20261020/statistics/results.md)：第20轮原模型8组×3种子、1824次根搜索全部完成；56主行动一致25–29步、实际平均访问份额33.95%–43.89%；Q差受低访问影响，不是已验证的人类失误或棋力差 |
| 第17–19轮EMA回溯 | [完整结果](entity_attention_ema_17_19_20261020/results.md)：H1000获选；第19轮EMA对同轮raw，greedy1000pair得49.70%，搜索500pair得48.45%，两组区间均跨50%；留出小幅改善，未确认棋力收益 |
| 搜索训练新功能工程验证 | [预算随机化、Forced Playouts与Policy Target Pruning](search_training_20261019/README.md)：183项通过、5项CUDA条件跳过；新版128两局完整CPU循环及混合/低预算训练恢复验证，不是棋力实验 |
| 新版128×2无辅助从零训练 | [结果](entity_attention_128x2_relations_no_aux_r24_r43_20261019/results.md)：68041步、选37423，greedy1000pair为53.775%，S1024搜索200pair为56.625% [51.750%,61.375%]；未隔离各结构／日程改动贡献 |
| 新版128×2的30619步补测 | [1000pair greedy对R43](entity_attention_128x2_relations_no_aux_r24_r43_20261019/greedy_step30619/results.md)：47.775% [45.600%,49.925%]；同开局对共同R43的得分比37423步低6.000pp，配对差区间[−8.950,−2.925]pp；未测该节点搜索棋力 |
| 小模型辅助消融 | [结果](entity_attention_r24_r43_aux_ablation_20261019/results.md)：两组22681步及四组面板均完成；旧双辅助减plain的greedy差+0.050pp、区间跨0，未确认辅助头额外收益 |
| R24–43原PSW／双辅助 | [原结果与后续对照](entity_attention_r24_r43_offline_objectives_20261013/results.md)：各1000pair greedy，历史结果保留；普通训练对照已在10/19独立专题完成，原双辅助仍未测搜索 |
| R43历史池离线 | [结果](entity_attention_r43_history_pool_20261012/results.md)：48776步及全部2300pair完成，＋24388步独立确认50.50% [47.35%,53.70%]，未确认超过原R43 |
| R24离线候选对R43 | [结果](entity_attention_r24_offline_vs_r43_20261012/results.md)：200pair、51.250% [46.125%,56.375%]，未确认超过R43；[原R24实验](entity_attention_r24_history_pool_20261012/results.md)对原R24的小幅收益仍限定原对手 |
| 256×4从零训练 | [中断说明](entity_attention_256x4_aux_r24_r43_20261014/README.md)：最近保存44227步，终点／搜索未完成；[34021步greedy](entity_attention_256x4_aux_r24_r43_early34021_20261014/results.md)45.45%；[结构验证](entity_attention_256x4_20261014/README.md)及[1024局面容量测试](entity_attention_capacity_1024_20261014/results.md)分别解释 |
| R43低预算／高预算／随机节点／CPW | [S128–512三组](entity_attention_r43_low_search_budget_arena_20261014/results.md)及[S1024／S4096／chance／CPW四组](entity_attention_r43_search_variants_arena_20261014/results.md)均完成，各500pair；未证明相应采集预算的训练收益 |
| R43终局信号与动作排序 | [64根×16终局信号](entity_attention_r43_value_signal_64_20261014/results.md)、[Greedy配对排序](entity_attention_r43_action_ranking_64_20261015/results.md)已完成；[S1024续局分支](entity_attention_r43_action_ranking_64_s1024_20261015/README.md)用户停止，保留部分结果，不作完整排序结论 |
| 公开购买关系与value多实体读取 | [结构与兼容性](entity_attention_relations_20261019/README.md)含市场／双方明保留牌、独立开关及有限验证；新版128训练结果属于上方独立专题 |
| R43起点S4096、policy2k/value6k | [会话与后续配置](entity_attention_r43_s4096_p2k_v6k_20261013/README.md)：已完成R44–R47，8小时会话到时停止；后续policy6k配置仅准备 |
| 较早在线研究 | [P4九轮完整报告](../docs/reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)、[在线主线截至R43](entity_attention_r9_online_lr2e5_1000_forever_20261011/results.md)、[M4整合报告](../docs/reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md)，分别保留分支与版本范围 |
| 产物集中与旧归档 | [本次7.03GiB／361文件](repository_archive_20261019/README.md)、[冷产物登记](../archive/README.md)；[旧45来源／721入口结论索引](repository_archive_20261012/conclusions.md)继续保留，旧真实归档已由用户外置、当前未挂载 |

更完整的按用途入口见[docs导航](../docs/index.md)。报告与原始统计保留原实验身份，本次没有重跑训练或评测。

## 未完成实验／确认

2026-10-20：[37423步起点混合预算持续迭代](entity_attention_random_budget_iteration_20261019/README.md)在本次EMA启动前已完成20轮、第21轮采集已开始；搜索为500pair、B500、12 worker／1000活跃局及owner单槽，采集仍双槽，保留旧200pair结果。 已汇总[17轮85个节点的学习轨迹](entity_attention_random_budget_iteration_20261019/learning_curves_20261020/results.md)，并完成[6–11／12–17轮共60节点全量固定留出回补](entity_attention_random_budget_iteration_20261019/fixed_holdouts_20261020/results.md)：前组value确认改善，后组改善被第17轮末尾反复抵消；固定旧teacher的policy KL持平／增加，非棋力评测。EMA另在上方独立专题运行，不改变在线teacher。

| 实验身份 | 阅读入口 |
| --- | --- |
| `mcts-sweep` | [MCTS 50-pair 对照的实际状态](../docs/reports/mcts_sweep_zh-CN.md) |
| `entity-attention-chance-depth-2026-10-07` | [大S深度初筛（用户取消后续）](entity_attention_chance_depth_20261007/README.md) |
| `entity-attention-iteration-trend-seed1-2026-10-08` | [B → C → D 单种子趋势实验](entity_attention_iteration_trend_seed1_20261008/README.md) |
| `entity-attention-critic-adaptation-strength-2026-10-08` | [16600原critic适配与后续搜索棋力检验](entity_attention_critic_adaptation_strength_20261008/README.md) |
| `entity-attention-avg4-behavior-preparation-2026-10-09` | [行为复用M4连续训练：记录](entity_attention_avg4_behavior_20261009/README.md) |

## 有效记录

| 实验身份 | 阅读入口 |
| --- | --- |
| `entity-attention-r30-value-weight-2026-10-11` | [第29轮终点：value权重1／0.25／0的训练与评测已完成](entity_attention_r30_value_weight_20261011/README.md)、[整网结果及归因限制](entity_attention_r30_value_weight_20261011/analysis.md) |
| `entity-attention-r9-lr-step-arena-2026-10-11` | [lr2e-5×1000步 vs lr1e-4×500步：完成200 pair直接搜索对战](entity_attention_r9_lr_step_arena_20261011/README.md)、[53.250%，95% CI [48.500%, 58.000%]](entity_attention_r9_lr_step_arena_20261011/results.md) |
| `entity-attention-r9-ofat-2026-10-10` | [第九轮步数／学习率／在线池／batch单变量完成结论](entity_attention_r9_ofat_20261010/analysis.md)、[100–2000步全部55点矩阵](entity_attention_r9_ofat_20261010/steps_100_2000.md)、[完整曲线与区间](entity_attention_r9_ofat_20261010/results.md) |
| `entity-attention-p4-p9-crossplay-2026-10-10` | [新P9＋旧V4、旧P4＋新V9对起点，完成各200pair](entity_attention_p4_p9_crossplay_20261010/README.md)、[完整结果](entity_attention_p4_p9_crossplay_20261010/results.md) |
| `entity-attention-r9-step-extension-2026-10-10` | [第九轮固定6000局窗口：500／1000／2000／5000步留出拟合](entity_attention_r9_step_extension_20261010/README.md)、[结果表与差值区间](entity_attention_r9_step_extension_20261010/results.md) |
| `entity-attention-p4-pool-iteration-2026-10-10` | [九轮完整报告](../docs/reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)、[结果与复算证据](entity_attention_p4_pool_iteration_20261010/README.md) |
| `az-loop-seed1` | [AlphaZero seed1：五轮训练与逐代评测](../docs/reports/alphazero_seed1_zh-CN.md) |
| `native-inference-08` | [Native 推理吞吐：有效基线与结论边界](../docs/reports/inference_throughput_zh-CN.md) |
| `transfer-profile-09` | [Native 推理吞吐：有效基线与结论边界](../docs/reports/inference_throughput_zh-CN.md) |
| `gpu-pool-09` | [多搜索 worker 供给单 GPU：窗口与整轮结果](../docs/reports/multiworker_gpu_zh-CN.md) |
| `pipeline-profile-2026-10-02` | [采集、learner 与固定评测：分项时间统计](../docs/reports/pipeline_profile_zh-CN.md) |
| `end-to-end-speed-2026-10-03` | [端到端速度优化与完整验收](../docs/reports/end_to_end_speed_optimization_zh-CN.md) |
| `entity-attention-v1-2026-10-04` | [实体注意力模型与离线搜索经验学习](../docs/guides/entity_attention_offline_zh-CN.md) |
| `offline-training-cache-speed-2026-10-04` | [离线训练缓存瓶颈与执行参数优化](offline_training_speed_20261004/README.md) |
| `entity-attention-self-iteration-preparation-2026-10-04` | [新模型自迭代：10×1000](../docs/guides/entity_attention_self_iteration_zh-CN.md) |
| `entity-attention-execution-speed-2026-10-05` | [新模型固定形状、kernel启动与独立双缓冲](../docs/reports/entity_attention_execution_speed_zh-CN.md) |
| `entity-attention-search-screen-2026-10-05` | [Attention 自迭代教师 S/N 有限筛选（2026-10-05）](entity_attention_search_screen_20261005/README.md) |
| `entity-attention-hybrid-fork-2026-10-05` | [从第九轮创建 S512/N2 与 S1024/N2 训练分支](../docs/guides/entity_attention_hybrid_fork_zh-CN.md) |
| `entity-attention-worker-cleanup-2026-10-05` | [OOM 后孤儿进程清理与 owner 死亡退出修复](entity_attention_worker_cleanup_20261005/README.md) |
| `entity-attention-hybrid-graph200-2026-10-06` | [200局采集的Graph尺寸与恢复验证](entity_attention_graph200_20261006/README.md) |
| `entity-attention-memory-window-2026-10-06` | [S1024/N1 第31轮 RAM OOM：修复与历史窗口分析](entity_attention_memory_window_20261006/README.md) |
| `entity-attention-strong-anchor-2026-10-06` | [原第9轮与第50轮：共同PPO锚点补测及最新数据方案](entity_attention_strong_anchor_20261006/README.md) |
| `entity-attention-policy-value-ablation-2026-10-06` | [Entity attention：固定数据 policy / value 消融（2026-10-06）](entity_attention_policy_value_ablation_20261006/README.md) |
| `entity-attention-split-training-2026-10-06` | [分头窗口与detach value实现验证](entity_attention_split_training_20261006/README.md) |
| `entity-attention-target-noise-2026-10-07` | [冻结 attention 教师的目标噪声 A/B/C/D 实验](entity_attention_target_noise_20261007/README.md) |
| `entity-attention-chance-search-2026-10-07` | [随机节点／信息历史搜索：同预算 A/B/C/D 复测](entity_attention_chance_search_20261007/README.md) |
| `entity-attention-chance-strength-2026-10-07` | [同一冻结模型：随机节点与确定化搜索的100 pair棋力对照](entity_attention_chance_strength_20261007/README.md) |
| `entity-attention-chance-pw-2026-10-07` | [Chance progressive widening：实现与有限诊断](entity_attention_chance_pw_20261007/README.md) |
| `entity-attention-chance-pw-noise-2026-10-07` | [CPW k=0.25、α=0.5：A/B/C/D目标稳定性复测](entity_attention_chance_pw_noise_20261007/README.md) |
| `entity-attention-chance-pw-strength-2026-10-07` | [CPW k=0.25、α=0.5：同模型100 pair棋力对照](entity_attention_chance_pw_strength_20261007/README.md) |
| `entity-attention-chance-pw-greedy-2026-10-07` | [CPW k=0.25、α=0.5：对同权重greedy的100 pair](entity_attention_chance_pw_greedy_20261007/README.md) |
| `entity-attention-chance-training-probe-2026-10-07` | [随机节点／CPW 搜索与冻结教师短训](../docs/reports/entity_attention_search_teachers_zh-CN.md) |
| `entity-attention-target-averaging-2026-10-07` | [随机节点／CPW 搜索与冻结教师短训](../docs/reports/entity_attention_search_teachers_zh-CN.md) |
| `entity-attention-step-dose-2026-10-07` | [步数、有效学习率、value 与标签平均：200 局短训剂量对照](entity_attention_step_dose_20261007/README.md) |
| `entity-attention-formal-lr-step-2026-10-08` | [正式回放账本上的学习率×步数短训探针](entity_attention_formal_lr_step_20261007/README.md) |
| `entity-attention-iteration-suite-validation-2026-10-08` | [最终套件有限 CUDA 验证](entity_attention_iteration_suite_validation_v2_20261008/README.md) |
| `entity-attention-stage-search-2026-10-08` | [只搜后期／只搜前中期：同权重 greedy 对照](entity_attention_stage_search_20261008/README.md) |
| `entity-attention-critic-only-replay-2026-10-08` | [固定encoder，只练critic：B2的50步账本回放](entity_attention_critic_only_replay_20261008/README.md) |
| `entity-attention-encoder-critic-refit-2026-10-08` | [冻结16500与16600 encoder，重新拟合相同critic](entity_attention_encoder_critic_refit_20261008/README.md) |
| `decision-probe-v1-2026-10-09` | [固定 256 个独立真实局面的决策诊断](decision_probe_v1_20261009/README.md) |
| `entity-attention-policy-history-fixed-m4-2026-10-09` | [固定M4数据：历史采样与policy梯度诊断](entity_attention_policy_history_20261009/README.md) |
| `entity-attention-avg4-independent-delta-kl-2026-10-09` | [行为+M4 分支：独立 M4 配对 ΔKL 诊断（2026-10-09）](entity_attention_avg4_delta_kl_20261009/README.md) |
| `entity-attention-m4-diagnostics-conclusions-2026-10-09` | [M4 连续训练与固定数据诊断：完整结论（2026-10-09）](../docs/reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md) |
| `entity-attention-clip-sweep-2026-10-09` | [policy 组 clip 阈值回放探针（2026-10-09）](entity_attention_clip_sweep_20261009/README.md) |
| `entity-attention-scaleup-2026-10-09` | [不确定结果扩样实验（2026-10-09）](entity_attention_scaleup_20261009/README.md) |
| `entity-attention-hybrid-kl-trend-2026-10-09` | [5000步起点→9轮：固定留出、9代矩阵与基线复核（2026-10-09）](entity_attention_hybrid_kl_trend_20261009/README.md) |
| `entity-attention-hybrid-offline9500-2026-10-09` | [离线9500重放、固定留出与200pair面板（2026-10-09）](entity_attention_hybrid_offline9500_20261009/README.md) |
| `entity-attention-recent-pool-2026-10-09` | [29,000局 attention S1024/N1：34398步得分86.125%，终点55.50%；最终policy＋34398步value组合200pair得分92.25%，较34398步的配对差CI高于零](entity_attention_recent_pool_20261009/README.md) |

## 历史有效记录

| 实验身份 | 阅读入口 |
| --- | --- |
| `python-engine-rules-v2` | [Python 规则引擎历史验收](../docs/reports/engine_validation_zh-CN.md) |
| `ppo-5m-seed1` | [PPO 5M 单 seed 训练报告](../docs/reports/ppo_5m_zh-CN.md) |
| `az-theta5-10x1000-seed1-preparation` | [旧θ5十轮的历史准备快照](../docs/archive/next_az_training_zh-CN.md)，不代表正式运行仍未启动 |
| `az-theta5-10x1000-seed1-completed` | [旧θ5十轮完成事实、锚点结果与未完成面板](../docs/reports/az_theta5_10x1000_zh-CN.md) |
| `native-early-panel` | [Native 推理吞吐：有效基线与结论边界](../docs/reports/inference_throughput_zh-CN.md) |
| `entity-attention-hybrid-s1024-n1-2026-10-06` | [原第九轮 S1024/N1 小步更新配置验证](entity_attention_hybrid_s1024_n1_20261006/README.md) |
| `entity-attention-hybrid-s1024-n1-replay3000-2026-10-06` | [原第九轮 S1024/N1 小步更新配置验证](entity_attention_hybrid_s1024_n1_20261006/README.md) |

## 失效或被替代

| 实验身份 | 阅读入口 |
| --- | --- |
| `inference-06-07` | [Native 推理吞吐：有效基线与结论边界](../docs/reports/inference_throughput_zh-CN.md) |
| `entity-attention-iteration-suite-2026-10-08` | [连续自迭代实验套件交付（历史方案）](entity_attention_iteration_suite_20261008/README.md) |
| `entity-attention-iteration-trend-2026-10-08` | [B → C → D 趋势实验交付（历史双种子准备）](entity_attention_iteration_trend_20261008/README.md) |

## 未核验记录

| 实验身份 | 阅读入口 |
| --- | --- |
| `native-multi-round-training` | [AlphaZero 式 policy/value 循环](../docs/guides/alphazero_zh-CN.md) |

索引聚焦已复核的专题；未列入的既有基线和搜索汇总仍在原目录，未列入不代表失效。每个目录内的README、配置、原始汇总与分析入口共同限定其版本。目录身份和恢复产物不因本次导航整理而重建。
