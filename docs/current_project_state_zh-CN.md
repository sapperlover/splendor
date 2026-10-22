# 当前项目状态

2026-10-22的[MLP角色初始化标定及同8万局离线训练](../artifacts/entity_set_mlp_cross64_role005_offline_80k_20261022/README.md)：906434参数，三个角色embedding各std0.05，合成RMS0.0864、内容0.0909；乘积及动作旁路不变。继承旧MLP数据／抽样及配置（lr1e-4、16GiB、270557步、每5%留出），95项CPU和实际GPU短验证通过。最初按要求仅交付命令，用户后来自行启动，现执行记录为中断、已保存81168步，六个百分比节点完成。完整留出KL＋MSE最低为54112步0.472901／0.772611；value最低40584步0.769011，此后升至81168步0.797616。独立授权的[54112步对R20 greedy1000pair](../artifacts/entity_set_mlp_cross64_54112_vs_r20_greedy_20261022/README.md)已完成：858胜／13和／1129负，43.225% [41.075%,45.325%]，弱于R20；无异常／非法动作／工程截断。本次未恢复原训练或测搜索，也未以对战重新选点；原270557步计划仍未完成。此前std1 cross历史验证保留。

2026-10-22按用户要求已后台启动[第20轮继承续跑分支](../artifacts/entity_attention_r20_s256n4_noble12_iteration_20261022/README.md)：raw R20累计57423步、AdamW/RNG及24k活动池完整继承，从第21轮连续运行；25%高预算改为S256N4（总1024），75%低预算仍S128N1，新采集及评测使用12贵族。其余在线训练／评测设置沿用，历史10贵族标签只读保留并按每轮4k淘汰。37项CPU回归、追加恢复回归及2局实际GPU混合世界采集／回放通过；生产manifest已核验，工程验证不构成新棋力结论。旧第21轮2591局残留已按授权删除，未动第20轮状态及活动池。

2026-10-22加入[MLP动作cross-attention](../artifacts/entity_set_mlp_256x4_20261021/action_cross_64x2_20261022/README.md)：当前模型预设906,434参数，单层64维／2头，Query读动作及共享状态，K/V读25个未汇总实体并按局面共享；value-only跳过新增分支。保留原动作MLP和乘积及独立开关，未证明其有益或有害。197项通过、1项CUDA跳过，实际预设CPU编译／checkpoint／原生短验证通过；未运行新正式训练、消融或棋力赛。原8万局MLP仍是冻结855,682参数结构及其原结果，未升级或恢复。

2026-10-21按用户授权先清理3.4335GB缓存及96个MLP恢复节点，随后完成[161个副本去重与1885个冷模型外置](../artifacts/repository_archive_20261021/README.md)：T7包`archive-20261021.tar.zst`为10.80GB，全部模型SHA／纳秒mtime已核对，再释放12.8474GB本地实体。MLP的9个指定模型、R20／R43／37423锚点及当前attention保持本机；8万局数据和账本未迁移。EMA第12–19轮168个来源及9个导出均保存在新包中，使用冷模型前需解压。原20261019保留24个关键点中23个此次已迁入新包，按两批清单查找，不能把历史保留数视为当前本机实体数。

[Attention128同8万局离线训练对照](../artifacts/entity_attention_128x2_offline_80k_20261021/README.md)最近提交135279步，核验时约135600步以KeyboardInterrupt退出，未自动恢复；原270557步计划未完成。模型128×2／4头／FFN512、583682参数，保留购买关系及value读取，无辅助／PSW；沿用MLP数据、抽样、种子、lr1e-4、16GiB缓存及5%留出。本次未修改或中断训练。随后按独立授权完成[预先选定94695步对R20 greedy1000pair](../artifacts/entity_attention_128x2_94695_vs_r20_greedy_20261021/README.md)：54.400%，95%整pair bootstrap区间[52.299%,56.525%]，1078胜／20和／902负；支持该候选greedy强于R20，搜索及value棋力未测。

[MLP 94695步对R20的greedy1000pair](../artifacts/entity_set_mlp_94695_vs_r20_greedy_20261021/README.md)已按新授权完成：MLP得37.075%，95%整pair bootstrap区间[35.000%,39.175%]，736胜、11和、1253负；当前12贵族池。该候选明显落后，单一训练经历对照不能单独归因于结构。

[完整实体MLP的8万局离线训练](../artifacts/entity_set_mlp_256x4_offline_80k_20261021/README.md)仍停止，保留97000步checkpoint、账本及日志，原270557步计划未完成，原自动面板取消，未执行搜索对战。855682参数、lr1e-4、无辅助／PSW；七个已评估节点内94695步完整留出KL+MSE最低（0.530577/0.782832），value最佳在40584步（MSE 0.768390）。最近两个节点联合指标只改善0.000942。

停训后按用户要求仅做raw R20相同留出推理：完整641723局面（134304条policy标签）得到KL 0.445110、MSE 0.755265，两项均优于该MLP已评估节点；固定4096子集为0.416234/0.763407。复用冻结的评估源码、原标签和掩码，原始记录见专题目录`r20_same_holdout.json`；该比较没有控制初始化／训练经历，不单独归因于模型结构。

2026-10-21实现[实体汇总＋256宽4残差块MLP](../artifacts/entity_set_mlp_256x4_20261021/README.md)：当时预设855,682参数、无attention，动作head保留原始属性直连，新增90→32→128非线性分支及逐通道状态×动作乘积；原特征/目标、pooling和残差主干保留。[本次动作头验证](../artifacts/entity_set_mlp_256x4_20261021/actor_upgrade_12/README.md)与初版848,418参数的101项CPU／CUDA历史验证分开；未启动正式MLP训练或棋力赛。先行[S1024N1对Greedy 1000pair](../artifacts/entity_attention_r20_s1024n1_vs_greedy_20261021/README.md)使用冻结源码、与已完成S256N4共用1000个开局种子，运行状态以该实验产物为准，不受新模型开发影响。

2026-10-21搜索对greedy可复用采集的12 worker GPU pool，greedy通过root-only native批处理作业前向并argmax，零搜索世界/模拟/转移；50项相关回归和真实GPU短验证通过。[第20轮S256N4对Greedy](../artifacts/entity_attention_r20_s256n4_vs_greedy_20261021/README.md)为1000pair、当前12贵族池，正式评测已后台启动；启动不代表已完成或已确认棋力收益。

2026-10-21规则修正：[贵族池统一为12张](../artifacts/noble_pool_bga12_20261021/README.md)，新增N11蓝绿黑333、N12红白44，均3分，双人仍抽3张。Python、native、BGA映射、新采集和评测使用同一规范表；采集／评测恢复身份冻结catalog fingerprint，不能在旧10张池目录直接续接新规则。原模型权重形状不变，旧实验保留原规则身份。下方“原脚本可恢复”等较早记录，仅适用于当时冻结的代码与牌池。

2026-10-20本次EMA启动前核对：[37423步起点混合预算持续迭代](../artifacts/entity_attention_random_budget_iteration_20261019/README.md)已完成20轮，第21轮采集已开始；此前“17轮完成／第18轮中断”是较早时点。4k新局／24k池、lr2e-5／1000步、FP/PTP关闭，采集B500/pipeline2。新搜索评测从500pair起步，复用采集的12 worker／1000活跃局／B500与CUDA Graph，owner为单槽（采集仍双槽），保留双方全程S1024与固定配对裁决；旧200pair不改写。原脚本可恢复，已开始采集/评测沿用各自冻结配置。

2026-10-20按用户要求启动[第17–19轮EMA回溯测试](../artifacts/entity_attention_ema_17_19_20261020/README.md)：每轮使用自己的原2048条留出，三轮统一预选250／500／1000更新半衰期，随后只测第19轮EMA对同轮raw的greedy1000pair、S1024搜索500pair。每50步checkpoint近似EMA，从第12轮训练前累计，不平均AdamW、不改变当前在线学生或教师。7项有限回归和12个native S2根通过，后台实验尚未完成，不能作为EMA棋力收益结论。

2026-10-20完成该分支[六轮全量固定留出回补](../artifacts/entity_attention_random_budget_iteration_20261019/fixed_holdouts_20261020/results.md)：第1–6轮2397局原始留出评估第6–11轮，第6–11轮2406局评估第12–17轮，各30节点。前组value MSE 0.763401→0.751138，整局配对区间确认改善；后组0.768743→0.770732，净变化区间跨0，但第12–16轮各轮终点仍下降，第17轮500步0.762603后回升。固定旧teacher的policy KL前组持平、后组增加0.007422。限定历史标签与本训练轨迹，不能据此推断棋力变弱、训练完全停止或独立局数不足；先前滚动小面板“后期泛化收益减弱”的结论须保留这一细分。

2026-10-19新增[搜索预算随机化、Forced Playouts与Policy Target Pruning](guides/search_training_zh-CN.md)：现有采集和在线训练可配置开启，默认不改变旧行为；新数据schema v3支持复核每步预算、有效policy样本与裁剪标签，旧v1/v2可继续混池。完成有限工程验证，未启动正式训练，未验证棋力收益。

2026-10-19完成[新版128×2无辅助头从零训练与对战](../artifacts/entity_attention_128x2_relations_no_aux_r24_r43_20261019/results.md)：583,682参数，购买关系与value多实体读取开启，PSW和辅助头关闭、lr1e-4；原R24–43两万局、68041步，按完整留出KL＋MSE选37423步。对原R43的1000pair greedy为53.775% [51.600%,55.900%]，200pair S1024/N1为56.625% [51.750%,61.375%]。支持这套配方在已测面板超过R43，限定单训练seed与固定对手／预算，未独立归因于关系特征、value读取或日程；终点value MSE回升至0.909698，不能用终点代替预先定义的选点。

2026-10-19完成[公开卡牌购买关系与value多实体读取实现](../artifacts/entity_attention_relations_20261019/README.md)：市场及双方明保留牌共享8维关系投影，盲保留不追加，不改变原始schema与参数量；value及启用的辅助头分别读取全局、己方、对手和有效市场卡均值。新128×2／256×4预设分别583,682／3,828,226参数（无辅助），两个独立开关随checkpoint保存，省略维持旧行为；结构交付阶段仅做有限验证，后续新版128训练见上方独立专题。

2026-10-19完成[R24–43小模型辅助目标消融](../artifacts/entity_attention_r24_r43_aux_ablation_20261019/results.md)：plain和turns-only同R43起点／AdamW／两万局／抽样，各22681步；greedy对R43为53.850%／53.150%，旧双辅助为53.900%。旧双辅助减plain为+0.050pp [−0.900,+1.025]pp，turns-only减plain为−0.700pp [−1.475,+0.075]pp，未确认辅助头额外收益。两组搜索为46.750%／47.875%，区间均跨50%；共同对手差不是候选直接互打。旧双辅助未补搜索，单seed、复用旧greedy DEV、未做多重比较校正。

2026-10-15的[原64根排序R43＋S1024续局分支](../artifacts/entity_attention_r43_action_ranking_64_s1024_20261015/README.md)已按用户要求停止，监督与续局进程均退出。保留3505条自然终局、1745份完整双臂配对、15条已完成未配齐单臂，每动作共同前26份世界（各根完整前缀26–29份），原轨迹另作压缩归档。原32→128协议与正式全量回放／汇总未完成；此前对话中的13／20／21份面板仍只代表各自时点，不能当作停止时的新结果。配置及20项有限测试、短回放与单根S1024预算验证保留，Greedy分支不变。

2026-10-15完成[原64根Greedy配对动作排序诊断](../artifacts/entity_attention_r43_action_ranking_64_20261015/results.md)：64根含60个不同动作比较和4同选控制，50根补至128世界，共6848配对世界、13696条自然终局，全部回放、128条checked回放，40/60根仍未分辨。V、根Q和访问选择的排序指标只代表该冲突采样与Greedy续局协议，不是总体棋力或value的独立因果结论。

2026-10-15完成[R43小模型64根×16次终局信号诊断](../artifacts/entity_attention_r43_value_signal_64_20261014/results.md)：原R24–43留出中64局各一根，统一R43 S1024/N1、根噪声0.25、全程T0。单次MSE0.7555，结果方差0.6936，修正后期望value误差0.0620 [0.0250,0.1034]；前者占自身MSE约91.8%，模型与16次均值相关0.829。支持单局监督高方差，不能仅凭单次MSE平台断言value没有进步；仍存在期望预测误差，也没有去噪标签训练后的棋力结论。1024个终局全部回放、64条checked回放及同种子控制通过。采样为现有均匀隐藏世界重建，非历史后验；原主线开局T1，与本实验T0不同，限定说明见专题。

256×4双辅助头离线链路与续跑命令已交付，见[专题](../artifacts/entity_attention_256x4_aux_r24_r43_20261014/README.md)：R24–43固定20000局、PSW关闭、68041步，每5%完整留出评估并选KL＋胜负MSE最小点，再对R43测greedy1000pair／S1024搜索200pair。用户已执行至约44448步后中断，最近完整验证44227步；value MSE较30619步的0.765951回升至0.797854，policy KL仍缓降至0.458084。终点选点／自动对战未完成；首次中断缺进度文件造成的旧0步汇总已于2026-10-19按保存checkpoint／完整留出和中断记录修正，原测量数值未变，详见专题说明。

2026-10-14新增[256维×4层实体注意力模型预设及有限正确性验证](../artifacts/entity_attention_256x4_20261014/README.md)：8个注意力头、FFN1024，主网络3,628,034参数；带既有双辅助头为3,761,156。工厂、AZ／纯权重／冻结加载按完整尺寸工作，历史128×2默认及原权重身份保留。22项测试、真实样本训练恢复、CUDA编译推理与S8原生搜索冒烟通过。同日该模型从零训练的中期面板（step 34,021＝50%预算）对原R43的1000pair greedy为**45.45% [43.32%,47.60%]**（2000局，0异常），未超过现有主线；训练终点与搜索面板尚未完成，见[早期面板](../artifacts/entity_attention_256x4_aux_r24_r43_early34021_20261014/results.md)。

近期结果与归档核对：2026-10-19；早期核验日期：2026-10-09；P4九轮完成结果与导航补充：2026-10-10；第9轮终点长期在线入口补充：2026-10-11；第43轮状态与第24轮历史池离线流程补充：2026-10-12。本文是导航快照，详细事实以当前实现、冻结配置和可复核原始产物为准。完成模型训练、完成评测、确认收益是三个不同状态；历史日期和测试数量不代表当前工作树重新验收。

2026-10-19完成[7个离线run冷产物集中](../artifacts/repository_archive_20261019/README.md)：334个中间checkpoint及27个账本／逐步指标，共7.0324GiB移入`archive/20261019/`；24个关键模型及两份共72000局池留在原位，旧路径相对链接可读。20261012旧归档已由用户移到其他位置，本机未挂载，具体外置位置未提供；[旧清单与45来源结论](../artifacts/repository_archive_20261012/README.md)保留，721个旧链接不可读属已外置状态，不作为本次新增丢失。

2026-10-13的[R43起点S4096/N1、policy2k/value6k分支](../artifacts/entity_attention_r43_s4096_p2k_v6k_20261013/README.md)已完成R44–R47共4000局／4000步，8小时会话到时停止；summary的in_progress表示无固定轮数终点，不表示进程仍在运行。后续R47起点恢复policy6k的配置仅准备，未启动；没有独立隔离采集预算和缩池贡献。

## 当前研究结果

2026-10-13完成[R24–43固定数据离线及1000pair greedy扩样](../artifacts/entity_attention_r24_r43_offline_objectives_20261013/results.md)：两组原R43起点、同20000局／同抽样各22681步。final对原R43的PSW得分51.350% [49.225%,53.450%]；双辅助53.900% [51.675%,56.075%]。辅助组对R43有提升证据，PSW未确认；辅助减PSW的共同对手得分差+2.550pp [−0.050,+5.200]pp，未确认候选之间的差异。前200pair保留，各新增800pair；3999局自然终局、1局policy cycle draw、零工程失败。该次未含普通同预算训练对照，后续[10/19消融](../artifacts/entity_attention_r24_r43_aux_ablation_20261019/results.md)已补普通对照，未确认辅助头额外收益；原双辅助和PSW仍无搜索棋力结论。

2026-10-13完成[R43固定4k S4096的Policy Surprise Weighting对照](../artifacts/entity_attention_r43_policy_surprise_s4096_20261013/results.md)：原R43继承AdamW／RNG，两个头同池、两分支同抽样，各1000步；不加权／50%局内加权的普通留出KL为0.568885／0.588634，加权留出KL为0.915460／0.898535。说明加权改变了拟合重点，未确认总体或棋力收益：greedy对P4V4为80.875%／81.625%，配对差+0.750pp [−3.875,+5.375]。配置是可选开关，原自迭代配置未自动启用；该次实验执行时尚未实现两个value辅助头。

较早已完成在线方案见[P4九轮完整报告](reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)：新增在线方案在已测S1024/N1面板胜过起点、同预算旧池对照及P12＋V4；旧池拟合退化与棋力提升并存，value改善不均衡，单训练seed和固定DEV尚不支持普遍最优结论。[M4 与固定数据诊断完整结论](reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md)限定较早分支：稳定fresh净泛化与持续棋力收益未确认；最新数据优先、删除历史、较小学习率仍非独立确认的最佳日程。早期5000→9500混合run有累计留出拟合与对θ5000的棋力收益，见[9代矩阵及基线复核](../artifacts/entity_attention_hybrid_kl_trend_20261009/README.md)，各阶段结论不能混用。

| 实验／分支 | 已完成与限制 | 依据 |
| --- | --- | --- |
| R43历史池离线与前置直接对战（2026-10-19核对完成） | R24离线候选对R43的200pair为51.250% [46.125%,56.375%]，未确认超R43。R43同43000局池48776步、5组greedy／4组搜索DEV及独立500pair均完成；DEV选＋24388步，独立确认50.50% [47.35%,53.70%]，未确认提升。完整留出KL从0.474121降至0.430380，MSE先降后升至0.758653；拟合改善不能替代独立棋力确认 | [流程](../artifacts/entity_attention_r43_history_pool_20261012/README.md)、[完成结果](../artifacts/entity_attention_r43_history_pool_20261012/results.md)、[前置结果](../artifacts/entity_attention_r24_offline_vs_r43_20261012/results.md) |
| 第24轮历史池离线实验（2026-10-12） | 43000局池、Train38583局/3121619局面、原留出4417局；lr2e-5、B256追加48776步及10组面板共2300pair全部完成。＋24388步对原R24独立500pair得分54.00% [50.80%,57.10%]，支持小幅收益；完整留出KL持续下降、value MSE先降后升，尚未确认超过R43 | [数据、流程和命令](../artifacts/entity_attention_r24_history_pool_20261012/README.md)、[完成结果](../artifacts/entity_attention_r24_history_pool_20261012/results.md) |
| 第9轮终点长期在线续训（2026-10-11） | 已完成绝对第43轮（相对第9轮续跑34轮），不再是“正式未启动”。每轮1000局／1000步、lr2e-5、6000局窗口，教师无条件升级。第43轮greedy对P4得分82.25%，第42轮S1024/N1对P4为65.75% [61.375%,70.125%]；第30轮以后尚未确认新的棋力提升。单训练seed与复用DEV，不能断言完全停止学习 | [当前原始结果](../artifacts/entity_attention_r9_online_lr2e5_1000_forever_20261011/results.md)、[流程与命令](guides/entity_attention_p4_pool_forever_plan_20261011_zh-CN.md) |
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
| 旧θ5的10×1000源数据 | 轮0–9、10000局／12050更新及20个常规面板完成；770870训练／80699留出位置用于早期attention离线学习。终点对旧θ5搜索200pair为46.625% [41.875%,51.250%]，未确认提升；终点对PPO仅完成100/200pair。此次核对保存结果，未新跑对战 | [完成报告](reports/az_theta5_10x1000_zh-CN.md)、[数据统计](../artifacts/entity_attention_v1_20261004/data_summary.json)、[历史准备](archive/next_az_training_zh-CN.md) |
| 实体注意力离线 | 532738参数、12045步及六个PPO面板完成；final搜索59.5%、greedy61.75%，重复DEV | [指南](guides/entity_attention_offline_zh-CN.md) |
| 新模型10×1000、混合训练 | 10×1000曾运行，已有轮0–2完成记录；混合run完成9轮／9500步；9代各2048留出局面，相对θ5000均改善，相对各代训练前为8降1升，本轮500步为5降4升；N1到14500对原九代均比各自训练前更差，当前窗口拟合未由此验证；离线重放9500对θ5000为67.38%，hybrid第3/6/9轮为58.88/66.75/74.38%，第6轮与离线9500不可区分、第9轮高+7.00pp[+0.62,+13.25]；不同分支不能互称完成 | [自迭代](guides/entity_attention_self_iteration_zh-CN.md)、[混合](guides/entity_attention_hybrid_zh-CN.md)、[KL趋势](../artifacts/entity_attention_hybrid_kl_trend_20261009/README.md)、[离线9500对照](../artifacts/entity_attention_hybrid_offline9500_20261009/README.md) |
| 第九轮N2与N1 fork | N2至少33新轮／12800步记录，曾OOM；N1完成50新轮／14500步后停止；原9／50轮共同PPO为63.75／61.75%，差值区间跨0 | [N2](guides/entity_attention_hybrid_fork_zh-CN.md)、[N1](guides/entity_attention_hybrid_s1024_n1_zh-CN.md)、[补测](../artifacts/entity_attention_strong_anchor_20261006/README.md) |
| 分头窗口、value detach | 20新轮／16500步完成；policy近4轮、value3000局近期50%；长期净收益未确认 | [指南](guides/entity_attention_split_training_zh-CN.md) |
| 搜索教师比较 | 目标稳定性不等于棋力；同S1024随机节点31.25%、CPW37%均弱于旧native；正式教师未切换 | [系列报告](reports/entity_attention_search_teachers_zh-CN.md) |
| 性能与进程管理 | 多worker、CUDA Graph、缓存与退出修复有有限验证；各报告限定形状／机器／版本，未确定全局最快配置 | [性能验收](reports/end_to_end_speed_optimization_zh-CN.md)、[新模型执行](reports/entity_attention_execution_speed_zh-CN.md)、[OOM修复](../artifacts/entity_attention_worker_cleanup_20261005/README.md) |

所有文档与配置见 [导航](index.md)。状态分类及原件身份见 [实验索引](../artifacts/experiment_index.json)；本次路径和结论维护见 [整理记录](../artifacts/repository_cleanup_20261009/README.md)。
