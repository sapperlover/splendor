# 当前项目状态

2026-10-29按用户更正，新增[固定R37初值104k／lr2e-5的140221步对原R37两组10k评测](../artifacts/r37_lr2e5_step140221_vs_r37_10k_20261029/README.md)。该节点完整留出主KL＋MSE为0.9286851476，在当前13个完整节点中最低；训练仍中断于228000步，本任务不续训、不取最新节点。准备串行greedy10000pair→双方S128N1（总128）10000pair；15项回归和CPU根验证通过，仅给命令，正式评测未启动。

2026-10-29完成[R37权重起点、104k池、lr2e-5当前最佳点的独立greedy](../artifacts/entity_attention_r37_104k_lr2e5_best_vs_r37_greedy_20261029/results.md)：固定140221步，对raw R37共1000 pair／2000局，1015胜／14和／971负，得分**51.10%**，95%整pair区间**[49.000%,53.175%]**，未确认提升。CPU8、零搜索、工程错误0；训练记录中断于228000步，本次未恢复或测搜索。

2026-10-27新增[R37初值的两组104k无辅助离线实验](../artifacts/entity_attention_r37_104k_offline_lr_20261027/README.md)：lr1e-4及2e-5，仅加载原R37权重，使用新AdamW和原离线RNG；原数据／split、18GiB／4共享worker、12次曝光及5%完整主KL＋MSE早停选点保持。两组真实CPU有限更新／恢复及实际冻结CLI已验证，仅准备命令，未启动正式训练／评测，不接现有队列。

2026-10-27[五个104k离线best对raw R37十组评测](../artifacts/offline5_vs_r37_10k_20261027/README.md)全部完成，共100000 pair／200000局。固定原选点，无辅助122694步、其余140221步；各模型greedy与双方S128N1（总128）各10000 pair。Greedy辅助组点估计略高；S128N1集中于56.29%–57.10%，未明显拉开。六组原训练、选点和原S1024N1评测也均已完成，见[指标与结果汇总](reports/auxiliary_offline_results_20261027_zh-CN.md)。下方按日期保留准备、停止、等待及发布过程，不作为当前进度。

2026-10-27完成核对：[原104k无辅助实验](../artifacts/entity_attention_128x2_noaux_r21_r37_20261026/README.md)在policy＋得分之后串行完成，210332步早停、选中122694步，主KL0.212048／MSE0.736596；原greedy1000 pair得51.075%，原S1024N1搜索500 pair得57.50%。后继队列execution为complete。

2026-10-27完成核对：[104k policy＋得分辅助实验](../artifacts/entity_attention_128x2_policy_score_r21_r37_20261026/README.md)192804步早停、选中140221步，主KL0.205600／MSE0.734649；原greedy／S1024N1为53.475%／56.80%。原池、初始化、优化器、18GiB／4 worker和辅助0.15／0.25设置按冻结配置。

2026-10-26实现[到终局8路分数归属辅助](guides/score_ownership_zh-CN.md)：self/opponent×一级／二级／三级／贵族的未来新增分；支付前后及终局尾步按决策前状态精确分配。复用continuation模型／cache／4 worker／state-only训练及主指标选点，新增分路误差和编码器梯度；旧配置身份保持。独立仅得分辅助的104k预设已准备，验证边界见[工程记录](../artifacts/score_ownership_20261026/README.md)，policy＋得分组合的正式训练和评测已完成，结果见文首；仅得分辅助预设仍只完成准备与工程验证。

2026-10-25已完成[四组离线普通模型／逐步日志和原80k局池归档](../artifacts/repository_archive_20261025/README.md)：672个普通checkpoint、4份主日志、80000个NPZ，包8.84GiB，净释放11.06GiB。最优／完整留出／initial/final模型和当前在线输入保留；原80k离线复训或完整复核须先恢复本包。下方较早“原80k留本机”是当时记录，当前位置以本批清单为准。

2026-10-25已运行[policy辅助最优离线模型的60k在线分支](../artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/README.md)，未接入现有离线队列。起点140221步，完整留出主KL+MSE0.9413569281，原权重／AdamW／全部RNG继承；继续对手policy辅助0.15、短期value关闭，lr2e-5／每轮3000步／10000新局。原R31–R37完整64k历史和原split保留；首采后74k按原seed随机淘汰14k历史至60k，后续随机原始历史优先／在线FIFO。采集12worker／active2000／B500／mb128，评测A active1000／精简桶／owner双槽；用户确认固定raw R37，每轮greedy1000pair、每3本地轮搜索500pair。每六轮完整前六轮留出，首段R32–R37的6092局；原2048轮内面板与500步周期点保留。当前训练器流式统计、state-only传输／批量回读和轻量SHA记录已冻结；在线16GiB LRU，不套用离线固定池4worker。29项相关回归、实际--check及有限真实CPU模型验证见专题；state已提交13轮；第12轮greedy66.075%、S1024N1搜索已完成500 pair得60.20%，第13轮greedy63.75%。第8轮后短期平台迹象较强，不能归因为policy改制收益全部耗尽；见[配对统计](../artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/plateau_review_20261025/results.md)。当前入口已切换[训练结束释放LRU／采样引用并归还CPU堆页](../artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/cache_release_20261025/README.md)的新冻结发布；下一轮需冷加载。本次未恢复正式运行。

2026-10-24发布记录：按用户要求，104k双辅助及44k仅value已另建执行发布并接入原脚本，本次均18GiB、4 worker、流式累计及传输／同步优化；当前policy仍用上一优化发布。后三项已设为以后新建离线训练的默认，缓存按池大小配置。183项有限回归和两组真实CUDA三次更新逐位对照通过，原计划／replay identity／固定留出保持，749个受保护原件哈希保持。本次未启动正式训练，当前policy恢复点115000步。[发布与命令](../artifacts/offline_training_defaults_20261024/README.md)，[通用默认](guides/offline_execution_zh-CN.md)。下方后两组未接入等为此前时点记录。

2026-10-24发布记录：按用户要求停止当前104k仅policy训练及串行队列，最后提交115000步。已准备流式会话统计、state-only value／辅助传输及指标批量回读的独立发布，87项回归及真实CUDA逐位一致性验证通过；仅给恢复命令，未重启。18GiB／4 worker保持，短测速度改善很小，不能据此声称全池明显提速。[改动、原件及命令](../artifacts/training_streaming_20261024/README.md)。下方11:42等运行状态为此前时点记录。

2026-10-24 11:42恢复核验：当前仅policy组已从102000步继续并提交104000步checkpoint，4 worker读取同一份只读共享池，最近605步miss为0。训练稳态约22.49步/秒，原100000→102000步约3.37步/秒；组合提速约6.68倍，不含一次性预热和验证。共享区16.64GiB，进程树PSS约20.90GiB，系统可用约3.35GiB。[原始核验与适用范围](../artifacts/training_prefetch_20261024/README.md#正式恢复核验2026-10-24-1142)。

2026-10-24后续执行更新：104k仅value辅助已完成并交接到仅policy组。按用户要求暂停后，当前仅policy组改用18GiB缓存和4个共享只读CPU worker，从102000步恢复；独立验证缓存不驱逐训练数据。模型／AdamW／采样RNG在真实checkpoint短测中逐位一致，63项回归通过，热缓存小数据测速约1.16倍，完整池速度以运行记录为准。[执行发布、配置与核验](../artifacts/training_prefetch_20261024/README.md)。原队列及后续两组保持，下面132000步／16GiB／等待首组等描述为此前时点记录。

2026-10-24，用户自行恢复的R21–R37仅value辅助实验在本任务首次核验时正在训练，最近提交132000步；下文2026-10-23“已停止／0步”是当时记录。新增[串行后续三组](../artifacts/entity_attention_auxiliary_queue_20261024/README.md)：104k仅policy辅助0.15 → 104k双辅助0.25／0.15 → 原R21–R31的44k仅value辅助0.25。全部从零、新AdamW、原seed／lr1e-4／16GiB／两头B256／12次曝光／5%完整主KL+MSE早停选点；各自完成对R37 greedy1000pair和搜索A500pair才启动下一组。90项有限回归与3次真实CPU更新／9个S8N1原生根（总72模拟）通过。后续等待原监督chain.lock并严格检查整条流程成功，失败／中断停止；本次不修改当前进程或冻结实现。实际挂起身份见队列launch。

2026-10-23首次准备／停止记录：[attention128仅短期value的R21–R37离线实验](../artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023/README.md)，80项回归和真实GPU短验证通过后曾启动；用户随后更正为仅给命令，现已SIGINT停止该队列，保留initial.pt（已提交0步），未重启。104000原局／原split，Train7478428位置／Validation835367位置；从零／新AdamW、lr1e-4、两头B256、16GiB，H2/H6/H14权重0.25、对手policy辅助关闭。12次value曝光上限350552步，每5%完整主KL+MSE连续三次严格上升早停并按最低点固定整个模型，随后只对raw R37 greedy1000pair与配置A搜索500pair。采用当前轻量参考SHA记录；支持逐轮4000／10000局数映射，旧整数格式兼容。当时启动与停止身份见专题launch／verification；该组后续已完成训练、原对战及本次大面板，当前结果见文首。

2026-10-23实现[短期value与对手policy辅助](guides/continuation_auxiliary_zh-CN.md)：KataGo式2／6／14玩家回合指数标签；对手下一主行动4类软CE及购买／保留各自的牌分布，可见牌槽＋不可见类。离线入口`offline_continuation`接通缓存、训练、验证、精确恢复与主KL＋MSE选点；当前在线driver未启用。已有[有限CPU/native工程验证](../artifacts/continuation_auxiliary_20261023/README.md)，未启动正式训练或棋力赛。单候选根无value会使该回合及之前的短期标签屏蔽，日志显式记录覆盖率；旧进展目标仍关闭。

2026-10-23按用户要求，[训练记录与checkpoint策略](guides/training_artifacts_zh-CN.md)已调整：当前在线／近期离线默认停写两份`.bin`、两份`.steps.jsonl`及恢复时的整份指标副本，逐步KL／MSE／梯度等仍在主日志。R31在线改为每500步保存＋0／100／250／500／1000／2000／3000七个诊断点，正常每轮10个模型文件、9个不同节点；新启动快照为`source_recording_checkpoint_20261023`。旧产物、原始run配置及state不回写，不自动续训；[交付与验证](../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/recording_checkpoint_20261023/README.md)。

2026-10-23已完成[五类冷产物T7迁移](../artifacts/repository_archive_20261023/README.md)：223个离线普通checkpoint、1011个抽样账本／逐步日志、76038个旧NPZ，流式包5.8104GB，完整SHA／纳秒mtime校验后净释放13.9211GB。原80k局、父分支46142局、当前60k局及所有未选模型保留；旧局池复训／完整诊断需先恢复归档。本次未启动训练或棋力评测。

2026-10-23按用户更正，在线留出已改为[每六轮固定使用此前六轮](guides/evaluation_zh-CN.md#在线全量固定留出)：R32–R37使用R26–R31；R38–R43改用R32–R37；R44–R49再用R38–R43，依次推进。第二段6092局／101425 policy／485932 value已冻结；原第一段2412局、42节点及累计MSE下降0.006722的结果保留。新启动脚本、配置、恢复补齐已接入，训练不因诊断改变；[验证与产物](../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/six_round_holdout_20261023/README.md)。R20–R31补评因冷checkpoint外置，用户已暂缓，本次未运行；当前无训练进程，未续训。

2026-10-23的[raw R31起点10k／60k池分支](../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/README.md)重置后已由用户启动，原始state已完成6轮至谱系R37，第7轮已建立采集配置。R37对固定原R43的greedy1000 pair为68.90%，搜索S1024N1（总1024）500 pair为74.55% [71.75%,77.25%]。原R31累计68423步及完整AdamW/RNG、R22–R31的40k历史起点不变；每轮10000新局／3000步／lr2e-5，首轮50k、第二轮60k，历史policy固定25%子集及随机历史优先／在线FIFO保留。采集继续12worker／active2000／B500／microbatch128／owner双槽和16GiB compact缓存。

按2026-10-23最新要求，[日常及在线新搜索评测统一配置A](guides/evaluation_zh-CN.md)：12 search worker、1000活跃局、B500、local128、owner双槽、B桶[64,128,256,500]、C桶[32,48,64]、等待5ms、shared slots2、FP32/native-cudagraph；默认500 pair、双方S1024N1总1024、无根噪声／温度0。日常greedy与每轮在线greedy保持CPU8 worker／1000 pair，在线搜索节奏仍每3个本地轮次。源配置在新轮界重读后供新面板使用，已有冻结面板与原始结果保持历史身份；本次不启动或重启实验。采集、日常及在线两类评测的配置／种子／命令集中于上述指南。

三组离线候选已另对同R43完成500 pair搜索：[80k/94695为74.05%，44k/59453为70.70%，44k低lr累计63453为72.20%](../artifacts/eval_speed_20261023/README.md)。与R37是共同对手比较，80k与R37得分接近，尚未确认谁更强；不能由此隔离扩池、学习率或12贵族适应收益。三组核心协议／SHA及实际预算一致，错误计数为0；新默认A是已验证可用配置，最优吞吐及提速因果尚未建立。下方历史“未测搜索”指当时或原R20／R31面板，当前对R43的独立补测以本段为准。

2026-10-23用户进一步撤回[全部新增购买结算特征](../artifacts/entity_attention_first_round_20261022/README.md)，包括其中终局标记。已删除购买模块、actor query分支、配置字段及对应首轮候选入口／配置／启动脚本，原关系attention128恢复583682参数；短期进展辅助保持关闭，原有market_relations及value读取不变。旧组合／纯购买候选都未正式训练，准备目录、CPU／CUDA短验证原件及撤回前源码快照保留；历史实验权重须用对应旧快照解释，当前不增加兼容忽略或自动权重转换。未新增终局设计、未启动训练／采集／棋力赛；当前有限验证见撤回记录。

2026-10-22的[原59453步attention128降lr续训](../artifacts/entity_attention_128x2_r21_r31_lr2e5_20261022/README.md)已正式运行并早停：继承模型、AdamW和全部RNG，lr2e-5，同44k原标签／split、两头B256及16GiB，每2000新增步评估同一4096子集与完整349435留出。新增4000步联合指标最低0.976699（KL0.217213、MSE0.759487，累计63453步）；6000／8000／10000步连续上升，于10000步（累计69453）正常结束并固定整个4000步checkpoint，未跑满新增148632步上限。相对父起点value MSE下降0.008527，尚不能换算棋力；原训练入口无对战，用户后续追加的R31 greedy1000pair采用同原面板开局单独执行，配置、原始结果及核验见专题。父run与已完成结果只读保留。

上述追加greedy已完成：992胜／22和／986负，50.150% [48.025%,52.300%]；相对父模型同R31／开局的49.525%，配对差+0.625pp [-2.051,+3.326]pp，未确认降lr带来greedy提升或超过R31。2000局异常／非法行动／工程截断／不变量失败均0，模型／规则／交换先后手／零搜索及整pair区间已核验；复用已查看面板，单训练seed，未测搜索或value棋力。独立结果见专题greedy报告。

2026-10-22后续用户指令已停止[第21–31轮attention128离线训练](../artifacts/entity_attention_128x2_offline_r21_r31_20261022/README.md)：SIGINT退出原监督／训练进程，保留84000步恢复点，原148632步预算未完成、未触发自动早停。11个已完成完整留出节点内最低联合指标为59453步（KL0.227124、MSE0.768013、合计0.995137），对战前固定整个checkpoint。对raw R20／R31各1000pair greedy均已完成：53.400% [51.250%,55.550%]／49.525% [47.375%,51.601%]，支持候选greedy超R20，未确认超R31；4000局异常／非法动作／工程截断／不变量失败均0，模型／规则／交换先后手／实际零搜索／整pair区间已核验。两组搜索按最新指令取消，未测value或搜索棋力；训练不恢复。范围变更、仅greedy命令与原始结果见专题；下段原完整队列是启动时历史记录。

2026-10-22已后台启动[第21–31轮固定44000局attention128离线早停实验](../artifacts/entity_attention_128x2_offline_r21_r31_20261022/README.md)：全新模型／AdamW，lr1e-4，保留旧attention离线种子、B256两头独立抽样、16GiB和12次value曝光上限148632步。原整局划分为39629训练局／4371留出局，完整留出349435位置；每5%KL+MSE连续三次严格上升即停止，持平／下降重置，按已到达节点最低联合指标固定候选。随后先对raw R20／R31各1000pair greedy，再各500pair双方S1024N1总1024搜索，12worker／B500／owner单槽。72项回归及真实GPU20步／8局短验证通过，一次启动核验阶段为train，无持续轮询；阶段与启动身份以实验launch、runtime/live和runs/progress为准。旧8万局与在线分支均不恢复、不改配置，尚无本实验棋力结论。

2026-10-22完成[第21–30轮及Cross64离线冷产物T7归档](../artifacts/repository_archive_20261022/README.md)：291个模型及43份唯一账本外置到`archive-20261022.tar.zst`（2.4473GB），12个重复账本改为相对别名，逐文件SHA／大小／纳秒mtime及压缩包SHA通过后净释放4.3250GB。按用户追加指令，第21–31轮全部44000个采集NPZ（包括淘汰训练局）、第32轮2142个已采NPZ及原80k局全留本机，逐文件身份释放前后相同；各轮final、离线初始及6个百分比节点、活动24k池和最新R31模型不动。在线归档核对时没有运行进程，未启动续跑；原启动段落是历史记录。在线恢复不依赖本包，旧轮中间模型／逐步分析及离线续训账本需按专题恢复。

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
| 本地人机网页（2026-10-29） | 支持固定策略、greedy、固定深度搜索及原生 MCTS；预算可开局／局中设置，CPU单线程；有模型value评估条与悔棋。新启动的服务在每次操作和AI回应完成后自动保存完整回放至 `runtime/web_replays/`，悔棋同步当前分支；支持 `--replay-dir`。旧内存服务可用专用脚本无重启导出。policy在线第9轮S1024N4命令见指南 | [评测与网页指南](guides/evaluation_zh-CN.md#本地人机网页) |
| PPO seed1、AZ seed1 | PPO 5M本轮decisions；AZ5×500及θ1–θ5扩展面板完成，DEV best不等于独立FINAL best | [PPO](reports/ppo_5m_zh-CN.md)、[AZ](reports/alphazero_seed1_zh-CN.md) |
| 旧θ5的10×1000源数据 | 轮0–9、10000局／12050更新及20个常规面板完成；770870训练／80699留出位置用于早期attention离线学习。终点对旧θ5搜索200pair为46.625% [41.875%,51.250%]，未确认提升；终点对PPO仅完成100/200pair。此次核对保存结果，未新跑对战 | [完成报告](reports/az_theta5_10x1000_zh-CN.md)、[数据统计](../artifacts/entity_attention_v1_20261004/data_summary.json)、[历史准备](archive/next_az_training_zh-CN.md) |
| 实体注意力离线 | 532738参数、12045步及六个PPO面板完成；final搜索59.5%、greedy61.75%，重复DEV | [指南](guides/entity_attention_offline_zh-CN.md) |
| 新模型10×1000、混合训练 | 10×1000曾运行，已有轮0–2完成记录；混合run完成9轮／9500步；9代各2048留出局面，相对θ5000均改善，相对各代训练前为8降1升，本轮500步为5降4升；N1到14500对原九代均比各自训练前更差，当前窗口拟合未由此验证；离线重放9500对θ5000为67.38%，hybrid第3/6/9轮为58.88/66.75/74.38%，第6轮与离线9500不可区分、第9轮高+7.00pp[+0.62,+13.25]；不同分支不能互称完成 | [自迭代](guides/entity_attention_self_iteration_zh-CN.md)、[混合](guides/entity_attention_hybrid_zh-CN.md)、[KL趋势](../artifacts/entity_attention_hybrid_kl_trend_20261009/README.md)、[离线9500对照](../artifacts/entity_attention_hybrid_offline9500_20261009/README.md) |
| 第九轮N2与N1 fork | N2至少33新轮／12800步记录，曾OOM；N1完成50新轮／14500步后停止；原9／50轮共同PPO为63.75／61.75%，差值区间跨0 | [N2](guides/entity_attention_hybrid_fork_zh-CN.md)、[N1](guides/entity_attention_hybrid_s1024_n1_zh-CN.md)、[补测](../artifacts/entity_attention_strong_anchor_20261006/README.md) |
| 分头窗口、value detach | 20新轮／16500步完成；policy近4轮、value3000局近期50%；长期净收益未确认 | [指南](guides/entity_attention_split_training_zh-CN.md) |
| 搜索教师比较 | 目标稳定性不等于棋力；同S1024随机节点31.25%、CPW37%均弱于旧native；正式教师未切换 | [系列报告](reports/entity_attention_search_teachers_zh-CN.md) |
| 性能与进程管理 | 多worker、CUDA Graph、缓存与退出修复有有限验证；各报告限定形状／机器／版本，未确定全局最快配置 | [性能验收](reports/end_to_end_speed_optimization_zh-CN.md)、[新模型执行](reports/entity_attention_execution_speed_zh-CN.md)、[OOM修复](../artifacts/entity_attention_worker_cleanup_20261005/README.md) |

所有文档与配置见 [导航](index.md)。状态分类及原件身份见 [实验索引](../artifacts/experiment_index.json)；本次路径和结论维护见 [整理记录](../artifacts/repository_cleanup_20261009/README.md)。
