# 文档导航

2026-10-08更新[连续步数、学习率与policy窗口趋势实验](guides/entity_attention_iteration_suite_zh-CN.md)：按B→C→D顺序运行三组各seed1×16轮，2/4/8/12/16轮对冻结起点各100pair。取消A/PPO/0轮自测、专用完整性审计和磁盘门槛，保留16GiB内存上限与断点恢复。已完成B前两轮与首次评测51.25%（44.5%–58.0%），未确认净收益；C/D未运行。旧四组方案仅保留为历史记录。

同日新增[只搜后期／只搜前中期对greedy](../artifacts/entity_attention_stage_search_20261008/README.md)：冻结16500步模型，最高公开分≥10切换，两组各500pair，独立串行/恢复入口，active200＋CUDA Graph且不设RAM硬上限；两组500pair已完成：后期67.0%、前中期69.35%，差值区间跨0；新增全程搜索500pair为81.8% [79.3%,84.3%]，两项配对差值均大于0；本次耗时仅比前中期组多5.1%。

2026-10-06更新：[实体注意力离线训练](guides/entity_attention_offline_zh-CN.md)已完成12045步及六个PPO面板；[原第9轮/第50轮共同PPO补测](../artifacts/entity_attention_strong_anchor_20261006/README.md)完成400局；[固定数据policy/value消融](../artifacts/entity_attention_policy_value_ablation_20261006/README.md)完成17面板2700局，联合分支搜索退化可通过替换完整value恢复。[分头窗口与detach value](guides/entity_attention_split_training_zh-CN.md)已实现并通过有限CPU/GPU验证，正式长训未启动；取代此前[最新单轮份额阶梯候选](guides/entity_attention_fresh_share_zh-CN.md)，历史设计保留。[旧θ5起点10×1000配置与命令](guides/next_az_training_zh-CN.md)、[记录字段/分母清单](guides/next_az_record_fields_zh-CN.md)保留准备时记录，该十轮源run已完成。

docs 只保留需要阅读和维护的规范、操作指南、可复核报告。入口为 [当前状态](current_project_state_zh-CN.md)，实验分类与依据见 [实验索引](../artifacts/experiment_index.json)。文件中的实验日期与适用版本独立于本次整理日期。

2026-10-07运行状态更新：分头训练已完成20新轮/16500步，替代上文2026-10-06交付时的“正式长训未启动”。[冻结模型A/B/C/D目标噪声实验](../artifacts/entity_attention_target_noise_20261007/README.md)完成6400次搜索；随后[随机节点后端同预算复测](../artifacts/entity_attention_chance_search_20261007/README.md)又完成6400次，C组JS下降90.8%、D下降63.8%，耗时1.52倍、搜索更浅。新增[同模型100 pair棋力对照](../artifacts/entity_attention_chance_strength_20261007/README.md)：新搜索31.25%，95%区间24.5%–38.5%，200局有效，当前S1024实现明显弱于旧确定化；正式训练教师仍用旧native，长期训练影响未验证。本系列整合报告见[随机节点／CPW搜索与冻结教师短训](reports/entity_attention_search_teachers_zh-CN.md)。

同日新增[Chance progressive widening有限诊断](../artifacts/entity_attention_chance_pw_20261007/README.md)，相关45项回归与六局checked smoke通过；随后完成[k0.25/α0.5的6400次稳定性复测](../artifacts/entity_attention_chance_pw_noise_20261007/README.md)，D组深度2.560→3.401、JS0.03072→0.04835，保留部分相对旧确定化的稳定性收益。[同模型100 pair棋力](../artifacts/entity_attention_chance_pw_strength_20261007/README.md)为37.0%（95%区间30.75%–43.25%），仍弱于旧确定化；相对不扩宽31.25%的配对改善区间跨0。当次棋力实验未训练；后续冻结采集＋短训见下方新报告，正式教师未切换。用户取消了[大S初筛后续](../artifacts/entity_attention_chance_depth_20261007/README.md)，原始已完成结果保留。

2026-10-07补充与复核：[多搜索平均标签](../artifacts/entity_attention_target_averaging_20261007/README.md)保持每份S1024/N1，将标签总预算增加至S4096；相同抽样和优化配置不意味着相同更新轨迹或value函数。修订按77个来源对局聚类后，100/300/500步对独立经验均值的KL优势仍成立；不同学生权重、相同旧搜索直接对打60.5%（119胜、4和、77负，100 pair，区间54.25%–66.75%）。对起点47%（区间37%–56.5%，跨50%）、greedy49%均未确认净提升。本次额外四次重标注加原采集，总模拟量约5倍；正式平均标签入口未实现。[报告第九节](reports/entity_attention_search_teachers_zh-CN.md#九如何改善当前自迭代先改变标签再判断剩余限制)给出保持现有窗口和小步更新、先引入平均标签再辨别value漂移的候选方案，不追加fresh-M1对照。

2026-10-07剂量复核：[200局短训的步数/学习率/value/标签平均对照](../artifacts/entity_attention_step_dose_20261007/README.md)在同一batch账本、同一起点上发现棋力峰值在**50步附近**（avg4 56.5%、single 58.0%），100步（47%/45%）已明显回落；1e-4学习率偏大，lr2.5e-5@100为54.0%（+7pp [−5.5,+20]）；冻结critic head与detach无差异（48%对47%）；拟合最优点（avg4 100–200步、低LR）与棋力最优点（50步）不一致。**200局/100步/lr1e-4恰好落在本批数据的最差区间，降步数或降学习率不增加搜索成本，应先于标签平均的×4成本**；单轮最好节点仍只略高于50%，自迭代启动仍未证明。报告[第十节](reports/entity_attention_search_teachers_zh-CN.md#十剂量复核步数学习率value标签平均2026-10-07-补充)已同步。

2026-10-08正式回放复核：[学习率×步数短训探针](../artifacts/entity_attention_formal_lr_step_20261007/README.md)在同一16500起点逐条重放round-019正式账本（policy近4轮、value3000局近期半、detach/batch/裁剪不变），三条轨迹只改lr 2.5e-5/5e-5/1e-4，节点25/50/100步，并用同口径第二固定账本把边界节点延伸到150步；从round-018重放与正式round-019**逐张量一致**，起点诊断与正式validation一致。2.5e-5的recent留出KL随步数降到0.40525@150（起点0.42109、1e-4@100=0.41977），但1e-4@100净策略变化近0（KL0.00071、top1 2.3%）、125/150又变大且bank KL变差；三档lr的policy梯度99–100%步被裁剪。50-pair面板无显著差异，DEV最高节点2.5e-5@150=60.0%在独立HEAD种子上41.5%（对照46.5%，配对差−5.0pp[−20,+10]），未复现。**本轮未确认优于现行1e-4/100步/200局，正式配置未改**；低学习率长期效果需执行级lr覆盖的多轮自迭代分支，未实现未运行。

## 规则与现行契约

- [标准规则与规范数据](splendor_standard_rules_zh-CN.md)：唯一玩法依据，包含双人本地 forced-skip 增补。
- [引擎、Observation 与阶段机](specs/engine_contract_zh-CN.md)：公开边界、私有状态、replay 和 native 差异。
- [数值特征 schema](specs/feature_schema_zh-CN.md)：维度、mask、缩放、隐藏信息和候选映射。
- [Actor–Critic、学习环境与 PPO](specs/learning_contract_zh-CN.md)：网络、transition、GAE、更新与 checkpoint。
- [PPO 训练、评测与历史池协议](specs/ppo_training_protocol_zh-CN.md)：5C run-format v4、种子域、DEV/promotion/FINAL。

## 操作指南

- [冻结网络搜索](guides/search_zh-CN.md)：独立确定化／随机节点信息历史PUCT与固定深度算法/API。
- [自对弈数据与恢复](guides/selfplay_data_zh-CN.md)：v1/v2 数值归档、compact trace、checked 重放。
- [AlphaZero 循环](guides/alphazero_zh-CN.md)：replay、损失、跨轮状态、恢复和命令。
- [实体注意力与离线学习](guides/entity_attention_offline_zh-CN.md)：模型架构、全量所选 train 抽样、验证/恢复和运行命令。
- [新模型10×1000自迭代](guides/entity_attention_self_iteration_zh-CN.md)：1000局同时调度，固定起点双面板，正式命令与有限CPU/GPU验证。
- [从5000步接入连续自迭代](guides/entity_attention_hybrid_zh-CN.md)：状态连续、两池日程、固定地址Graph；原run已完成9轮/9500步与有效棋力面板。
- [第九轮N2分支与逐步数据替换](guides/entity_attention_hybrid_fork_zh-CN.md)：S512/N2、S1024/N2各100局/100步，继承优化器/RNG，20新轮替换在线N1、61新轮起纯在线；2026-10-06本次检查S512/N2至少已完成33新轮/12800步；此前RAM OOM的孤儿清理与退出修复见[验证记录](../artifacts/entity_attention_worker_cleanup_20261005/README.md)。
- [从原第九轮恢复S1024/N1小步更新](guides/entity_attention_hybrid_s1024_n1_zh-CN.md)：完成50新轮/14500步后用户停止；原第9轮/第50轮对PPO补测63.75%/61.75%，后续增益与退化均未确认；[内存修复](../artifacts/entity_attention_memory_window_20261006/README.md)保留截至30轮证据。
- [分头窗口与detach value续训](guides/entity_attention_split_training_zh-CN.md)：第50轮fork，policy近4轮、value3000局/近期50%，独立clip、每路batch256，200局/100步、固定PPO200局/5轮；已完成20新轮/16500步，长期收益未确认。
- [连续自迭代对照实验套件](guides/entity_attention_iteration_suite_zh-CN.md)：共同16500步起点，B→C→D三组各seed1、16轮、冻结起点趋势、内存上限与自动恢复；B前两轮和首次评测已完成，整体incomplete。
- [分阶段搜索对greedy](../artifacts/entity_attention_stage_search_20261008/README.md)：只搜后期与只搜前中期各500pair，冻结16500步同权重、公开分数阈值10、可中断续跑；三组各500pair已完成；全程81.8%明确高于后期67.0%与前中期69.35%。
- [旧最新单轮数据配额设计](guides/entity_attention_fresh_share_zh-CN.md)：12.5%→18.75%→25%的历史候选，已由分头窗口替代，保留补测时设计。
- [原生后端与推理配置](guides/native_search_zh-CN.md)：构建、后端、batch/传输与版本兼容。
- [多 worker GPU](guides/multiworker_gpu_zh-CN.md)：共享槽、容量、故障与恢复。
- [固定配对评测](guides/evaluation_zh-CN.md)：计分、种子、bootstrap、sweep 和结果解释。

## 可复核报告

- [搜索目标噪声A/B/C/D](../artifacts/entity_attention_target_noise_20261007/README.md)：固定最新16500步模型、200个留出局面、8次重复，控制世界抽样和根噪声，报告JS、动作一致性、平均目标及KL分解。
- [随机节点搜索同预算A/B/C/D](../artifacts/entity_attention_chance_search_20261007/README.md)：复用相同模型/局面/种子，D组JS0.08477→0.03072、Top-1一致率67.1%→85.2%；39项回归和短GPU调用链通过，报告耗时与深度代价，不代表棋力提升。
- [随机节点与确定化搜索100 pair棋力](../artifacts/entity_attention_chance_strength_20261007/README.md)：同16500步权重、均S1024，新搜索62胜1和137负、得分31.25%（95%区间24.5%–38.5%）；200局自然终局，零工程失败。
- [CPW k0.25/α0.5的A/B/C/D](../artifacts/entity_attention_chance_pw_noise_20261007/README.md)：同200局面×8重复，C/D JS相对旧下降54.6%/43.0%，相对不扩宽上升，D Top-1一致率76.43%；报告深度、成本、KL与两份配对对照。
- [CPW同模型100 pair棋力](../artifacts/entity_attention_chance_pw_strength_20261007/README.md)：73胜2和125负、得分37.0%，仍弱于旧确定化；相对不扩宽+5.75个百分点，配对差值95%区间[−3.75,+15.00]个百分点，未确认改善。200局自然终局，零工程失败。
- [CPW对同权重greedy的100 pair](../artifacts/entity_attention_chance_pw_greedy_20261007/README.md)：当前16500步模型、S1024/k0.25/α0.5为145胜1和54负，纯胜率72.5%、得分72.75%（配对95%区间66.5%–78.5%），明显强于裸网。200局自然终局，零工程失败；新增混合native/greedy批量评测并通过35项相关回归，正式训练教师未切换。
- [冻结教师采集与短训对照](../artifacts/entity_attention_chance_training_probe_20261007/README.md)：native/chance/CPW各200局＋500步，value detach、分组clip1；16主面板1600局已完成，未确认新教师训练棋力收益。CPW100步独立100 pair复测49.75%对native50%，差值区间跨0，初筛优势未复现；共2000局通过完整审计；完整拟合／留出／阶段／梯度／采样覆盖／value漂移与成本记录。
- [随机节点／CPW搜索与冻结教师短训：2026-10-07系列整合报告](reports/entity_attention_search_teachers_zh-CN.md)：汇总目标噪声、搜索稳定性和棋力、固定数据短训与平均标签；修订模型身份、统计单位和计算成本，区分已测事实与机制假设；第九节分析正式20轮留出变化并给出自迭代候选方案。
- [多搜索平均标签对照](../artifacts/entity_attention_target_averaging_20261007/README.md)：四份完整S1024/N1平均，修订对局聚类KL优势仍成立，学生直接对原单次学生得分60.5%；尚未确认相对起点净提升。原统计保留，修订统计另存；总模拟量包括原采集约5倍，生产复用行为搜索属于另一个候选。长期训练、M2学生与等成本收益未验证。
- [200局短训剂量对照](../artifacts/entity_attention_step_dose_20261007/README.md)：步数/学习率/value/标签平均四轴；棋力峰值在50步附近（avg4 56.5%、single 58.0%），100步回落（47%/45%）；lr2.5e-5@100为54.0%；冻结critic head与detach无差异；拟合与棋力最优点不一致。当前200局/100步/lr1e-4处于最差区间，优先降步数或学习率。
- [正式回放学习率×步数探针](../artifacts/entity_attention_formal_lr_step_20261007/README.md)：16500起点、round-019正式账本固定重放，lr2.5e-5/5e-5/1e-4×25/50/100步，边界延伸到150步；重放链逐张量复现正式round-019。低lr的留出KL更低且随步数继续下降，1e-4@100净策略变化近0、长步数bank KL变差；50-pair面板无显著差异，DEV最高节点未在独立种子复现。结论：不据此改正式日程，多轮验证未运行。
- [固定数据policy/value消融](../artifacts/entity_attention_policy_value_ablation_20261006/README.md)：同起点与batch三分支、100/500步裸网与搜索、固定完整value、训练/留出KL及整局区间、2000步固定policy和256局面探针。正式训练未恢复，窗口/长期分离收益未证明。
- [Python 引擎历史验收](reports/engine_validation_zh-CN.md)。
- [PPO 5M seed1](reports/ppo_5m_zh-CN.md)。
- [AZ 五轮与逐代扩展评测](reports/alphazero_seed1_zh-CN.md)。
- [推理吞吐有效基线与失效结论](reports/inference_throughput_zh-CN.md)。
- [多 worker GPU 的窗口与整轮对照](reports/multiworker_gpu_zh-CN.md)。
- [采集、learner 与固定评测的分项统计](reports/pipeline_profile_zh-CN.md)。
- [端到端性能优化与完整1000局/50 pairs验收](reports/end_to_end_speed_optimization_zh-CN.md)。
- [新模型固定形状、kernel启动与独立双缓冲](reports/entity_attention_execution_speed_zh-CN.md)。
- [MCTS sweep 实际完成状态](reports/mcts_sweep_zh-CN.md)。

## 历史归档和维护边界

旧工作包、实施大纲、修复流水账以及混杂的性能附录已完整打包到 `runtime/document_archive/2026-10-01/docs_before_cleanup.tar.gz`，逐文件 SHA 位于相邻 `manifest.json`。其中仍有效的接口/算法已提取到以上文档；原件不继续作为 docs 或 Git 中的现行资料。归档仅存在本机，迁移需备份。

`prompt/` 是被忽略的本地历史任务资料，只有用户本次明确采用时才是需求，docs 不依赖它才能读懂。实验原件保留策略见 [artifacts 约定](../artifacts/README.md)，配置用途见 [configs 导航](../configs/README.md)。保留原实验目录身份，日志、checkpoint 和恢复 sidecar 不在本次整理中重组。
