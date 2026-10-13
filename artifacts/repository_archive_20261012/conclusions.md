# 归档内容、已保留结论与证据索引

核对日期：2026-10-12。本说明对应本批721个迁移入口，按45个来源／用途组覆盖；完整逐项映射与文档／证据SHA见 [conclusion_index.json](conclusion_index.json)。这是已保存结果的文档核对，不是新训练、对战或全量规则重放。

归档保存真实数据；报告继续在 `docs/reports/`、对应 `artifacts/<实验身份>/` 或明确写出结论的专题指南维护。表内结论是截至核对日期的快照，原始数字、版本和完成状态以列出的证据为准。无棋力结论是工程数据的明确分类，不是遗漏。

## 数据与研究分支

| 原来源／归档内容 | 入口数 | 已保留结论 | 适用限制 | 正式结论位置 |
| --- | ---: | --- | --- | --- |
| 原AZ五轮／2500局；`runs/az_loop_seed1` | 5 | 原五轮完成。θ5对原PPO的greedy为53.65% [50.40%,56.85%]、500pair；搜索56.50% [47.00%,66.00%]、50pair。 | greedy有当前DEV正向信号；搜索未确认，跨代并非单调，单训练seed。归档的是采样，不是新训练结果。 | [alphazero_seed1_zh-CN.md](../../docs/reports/alphazero_seed1_zh-CN.md) |
| 旧θ5续训十轮／10000局；`runs/az_theta5_10x1000_seed1` | 10 | 正式十轮、12050次更新和20个50pair常规面板完成；终点对旧θ5搜索200pair为46.625% [41.875%,51.250%]，未确认超过起点。 | 终点对PPO只完成100/200pair；59.25%是未完成点估计，不能视为200pair确认。与原PPO→θ5及attention蒸馏分开解释。 | [az_theta5_10x1000_zh-CN.md](../../docs/reports/az_theta5_10x1000_zh-CN.md)、[az_theta5_10x1000_seed1](../../artifacts/az_theta5_10x1000_seed1/README.md) |
| 早期S256/N2单次采集／300局；`runs/az_selfplay_seed1_round000` | 1 | 原summary完成300/300，Train268局／Validation32局，旧v1格式。 | 采集完成不是棋力提升；预算、协议不同，不直接合并到当前S1024/N1池。本次未全量重放。 | [archived_engineering_experiments_20261012_zh-CN.md](../../docs/reports/archived_engineering_experiments_20261012_zh-CN.md)、[selfplay_data_zh-CN.md](../../docs/guides/selfplay_data_zh-CN.md) |
| M4追加搜索标签／12轮；`runs/entity_attention_avg4_behavior_20261009/seed1_avg4` | 12 | 行为搜索复用为四份之一，追加三份；12/16轮完成，2400局／1200更新。扩样后r4、r12各300pair仍未确认超过冻结起点。 | 482982个文件不是48万局新采集；2400局行为原件保留。固定数据可拟合，不等于稳定fresh泛化或长期棋力收益。 | [entity_attention_m4_diagnostics_conclusions_zh-CN.md](../../docs/reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md)、[entity_attention_avg4_behavior_20261009](../../artifacts/entity_attention_avg4_behavior_20261009/README.md) |
| Chance／CPW采样与6份smoke；`runs/entity_attention_chance_training_probe_20261007` | 5 | 归档Chance与CPW各200局及6份smoke；当前固定预算探针未支持切换正式教师，CPW初筛优势在独立确认中未复现。 | native 200局未迁移、仍属于旧29000局池；标签更稳定或更好拟合不是更强教师的证明。 | [entity_attention_chance_training_probe_20261007](../../artifacts/entity_attention_chance_training_probe_20261007/README.md)、[entity_attention_search_teachers_zh-CN.md](../../docs/reports/entity_attention_search_teachers_zh-CN.md) |
| S512/N2训练分支；`runs/entity_attention_hybrid_s512_n2_from_round9_seed1` | 1 | 33新轮已提交、12800步；3300局来自完成采集轮，下一轮仅5份已提交NPZ。 | 采集使用S512/N2，评测统一S1024/N1；没有只改变N的长期因果对照。不能把归档或未纳入新池解释成已证明数据无效。 | [entity_attention_hybrid_fork_zh-CN.md](../../docs/guides/entity_attention_hybrid_fork_zh-CN.md)、[entity_attention_hybrid_fork_20261005](../../artifacts/entity_attention_hybrid_fork_20261005/README.md) |
| 固定M4的A/B/C采样与梯度；`runs/entity_attention_policy_history_20261009` | 1 | 三起点、4500更新完成。只训最新数据可降低Train KL，却未改善未见局面并牺牲旧代；未发现普遍强烈历史梯度冲突。 | B保留为候选，不能把一阶梯度余弦当实际收益；后续fresh扩样需与原stored结果共同解释，无棋力面板。 | [entity_attention_policy_history_20261009](../../artifacts/entity_attention_policy_history_20261009/README.md)、[entity_attention_m4_diagnostics_conclusions_zh-CN.md](../../docs/reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md) |
| 第九轮单变量参数扫描；`runs/entity_attention_r9_ofat_20261010` | 1 | 10个新分支及原配置对照完成，55个唯一配置—节点。较小lr减轻短训policy回退；1k池value有明显训练—留出分化。 | 100／250步没有优于原500步的证据，500步也不是全局最优。lr2e-5×1000对原500步直接搜索53.25% [48.50%,58.00%]，未确认棋力优势。 | [analysis.md](../../artifacts/entity_attention_r9_ofat_20261010/analysis.md) |
| 第九轮500→5000步扩展；`runs/entity_attention_r9_step_extension_20261010` | 1 | 同一轨迹500／1000／2000／5000步已测完整第九轮留出。5000步MSE较500步升0.113341 [0.059843,0.170553]。 | 固定lr1e-4联合长训未带来一致留出收益；不能推出任意学习率下都练多了，也没有直接测这些节点棋力。 | [entity_attention_r9_step_extension_20261010](../../artifacts/entity_attention_r9_step_extension_20261010/README.md) |
| 第29轮起点的BCD分池；`runs/entity_attention_r30_head_pools_20261011` | 1 | 三分支各1000步、200pair greedy完成。policy缩到3k／value扩到9k未获得已确认的改进，B的微弱正向点估计未确认。 | Train拟合改善未同步到留出；greedy不用value决定动作，因此未检验critic的搜索价值。 | [conclusions.md](../../artifacts/entity_attention_r30_head_pools_20261011/conclusions.md) |
| 迭代套件验证v1；`runs/entity_attention_iteration_suite_validation_20261008` | 1 | 两条分支共16新局/S2、8更新、7个1pair面板；初版源码只作有限CUDA工程验证。 | 小样本分数不作科学结论；不能把缩小负载峰值当正式S1024内存上界。旧源码身份与恢复适用范围保留。 | [entity_attention_iteration_suite_validation_20261008](../../artifacts/entity_attention_iteration_suite_validation_20261008/README.md) |
| 迭代套件验证v2；`runs/entity_attention_iteration_suite_validation_v2_20261008` | 1 | 最终套件16新局/S2、8更新、7个1pair面板；完整调用链、重复启动与退出清理已有限验证。 | 小样本分数不作科学结论；不能把缩小负载峰值当正式S1024内存上界。旧源码身份与恢复适用范围保留。 | [entity_attention_iteration_suite_validation_v2_20261008](../../artifacts/entity_attention_iteration_suite_validation_v2_20261008/README.md) |
| 长期在线主线的544个中间checkpoint；`runs/entity_attention_r9_online_lr2e5_1000_forever_20261011` | 544 | 归档时续训34轮至绝对第43轮；该节点greedy对P4为82.25% [78.25%,86.00%]、200pair。 | 第24↔30轮直接搜索及1000pair greedy均未确认强弱，不能据此断言整个24–43轮完全平台。保留主要节点和采样；未测每个50步节点棋力。 | [entity_attention_r9_online_lr2e5_1000_forever_20261011](../../artifacts/entity_attention_r9_online_lr2e5_1000_forever_20261011/README.md) |
| 29000局离线蒸馏的中间模型与账本；`runs/entity_attention_recent_pool_20261009` | 106 | 103194步完成。P4完整模型对θ5000为86.125%；最终policy＋P4 value双网络92.25% [89.625%,94.75%]、200pair。后期policy进步与value退化可并存。 | 固定对手、复用DEV及事后节点比较，不证明普遍最优。原池NPZ、P4/P8/P12与组合仍保留；归档中间权重不代表节点无科学价值。 | [entity_attention_recent_pool_20261009](../../artifacts/entity_attention_recent_pool_20261009/README.md) |

## 工程、诊断与运行现场

以下31个runtime来源逐一对应；其中的smoke对局、失败初版和探索微基准不混入正式棋力或速度结论。

| 原runtime来源 | 已保留结论／用途 | 限制 | 正式结论位置 |
| --- | --- | --- | --- |
| `runtime/action_benchmark_cli_check` | 200局面prior输出与83局面S1024/N4子集比较保留；对应协议模块不在当前源码树。 | 未核验调试输出，无可靠棋力校准结论；不把JSD或动作变化当胜率。 | [archived_engineering_experiments_20261012_zh-CN.md](../../docs/reports/archived_engineering_experiments_20261012_zh-CN.md) |
| `runtime/az_loop_smoke` | 两轮各4局完成，原采集汇总无记录异常。 | 仅小规模闭环，不替代正式五轮或证明棋力。 | [archived_engineering_experiments_20261012_zh-CN.md](../../docs/reports/archived_engineering_experiments_20261012_zh-CN.md) |
| `runtime/az_native_cuda_pool_smoke` | 一轮2局完成，小规模原生GPU池闭环。 | 不能外推正式负载内存／吞吐／棋力。 | [archived_engineering_experiments_20261012_zh-CN.md](../../docs/reports/archived_engineering_experiments_20261012_zh-CN.md)、[multiworker_gpu_zh-CN.md](../../docs/guides/multiworker_gpu_zh-CN.md) |
| `runtime/end_to_end_speed_2026-10-03` | 各1000局采集、50pair评测及128步learner对照完成，保存checked replay与更新等价性。 | 单机每组一次；重仪器中断基线被排除，不能声称新的棋力或多轮AZ收益。 | [end_to_end_speed_optimization_zh-CN.md](../../docs/reports/end_to_end_speed_optimization_zh-CN.md)、[end_to_end_speed_2026-10-03](../../artifacts/end_to_end_speed_2026-10-03/README.md) |
| `runtime/entity_attention_avg4_behavior_20261009` | 行为＋追加3份的多进程标签与单进程逐项一致；有限中断复用与worker清理已验证。 | 两局短测未显示加速，不能当200局整轮吞吐证明；准备快照与正式12轮分开。 | [entity_attention_avg4_behavior_20261009](../../artifacts/entity_attention_avg4_behavior_20261009/README.md) |
| `runtime/entity_attention_avg4_delta_kl_20261009` | 修正过cluster bootstrap重复权重和遗漏补充组；均匀Train样本含已抽中局面，不能称纯未见组。 | 扩样结论以scaleup复核为准，旧跨0结果不是等效；不能跨不同目标直接相减绝对KL。 | [entity_attention_avg4_delta_kl_20261009](../../artifacts/entity_attention_avg4_delta_kl_20261009/README.md)、[entity_attention_m4_diagnostics_conclusions_zh-CN.md](../../docs/reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md) |
| `runtime/entity_attention_chance_pw_noise_20261007` | CPW保留部分随机节点标签稳定性收益，但较不扩宽随机节点更抖。 | 稳定性不是强度，A/B/C/D并非严格可加的方差分解。 | [entity_attention_chance_pw_noise_20261007](../../artifacts/entity_attention_chance_pw_noise_20261007/README.md) |
| `runtime/entity_attention_chance_search_20261007` | 随机节点搜索可改变目标稳定性，但有限预算分散展开；对战未支持替换正式native教师。 | 固定根目标诊断与整局棋力分别解释，不能只按标签JS选择教师。 | [entity_attention_chance_search_20261007](../../artifacts/entity_attention_chance_search_20261007/README.md)、[entity_attention_search_teachers_zh-CN.md](../../docs/reports/entity_attention_search_teachers_zh-CN.md) |
| `runtime/entity_attention_clip_sweep_20261009` | 只改policy裁剪，clip1对照精确复现；较小学习率／裁剪改变更新幅度和固定标签拟合。 | 未做棋力面板，不证明关闭裁剪或某个组合全局最佳。 | [entity_attention_clip_sweep_20261009](../../artifacts/entity_attention_clip_sweep_20261009/README.md) |
| `runtime/entity_attention_encoder_critic_refit_20261008` | 相同新head、账本和500步预算，16500／16600 encoder留出MSE约0.79247／0.79242，差值区间跨0。 | 仍在下降，未测绝对拟合上限，也未证明等效或棋力提升。 | [entity_attention_encoder_critic_refit_20261008](../../artifacts/entity_attention_encoder_critic_refit_20261008/README.md) |
| `runtime/entity_attention_execution_speed_20261005` | 有限真实窗口及GPU数值／完整小局验证支持执行路径优化；最终同源码窗口才用于比较。 | 一次重叠CPU回归窗口失效，不能把早期窗口或smoke当完整1000局验收。 | [entity_attention_execution_speed_zh-CN.md](../../docs/reports/entity_attention_execution_speed_zh-CN.md)、[entity_attention_execution_speed_20261005](../../artifacts/entity_attention_execution_speed_20261005/README.md) |
| `runtime/entity_attention_hybrid_cpu_smoke_20261005` | 缩小数据、离线4步、两轮各2局／2步，在线validation设0。 | 只验证小链路，不能代替正式GPU规模／留出或棋力验证。 | [entity_attention_hybrid_zh-CN.md](../../docs/guides/entity_attention_hybrid_zh-CN.md) |
| `runtime/entity_attention_hybrid_fork_validation_20261005` | 两次initialize-only继承9500步模型、AdamW/RNG及2000局来源；有限CPU恢复验证。 | 这批初始化副本本身没有正式新局更新；后续S512/N2训练是独立来源。 | [entity_attention_hybrid_fork_20261005](../../artifacts/entity_attention_hybrid_fork_20261005/README.md) |
| `runtime/entity_attention_memory_gpu_smoke_20261006` | 修复窗口外解码缓存滞留、纯在线前离线缓存保留及阶段编译状态释放；有限GPU恢复检查。 | 数组预算不是进程RSS硬上界，不能保证长期无OOM；历史OOM原始证据仍保留。 | [entity_attention_memory_window_20261006](../../artifacts/entity_attention_memory_window_20261006/README.md) |
| `runtime/entity_attention_p4_pool_iteration_20261010` | 中断／连续更新等价、窗口滚动和低分仍升级经过有限验证，后续渐进配额和Graph档位另有验证。 | 旧smoke早于配额修订，不把它当新正式配置完整验收；正式九轮有独立报告。 | [entity_attention_p4_pool_iteration_20261010](../../artifacts/entity_attention_p4_pool_iteration_20261010/README.md) |
| `runtime/entity_attention_policy_history_20261009` | 保留最新／前三代梯度、实际AdamW位移及抽样引用；局部head冲突不是所有起点普遍冲突。 | 一阶方向不保证有限步KL下降；相同value账本不等于相同value函数。 | [entity_attention_policy_history_20261009](../../artifacts/entity_attention_policy_history_20261009/README.md)、[entity_attention_m4_diagnostics_conclusions_zh-CN.md](../../docs/reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md) |
| `runtime/entity_attention_r9_online_validation_20261011` | 第9轮完整模型有限2步训练与2局S8采集，验证fork、原split继承及恢复。 | 2 active／batch2／workers2不代表B500/S1024的长期资源上界。 | [entity_attention_p4_pool_forever_plan_20261011_zh-CN.md](../../docs/guides/entity_attention_p4_pool_forever_plan_20261011_zh-CN.md) |
| `runtime/entity_attention_recent_pool_20261009` | 保留固定池、拟合和恢复的验证现场；正式103194步及组合棋力有独立原始汇总。 | 短验证不独立提供棋力结论，未完成的历史状态须由完成报告解释。 | [entity_attention_recent_pool_20261009](../../artifacts/entity_attention_recent_pool_20261009/README.md) |
| `runtime/entity_attention_self_iteration_smoke` | 真实小规模GPU两轮：8采集局、4更新及四个1pair面板，用于循环与恢复行为。 | 与后来正式3×1000采样分开，smoke分数不是棋力证据。 | [entity_attention_10x1000_seed1](../../artifacts/entity_attention_10x1000_seed1/README.md) |
| `runtime/entity_attention_target_averaging_20261007` | 100步平均标签组对单次标签组直接搜索60.5% [54.25%,66.75%]、100pair；相对训练起点净收益未确认。 | 这里是额外4份搜索、总成本约5倍；与行为＋3份的连续M4口径不同。 | [entity_attention_target_averaging_20261007](../../artifacts/entity_attention_target_averaging_20261007/README.md) |
| `runtime/entity_attention_target_noise_20261007` | 200根×4组×8重复；世界抽样目标变化大于根噪声，真实联合条件随机分歧占该样本平均KL约38.9%。 | 这不是长期平台因果证明；固定模型／样本／种子，不能当所有训练误差的噪声占比。 | [entity_attention_target_noise_20261007](../../artifacts/entity_attention_target_noise_20261007/README.md) |
| `runtime/entity_attention_v1_smoke` | 8步GPU训练4+恢复4、2局native采集和1pair混合对战；原10000局仅metadata盘点。 | 两种形状短测不代表正式拟合或棋力收益，本次不重跑。 | [entity_attention_v1_20261004](../../artifacts/entity_attention_v1_20261004/README.md) |
| `runtime/gpu_pool_compiled_dev` | 原summary完成1/1局，failed／exceptions／工程截断为0。 | 仅开发短链路，不能推导无回退、生产吞吐或模型强度。 | [archived_engineering_experiments_20261012_zh-CN.md](../../docs/reports/archived_engineering_experiments_20261012_zh-CN.md) |
| `runtime/gpu_pool_dev_smoke` | 原summary完成2/2局，failed／exceptions／工程截断为0。 | 仅开发短链路，不能推导无回退、生产吞吐或模型强度。 | [archived_engineering_experiments_20261012_zh-CN.md](../../docs/reports/archived_engineering_experiments_20261012_zh-CN.md) |
| `runtime/inference_throughput` | 08正确row-stride与09传输基准有适用范围；06／07旧错误输入测量保留为审计而非当前基线。 | 必须按原报告区分有效、被替代与窗口负载，不把全部归档文件一起用于排名。 | [inference_throughput_zh-CN.md](../../docs/reports/inference_throughput_zh-CN.md)、[inference_throughput](../../artifacts/inference_throughput/README.md) |
| `runtime/mcts_step2_smoke` | 4/4局归档完成，原summary未记失败。 | 采集正确性短验证，不是棋力面板。 | [archived_engineering_experiments_20261012_zh-CN.md](../../docs/reports/archived_engineering_experiments_20261012_zh-CN.md)、[selfplay_data_zh-CN.md](../../docs/guides/selfplay_data_zh-CN.md) |
| `runtime/module_b` | 两组完整500局与开局窗口用于多worker供给单GPU的速度比较。 | 每组一次、硬件／代码身份记录不完整；不承诺重复稳定倍率，不提供棋力结论。 | [multiworker_gpu_zh-CN.md](../../docs/reports/multiworker_gpu_zh-CN.md) |
| `runtime/native_search` | 保存三个500局档位及smoke；正确输入08系列有完整摘要，较早错误输入测量排除。 | 仅原模型／机器／形状，不能直接认定当前attention模型最优参数。 | [inference_throughput_zh-CN.md](../../docs/reports/inference_throughput_zh-CN.md)、[inference_throughput](../../artifacts/inference_throughput/README.md) |
| `runtime/offline_cache_speed_20261004` | 扩缓存恢复保持模型／AdamW／RNG与样本引用一致；缩小工作集减少draw阶段miss。 | 12.3倍是理论上限，未测真实8GiB整轮加速，不把解码字节当物理读取。 | [offline_training_speed_20261004](../../artifacts/offline_training_speed_20261004/README.md) |
| `runtime/search_accel` | 保留固定局面数值／性能与串行／合批短测，供旧版本行为复核。 | 未统一复算认证，CPU/GPU排名不适用于当前native配置，无棋力结论。 | [archived_engineering_experiments_20261012_zh-CN.md](../../docs/reports/archived_engineering_experiments_20261012_zh-CN.md)、[search_zh-CN.md](../../docs/guides/search_zh-CN.md) |
| `runtime/speed_profile_2026-10-02` | 既有完整运行汇总加24步learner、1pair CPU及16局诊断定位成本；原相对路径失效初版由v2替代。 | 只归因，非已实施优化或完整生产验收；profiler成本不混入正式计时。 | [pipeline_profile_zh-CN.md](../../docs/reports/pipeline_profile_zh-CN.md)、[speed_profile_2026-10-02](../../artifacts/speed_profile_2026-10-02/README.md) |

## 这次补齐和修正的结论入口

- 旧θ5十轮从“未启动”修正为训练完成，补充正式报告，单独标出终点PPO只有100/200pair；原准备清单和数值原件不改。
- 长期在线主线新增截至第43轮的结果快照；区分对P4的绝对优势、24↔30轮未确认的代际差异与未测的50步中间节点。
- 早期300局、开发GPU池、旧Python搜索及临时校准器遗留补入工程报告；没有独立证据的项目明确写无正式棋力结论。
- M4、A/B/C、OFAT、步数扩展、BCD、critic与噪声等已有报告保留原结论，通过本索引连接归档，不重新宣称实验完成或修改原始SHA。

## 长期保存与外置

`archive/20261012/CONCLUSIONS.md` 是本文件的同日只读快照；同目录另有本批迁移清单与机器结论索引快照。维护位置为 `artifacts/repository_archive_20261012/`，后续修订应重新同步快照并更新核对时间。外置盘单独打开本表可阅读核心结论；跳转专题全文仍需主仓库中的docs/artifacts，需独立复核时一并保留表列出的报告和机器索引中的证据。

两份冻结训练池72000局、主要模型及当前链路未在此批迁移。恢复路径、文件数统计及核验详见 [归档说明](README.md)。
