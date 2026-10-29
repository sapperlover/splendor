# 运行与分析工具

[`export_running_web_game.py`](export_running_web_game.py)：一次性导出旧版人机网页仍在内存中的指定对局。需要允许GDB附加目标进程的权限；排入Python待执行任务后立即断开，由原服务原子保存完整状态与回放，不重启或改变棋盘。新启动的网页服务已默认自动保存，无需此工具。命令与恢复／校验方式见[本地人机网页指南](../docs/guides/evaluation_zh-CN.md#本地人机网页)。

[`run_r37_lr2e5_step140221_vs_r37_10k_20261029.sh`](run_r37_lr2e5_step140221_vs_r37_10k_20261029.sh)：[固定R37初值／104k池／lr2e-5的140221步对raw R37](../artifacts/r37_lr2e5_step140221_vs_r37_10k_20261029/README.md)。先10000pair greedy，再10000pair双方S128N1（总128），CPU8／搜索配置A及原10k种子库；固定模型不追踪最新。默认run、prepare只准备、status只读，同命令恢复，失败／中断停止。未启动正式评测。

[`run_r37_104k_offline_lr1e4_20261027.sh`](run_r37_104k_offline_lr1e4_20261027.sh)与[`run_r37_104k_offline_lr2e5_20261027.sh`](run_r37_104k_offline_lr2e5_20261027.sh)：[R37权重起点的104k离线训练](../artifacts/entity_attention_r37_104k_offline_lr_20261027/README.md)，新AdamW／原离线RNG、无辅助，其余原离线配置逐字段保持，包含5%完整留出早停。默认训练→选点→原R37 greedy1000pair／搜索A500pair，train仅训练选点，prepare只准备，status汇总；本次未启动，不接现有队列。

2026-10-27完成状态：[六组原离线训练与原R37对战](../docs/reports/auxiliary_offline_results_20261027_zh-CN.md)全部结束；[五模型十组10000 pair评测](../artifacts/offline5_vs_r37_10k_20261027/README.md)也已完成。下方按日期记录的等待、115000步停止和首次未启动为历史过程；执行参数与命令保持。

[`run_offline5_vs_r37_10k_20261027.sh`](run_offline5_vs_r37_10k_20261027.sh)：[五个原104k离线best对raw R37](../artifacts/offline5_vs_r37_10k_20261027/README.md)，先5组greedy各10k pair，再5组双方S128N1各10k pair（每世界128×1，总128），严格串行；greedy CPU8、搜索配置A、共用新开局种子，原选点固定。默认run，prepare只准备，status只汇总；完整pair恢复，失败／中断停止。十组正式评测均已完成，共100000 pair。

[`run_attention128_noaux104k_queue_20261026.sh`](run_attention128_noaux104k_queue_20261026.sh)：[串行等待当前policy＋得分辅助完整成功，再运行104k无辅助实验](../artifacts/entity_attention_noaux104k_queue_20261026/README.md)。后继单组脚本为[`run_attention128_noaux104k_20261026.sh`](run_attention128_noaux104k_20261026.sh)，原lr1e-4／18GiB／4 worker／5%早停选点／R37两组评测保持，从零基础attention128，无任何辅助；支持prepare／默认run／train／status。同命令恢复。

[`run_attention128_policy_score104k_20261026.sh`](run_attention128_policy_score104k_20261026.sh)：[原104k池policy＋得分归属辅助离线实验](../artifacts/entity_attention_128x2_policy_score_r21_r37_20261026/README.md)，从零attention128、辅助0.15／0.25、短期value关闭，其余沿用原离线设置。支持`prepare`／默认`run`（训练→选点→R37 greedy1000pair→搜索A500pair）／`train`（仅训练及选点）／`status`，同命令恢复；独立冻结源码；训练、选点及两组对战已完成。

2026-10-26，`validate_continuation_auxiliary.py`支持按实际启用的头验证分数归属任务：3局checked重放、已购牌／贵族ID独立oracle、一次CPU更新及3个S8N1根，命令与边界见[新目标工程记录](../artifacts/score_ownership_20261026/README.md)。

[`run_policy140221_r31_r37_p60k_iteration_20261025.sh`](run_policy140221_r31_r37_p60k_iteration_20261025.sh)：[140221步policy辅助离线模型的独立在线分支](../artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/README.md)，R31–R37历史64k、每轮新10k／容量60k／3000步／lr2e-5，继续policy辅助0.15，首轮显式淘汰14k原历史。每轮对raw R37 greedy1000pair，每3轮双方S1024N1（总1024）搜索A500pair；每六轮完整此前六轮留出。`--check`只读；用户已完成13轮，第12轮搜索500 pair完整得分60.20%，第13轮greedy63.75%。当前指向[训练结束释放缓存的新冻结发布](../artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/cache_release_20261025/README.md)，评测及下一轮采集期间训练LRU为空，下一轮训练冷加载；本次未续跑。

2026-10-24发布记录：`run_attention128_both104k_20261024.sh`和`run_attention128_value44k_20261024.sh`已接入[后两组独立执行发布](../artifacts/offline_training_defaults_20261024/README.md)，本次均18GiB／4 worker，启用流式累计和传输优化。总队列命令保持，当前policy发布保留；三组均已采用这些优化，本次未启动。新建离线默认与缓存规则见[指南](../docs/guides/offline_execution_zh-CN.md)。下文后两组仍用旧发布为此前记录。

2026-10-24发布记录：`run_attention128_policy104k_20261024.sh`已改为[流式统计与传输优化发布](../artifacts/training_streaming_20261024/README.md)。当前组和外层队列已停止，自动恢复点115000步，本次未重启；单组完整链与原串行队列命令见该文档。18GiB／4 worker保持，后续两个实验仍用各自原冻结发布。下文prefetch入口描述为上一发布。

2026-10-24，`run_attention128_policy104k_20261024.sh`已指向[18GiB＋4 worker执行发布](../artifacts/training_prefetch_20261024/README.md)，从原checkpoint恢复；总队列入口与串行成功边界保持。其余两个后续实验脚本仍使用原配置。

[`run_attention128_auxiliary_queue_20261024.sh`](run_attention128_auxiliary_queue_20261024.sh)：[等待当前实验完整结束，再依次运行三组消融](../artifacts/entity_attention_auxiliary_queue_20261024/README.md)。单组脚本依次为`run_attention128_policy104k_20261024.sh`、`run_attention128_both104k_20261024.sh`、`run_attention128_value44k_20261024.sh`；各自包含训练／选点／R37 greedy1000pair／搜索A500pair。等待监督锁，不轮询训练进度；失败或中断停止后续。

[`run_attention128_shortterm_value_r21_r37_20261023.sh`](run_attention128_shortterm_value_r21_r37_20261023.sh)：[104k局R21–R37从零短期value训练与R37两组对战](../artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023/README.md)。默认串行训练→主KL+MSE选点→greedy1000pair→配置A搜索500pair；`prepare`只准备，`train`只训练与选点，`status`汇总已有原件。最新轻量记录，无大抽样账本；同命令恢复。

`validate_continuation_auxiliary.py`：读取3局真实历史轨迹，验证短期value／对手policy标签与一次CPU更新、严格加载及3个原生S8N1根（共24次模拟），确认搜索不执行辅助头。参数、完整命令和已保存结果见[工程验证](../artifacts/continuation_auxiliary_20261023/README.md)，非正式棋力评测。

2026-10-23当前R31入口已发布停写双份抽样／逐步日志、500步恢复点＋七个固定诊断点的更新；同一命令恢复，旧源码与已完成产物保持。详见[记录规则](../docs/guides/training_artifacts_zh-CN.md)及[交付](../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/recording_checkpoint_20261023/README.md)。

当前R31启动脚本使用独立`source_recording_checkpoint_20261023`快照：每六轮使用此前六轮完整validation，段内固定、换段自动冻结、恢复补齐；R38–R43的新面板已冻结。只补齐本分支已完成节点使用`python -m splendor.six_round_holdout`，完整冻结PYTHONPATH命令见[评测指南](../docs/guides/evaluation_zh-CN.md#在线全量固定留出)。原静态评测器及42项结果保留，R20–R31补评已按最新指示暂缓，本次未续训。

[`run_r31_10k_p60k_iteration_20261023.sh`](run_r31_10k_p60k_iteration_20261023.sh)：[raw R31＋40k历史的自迭代分支](../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/README.md)，每轮新采10k、池60k、lr2e-5／3000步；首轮50k、第二轮满60k。已完成6轮至谱系R37。采集12worker／2000活跃局／B500／microbatch128／双槽、16GiB回放缓存保持；后续搜索使用A（active1000、双槽、精简B/C桶），在线greedy仍CPU8 worker／1000 pair，搜索每3个本地轮次500 pair。新轮界读取源配置，已准备面板沿用冻结配置；本次不重启任务，默认无限，同命令恢复，`--check`仅检查输入。

[`run_recorded_evaluation.py`](run_recorded_evaluation.py)：日常独立评测及调用／源码身份／资源记录；新增[greedy预设](../configs/entity_attention_v1/evaluation_greedy_r43.json)与[搜索A预设](../configs/entity_attention_v1/evaluation_search_a_r43.json)，候选示例固定R37、对手R43，不自动选择最新模型。完整命令、规模、预算、种子及五种场景对照见[评测指南](../docs/guides/evaluation_zh-CN.md)。

2026-10-23已删除购买结算特征候选的`run_attention_first_round_20261022.sh`及专用配置／入口，方案撤回，未启动正式训练。原准备／短验证和删除前脚本见[历史记录](../artifacts/entity_attention_first_round_20261022/README.md)。

[`run_attention128_r21_r31_lr2e5_20261022.sh`](run_attention128_r21_r31_lr2e5_20261022.sh)：[第21–31轮最低节点降lr续训](../artifacts/entity_attention_128x2_r21_r31_lr2e5_20261022/README.md)已在新增10000步三次上升早停，选中新增4000步／累计63453步。原59453步继承AdamW/RNG、lr2e-5，每2000步完整留出，原148632步上限未跑满；该入口只训练／选点，已完成时返回。用户后续授权的R31 greedy1000pair为独立评测命令，见专题；不恢复训练或自动搜索。

[`run_attention128_r21_r31_offline_20261022.sh`](run_attention128_r21_r31_offline_20261022.sh)保留[第21–31轮attention离线实验](../artifacts/entity_attention_128x2_offline_r21_r31_20261022/README.md)的原完整队列。最新用户指令已停止训练于保存84000步，固定59453步，仅执行对R20／R31各1000pair greedy，搜索取消；本次使用专题中的`run_greedy_after_stop.py run`及冻结PYTHONPATH命令，不用原shell恢复训练。新入口支持`prepare`／`run`／`status`，实际结果见专题。

2026-10-22存储说明：[第21–30轮及Cross64离线冷产物已外置](../artifacts/repository_archive_20261022/README.md)。在线最新模型／活动池及所有采集对局保留，原在线恢复入口可继续使用；Cross64离线的普通模型和大账本保留为冷归档链接，恢复该离线训练前须先按清单恢复会读写的文件为原位真实文件。

[`run_mlp_cross64_role005_offline_20261022.sh`](run_mlp_cross64_role005_offline_20261022.sh)：[角色初始化标定后的906434参数MLP离线入口](../artifacts/entity_set_mlp_cross64_role005_offline_80k_20261022/README.md)，同旧8万局／lr1e-4／16GiB缓存／270557步／5%留出。默认训练→完整留出选点→R20 greedy1000pair及搜索500pair；`train`只训练＋选点，`status`只读汇总。最初仅交付命令，用户后来启动，现中断、已保存81168步；本命令会恢复原训练。独立授权的54112步对R20 greedy1000pair得43.225% [41.075%,45.325%]，不代表原自动流程完成。

[`run_r20_s256n4_iteration_20261022.sh`](run_r20_s256n4_iteration_20261022.sh)：[第20轮继承续跑](../artifacts/entity_attention_r20_s256n4_noble12_iteration_20261022/README.md)，完整继承AdamW/RNG及24k活动池；25%高预算S256N4总1024、75%低预算S128N1总128，新局用12贵族。从第21轮无限迭代；同命令恢复，`--check`只校验。使用独立冻结源码与数据，不修改原分支。

[`run_attention128_offline_20261021.sh`](run_attention128_offline_20261021.sh)：[同8万局、同抽样与优化的attention128离线对照](../artifacts/entity_attention_128x2_offline_80k_20261021/README.md)，用户后来启动，现以KeyboardInterrupt退出，最近保存135279步，原270557步计划未完成。lr1e-4，每5%固定子集和完整留出，默认只训练＋KL+MSE选点，不自动对战。支持`prepare`、`status`、`select`及同命令恢复；本次新实验不恢复或修改旧run。

[`run_entity_set_mlp_offline_20261021.sh`](run_entity_set_mlp_offline_20261021.sh)：855682参数MLP的8万局离线入口，原270557步预算已按用户要求提前停止并保留97000步。原自动面板取消，之后单独授权的94695步对R20 greedy1000pair已完成，MLP得37.075%。默认入口仍会恢复原训练并在完成后对战，当前不执行。[配置与停止记录](../artifacts/entity_set_mlp_256x4_offline_80k_20261021/README.md)。

新增[`run_random_budget_iteration_20261019.sh`](run_random_budget_iteration_20261019.sh)：37423步起点、每轮4k局、24k混合预算池，policy历史固定25%子集及新局1/6、1/3…配额；[配置与恢复](../artifacts/entity_attention_random_budget_iteration_20261019/README.md)。已启动，完成4轮、第5轮采集中；源配置为B500/pipeline2，第5轮保持冻结768/2，第6轮自动切换500/2，无需重启。入口在下一新轮次读取批大小、batch buckets及pipeline slots的修改；追加`--check`只做输入检查。

从仓库根目录、Conda `splendor`、`PYTHONPATH=src` 执行。先读对应指南，再按目标机器核对依赖、数据和设备。`benchmark`／`profile`／`validate` 会执行工作负载；名称不表示只读。

| 工具 | 用途与参考 |
| --- | --- |
| `analyze_az_run.py` | 汇总已有训练／资源／抽样记录，不启动训练或对战；[AZ指南](../docs/guides/alphazero_zh-CN.md) |
| `run_recorded_evaluation.py`、`run_az_generation_evals.py` | 固定配对面板及会话记录、逐代比较；[评测](../docs/guides/evaluation_zh-CN.md) |
| `run_mcts_50pair_sweep.py` | 历史固定搜索面板；[sweep报告](../docs/reports/mcts_sweep_zh-CN.md) |
| `benchmark_native_search.py`、`benchmark_batched_search.py` | 同checkpoint、固定位置的搜索执行比较；[native指南](../docs/guides/native_search_zh-CN.md) |
| `benchmark_collect_throughput.py`、`benchmark_large_batch_inference.py` | 真实请求采集窗口与大batch吞吐；[多worker报告](../docs/reports/multiworker_gpu_zh-CN.md) |
| `benchmark_inference_modes.py`、`benchmark_transfer_modes.py`、`profile_inference_pipeline.py`、`compare_speed_inference.py` | 推理、传输、真实输入和数值比较；[推理报告](../docs/reports/inference_throughput_zh-CN.md) |
| `profile_speed_pipeline.py` | 有限采集／learner／评测分项诊断；[时间口径](../docs/reports/pipeline_profile_zh-CN.md) |
| `run_speed_validation.py`、`summarize_speed_validation.py` | 执行性能面板、从已有产物重建汇总；[端到端报告](../docs/reports/end_to_end_speed_optimization_zh-CN.md) |
| `validate_entity_attention.py` | 实体注意力有限集成smoke与GPU形状；[离线指南](../docs/guides/entity_attention_offline_zh-CN.md) |
| `prepare_next_az.py`、`run_search_accel_short_benchmark.py` | 旧10×1000准备工具、旧Python搜索执行基准；[历史准备](../docs/archive/next_az_training_zh-CN.md)、[搜索指南](../docs/guides/search_zh-CN.md) |

特定实验的 `run.py`／`experiment.py`／`probe.py` 留在对应 `artifacts/<实验身份>/`，与冻结配置、报告一起复核；不因名字相似合并。训练与搜索的稳定模块入口在 `src/splendor/`，运行日志／临时配置写 `runtime/`。
