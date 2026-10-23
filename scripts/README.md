# 运行与分析工具

[`run_r31_10k_p60k_iteration_20261023.sh`](run_r31_10k_p60k_iteration_20261023.sh)：[raw R31＋22–31轮40k历史的独立自迭代分支](../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/README.md)，每轮新采10k、池容量60k、lr2e-5／3000步；首轮50k即可训练、第二轮满60k，沿用历史固定25%policy子集、原随机历史淘汰／在线FIFO、混合预算及R43轮间评测。执行快照沿用父分支，只替换支持未满池的driver。采集更新为最新12worker／2000活跃局／B500／microbatch128，回放缓存16GiB；首次未完成采集已清除，重置后未重启。默认无限，同命令启动／恢复，`--check`只检查输入。

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
