# R21–R37双辅助离线训练

2026-10-27完成状态核对：训练、完整留出选点及原两组对战均已完成；192804步早停，选中140221步。原raw R37 greedy1000 pair得分53.275%，双方S1024N1（总1024）500 pair为53.100%，见[原结果](results.md)。同一候选的新版10000 pair greedy／S128N1结果见[大面板](../offline5_vs_r37_10k_20261027/README.md)。下方按日期保留首次准备、停训和恢复过程，不再代表当前进度。

2026-10-25存储更新：本实验训练／选点及greedy、搜索两组对战均已完成，结果见[results.md](results.md)。按本次授权将192个普通checkpoint及`training_metrics.jsonl`外置至[归档20261025](../repository_archive_20261025/README.md)，原路径保留恢复链接；选中140221步、全部完整留出节点、initial/final以及配置／源码／结果保留。细查逐步曲线或普通模型须恢复对应文件，继续写入前须恢复为原位真实文件；下文运行进度及旧存储描述保留为历史记录。

2026-10-24执行更新：本组已接入[独立执行发布](../offline_training_defaults_20261024/README.md)，实际配置`offline_continuation_r21_r37_prefetch_20261024.json`为18GiB／4 worker，启用流式统计及传输／同步优化。原16GiB配置、原源码与plan保留历史身份，种子／目标／优化器／留出／评测保持；已完成真实CUDA短对照，本次未启动正式训练。下方16GiB和源码未变描述对应首次交付。

2026-10-24，属于[三组串行消融队列](../entity_attention_auxiliary_queue_20261024/README.md)，按用户指定顺序排在当前完整实验之后；不并行启动。

- 104000局，独立从零初始化及新AdamW；短期value辅助0.25、对手policy辅助0.15，主policy／终局value仍为1/1。
- lr1e-4、16GiB、两路B256、原seed、12次value曝光上限350552步；每5%完整主KL+MSE连续三次上升早停，最低节点固定整个模型。
- 随后对同一raw R37 greedy1000pair，再双方S1024N1（总1024）搜索A500pair，模板和种子与当前实验完全相同。
- 最新轻量指标与参考SHA记录、每1000步及全部百分比节点保存；不写大抽样账本。配置、数据、留出、原始计划与冻结源码身份见本目录plan／source。

该单组手动入口是`bash scripts/run_attention128_both104k_20261024.sh`，支持prepare／run／train／status。此次使用总队列`bash scripts/run_attention128_auxiliary_queue_20261024.sh`，确保前一组训练、选点及两组对战全部成功结束后才启动。失败或中断停止后续队列。

有限回归与真实数据CPU／原生搜索验证统一保存在[队列目录](../entity_attention_auxiliary_queue_20261024/README.md)。当前任务的冻结执行源码及进程保持不变。
