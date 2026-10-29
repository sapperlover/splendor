# R21–R37仅policy辅助离线训练

2026-10-24，属于[三组串行消融队列](../entity_attention_auxiliary_queue_20261024/README.md)，按用户指定顺序排在当前完整实验之后；不并行启动。

- 104000局，独立从零初始化及新AdamW；短期value辅助0.0、对手policy辅助0.15，主policy／终局value仍为1/1。
- lr1e-4、16GiB、两路B256、原seed、12次value曝光上限350552步；每5%完整主KL+MSE连续三次上升早停，最低节点固定整个模型。
- 随后对同一raw R37 greedy1000pair，再双方S1024N1（总1024）搜索A500pair，模板和种子与当前实验完全相同。
- 最新轻量指标与参考SHA记录、每1000步及全部百分比节点保存；不写大抽样账本。配置、数据、留出、原始计划与冻结源码身份见本目录plan／source。

该单组手动入口是`bash scripts/run_attention128_policy104k_20261024.sh`，支持prepare／run／train／status。此次使用总队列`bash scripts/run_attention128_auxiliary_queue_20261024.sh`，确保前一组训练、选点及两组对战全部成功结束后才启动。失败或中断停止后续队列。

有限回归与真实数据CPU／原生搜索验证统一保存在[队列目录](../entity_attention_auxiliary_queue_20261024/README.md)。当前任务的冻结执行源码及进程保持不变。
