# 新模型自迭代配置的有限验证

2026-10-23归档位置更新：本专题关联`runs/entity_attention_10x1000_seed1`中的9个抽样账本／逐步日志、3003个旧NPZ已按用户授权外置至[当日T7归档](../repository_archive_20261023/README.md)，原路径保留相对链接。复算这些原件前需解包；复训前将需写入的文件恢复为原位真实文件。历史指标、正式结果及来源身份不改写。

2026-10-04。交付时正式10×1000未启动；后续已有轮0–2完成记录，未在本次重新验收全部十轮；配置与命令见 [自迭代指南](../../docs/guides/entity_attention_self_iteration_zh-CN.md)。

- `verification.json`：11个循环相关CPU用例、2个调度/历史配置用例及GPU链路断言；1000局调度为8进程各125局，独立于GPU batch500。
- `verification.json.admission_and_legacy_config_tests`：调度和历史配置2项检查的命令／退出码／计数；冗余stdout本地保全，路径与SHA见[整理清单](../repository_cleanup_20261009/manifest.json)。
- `round-000/`、`round-001/`：真实新模型GPU短测各搜索/greedy评测原始summary，双方交换先后手各1 pair，四个面板均passed；对手始终是离线final。
- `session-00.json`、`source_identity.json`：GPU会话命令、实际耗时、设备、源码身份和冻结smoke配置。采集8局、训练4步，S16/N1，仅验证调用链。

重型短测归档、checkpoint、sidecar、TensorBoard与源码快照在忽略的 `runtime/entity_attention_self_iteration_smoke/`。正式配置为 `configs/entity_attention_v1/online_10x1000.json`，短测配置为同目录 `online_smoke.json`；配置中保存完整种子与设置。短测胜负不能解释为棋力结论；未测正式1000活跃局的速度收益。
