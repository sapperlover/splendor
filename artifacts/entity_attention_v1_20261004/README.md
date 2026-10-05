# Entity attention v1 有限验证

对应 [结构、设置与命令](../../docs/guides/entity_attention_offline_zh-CN.md)，2026-10-04。正式新模型拟合、棋力面板和三轮在线任务没有执行。

- `integration.json`：8步真实归档训练（4+恢复4）、2局 native_cuda_pool、1 pair 新旧混合 native_batched；compiled/reused，编译失败 fallback 为0。
- `gpu_shapes.json`：RTX4050Laptop、FP32、B64/C32与B500/C64的首次/预热延迟、显存及倍率。首调用不等于纯编译时间，reserved跨模型保留。
- `data_summary.json`：十轮10000局原split，train770870/validation80699；metadata-only统计，decode_count=0。
- `model_identity.json`：实际新8步导出权重与旧θ5的架构配置、已记录model SHA和训练步数。
- `verification.json`：相关测试及后续记录字段有限检查、执行范围。
- `configs/`、`commands.json`：冻结配置副本、命令、初始化/抽样/验证seed；正式命令前置条件在专题中。
- `source_identity.json`：8步GPU链路和形状短测实际源码身份，包含git基点、dirty源码身份与native扩展。
- `delivery_source_identity.json`：交付源码身份；离线入口补充了GPU名称、全留出耗时/数据身份、总会话耗时、resume时latest刷新及冻结文件摘要校验。这些改动由追加1步CPU和完成后零步resume检查验证，模型和GPU路径没有改动。
- `offline_smoke/`：原8步两次会话、曲线、使用覆盖和训练摘要；`recording_check/`：全留出/完成后零步resume记录。

本次源码与配置本就有未提交工作，保留其状态，没有提交或清理。重型临时NPZ、checkpoint和sidecar保留在 `runtime/entity_attention_v1_smoke/`（忽略），正式模型和数据输出使用 `runs/`。短测得分不解释为棋力。实验索引只引用本次必要摘要，不重新扫描或checked replay旧10000局。
