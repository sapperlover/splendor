# B → C → D 单种子趋势实验

2026-10-23归档位置更新：本专题关联`runs/entity_attention_iteration_trend_seed1_20261008`中的14个抽样账本／逐步日志、400个旧NPZ已按用户授权外置至[当日T7归档](../repository_archive_20261023/README.md)，原路径保留相对链接。复算这些原件前需解包；复训前将需写入的文件恢复为原位真实文件。历史指标、正式结果及来源身份不改写。

2026-10-08，按用户要求去掉seed2，只保留seed1。现已完成B/steps50前两轮和第2轮首次评测；C/D尚未运行。当前[配置](../../configs/entity_attention_v1/iteration_trend_20261008.json)和[完整流程及启动命令](../../docs/guides/entity_attention_iteration_suite_zh-CN.md)已同步。

首次节点累计400新局/100更新、16500→16600步，对冻结起点102胜1和97负，得分51.25%（100pair，95%区间44.5%–58.0%），尚未确认净收益。200局原始JSONL复算一致且均自然终局；两轮各自当前留出policy KL与value MSE略上升，不能据两轮判定长期失败。详见[first_evaluation_review.json](first_evaluation_review.json)。分阶段搜索归因的独立新实验见[只搜后期／只搜前中期](../entity_attention_stage_search_20261008/README.md)。

新增独立[critic-only诊断](../entity_attention_critic_only_replay_20261008/README.md)：从B1终点16550冻结完整encoder/policy，保留AdamW状态重放B2原有50步value批次。15代3000局完整评估，整体留出MSE 0.802914→0.789075，正式B2为0.808967；支持共享表征变化干扰value拟合。最新B2的12局留出改善区间跨0，未评估诊断模型棋力；不推进或替换本套件正式模型。

从共同16500步起点分别fork，完整完成B的16轮及评测后运行C，最后D；每轮200新局，B50步/lr1e-4/P4，C100步/lr2.5e-5/P4，D50步/lr1e-4/P1。value采样口径、内存限制和断点恢复机制沿用现有实现。

第2/4/8/12/16轮统一对冻结起点测100pair，保留共同评测种子及相邻节点配对差。总计48训练轮、9600新局、3200更新、15个面板/1500pair/3000评测局，粗估8–12小时。每组只有一条训练轨迹，不再提供跨训练种子的重复性比较。

为保留[双种子准备原件](../entity_attention_iteration_trend_20261008/README.md)，当前输出目录改为`entity_attention_iteration_trend_seed1_20261008`。命令仍使用同一个`iteration_trend_20261008.json`；首次启动自动保存新的manifest，旧冻结manifest不重写。

当前配置快照见`configuration_snapshot.json`。准备阶段仅检查单种子配置与调度；随后用户实际运行了上述两轮及首次棋力面板，本次核验读取其既有原始产物，不追加训练或评测。既有真实CPU断点短流程本身使用一个训练种子，先前CUDA检查只检查环境和内存限制，记录均保留在旧目录。

结果在`artifacts/entity_attention_iteration_trend_seed1_20261008/analysis.json`，模型/NPZ在`runs/entity_attention_iteration_trend_seed1_20261008/`，日志和状态在`runtime/entity_attention_iteration_trend_seed1_20261008/`。第一次启动与中断续跑使用同一命令。
