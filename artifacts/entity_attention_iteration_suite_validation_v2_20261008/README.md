# 最终套件有限 CUDA 验证

2026-10-08。最终源码加入父进程退出时向直属实验进程发送SIGINT，并处理SIGTERM/SIGHUP。工程验证采用独立目录、16新局/S2、8更新步骤、7个1-pair面板（14评测局），不启动正式128轮长实验。

完整流程通过，实测阶段cgroup峰值2.841GiB（已受编译磁盘缓存暖态影响）。相同启动命令再次执行，仅preflight/analysis执行，未追加采集、训练或评测；源码哈希与正式套件冻结身份一致。运行结束无遗留本套件scope或CUDA计算进程。不能把这个小规模峰值外推为正式200局/S1024的内存上界；正式阶段另受16GiB硬限额约束。

完整调用链、重复启动、cgroup限额与中断恢复证据汇总见 [verification.json](../entity_attention_iteration_suite_20261008/verification.json)。本目录 `manifest.json` 冻结实际源码与参数；`analysis.json`、`round_checks/`、`evaluation_checks/` 和 `resources/` 保留可复核原始汇总。模型/NPZ和日志仍分别放在ignored runs/runtime。

这里只验证工程行为，缩小参数后的胜率不用于任何科学结论，也不代表已经测得正式长期资源峰值。
