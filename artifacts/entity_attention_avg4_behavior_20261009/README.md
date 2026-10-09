# 行为复用M4连续训练：记录

状态：`incomplete`。已完成12轮、2400新局、1200更新步／17700步；原套件中断，第2/4/8轮完整100pair得分48.5%／56.25%／53.0%，区间均跨50%。第12轮只完成61个完整pair，配对得分50%，另有2个单局，不作为完整结果。`analysis.json`仍是第8轮之后的自动汇总。追加标签多进程验证记录保留；下文首轮115局迁移属于历史准备状态。

流程、启动与恢复命令见[专题指南](../../docs/guides/entity_attention_avg4_iteration_zh-CN.md)，计划见[配置](../../configs/entity_attention_v1/iteration_avg4_behavior_20261009.json)。

从原16500步checkpoint分支，保留AdamW／采样RNG；16轮，每轮200局／100步／lr1e-4。行为搜索复用为第1份，额外3次完整S1024/N1，四份平均仅替换policy监督目标。继承旧单次标签历史并按原窗口退出：policy近4轮、value3000局／近期128＋历史128。第2/4/8/12/16轮各对冻结起点100pair；每轮另做固定256局面诊断。此分支不设硬RAM上限或磁盘门槛。

`manifest.json`冻结配置、起点和五个对战节点；`training_configs/`为生成的实际hybrid配置；`preflight.json`是正式配置CUDA准备检查。`verification.json`保留原单进程准备验证；多进程当前依据为`pool_verification.json`、`pool_tests.xml`和`pool_cuda_smoke.json`。8个CPU worker、总256活跃根、GPU batch200、micro64、共享槽2，单GPU owner／FP32 eager；每根完成即保存，worker每轮一次启动，恢复跳过已完成根。`cuda_smoke.json`是两局自然CUDA对局、S8/N1行为＋追加3份标签验证，原始档案和标签在忽略的`runtime/…/cuda_smoke_final/`；`cuda_smoke.py`保存执行脚本原件。脚本固定写入该验证路径，仅用于复核原执行；重复运行需另指定输出路径，正式训练使用suite入口。

预计训练与五次完整对战总计约10–12小时，来自旧采集、重标注及对战耗时外推，正式全程尚未测量。旧“额外四份平均标签”实验总搜索约5倍，本次行为＋3份总4倍是另一种生成口径，净棋力收益尚未验证。

`pool_migration.json`记录停止状态的一次性执行配置迁移和115局SHA；原配置/manifest/准备报告快照在忽略的`runtime/…/before_pool_switch/`。当时原命令保留已有115局并补完剩余85局。`migrate_to_pool.py`是这一次迁移的可复核脚本，不是通用恢复入口。`pool_cuda_smoke.py`只在独立runtime副本上做有限验证，不会重启正式实验。

最终CUDA验证：两局数据副本共110个真实可搜索根、330次追加完整S1024/N1／337920次模拟；8个worker均未初始化CUDA，标签与单进程逐项相同；中断后复用85个已完成任务并清理worker／共享内存。两局短测单进程13.77秒、多进程16.00秒，尚未显示加速，200局整轮吞吐未验证。

当前固定数据诊断见[历史采样与policy梯度实验](../entity_attention_policy_history_20261009/README.md)：使用第4/8/12轮训练前checkpoint与固定M4数据，A/B/C三种policy抽样、0/50/200/500步，value规则及优化配置保持原样；检查训练／留出KL与继承AdamW的真实更新方向，不进行搜索对战。

版本说明：`preflight.json`和`code_version.json`会在后续suite启动／准备时更新；`pool_verification.json`保留当时有限验证及当时文件SHA，不等同于当前运行准备摘要。实验索引的文件SHA对应当前保留文件，历史验证记录不改写。
