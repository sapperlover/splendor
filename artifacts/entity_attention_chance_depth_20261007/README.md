# 大S深度初筛（用户取消后续）

2026-10-07。在完成12根预算初筛后，用户取消继续测深度／棋力，改为[chance progressive widening](../entity_attention_chance_pw_20261007/README.md)。本目录仅保存已完成的原始初筛；原计划的200根确认、匹配深度棋力赛和大S采集计时均未执行。

同一16500步模型、同一分层抽取12根、根噪声关闭、N1；200根输入来自此前冻结留出库，12根包含early/middle/late各4根、其中5根有对手暗保留牌，selection seed16800000，搜索seed16710000／16720000。源码／扩展身份见[manifest.json](manifest.json)，原始JSON在`cases/`及`graph/cases/`；源码、原生扩展快照与日志在`runtime/entity_attention_chance_depth_20261007/`。

| 推理路径 | 搜索预算 | 12根平均逻辑深度 | 12根耗时 |
| --- | --- | ---: | ---: |
| FP32 eager | 旧native S1024 | 4.630 | 1.83s |
| FP32 eager | native_chance S1024 | 2.399 | 1.99s |
| FP32 eager | native_chance S4096 | 2.886 | 7.72s |
| FP32 eager | native_chance S16384 | 3.501 | 32.38s |
| CUDA Graph | 旧native S1024 | 4.630 | 18.19s（含冷编译） |
| CUDA Graph | native_chance S1024 | 2.396 | 2.94s |
| CUDA Graph | native_chance S16384 | 3.505 | 12.34s |
| CUDA Graph | native_chance S65536 | 4.194 | 54.72s |
| CUDA Graph | native_chance S131072 | 4.514 | 159.83s |

S131072在这个12根Graph初筛中距旧深度约−2.5%，进入原预设±5%范围。它使用batch6，其余batch12；不能把该比值当作完整200根深度匹配、eager匹配或等量采集成本，更没有棋力结论。高S需要限制并发以控制树内存；S131072此进程峰值RSS约5.77GiB。

编译路径会改变微小浮点数，搜索会放大这些差异：旧Graph根prior与eager最大绝对差约5.50e-7，但12根target／visit counts／选招一致；新S1024和S16384最大target分量差分别约0.0625和0.1620，平均深度差则很小。Graph仅作为单独记录的性能／深度路径，不能声称与eager搜索目标逐位一致。

首次eager命令为`probe.py --backend native_chance --subset pilot --sims 1024 4096 16384 --batch-size 12`；Graph为`graph_probe.py`同参数接口，S65536使用batch12、S131072使用batch6。均从根目录以`conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u`运行。首次Path类型preflight错误在任何结果产生前修复，失败日志保留；后续真实结果均完整使用声明预算。

这是被用户停止的未完成实验，保留初筛事实，不作为新的正式搜索配置依据。
