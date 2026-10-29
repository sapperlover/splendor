# 18GiB缓存与4 worker共享只读供数

2026-10-27后续状态：这里记录的停止／准备／仅当前组适用属于当时工程发布范围；后续104k仅policy、双辅助及44k仅value均采用已发布优化完成训练、选点和两组对战，见[完成汇总](../../docs/reports/auxiliary_offline_results_20261027_zh-CN.md)。原工程验证数值与冻结发布不改写。

2026-10-24后续状态：本发布曾恢复的policy组现已按用户要求停止于115000步。当前脚本已改指向[流式统计／传输优化发布](../training_streaming_20261024/README.md)，18GiB／4 worker配置保持；本目录源码、身份及运行记录保留为上一发布历史。新任务只准备命令，未重启。下文“仍运行／继续”均对应此前交付时点。

2026-10-24，用户要求停止当前训练，将缓存改为18GiB，尝试4 worker共享只读供数并恢复。本次仅应用于正在运行的104k对手policy辅助组；后续双辅助、44k组保留原执行配置，串行顺序和完整评测边界保持。

## 入口与身份

仍使用`bash scripts/run_attention128_auxiliary_queue_20261024.sh`。其中policy104k脚本改为调用本目录`resume_policy.py`，使用独立`source_prefetch_20261024`发布和[实际执行配置](../../configs/entity_attention_v1/offline_opponent_policy_r21_r37_prefetch_20261024.json)：`cache_mib=18432`、`training_workers=4`。

原16GiB配置、冻结源码、历史checkpoint和完整留出保留。新会话source记录指向[source.json](source.json)，训练会话另保存实际配置和源码。本次从102000步恢复模型、AdamW、采样及Torch RNG；之后未提交日志按原恢复机制撤回重算。原NPZ／标签／整局split、lr1e-4、两路B256、辅助权重0/0.15、5%验证／早停／选点及R37两组棋力面板不变。

## 供数和内存

- 启动时4个CPU进程并行读取全部训练局、生成原有标签并压缩；主进程写入一个18GiB上限的Linux memfd。它不占磁盘文件或`/dev/shm`挂载配额，实际RAM按触及的页使用。
- 随后4个spawn进程只读映射同一份数组。每个worker仅保留最多1024局的数组视图／元数据和batch，不各建18GiB LRU，不通过fork继承CUDA上下文。
- 主进程按原顺序抽样，最多预取4组batch；checkpoint仅保存已消费batch对应RNG。未消费预取在中断后丢弃，不跳样本。
- 验证使用独立256MiB流式缓存，不驱逐共享训练数据。训练池放不进共享区会明确报错。18GiB含共享区对齐开销，不是整个进程树RSS上限；多进程RSS重复计入共享映射，应同时看PSS和MemAvailable。
- 此Conda Python没有`os.memfd_create`，实现直接调用目标Linux的glibc接口；初次接口检查失败记录保留于`initial_memfd_tests.xml`。

`training_workers=0`保留原串行路径。缓存容量和worker数是执行参数，恢复允许改变；其他算法字段仍严格核对。旧checkpoint／数据digest沿用原冻结配置，实际执行值见新配置和每会话source。

## 验证与测速

- [tests.xml](tests.xml)：63项相关回归通过，包括只读共享、4 worker、引用与batch顺序、未消费预取、改变缓存／worker恢复、模型／AdamW／RNG逐位一致和退出回收。
- [gpu_smoke.json](gpu_smoke.json)：68局真实源数据、正式B256，从原102000步checkpoint分别进行8次串行／并行CUDA更新，模型、AdamW及采样RNG逐位一致。各20次预热＋160次计时更新：热缓存串行22.01步/秒、4 worker25.49步/秒，约1.16倍。小数据工程测速不代表全池吞吐或棋力。
- [before.json](before.json)保存改动前备份和234个受保护历史文件哈希；`recovery_check.json`核对真实恢复身份。停止身份见`stop_request.json`，后续实际启动／吞吐另记录。

预热进度见`runs/entity_attention_128x2_opponent_policy_r21_r37_20261024/prefetch_progress.json`。每步新增`prefetch.worker_pid`及`worker_collate_seconds`；原`seconds.collate`现在只计主进程耗时，速度比较以稳态墙钟为准。

## 正式恢复核验（2026-10-24 11:42）

已从102000步继续并提交103000、104000步checkpoint，核验时到104473步，串行队列及当前完整训练链仍运行。共享区实际数组17764760952字节（16.54GiB），含对齐17869150492字节（16.64GiB）；93537个训练局全部常驻，预热379.48秒。

原100000→102000步墙钟594.16秒，为3.37步/秒；新103000→104000步44.46秒，为22.49步/秒，当前稳态组合提速约6.68倍。该比较排除一次性预热和百分比验证，不等于整条训练／评测链加速比；大部分收益来自18GiB消除反复解码，4 worker独立热缓存短测约1.16倍。

最近605个新训练步miss为0，4个worker均实际供数。主进程和4个worker映射同一memfd inode，worker权限全部`r--s`；进程树PSS约20.90GiB，MemAvailable约3.35GiB。234个历史checkpoint／留出／原冻结源码文件哈希全部保持。此为有限运行核验，不代表完成长期内存稳定性实验。

原件：[running_verification.json](running_verification.json)、[verification.json](verification.json)、[gpu_running.json](gpu_running.json)、[launch.json](launch.json)。本次核验结束后继续原串行任务，不新增评测或配置矩阵。
