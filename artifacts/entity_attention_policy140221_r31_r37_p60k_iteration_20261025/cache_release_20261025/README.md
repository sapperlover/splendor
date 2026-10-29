# 在线训练结束后释放训练缓存

2026-10-25，按用户要求，`random_budget_iteration`在完整轮训练结束、final保存成功后，进入greedy／搜索前清空compact LRU，并删除训练池和抽样验证视图的引用，再执行原有GC／GPU缓存清理，以及glibc `malloc_trim(0)`向系统归还空闲堆页。发出`training_cache_released`事件，记录释放游戏数、字节数、清空后的缓存字节数和`heap_trimmed`。下一轮采集期间保持空缓存，训练按需重新加载。

实际机器已核实为Linux/glibc 2.35并提供`malloc_trim`。本在线入口需要glibc的该接口，不做静默allocator回退。1024局真实CPU检查证明仅删除对象／GC后RSS没有下降，因此归还空闲堆页是针对已复现的问题；RSS仍包含存活面板、索引和运行时，不等于LRU字节数。

主训练指标中的`replay_cache`仍记录释放前的实际训练占用，避免把训练缓存误报为0。原始NPZ、回放活动池、固定历史子集／小型eligibility索引、模型、AdamW及RNG不变；完整固定留出面板仍单独常驻CPU，本次没有扩大到它的释放。

实际入口已切换：

```bash
bash scripts/run_policy140221_r31_r37_p60k_iteration_20261025.sh --check
# 手动续跑；本次未执行该命令：
bash scripts/run_policy140221_r31_r37_p60k_iteration_20261025.sh
```

新冻结发布是`runtime/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/source_cache_release_20261025`。从旧`source`复制120个文件，仅替换`src/splendor/random_budget_iteration.py`；旧发布、旧会话和已产出的训练／棋力结果保留原身份。源码默认行为已更新；其他历史冻结入口不会被隐式替换。

代价是失去跨轮热缓存。本次实际训练缓存约10.04GiB，后续每轮会重新解码压缩；此前暖缓存3000步约5.7分钟，重启后冷缓存约13.5分钟，可作量级参考，不保证新实现每轮都恰好增加此时长。搜索预算、并发和配置A不变。

- `source.json`：旧／新发布全部文件SHA和唯一差异。
- `tests.xml`：源码30项相关回归。
- `frozen_tests.xml`：实际发布的缓存释放、完整AdamW/RNG精确恢复、六轮留出恢复验证。
- `final_tests.xml`：加入CPU堆页归还后，最终冻结发布的全部30项相关回归。
- `input_check.json`：通过实际启动脚本完成的正式64k输入只读检查。
- `memory_smoke.json`、`memory_smoke.py`：CPU只读使用1024个真实训练对局的缓存分配／GC／归还堆页检查，不执行更新或对战。未加归还堆页时的原始结果保留为`memory_smoke_without_heap_trim.json`。

当前正式训练未恢复；第12轮仍保留已训练完、greedy完成、搜索未完成的原始恢复边界。续跑会先沿原协议补齐尚缺搜索，不将中断面板改为成功。
