# 新模型采集执行优化：2026-10-05

结论和边界见[报告](../../docs/reports/entity_attention_execution_speed_zh-CN.md)。这是有限窗口与正确性实验，不是优化版1000局或棋力验收。

- `summary.json`、`summarize.py`：有效窗口、最终源码第3组对照、数值/kernel统计与验证汇总。
- `configs/`、`*_r*.json`：配置快照与原始owner/worker窗口。baseline为原shared1；shared2控制只增加worker槽；static为B256/512＋reduce-overhead；static500为B64/256/500；direct为固定地址Graph，direct_double另开owner2。
- `*_r*.command.json`、`run_windows.py`：后续窗口的精确命令、源码身份、环境、GPU前后状态与输出。早期baseline/static/double/shared2窗口的命令参数均为对应config、games-per-worker125、warmup3、seconds15、scratch runtime/entity_attention_execution_speed_20261005；代码变更期间的早期测量仅作探索，最终同源码对照使用repeat3。
- `gpu_*.json`、`gpu_commands*.json`、`verify_gpu.py`：真实请求policy/value/value-only的FP32检查、10批CUDA API统计。最后direct检查与CPU回归并行，微测计时不用于性能结论；数值和调用数验证有效。大Profiler trace保留在runtime。
- `gpu_complete_smoke.json`、`gpu_complete_smoke_direct.json`、`run_gpu_checks.py`：每模式4局S8/N2、双缓冲及checked replay。
- `cpu_tests.xml`：最终代码55项相关CPU测试；`cpu_tests_pre_direct.xml`保留较早测试记录。
- `source_identity.json`、`final_source_identity.json`：实验过程及交付源码身份；源码副本和dirty.patch在runtime的source/final_source。
- `round2_phase_context.json`：原正式run的完整轮2阶段计时，仅用于解释优化优先级。
- `warmup_initial_failure.json`：初始静态形状命中默认图限额，随后改为配置决定的局部有限限额。
- `invalid_window.json`：一次static500窗口与CPU回归重叠，排除比较，原件保留在runtime/invalid，使用repeat2替代。

没有更改已有run、checkpoint、训练设置或用户进程；没有启动完整新训练。新采集预设为[collection_static_graph.json](../../configs/entity_attention_v1/collection_static_graph.json)。
