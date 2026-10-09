# 运行与分析工具

从仓库根目录、Conda `splendor`、`PYTHONPATH=src` 执行。先读对应指南，再按目标机器核对依赖、数据和设备。`benchmark`／`profile`／`validate` 会执行工作负载；名称不表示只读。

| 工具 | 用途与参考 |
| --- | --- |
| `analyze_az_run.py` | 汇总已有训练／资源／抽样记录，不启动训练或对战；[AZ指南](../docs/guides/alphazero_zh-CN.md) |
| `run_recorded_evaluation.py`、`run_az_generation_evals.py` | 固定配对面板及会话记录、逐代比较；[评测](../docs/guides/evaluation_zh-CN.md) |
| `run_mcts_50pair_sweep.py` | 历史固定搜索面板；[sweep报告](../docs/reports/mcts_sweep_zh-CN.md) |
| `benchmark_native_search.py`、`benchmark_batched_search.py` | 同checkpoint、固定位置的搜索执行比较；[native指南](../docs/guides/native_search_zh-CN.md) |
| `benchmark_collect_throughput.py`、`benchmark_large_batch_inference.py` | 真实请求采集窗口与大batch吞吐；[多worker报告](../docs/reports/multiworker_gpu_zh-CN.md) |
| `benchmark_inference_modes.py`、`benchmark_transfer_modes.py`、`profile_inference_pipeline.py`、`compare_speed_inference.py` | 推理、传输、真实输入和数值比较；[推理报告](../docs/reports/inference_throughput_zh-CN.md) |
| `profile_speed_pipeline.py` | 有限采集／learner／评测分项诊断；[时间口径](../docs/reports/pipeline_profile_zh-CN.md) |
| `run_speed_validation.py`、`summarize_speed_validation.py` | 执行性能面板、从已有产物重建汇总；[端到端报告](../docs/reports/end_to_end_speed_optimization_zh-CN.md) |
| `validate_entity_attention.py` | 实体注意力有限集成smoke与GPU形状；[离线指南](../docs/guides/entity_attention_offline_zh-CN.md) |
| `prepare_next_az.py`、`run_search_accel_short_benchmark.py` | 旧10×1000准备工具、旧Python搜索执行基准；[历史准备](../docs/archive/next_az_training_zh-CN.md)、[搜索指南](../docs/guides/search_zh-CN.md) |

特定实验的 `run.py`／`experiment.py`／`probe.py` 留在对应 `artifacts/<实验身份>/`，与冻结配置、报告一起复核；不因名字相似合并。训练与搜索的稳定模块入口在 `src/splendor/`，运行日志／临时配置写 `runtime/`。
