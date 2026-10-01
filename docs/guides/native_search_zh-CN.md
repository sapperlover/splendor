# 原生 PUCT 后端与推理配置

`native_search.cpp` 是 C++17/pybind11 后端，包含规则、数值编码、PUCT 树和状态推进。Python checked 引擎与 `features.py` 是差分参照。真实 native 采集由 `NativeMatchStepper` 推进；固定对战 runner 的真实对局仍为 Python，搜索世界可选择 native。搜索公开入口只使用 `DecisionRequest`，不读取真实暗牌。

## 构建

从仓库根目录、Conda `splendor` 环境执行：

```bash
conda run -n splendor python -m pip install -e . --no-deps
conda run -n splendor env PYTHONPATH=src python -m pytest -q tests/test_native_search.py tests/test_native_batch.py
```

构建使用 C++17、`-O3 -DNDEBUG`，没有 fast-math；`.so` 和 `build/` 被 Git 忽略。选择 native 而未构建时明确失败，不静默切换 Python。

## 后端与配置

| 后端 | 模型与搜索布局 | 配置例子 |
| --- | --- | --- |
| `native_cpu` | 多进程 CPU，各 worker 一个冻结模型 | `configs/native_search/alphazero_loop_cpu.json` |
| `native_cuda` | 单进程搜索与唯一 CUDA owner | `configs/native_search/alphazero_loop_cuda.json` |
| `native_cuda_pool` | 多搜索进程、唯一 CUDA owner | `configs/native_search/alphazero_loop_cuda_pool.json` |

当前单 owner CUDA 生产配置是 active=500、batch=500，不是旧的 active=8。CPU eager、CUDA compiled 的成本结论及有效范围见 [性能报告](../reports/inference_throughput_zh-CN.md)。pool 配置的容量语义见 [多 worker 指南](multiworker_gpu_zh-CN.md)。这些是可执行预设，不表示所有后端都已经完成多轮长训。

native 要求 `engine_mode=fast`、FP32。CUDA 单 owner 的 `workers=1`；pool 使用独立 `search_workers`，不能用外层 workers 复制 CUDA 模型。

| 字段 | 含义 |
| --- | --- |
| `batch_interface` | `native` 整批接口；`legacy` 逐叶参照 |
| `inference_mode` | `eager` 或 `compiled` |
| `actor_projection` | 复用 actor 首层状态投影 |
| `candidate_buckets` | 编译候选宽度档位，常见 `[16,32,48]`；宽度超档按已实现策略走 eager 并计数 |
| `compile_failure_policy` | `fallback` 或 `fail_fast`；必须检查实际 worker_compile，不能仅由配置声称全程 compiled |
| `native_transport` | `reused` 复用缓冲，或 `packed` 参照传输 |
| `pipeline_slots` | 单 owner 支持 1/2；pool 保持 1 |

## 整批内存与兼容

`NativeBatch` 直接编码待评估行，保存 `(world, simulation)` 身份并一次性导出；重复/stale 结果明确失败。policy/value-only 分开。`reused` CUDA 输入按 row stride 读取，float 与 mask 各一次 H2D；设备内融合输出后一次 D2H。CPU batch 可使用零拷贝 NumPy 视图。满批、尾批、宽度切换和 value-only 的 mask/padding 必须保持契约。

单 owner reused manifest 使用 `native_gpu_layout=row-stride-v2`；pool 使用自己的 `native_gpu_pool=shared-slots-v1` 协议。恢复校验模型 SHA、配置、规则/特征/RNG/搜索版本，不能续接不兼容半成品。

隐藏世界洗牌采用 SplitMix64；native 根噪声采用 C++ `mt19937_64` gamma。相同整数 seed 不保证 Python/native 的噪声和选招相同。跨设备/batch 的 FP32 舍入也可能改变近平票选招；checked replay 验证保存的真实动作轨迹，不等于重新执行神经搜索得到逐位相同结果。

## 有限验证与命令

差分与内存回归入口：`test_native_search.py`、`test_native_batch.py`、`test_inference_modes.py`；pool 见 `test_multiworker_gpu.py`。条件 CUDA 测试被跳过时，只能报告 CPU 覆盖。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast generate \
  --config configs/native_search/selfplay_cpu_smoke.json --output runtime/native_search/cpu_smoke
conda run -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast validate \
  --input runtime/native_search/cpu_smoke --replay
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.alphazero_loop \
  --config configs/native_search/alphazero_loop_smoke.json --output runtime/native_search/alphazero_loop_smoke
```

固定局面、采集窗口和传输画像脚本分别为 `benchmark_native_search.py`、`benchmark_collect_throughput.py`、`profile_inference_pipeline.py`。计时和输出参数使用 `--help` 核对；长训练或完整复测需要具体实验任务。
