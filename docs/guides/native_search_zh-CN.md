# 原生 PUCT 后端与推理配置

`native_search.cpp` 是 C++17/pybind11 后端，包含规则、数值编码、PUCT 树和状态推进。Python checked 引擎与 `features.py` 是差分参照。真实 native 采集由 `NativeMatchStepper` 推进；固定对战 runner 的真实对局仍为 Python，搜索世界可选择 native。搜索公开入口只使用 `DecisionRequest`，不读取真实暗牌。

2026-10-21默认贵族目录扩为12张，新增N11蓝绿黑333、N12红白44；Python开局仍随机选3张，native使用同一目录及入场列表。C++不再把“目录恰好10张”当作特征约束：只有每局3个贵族槽、每牌17维属性是输入约束。规范表更新后必须重建扩展并重启调用进程；本次已原子替换本机扩展，旧运行中进程不会热切换。旧模型参数形状不变，原10张目录只用于显式的历史复现。

同日新增`NativeGreedyPolicy`的root-only批处理作业，供12 worker评测复用采集GPU owner。根前向后直接按原始logit argmax结束，零模拟、不创建搜索世界；原搜索的`root_only`默认false，预算及搜索行为不变。与eager greedy的逐局对照及真实模型GPU核验见[S256N4对Greedy实验](../../artifacts/entity_attention_r20_s256n4_vs_greedy_20261021/README.md)。

2026-10-07新增`NativeChancePUCTSearchPolicy`及评测`search_backend="native_chance"`，在simulation内按概率抽牌，按双方各自的信息历史共享决策Q/N。要求N=1作为共享统计池，已有`native`仍是独立确定化；模式、随机数与数据字段含义见[搜索指南](search_zh-CN.md#采样随机节点与信息历史-puctnative_chance)，冻结模型复测见[实验记录](../../artifacts/entity_attention_chance_search_20261007/README.md)。真实游戏牌堆语义不变。

同日新增可选`ChanceWideningConfig`与策略JSON的`chance_widening`字段，限制真实抽牌事件的结果数随访问量增长。仅在`native_chance`启用，省略保持原行为，正式训练配置未切换。API、近似边界及短验证见[CPW说明](search_zh-CN.md#chance-progressive-widening可选)和[实验记录](../../artifacts/entity_attention_chance_pw_20261007/README.md)。

后续[k0.25/α0.5稳定性复测](../../artifacts/entity_attention_chance_pw_noise_20261007/README.md)及[同模型100 pair棋力](../../artifacts/entity_attention_chance_pw_strength_20261007/README.md)已完成：CPW增加深度、保留部分相对旧确定化的JS收益，但对旧native得分37.0%（95%区间30.75%–43.25%），仍明显较弱；相对不扩宽的提升未确认，未切换正式教师。

新增[CPW对同权重greedy的100 pair](../../artifacts/entity_attention_chance_pw_greedy_20261007/README.md)：CPW得分72.75%（区间66.5%–78.5%），明显强于当前16500步模型裸网。混合native_batched／greedy评测已支持，greedy零模拟、单请求eager argmax，使用同一冻结模型；详见[评测指南](evaluation_zh-CN.md)。C++搜索和二进制未改，35项相关回归、四局GPU smoke和200局正式checked对打通过。

独立`selfplay_fast`现在接受`search_backend="native_chance"`及可选`chance_widening`，要求native执行与N=1；多worker共享槽路径已接通。`record_search_diagnostics=true`只增加每根metadata，默认关闭，默认配置序列化及旧fingerprint保持兼容。三组200局冻结采集＋detach/分组clip短训的原始数据与有限结论见[实验报告](../../artifacts/entity_attention_chance_training_probe_20261007/README.md)；现有hybrid长训配置仍为旧教师。

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
| `batch_buckets` | 可选静态batch档位，要求compiled且覆盖B上限；补齐行不进入搜索或逻辑位置统计，省略时仍动态B |
| `compile_mode` | PyTorch模式或项目native专用`native-cudagraph`（default编译后捕获固定设备输入，要求静态B/reused）；[实际GPU对照](../reports/entity_attention_execution_speed_zh-CN.md) |
| `compile_failure_policy` | `fallback` 或 `fail_fast`；必须检查实际 worker_compile，不能仅由配置声称全程 compiled |
| `native_transport` | `reused` 复用缓冲，或 `packed` 参照传输 |
| `pipeline_slots` | native CUDA单进程与pool均支持1/2；pool的2是独立owner输入/输出缓冲，与worker `shared_slots`独立 |

编译图按模型、policy/value入口、candidate宽度和可选静态batch档位隔离Dynamo code object缓存。2026-10-05修复了此前 `functools.partial` 共用PyTorch包装函数缓存、采集后首次评测触发重编译上限的问题；不再用扩大共享 `recompile_limit` 代替隔离，`fail_fast` 仍会明确报告单个图的真实编译失败。事故与恢复命令见[混合训练指南](entity_attention_hybrid_zh-CN.md)。

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
