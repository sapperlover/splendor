# 精确输入缓存复核（2026-10-22）

状态：**只完成了正确口径的命中率与 GPU 图计时；没有实现生产缓存，也没有 cache off/on 的 A/B**。因此本目录给出的是**可行性上界**，不是实际缓存加速结论。

## 1. 为什么重做

上一版用 `forward_seconds` 推 GPU 计算占比是错的：`forward_seconds` 是异步 GPU 路径上的**主机提交/启动计时**，完成可能落在 `wait_seconds`。上一版还混用了 `gpu_owner.inference` 的累计值和 `clock.common_seconds` 的窗口值，并用 Python eager/B256 微基准代替生产 native/compiled/B500。那些数字不再使用。

## 2. 当前混合预算的真实命中率

数据：R20 attention 模型，96 个既有真实根 × 2 个搜索种子，分别跑 **S128N1** 和 **S256N4**，每次搜索从空缓存开始；只观察 exact input，不替换响应。

| 搜索 | 请求数 | 不同输入 | ready-hit | potential（含在途） |
|---|---:|---:|---:|---:|
| S128N1 | 23,894 | 22,722 | **4.905%** | 4.905% |
| S256N4 | 188,814 | 152,553 | **16.291%** | 19.205% |

按 R20 完整轮的模拟量加权：高预算 S256N4 占 **72.87%**、低预算 S128N1 占 **27.13%**，得到当前混合预算的请求加权：

- **ready-hit ≈ 13.20%**
- 含在途重复的 potential ≈ **15.33%**

旧的 15% 只来自 S1024N1/S256N4，未包含当前占 75% 决策的 S128N1；重测后 S128N1 只有 4.9%，不能再用旧的 15% 直接代表当前采集。当前 13.2% 是**每次搜索空缓存**的保守值，不含跨根、跨局或全局缓存。

## 3. 生产 CUDAGraph 计时（B500）

不再直接调用 `compiled._dispatch`。使用真实 `NativeBatchRunner` + `NativePackedTransfer`，在生产路径 `NativePackedTransfer._direct_graph`（[search_inference.py](../../src/splendor/search_inference.py#L589-L613)）的 `graph.replay()` 外加 CUDA event；每个宽度用 1000 个并发搜索 job 产生 B500 batch，取 6–7 个 steady B500 的中位数。表中每批 500 行：

| width | attention graph | attention total | MLP graph | MLP total |
|---:|---:|---:|---:|---:|
| 16 | 6.793 ms | 9.901 ms | 2.313 ms | 5.015 ms |
| 32 | 6.634 ms | 9.676 ms | 2.085 ms | 4.974 ms |
| 48 | 6.948 ms | 9.599 ms | 1.865 ms | 4.712 ms |
| 64 | 6.703 ms | 9.725 ms | 2.134 ms | 4.883 ms |

`graph` 是生产 CUDAGraph replay 的 event；`total` 是 `begin_policy_value` 到 `finish`，包含 H2D + graph replay + D2H，但不含 worker/owner 的 shared-memory gather/dispatch。

按真实宽度分布加权：

- attention 完整轮宽度分布：16 0.002%、32 1.60%、48 74.94%、64 23.45%，加权 **graph 6.885 ms/B500（13.77 µs/行）**、total 9.630 ms/B500（19.26 µs/行）；
- attention 60 秒 tuned 窗口：32 64.1%、48 34.4%、64 1.5%（其余宽度为 0），加权 graph 6.743 ms/B500（13.49 µs/行）；
- MLP 60 秒 tuned 窗口：32 41.6%、48 48.8%、64 9.6%，加权 **graph 1.983 ms/B500（3.97 µs/行）**、total 4.838 ms/B500（9.68 µs/行）。

**GPU graph 占墙钟**（只乘 graph，不乘 total；total 的 H2D/D2H 与 GPU 可重叠，直接乘会超过 100%）：

- attention 完整轮：13.77 µs/行 × 52,548 行/s = **72.4%**；
- attention 60 秒 tuned：13.49 µs/行 × 60,372 行/s = **81.4%**；
- MLP 60 秒 tuned：3.97 µs/行 × 126,768 行/s = **50.3%**。

这是生产 CUDAGraph 的 graph replay 投影，不是 `nvidia-smi` 的逐时刻 SM 占用，也不是完整 4000 局的逐批积分。

## 4. 缓存上界（不是实际收益）

用生产 CUDAGraph 的 graph-only 时间，假设每个 ready-hit 完全省掉该行 graph replay、且 key/合并/LRU 成本为零：

- attention 完整轮：13.20% × 72.4% ≈ **9.6%** 墙钟；
- attention 60 秒 tuned：13.20% × 81.4% ≈ **10.8%** 墙钟；
- MLP 60 秒 tuned：13.20% × 50.3% ≈ **6.6%** 墙钟。

这是**乐观上界**，不是实际加速：

- owner 侧缓存只省 graph replay；H2D、gather、dispatch 仍要支付；
- 若缓存放在 worker 侧、连 H2D 一起省，上界会更高，但实现和验证都更复杂；
- 没有计 key 构造、LRU、命中行合并、miss 行补齐；Python key 8.83 µs/行约 0.46 CPU 核当量，未换算墙钟；
- 每次搜索空缓存；跨根/跨局全局缓存可能更高，也可能因容量/污染更低；
- 没有真实 cache off/on，不能判断部分命中、合批和调度影响。

## 5. 结论

**缓存不能再被判定为“不值得”。** 正确生产路径下，当前混合预算的 per-search ready-hit 约 **13.2%**；attention 的 graph 占墙钟约 **72–81%**，缓存上界约 **9.6–10.8%**；MLP graph 只占约 **50%**，上界约 **6.6%**。是否真正提速，必须实现生产路径缓存后做：

1. 同 seed、同规则的 cache off/on 60 秒窗口；
2. 完整 4000 局 A/B（由用户执行）；
3. 用生产 CUDAGraph / CUDA events 计时，而不是 host forward 或直接 `compiled._dispatch`；
4. 分别记录 S128N1/S256N4 命中率和跨局缓存命中。

## 5.1 对性能结论的影响

- **实际吞吐、CPU 秒/核数、owner gather/dispatch**：不受 CUDAGraph 口径影响，仍可直接使用；
- **GPU graph 占比、graph 时间比、缓存上界**：现在使用生产 CUDAGraph B500 重算；旧 direct-dispatch 数字（15.01/3.58 µs、78.9%/45.3%、10.4%/6.0%）全部作废；
- **FLOPs 比**：不受影响，仍为 attention/MLP ≈ **5.82×**；
- **端到端 2.1×**：实际采集吞吐，仍成立；
- **MLP “GPU 约 45%”**：更新为生产 CUDAGraph graph 约 **50.3%**；CPU 仍约 42%；瓶颈仍是 owner gather/dispatch 与 worker 供给。

## 6. 规模与限制

- 命中率：96 根 × 2 seed × 2 面板 = 384 次搜索；每次搜索独立空缓存，不跨根/局。
- GPU graph：生产 `NativePackedTransfer` CUDAGraph replay；B500、width16/32/48/64；每宽度 6–7 个 steady B500 事件；按真实宽度分布加权。
- 没有生产缓存实现，没有 cache off/on，没有完整 4000 局；已改用生产 CUDAGraph B500 计时，但仍不是完整轮逐批积分。
- 单机 RTX 4050 Laptop 6GB；未测其他模型/硬件/批形状。
- 根来自既有搜索稳定性实验的 96 个真实根，不是本轮 4000 局的重采样；结论限定这些根和当前 R20 模型。

## 7. 产物

- 命中率原始汇总：[results/hit_rate_summary.json](results/hit_rate_summary.json)
- 逐根结果：[results/per_root.json](results/per_root.json)
- 生产 CUDAGraph 计时：[attention width16](results/production_graph_attention_w16.json) / [w32](results/production_graph_attention_w32.json) / [w48](results/production_graph_attention_w48.json) / [w64](results/production_graph_attention_w64.json)；[MLP width16](results/production_graph_mlp_w16.json) / [w32](results/production_graph_mlp_w32.json) / [w48](results/production_graph_mlp_w48.json) / [w64](results/production_graph_mlp_w64.json)
- 机器可读复核：[results/summary.json](results/summary.json)
- 复现脚本：[scripts/probe.py](scripts/probe.py)、[scripts/production_graph_timing.py](scripts/production_graph_timing.py)

复现命中率：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python runtime/cache_recheck_20261022/run.py run
conda run -n splendor env PYTHONPATH=src \
  python runtime/cache_recheck_20261022/run.py summarize
```

复现生产 CUDAGraph B500 计时：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python runtime/cache_recheck_20261022/production_graph_timing.py \
  --checkpoint runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt \
  --output runtime/cache_recheck_20261022/production_graph_attention_w48.json \
  --width 48 --jobs 1000 --simulations 8
```
