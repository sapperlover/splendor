# 相同局面上的精确推理输入缓存潜力

状态：**complete**。768次搜索、1,313,931条请求已核验。单次搜索内潜在命中率：S1024N1 14.62%、S256N4 19.20%、S1024N4 20.22%、随机节点总1024且CPW关闭2.69%；完整方法、阶段结果和限制见[结果报告](results.md)。

范围：用户授权测量 S1024N1、S256N4、S1024N4、随机节点四种搜索的输入复用率；不实现缓存、不替换网络返回值、不做实际测速或棋力对战。

复用稳定性实验的96个未按分歧筛选的基准根：开／中／后期各32个、每根来自不同对局；每组相同两个搜索seed，共768次搜索。阶段表示来源对局真实主动作序列的三等分，不是按分数划分。模型为第20轮原128×2，SHA `fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd`。

| 搜索 | 每世界预算 | 世界数 | 总预算 |
| --- | ---: | ---: | ---: |
| S1024N1 | 1024 | 1 | 1024 |
| S256N4 | 256 | 4 | 1024 |
| S1024N4 | 1024 | 4 | 4096 |
| chance_S1024 | 共享统计池1024 | 1 | 1024 |

随机节点为 `native_chance`，关闭CPW；四组均无根噪声、T0、无强制探索／裁剪，FP32、TF32关闭。采用实际冻结网络前向，CUDA Graph、16个活跃根、B上限500、owner单槽；运行时已有用户GPU进程，不停止它们，耗时与利用率不作性能结论。

## 统计定义

每个根／seed／搜索组独立空缓存，不跨seed、根或搜索模式共享。直接观察NativeBatch输入，以推理入口类型、有效候选数量、完整小端FP32特征与mask生成SHA256；保留候选／实体顺序，去除批大小和候选宽度padding。模型与特征版本由冻结plan命名空间隔离。不对浮点近似相等或集合置换做归一化。

- 总潜在命中率：`(requests - unique_inputs) / requests`，无限容量、允许同批请求合并的精确输入复用上限。
- 已完成结果命中率：请求出现前，之前某一批已返回相同输入的结果。
- 在途重复率：重复请求出现时，该输入此前只有尚未返回的请求；当前同步单槽路径主要是同批重复。需要合并在途请求，普通只查已完成结果的缓存不能直接消除。
- 世界内重复数：每个world内部重复次数之和。
- 跨世界额外重复数：`sum(unique_per_world) - unique_all_worlds`。两项与总重复数严格对账；根前向只算一次，world=-1。

主表按请求量加权，同时保存逐根等权均值／分位数和分阶段结果。随机节点已有信息历史节点共享，`information_set_hits`另存，不能再当作额外网络缓存命中。

本实验只测单次搜索生命周期的复用潜力，不测跨走子缓存；不保证改变批形状后的FP32搜索轨迹逐位相同，不由命中率推出墙钟加速。

## 产物与复现

`run.py`是独立诊断入口，不修改生产代码。`test_probe.py`覆盖padding排除、输入顺序／内容区分、已返回与在途重复、世界分解、命名空间隔离，以及native N1/N4／chance真实小预算搜索启用观察器前后的完整结果一致。

`plan.json`冻结输入／脚本／checkpoint／源码身份；`roots.json`保留完整96根。运行中状态、逐根恢复文件、逐请求指纹流及源码快照写入被忽略的`runtime/search_input_cache_rate_20261021/`；完成后`per_root.json`、`summary.json`、`execution.json`和`results.md`保存必要原始统计、汇总、环境／指纹流SHA和解释。

```bash
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q artifacts/search_input_cache_rate_20261021/test_probe.py
conda run -n splendor env PYTHONPATH=src python artifacts/search_input_cache_rate_20261021/run.py prepare
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u artifacts/search_input_cache_rate_20261021/run.py run
```

准备仅执行一次；中断后原命令`run`跳过已提交逐根结果。只重算现有统计使用`summarize`。已有正式输出不可覆盖为新实验。
