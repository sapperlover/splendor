# 工作包 5C：Multiprocessing Collector Preflight

## 结论

预检通过，正式 5M **未启动**。

在总 `active_envs=32` 不变时，multiprocessing 方案中吞吐最高的配置是：

- `num_workers=8`
- `envs_per_worker=4`
- 平均 `315.94 learning decisions/s`
- 相对原单进程 32 logical env 的 `341.63 decisions/s`，为 `0.925x`（约 `-7.5%`）
- 两次 8-worker 结果为 `292.68` 和 `339.21 decisions/s`，变异系数 `7.36%`

因此第一版 multiprocessing **不用于正式 5M**；正式配置继续采用原单进程 32 logical env。8 worker 确实把总 CPU 利用率从约 `100%` 提高到 `272%`，但 Python Queue/pickle IPC、主进程调度，以及 neural-opponent 请求通常只能形成约 `1.19` 的小 batch，抵消了并行环境推进的收益。按照本工作包约束，没有继续测试更多 worker，也没有引入共享内存、Ray 或其他复杂框架。

原始结果：`runs/5c_main_seed1/multiprocessing_preflight/benchmark.json`。

## 实现语义

- multiprocessing start method 固定为 `spawn`；
- CPU worker 只持有 `GameEngine`、合法动作、Observation、编码和固定非神经对手；
- worker 不持有 learner 或 opponent 网络，不执行 CUDA 操作；
- learner 与 latest/historical neural opponent 的推理都在中央主进程；
- learner 请求跨 worker 做一个 ragged batch；
- neural opponent 请求按冻结的 opponent model 分组后由中央推理；
- neural opponent 使用固定 worker row、固定 batch size 和按阶段固定的候选容量，机会式合批不会改变数值执行形状或抽样结果；
- candidate index 回传后，worker 仍用原 `DecisionRequest.legal_actions` 中的 handle 执行；
- neural opponent 的动作不会写入 learner trajectory；
- 一个 episode 的 opponent 类型、checkpoint、座位和 RNG stream 全程固定；
- rollout 期间 learner policy version 不变，PPO update 只在 rollout 完成后发生。

## RNG 与 checkpoint/resume

每个 episode 继续由 master training seed 和全局 episode index 确定性派生：

- game seed；
- learning-policy seed；
- opponent-policy seed；
- learning player；
- first player。

opponent mixture RNG 和 historical pool RNG 仍由主进程维护并按原契约保存。每个 worker 另有互不重复的 worker stream，checkpoint metadata 保存：

- collector 版本；
- `num_workers` / `envs_per_worker`；
- next episode index；
- 各 worker RNG snapshot；
- 各 worker 统计。

5C 正式 checkpoint 仍先 drain 到完整 episode boundary，因此 checkpoint 时不存在需要序列化的活跃 `GameEngine`。resume 会拒绝 worker 拓扑变化或非边界 snapshot，并从相同 next episode index、worker RNG、mixture RNG 和 pool RNG 继续。

训练 checkpoint 顶层格式版本没有变化；multiprocessing 状态加入原有可扩展的 `metadata.collector_rng.multiprocess_snapshot`。旧的 512112 source checkpoint 不需要转换。

## 语义测试

新增测试覆盖：

- 单进程 collector 与 spawn collector 在同模型、同 episode/action RNG 下得到相同动作序列和最终权威 state digest；
- P0/P1 两个学习座位均经过多进程路径；
- neural opponent 能完成完整对局，且冻结模型参数不变；
- worker candidate handle 与中央返回 index 一致；
- 0 IllegalAction、0 invariant failure、0 engineering truncation；
- episode-boundary snapshot/restore 后，后续完整对局与不中断执行一致；
- worker RNG streams 唯一且可恢复。

## Benchmark

RTX 4050 Laptop GPU、FP32、同一 `step_000512112.pt`、同一 Phase-A mixture 初态；每个 case 2,048 learning decisions，重复 2 次，并各执行一次相同 PPO update。

| 配置 | decisions/s | games/s | 相对单进程 | 总 CPU 利用率 | learner batch | IPC wait/s（每 case） |
|---|---:|---:|---:|---:|---:|---:|
| 原单进程 × 32 env | **341.63** | **5.34** | **1.000x** | 100% | 31.75 | 0.00 |
| 1 worker × 32 env | 197.73 | 3.09 | 0.579x | 105% | 31.75 | 7.47 |
| 2 workers × 16 env | 242.99 | 3.80 | 0.711x | 151% | 31.75 | 3.95 |
| 4 workers × 8 env | 293.45 | 4.59 | 0.859x | 203% | 31.75 | 2.28 |
| 8 workers × 4 env | 315.94 | 4.94 | 0.925x | 272% | 31.75 | 1.76 |

8-worker case 的平均中央 neural-opponent 实际请求 batch 只有 `1.19`。GPU peak allocated memory 约 `134 MiB`，PPO update 约 `0.39 s`，与 worker 数基本无关。worker 的 `ru_maxrss` 求和约 `13.8 GiB`，其中包含各 spawn 进程映射的共享库页面，不能等同于独占物理内存，但说明第一版 multiprocessing 的内存成本明显高于单进程。

## 正式配置与命令

`runs/5c_main_seed1/config.json` 的正式 backend 保持 `single_process`。若后续仅为调试 multiprocessing，当前最佳拓扑是 `8 × 4 = 32`；它不应覆盖正式 5M 配置。

启动：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.train_5c train \
  --config runs/5c_main_seed1/config.json \
  2>&1 | tee -a runs/5c_main_seed1/logs/train.log
```

恢复：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.train_5c resume \
  --config runs/5c_main_seed1/config.json \
  2>&1 | tee -a runs/5c_main_seed1/logs/train.log
```

本报告完成后停止；没有自动启动 5M。
