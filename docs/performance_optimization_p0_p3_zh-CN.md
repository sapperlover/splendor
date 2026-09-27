# P0–P3 训练、评测与搜索性能优化

> 状态：已实现并通过回归测试
>
> 日期：2026-09-21
>
> 适用范围：当前代码；不追溯改写已完成的 5M 运行和旧 preflight 数据

## 1. 结论

P0–P3 只改变调度、推理位置和引擎热路径的重复工作，没有改变游戏规则、Observation、模型特征、PPO 目标、对手混合比例、paired-seed 语义或 promotion 规则。

当前建议是：

- 普通 5C 评测使用 `evaluation_workers=8`；
- 新建训练配置使用 `multiprocessing_local_opponents`，`4 workers × 8 envs`；
- 搜索评测优先从 `--device cpu --workers 4` 或 `8` 起步，不把 worker 数当成越多越好；
- 真正的 CUDA learner 长跑吞吐仍需在可用 GPU 环境做短 preflight，本轮不声称已完成 CUDA 加速验收。

## 2. P0：paired evaluation 多进程并行

### 2.1 实现

- 以一个 paired seed 的两局换先手比赛为调度单位，不拆散 pair-level 统计；
- 使用 `spawn` 长寿命进程池，同一面板的多个对手可复用 worker；
- checkpoint 在每个 worker 中加载一次后缓存；
- worker 中的 PyTorch intra-op 和 inter-op 线程均限制为 1，避免进程与线程双重超额订阅；
- `evaluation.py` 的固定对局评测在 `workers > 1` 时使用 CPU 推理；单 worker 路径可选择 CPU 或 CUDA。训练期 `evaluate_5c` 的 worker 规则保持不变。

5C 训练配置的 `evaluation_workers` 取值范围是 1–12，新 `prepare` 默认为 8。固定评测 JSON 配置中的 `workers` 可设为正整数，但更大的值不一定更快。

### 2.2 短基准

同一 checkpoint、20 paired seeds / 40 games、`top-k=2`、`depth=2`、CPU 的单次短基准：

| workers | 墙钟时间 |
|---:|---:|
| 2 | 9.340 s |
| 4 | 5.732 s |
| 8 | 5.700 s |
| 12 | 5.544 s |
| 14 | 5.521 s |

4 之后边际收益已很小；14 workers 相比 4 workers 只快约 3.7%，却会占用更多 CPU 和模型内存。这只是短基准，不是通用的 worker 上限结论。

## 3. P1：单进程对手批处理和 critic-only

### 3.1 对手批处理

`BatchedRolloutCollector` 不再让每个 learner action 后的神经对手立即单独前向，而是：

1. 先执行同一轮 learner batch；
2. 收集所有等待对手的环境；
3. 按对手模型实例分组；
4. 同组的 latest/historical 请求合并成一次 ragged batch 前向；
5. 每局仍使用自己独立、可恢复的抽样 RNG。

固定启发式对手直接执行，不进入神经 batch。单进程 5C collector 路径默认开启该机制；`active_envs` 和同时抽到同一模型的环境数决定实际 opponent batch，没有需要用户单独调节的“对手 batch size”参数。

### 3.2 critic-only

rollout 边界 bootstrap 改用 `ActorCritic.value_only()`，只编码 state 并运行 critic，不再编码合法候选或运行 actor。这不改变 value 输出语义。

## 4. P2：worker-local CPU 神经对手

### 4.1 实现

`MultiprocessRolloutCollector` 现在支持两种 opponent inference backend：

- `central`：旧实现，worker 把每个神经对手决策发回主进程；
- `worker_cpu`：新实现，worker 在本地 CPU 完成神经对手推理。

worker-local 路径的约束是：

- historical checkpoint 在 worker 初始化或历史池变更时加载到 CPU；
- latest 权重每个 rollout 同步一次，rollout 内冻结；
- 对手每个动作不再经过 Queue 请求/响应；
- 中央 GPU 只负责 learner batch 和 PPO update；
- 每个 worker 的 PyTorch CPU 线程固定为 1；
- 对手身份、版本和 RNG 仍然按 episode 固定，checkpoint/resume 语义不变。

### 4.2 短基准

CPU、512 learning decisions、单次短基准：

| 配置 | worker-local | legacy central |
|---|---:|---:|
| 单进程 32 envs | 471.0 decisions/s | 465.3 decisions/s |
| 1 worker × 32 envs | 334.6 | 159.3 |
| 2 workers × 16 envs | 519.0 | 246.5 |
| 4 workers × 8 envs | **600.2** | 271.9 |
| 8 workers × 4 envs | 480.8 | 323.7 |

在该基准中，4×8 worker-local 是 legacy central 4×8 的约 2.21×，是同轮单进程对照的约 1.27×。8 workers 已经回退，说明 worker 数不是越多越好。

这组数字只有一次重复、决策数较小，且当时 CUDA 不可用。它足以证明取消逐动作 IPC 的方向有效，但不能代替真正长 rollout + CUDA learner 的 preflight。

### 4.3 当前配置

新执行 `python -m splendor.train_5c prepare` 时，默认生成：

```text
collector_backend = "multiprocessing_local_opponents"
num_workers = 4
envs_per_worker = 8
active_envs = 32
evaluation_workers = 8
```

`num_workers * envs_per_worker` 必须等于 `active_envs`。已完成的 `runs/5c_main_seed1/config.json` 仍然是 `single_process`，不会被新默认值静默改写，resume 也会拒绝冻结配置不匹配。

## 5. P3：引擎和评测热路径

已实现：

1. `DecisionRequest.legal_actions` 作为权威候选，执行时复用引擎合法动作缓存；每次成功状态转移后失效。
2. `GameEngine.current_state_digest` 缓存当前不可变 state 的 digest，下一步复用上一步的 after digest。
3. `GameEngine.fork()` 共享已验证的父 state/replay/digest/合法候选，不重复扫描父状态，但每个新子状态仍执行 invariant check。
4. evaluation runner 使用 `lean=True`，跳过只在压力测试需要的 coverage/checkpoint reservoir 统计。
5. cycle signature 直接从权威 `GameState` 字段生成稳定 SHA-256 摘要，不再每个决策构造大段 JSON。
6. 失败时的 seed、动作前缀、完整 state、replay 重建和 failure scene 仍然保留。

关键短基准：

- 5,000 次 cycle signature：旧 JSON + SHA 约 0.320 s，新直接摘要约 0.079 s，约 4.06×；
- 50 局 random runner：full 约 2.450 s，lean 约 2.348 s，约快 4.4%；
- 20 paired seeds / 40 games 的浅搜索：墙钟从 5.732 s 降至 5.550 s（约 -3.2%），合计搜索时间约 -3.9%；评测结果逐字段一致。

这些收益大小取决于搜索深度、合法动作数和评测对局数，不应线性外推到长训练。

## 6. 正确性与验收

新增或扩展的测试覆盖：

- 串行与 pair-parallel 评测结果一致；
- 对手 batch 不改变 episode RNG 和 learner trajectory 语义；
- worker-local historical/latest 模型生命周期、同步和可复现性；
- worker 的 PyTorch CPU 线程为 1；
- 合法动作只在状态变化后重算；
- fork 不重验父状态，但会验证子状态；
- lean/full runner 产生相同结果、replay 和 final digest；
- failure scene 仍可从 seed、state 和 replay prefix 重建。

完成 P0–P3 后的全量测试结果为 `248 passed, 2 skipped`；两个 skip 均因当时环境无可用 CUDA。

## 7. 历史文档的阅读方式

- [5C Multiprocessing Collector Preflight](work_package_5c_multiprocessing_preflight_zh-CN.md) 记录的是 legacy `central` backend，其数据和当时结论保留。
- [5C 现行协议](work_package_5c_current_protocol_zh-CN.md) 中的单进程 collector 描述对已完成的 `5c_main_seed1` 仍然正确。
- [5M 完成报告](work_package_5c_5m_training_report_zh-CN.md) 的耗时和吞吐是当时实际运行数据，P0–P3 不应被倒推到该次训练。
