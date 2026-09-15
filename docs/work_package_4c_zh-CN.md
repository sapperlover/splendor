# 工作包 4C：Rollout、GAE、PPO 更新与最小学习闭环

## 1. 完成范围

本工作包在不修改 3B 特征契约、4A 网络结构和 4B 环境语义的前提下，新增了一条最小完整学习路径：

```text
LearningDecision
  -> EncodedDecision
  -> ActorCritic Categorical 采样
  -> LearningEnvironment.step_learning_action
  -> CPU RolloutSample
  -> GAE / return
  -> ragged-candidate PPO minibatch update
  -> 原 ActorCritic 参数就地更新
  -> 完整训练 checkpoint
```

代码集中在 `splendor.ppo`。策略采样器只接收 `EncodedDecision`，不接收 `GameState`；动作仍由 4B 保存的 candidate index 映射回原 `DecisionRequest` 中的引擎 action。本阶段仅使用正式终局奖励 `win=+1 / loss=-1 / draw=0 / non-terminal=0`，没有 reward shaping。

## 2. PPOConfig

| 字段 | 默认值 |
|---|---:|
| `gamma` | `1.0` |
| `gae_lambda` | `0.95` |
| `learning_rate` | `3e-4` |
| `clip_coef` | `0.2` |
| `value_coef` | `0.5` |
| `entropy_coef` | `0.01` |
| `max_grad_norm` | `0.5` |
| `ppo_epochs` | `4` |
| `minibatch_size` | `512` |
| `target_kl` | `0.02` |
| `rollout_steps` | `8192` |
| `advantage_epsilon` | `1e-8` |

`gae_lambda` 可直接配置为 `0.95` 或 `1.0`。测试可缩小 rollout、minibatch 和 epoch，实现中没有散落的另一套超参数。

## 3. 采样与 RolloutSample

`PolicySampler` 在 `candidate_mask` 限定的 `Categorical` 分布中用概率采样，不做 argmax。采样 RNG 是与游戏 RNG、对手 RNG 分离的 CPU `torch.Generator`。它保存 candidate index、当时的 `old_logprob` 和 `old_value`。单候选时概率为 1、log-probability 为 0。

`RolloutSample` 字段为：

- `encoded`：完整原始 NumPy `EncodedDecision`，包括 candidate feature/mask 及非数值执行映射；
- `chosen_candidate_index`、`old_logprob`、`old_value`；
- `reward`、`terminated`、`truncated`；
- `episode_id`、`learning_player`；
- `bootstrap_value`。

没有保存 card embedding、state embedding 或其他旧神经表示。PPO 每个 minibatch 都用当前网络从原始特征重新前向计算。rollout 数据驻留 CPU，只有当前 minibatch 进入 CPU 或 CUDA 模型设备。

`RolloutCollector` 只记录固定 learning player 的 transition，对手决策由 4B 环境内部推进。一次 `collect` 中不调用 optimizer，测试逐参数确认采样前后网络未变。batch 在对局中途截止时，保存 4B 的真实同学习方后继观察的 value，并在 collector 中保留未完对局供下一批继续。

## 4. GAE 语义

对每个同学习方时间步：

```text
bootstrap_t = 0                         if terminated_t
              saved V(real next obs)   otherwise

delta_t = reward_t + gamma * bootstrap_t - old_value_t

adv_t = delta_t + gamma * lambda * adv_(t+1)
```

最后一项只在下一条样本同属一个 `episode_id` 和 `learning_player`，且当前样本不是 terminal/truncation 边界时连接。因此：

- 正式 terminal 不 bootstrap，不读取 reset 后的新局观察；
- engineering truncation 使用 4B 保存的真实 `next_decision` value，但 advantage recursion 不跨局；
- rollout 数量边界同样使用真实后继 value，不装作 terminal；
- `returns = advantages + old_values`。

优势在整个 rollout 上用 population variance 标准化；分母加 epsilon，单样本得到有限的 0，return 不标准化。手算测试已覆盖 terminal win/loss/draw、truncation bootstrap、batch boundary 及 `lambda=0.95/1.0`。

## 5. PPO 更新

Actor 采用标准 clipped surrogate：

```text
ratio = exp(new_logprob - old_logprob)
policy_loss = -mean(min(
    advantage * ratio,
    advantage * clip(ratio, 1-clip_coef, 1+clip_coef)
))
```

单元测试分别校验 `ratio=1`、正优势超上界、负优势超下界和负优势向错误方向变化时的 clipping。

- value loss：`MSE(new_value, return)`，不做 value clipping，critic 输出仍无 tanh；
- entropy：只对 `num_valid_candidates > 1` 的样本求平均并加入 bonus，单候选样本不稀释探索统计；
- total loss：`policy + value_coef * value - entropy_coef * entropy`；
- optimizer：Adam，每个 minibatch 清梯度、backward、检查 finite、记录未裁剪 norm、用 `max_grad_norm` 裁剪后 step；
- ragged candidate：每个 minibatch 调用 3B collator，仅 padding 到该 minibatch 局部最大候选数；
- 每个 epoch 用独立 shuffle generator 重排样本，`old_logprob/old_value` 不修改；
- approximate KL：`mean((ratio - 1) - logratio)`；超过 `target_kl` 时停止剩余 minibatch/epoch，已执行的 Adam state 保留；
- clip fraction：`abs(ratio - 1) > clip_coef` 的样本比例；
- explained variance 在 return 方差近 0 时返回 0，不产生 NaN。

`PPOUpdateStats` 输出 policy/value/entropy/total loss、approx KL、clip fraction、平均未裁剪 gradient norm、explained variance、epoch/minibatch 数、样本数、entropy-active 样本数、发生梯度裁剪的 minibatch 数及 KL early-stop 标记。NaN/Inf 会立即失败，不跳过更新。

## 6. 训练 checkpoint

checkpoint 格式版本为 1，保存：

- model config 和 `state_dict`；
- PPO config 和 Adam `state_dict`；
- feature schema version；
- global learning decision step 和 PPO update count；
- Python、NumPy、PyTorch CPU RNG state；
- CUDA RNG states（CUDA 可用时）；
- minibatch shuffle RNG state；
- 必传的 `PolicySampler` 独立 RNG state；
- 调用方提供的 training metadata。

加载时验证 checkpoint 格式和 feature schema，重建 model/Adam 并恢复全部计数和 RNG。CPU 确定性续训测试中，保存前后的下一次 Python/NumPy/Torch/采样 RNG 结果、下一轮 PPO 统计、模型参数、optimizer state 和计数严格一致。

## 7. 一步胜负和座位对称验收

测试构造了符合全部 authoritative invariant 的最终轮回应局面。当前学习方有且仅有两个正式 `MAIN_ACTION` 候选：

- `BuyReserved("055") -> PAYMENT`，付款后从 14 分到 16 分，正式终局获胜，reward `+1`；
- `TakeDistinct(white)`，正式完成最后回应回合但仍为 14 分，负于对手 15 分，reward `-1`。

两条路径都经过正式 `GameEngine -> Observation -> legal_actions -> 3B encode -> 4A ActorCritic -> 4B transition -> GAE -> PPO`；没有向模型直接提供正确标签。每个座位使用 6 轮、每轮 128 个学习决策的小型 CPU 验收，实测为：

| learning player | 必胜动作概率 | 必败动作概率 |
|---:|---:|---:|
| 0 | `0.5045 -> 0.9999` | `0.4955 -> 0.00008` |
| 1 | `0.4943 -> 0.9998` | `0.5057 -> 0.00017` |

这是管线方向性验收，不是棋力或长训结论。

## 8. Smoke test 与短性能 sanity

CPU 和 CUDA smoke 都使用 Random opponent 的真实环境数据，完成采样、GAE、PPO update，确认 loss/梯度有限、模型参数真实发生变化，且没有 `IllegalAction` 或 invariant failure。CUDA case 在普通工具沙箱外、同一 `splendor` Conda 环境的 RTX 4050 Laptop GPU 上通过。

短基准为单进程 FP32，未使用 multiprocessing、AMP、`torch.compile` 或自定义优化。该数字仅用于定位后续瓶颈：

| 项目 | 实测 |
|---|---:|
| CPU 环境推进（含随机对手） | `602.63 learning decisions/s` |
| CUDA model distribution，batch 32 | `1.754 ms/batch`，`18,245 decisions/s` |
| GAE，256 samples | `0.077 ms` |
| CUDA PPO update，256 samples / 4 epochs / minibatch 128 | `0.281 s` |
| CUDA peak allocated memory | `52.64 MiB` |
| 256-sample rollout NumPy numeric payload | `1.50 MiB` |

数值 payload 是所有 `EncodedDecision` NumPy array 的 `nbytes` 合计，不含 Python object 管理开销，因此是粗略下界。当前短基准显示 CPU 规则环境/逐决策收集是比 GAE 更明显的后续性能关注点；本阶段没有因此做性能重构。

## 9. 测试和未实现内容

新增 4C 测试覆盖：GAE 全部边界、PPO clip 手算、小 batch 稳定性、采样与 old log-probability 重建、采样期参数冻结、单候选 entropy 排除、ragged minibatch 更新、KL early stop、CPU/CUDA smoke、完整 checkpoint 续训以及两座位正式一步胜负学习。全部旧测试与 4A/4B 测试保持通过。

本阶段没有实现 reward shaping、历史 opponent pool、正式长期 self-play、多进程 sampler、CUDA inference server、行为克隆、ComprehensivePolicy、MCTS/search、长期训练、调度器或大规模超参搜索。工作包 4C 在最小学习闭环验收后停止。
