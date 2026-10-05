# Actor–Critic、学习环境与 PPO 契约

模型只消费 [数值特征](feature_schema_zh-CN.md)，不接收 `GameState`、执行 handle、种子或隐藏牌。本文区分纯权重、PPO 训练 checkpoint；AlphaZero checkpoint 和损失见 [AZ 指南](../guides/alphazero_zh-CN.md)。下表为默认模型和基础 `PPOConfig`，具体训练参数由冻结 run config 决定。

## 模型与权重

### 默认 Deep-Sets 网络结构

旧 Deep-Sets 的所有宽度集中在 `ActorCriticConfig`，使用 PyTorch `Linear` 的默认初始化。以下表格和集合聚合描述限于该架构。

| 模块 | 输入 | 结构 | 输出 |
|---|---:|---|---:|
| 共享 CardEncoder | 29 | Linear(29,64) → ReLU → Linear(64,64) → ReLU | 64 |
| 共享 NobleEncoder | 17 | Linear(17,32) → ReLU → Linear(32,32) → ReLU | 32 |
| StateEncoder | 570 | Linear(570,256) → ReLU → Linear(256,256) → ReLU | 256 |
| action representation | 82 拆分 | meta 36 + shared card 64 + shared noble 32 | 132 |
| Actor | 388 | Linear(388,128) → ReLU → Linear(128,1) | 每候选 1 个 logit |
| Critic | 256 | Linear(256,128) → ReLU → Linear(128,1) | 每样本 1 个 value |

派生维度为：

- `state_input_dim = 148 + 5 × (64 + 1) + (32 + 1) + 64 = 570`
- `action_representation_dim = 36 + 64 + 32 = 132`
- `actor_input_dim = 256 + 132 = 388`
- 总 trainable parameter count：`302,626`

Critic 只读取当前玩家视角的 state embedding，不读取候选；输出没有 tanh 限幅。Actor 对每个候选独立应用同一评分网络，没有候选位置参数、slot embedding、attention 或固定 252 维动作空间。

### 集合聚合与共享参数

三个 market tier、self reserve、opponent reserve 分别使用同一个 CardEncoder 后做 masked sum；market 除以固定容量 4，reserve 除以固定容量 3，并分别附加归一化有效数量。noble 使用同一个 NobleEncoder，masked sum 除以容量 3并附加数量。各集合没有 flatten 槽位。

pending card 和 action target card 复用同一个 CardEncoder；场上 noble 和 ClaimNoble target 复用同一个 NobleEncoder。无 pending 时 embedding 为精确零。无 card/noble target 的候选也会在共享编码后显式归零，避免 Linear bias 给不存在的目标产生伪信息。对手未知盲保留仍是 `valid=1, known=0` 的真实物件，会正常编码而不是按 padding 排除。

所有 mask 都在进入编码器前先将无效 raw feature 归零，并在编码后再次排除对应结果。因此即使 mask-false padding 被写入巨大数值，也不能通过溢出或 Linear bias 影响有效输出。

### Candidate mask 与前向接口

`tensorize_batch` 将特征契约中的 NumPy batch 转成 FP32/bool `TorchDecisionBatch`；该 tensor 类型故意不含 action 对象。`ActorCritic.forward(batch)` 返回：

```text
logits: [B, M]
values: [B]
```

`M` 只等于当前 batch 的最大候选数。`masked_logits` 将 padding logit 设为 `-inf`，`distribution` 仅据此构造 `torch.distributions.Categorical`。测试确认 padding 概率严格为 0、有效概率和为 1、单候选概率严格为 1；任一样本没有候选会明确失败，forced skip 仍完全由引擎处理。

### Entity attention v1

`EntityAttentionConfig` 固定 architecture=`entity_attention_v1`、hidden128、heads4、layers2、FFN512、dropout0；共532,738参数。沿用相同输入 schema，将全部全局字段分为双方各14维玩家块和120维上下文，并与市场、保留、贵族、pending 组成25个实体。两层 Pre-LN 残差 self-attention 后，每个候选独立做一层 cross-attention，保留 query 残差；critic 独立读取全局 token，无界输出。角色 embedding 无任意槽位位置，同角色集合置换不变、候选置换等变，unknown 不与 padding 混同。完整前向共享一次状态编码，value-only 不依赖候选。结构及配置见 [专题](../guides/entity_attention_offline_zh-CN.md)。

`build_actor_critic` 按模型配置分派。无 architecture 的旧格式构建 Deep-Sets，明确的 `deep_sets_v1` 同样可读；未知名称报错。新结构保存完整配置，pure weights、AZ build_model 和 frozen loader 均严格加载对应 state_dict。新结构推理必须 `actor_projection=false`，旧结构保留投影优化；混合架构评测分别路由。原 PPO 状态恢复机制仍只适用于原 PPO 训练格式。

### 保存与加载

`save_weights` 只保存：

- 权重格式版本；
- feature schema version；
- model config；
- `state_dict`。

`load_weights` 会验证格式和 schema，并严格加载 state dict。相同输入在 save/load 前后的 logits 和 values 逐元素一致。这不是训练 checkpoint，不含 optimizer、训练步数或 RNG。


## 学习环境

### LearningDecision 与 LearningTransition

`LearningDecision` 保存：

- 单调递增的 `decision_id`，用于拒绝 stale candidate；
- 原始 `DecisionRequest`，包含原顺序的完整合法候选；
- 同一 request 当场生成的 `EncodedDecision`。

`LearningTransition` 保存：

- `episode_id` 和固定的 `learning_player`；
- 当前 `LearningDecision`；
- `chosen_candidate_index`，并可由该索引无歧义取得 `chosen_action`；
- `reward`、`terminated`、`truncated`；
- 非终局时的同学习方 `next_decision`；
- 正式终局时的 `terminal_result`；
- 本步覆盖的 replay event 半开区间；
- 后继权威状态 digest 和双方已完成回合数。

保存的是特征契约中的原始数值特征和候选，不保存 Card/Noble neural embedding。环境适配器不预存 old log probability 或 old value；它们由 PPO 采样器保存。候选索引不重新排序，`EncodedDecision.candidate_actions[index]` 与实际交给引擎的 action 始终一致。

非法索引和过期 `decision_id` 会立即失败，并通过引擎 invariant guard 保存包含完整私有状态和 replay 前缀的可重放现场，不会负索引回绕、换动作或静默恢复。

### Episode 与 timestep 的精确定义

每个 episode 构造时固定 `learning_player ∈ {0,1}`，直到正式终局或工程截断都不会变化。一个 timestep 是：

```text
学习方当前 DecisionRequest
  → 选择一个 candidate index
  → GameEngine.apply
  → 必要时执行冻结对手的全部后续决策
  → 同一学习方下一 DecisionRequest / 正式终局 / 安全截断
```

如果学习方选择购买目标后立即进入自己的 PAYMENT，下一观察就是该 PAYMENT，形成独立 transition；PAYMENT 后进入自己的 CHOOSE_NOBLE 时同样再形成一个 transition。学习方完成真实回合后，对手的一到多个阶段不会写成学习 trajectory 的 timestep。

只有一个合法候选的真实 `DecisionRequest` 仍保留为学习 transition；引擎自动处理的单贵族和 forced skip 不伪造成候选。

### Opponent 自动推进

初始化时若当前座位是对手，或者学习动作结束后轮到对手，适配器执行以下循环：

1. 将当前原始 `DecisionRequest` 交给 episode 内固定的 opponent policy；
2. 将其返回 action 原样交给 `GameEngine.apply`；
3. 继续处理对手可能出现的 PAYMENT、RETURN_TOKENS、CHOOSE_NOBLE 或下一真实回合；
4. 遇到正式 terminal 立即停止；
5. 只有引擎再次返回属于固定 learning player 的 `DecisionRequest` 时才结束环境推进。

对手 policy 对象在一局内不替换。game seed、learning policy seed 和 opponent seed 独立记录；改变 PyTorch global RNG 不会改变使用独立确定性策略 RNG 的 episode replay。

### Base reward 与 terminal 归属

奖励始终相对固定 `learning_player`：

| 后继状态 | reward |
|---|---:|
| 非正式终局 | 0 |
| learning player 正式获胜 | +1 |
| learning player 正式失败 | -1 |
| 正式平局 | 0 |

实现直接使用 `TerminalResult.payoff(learning_player)`，不把座位 0 固定解释为正方。达到 15 分、final round 触发、购买、拿 token、贵族或 forced skip 本身都不发奖励。

学习方结束回合后，如果对手在最终回合购买并反超，环境尚未关闭学习方当前 transition；对手的全部决策完成并产生正式结果后，该 transition 才以 `reward=-1, terminated=true` 收尾，因此不会丢失发生在 opponent turn 的失败奖励。

### terminated、truncated 与 watchdog

- `terminated=true` 只来自正式 `TerminalResult`，包括 final-round 胜负/平局和连续两次 forced skip 的正式平局；终局 transition 没有 `next_decision`，不用于 bootstrap。
- `truncated=true` 只来自工程 watchdog；没有 `TerminalResult`、winner 或额外 base reward，并且 `terminated=false`。

watchdog 以双方完整玩家回合数之和为阈值。阈值即使在学习方完成回合时已经达到，也不会停在 opponent turn 或对手 PAYMENT 中间；环境继续推进，直到拿到同一 learning player 的真实下一 `DecisionRequest`，再把当前 transition 标为 truncated。该真实后继完整保存在 `next_decision`/`bootstrap_decision`，绝不 reset 后拿新局观察替代。

截断 episode 保存非终局 replay 前缀和权威状态 digest；测试通过 replay 重建同一状态，并验证重建后的 learning-player request 等于 bootstrap request。

### Episode runner 输出

`run_learning_episode` 可接收任意两个正式 Policy 实例，输出 `LearningEpisodeRun`，包含：

- episode/game/policy seeds、first player 和 learning player；
- 全部 learning transitions；
- terminal result、terminated/truncated、winner IDs；
- learning/opponent decision count；
- 双方及总完整玩家回合数；
- 完整 replay 和最终状态 digest；
- 截断时的 bootstrap decision。

runner 只把 `DecisionRequest` 传给两方 Policy。相同 game seed、座位、两个 policy seed 和策略动作序列会得到相同 replay、transition 顺序、编码、结果和最终 digest。


## PPO

### PPOConfig

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

### 采样与 RolloutSample

`PolicySampler` 在 `candidate_mask` 限定的 `Categorical` 分布中用概率采样，不做 argmax。采样 RNG 是与游戏 RNG、对手 RNG 分离的 CPU `torch.Generator`。它保存 candidate index、当时的 `old_logprob` 和 `old_value`。单候选时概率为 1、log-probability 为 0。

`RolloutSample` 字段为：

- `encoded`：完整原始 NumPy `EncodedDecision`，包括 candidate feature/mask 及非数值执行映射；
- `chosen_candidate_index`、`old_logprob`、`old_value`；
- `reward`、`terminated`、`truncated`；
- `episode_id`、`learning_player`；
- `bootstrap_value`。

没有保存 card embedding、state embedding 或其他旧神经表示。PPO 每个 minibatch 都用当前网络从原始特征重新前向计算。rollout 数据驻留 CPU，只有当前 minibatch 进入 CPU 或 CUDA 模型设备。

`RolloutCollector` 只记录固定 learning player 的 transition，对手决策由环境适配器内部推进。一次 `collect` 中不调用 optimizer，测试逐参数确认采样前后网络未变。batch 在对局中途截止时，保存环境适配器的真实同学习方后继观察的 value，并在 collector 中保留未完对局供下一批继续。

### GAE 语义

对每个同学习方时间步：

```text
bootstrap_t = 0                         if terminated_t
              saved V(real next obs)   otherwise

delta_t = reward_t + gamma * bootstrap_t - old_value_t

adv_t = delta_t + gamma * lambda * adv_(t+1)
```

最后一项只在下一条样本同属一个 `episode_id` 和 `learning_player`，且当前样本不是 terminal/truncation 边界时连接。因此：

- 正式 terminal 不 bootstrap，不读取 reset 后的新局观察；
- engineering truncation 使用环境适配器保存的真实 `next_decision` value，但 advantage recursion 不跨局；
- rollout 数量边界同样使用真实后继 value，不装作 terminal；
- `returns = advantages + old_values`。

优势在整个 rollout 上用 population variance 标准化；分母加 epsilon，单样本得到有限的 0，return 不标准化。手算测试已覆盖 terminal win/loss/draw、truncation bootstrap、batch boundary 及 `lambda=0.95/1.0`。

### PPO 更新

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
- ragged candidate：每个 minibatch 调用特征 collator，仅 padding 到该 minibatch 局部最大候选数；
- 每个 epoch 用独立 shuffle generator 重排样本，`old_logprob/old_value` 不修改；
- approximate KL：`mean((ratio - 1) - logratio)`；超过 `target_kl` 时停止剩余 minibatch/epoch，已执行的 Adam state 保留；
- clip fraction：`abs(ratio - 1) > clip_coef` 的样本比例；
- explained variance 在 return 方差近 0 时返回 0，不产生 NaN。

`PPOUpdateStats` 输出 policy/value/entropy/total loss、approx KL、clip fraction、平均未裁剪 gradient norm、explained variance、epoch/minibatch 数、样本数、entropy-active 样本数、发生梯度裁剪的 minibatch 数及 KL early-stop 标记。NaN/Inf 会立即失败，不跳过更新。

### 训练 checkpoint

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


## 维护入口

模型：`actor_critic.py`、`entity_attention.py`；离线 AZ：`offline_train.py`；环境：`training_env.py`；PPO：`ppo.py`。相关回归含 `test_entity_attention.py`、`test_actor_critic_4a.py`、`test_actor_critic_invariance_4a.py`、`test_training_env_4b.py`、`test_ppo_4c.py`。5C 的采样、选模、历史池和种子域见 [PPO 训练与评测协议](ppo_training_protocol_zh-CN.md)。
