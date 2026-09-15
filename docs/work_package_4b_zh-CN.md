# 工作包 4B：单学习方环境、Transition 与终止语义

## 1. 完成范围

本工作包新增 `LearningEnvironment` 和 `run_learning_episode`，在不修改正式规则的前提下，把现有 `GameEngine` 适配成固定单一学习座位的 episode。学习方和冻结对手仍只通过 `DecisionRequest → Policy.choose` 边界行动；所有资源变化、forced skip、合法性、阶段转换和终局都继续由 `GameEngine.apply` 处理。

本阶段仅实现 base reward、terminal/truncation 语义和未来 rollout 所需的数据结构。没有实现 reward shaping、return、GAE、PPO loss、optimizer、参数更新、自我对弈池或训练。

## 2. LearningDecision 与 LearningTransition

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

保存的是 3B 原始数值特征和候选，不保存 Card/Noble neural embedding。4B 不预存 old log probability 或 old value；它们留给真正采样模型的后续工作包。候选索引不重新排序，`EncodedDecision.candidate_actions[index]` 与实际交给引擎的 action 始终一致。

非法索引和过期 `decision_id` 会立即失败，并通过引擎 invariant guard 保存包含完整私有状态和 replay 前缀的可重放现场，不会负索引回绕、换动作或静默恢复。

## 3. Episode 与 timestep 的精确定义

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

## 4. Opponent 自动推进

初始化时若当前座位是对手，或者学习动作结束后轮到对手，适配器执行以下循环：

1. 将当前原始 `DecisionRequest` 交给 episode 内固定的 opponent policy；
2. 将其返回 action 原样交给 `GameEngine.apply`；
3. 继续处理对手可能出现的 PAYMENT、RETURN_TOKENS、CHOOSE_NOBLE 或下一真实回合；
4. 遇到正式 terminal 立即停止；
5. 只有引擎再次返回属于固定 learning player 的 `DecisionRequest` 时才结束环境推进。

对手 policy 对象在一局内不替换。game seed、learning policy seed 和 opponent seed 独立记录；改变 PyTorch global RNG 不会改变使用独立确定性策略 RNG 的 episode replay。

## 5. Base reward 与 terminal 归属

奖励始终相对固定 `learning_player`：

| 后继状态 | reward |
|---|---:|
| 非正式终局 | 0 |
| learning player 正式获胜 | +1 |
| learning player 正式失败 | -1 |
| 正式平局 | 0 |

实现直接使用 `TerminalResult.payoff(learning_player)`，不把座位 0 固定解释为正方。达到 15 分、final round 触发、购买、拿 token、贵族或 forced skip 本身都不发奖励。

学习方结束回合后，如果对手在最终回合购买并反超，环境尚未关闭学习方当前 transition；对手的全部决策完成并产生正式结果后，该 transition 才以 `reward=-1, terminated=true` 收尾，因此不会丢失发生在 opponent turn 的失败奖励。

## 6. terminated、truncated 与 watchdog

- `terminated=true` 只来自正式 `TerminalResult`，包括 final-round 胜负/平局和连续两次 forced skip 的正式平局；终局 transition 没有 `next_decision`，不用于 bootstrap。
- `truncated=true` 只来自工程 watchdog；没有 `TerminalResult`、winner 或额外 base reward，并且 `terminated=false`。

watchdog 以双方完整玩家回合数之和为阈值。阈值即使在学习方完成回合时已经达到，也不会停在 opponent turn 或对手 PAYMENT 中间；环境继续推进，直到拿到同一 learning player 的真实下一 `DecisionRequest`，再把当前 transition 标为 truncated。该真实后继完整保存在 `next_decision`/`bootstrap_decision`，绝不 reset 后拿新局观察替代。

截断 episode 保存非终局 replay 前缀和权威状态 digest；测试通过 replay 重建同一状态，并验证重建后的 learning-player request 等于 bootstrap request。

## 7. Episode runner 输出

`run_learning_episode` 可接收任意两个正式 Policy 实例，输出 `LearningEpisodeRun`，包含：

- episode/game/policy seeds、first player 和 learning player；
- 全部 learning transitions；
- terminal result、terminated/truncated、winner IDs；
- learning/opponent decision count；
- 双方及总完整玩家回合数；
- 完整 replay 和最终状态 digest；
- 截断时的 bootstrap decision。

runner 只把 `DecisionRequest` 传给两方 Policy。相同 game seed、座位、两个 policy seed 和策略动作序列会得到相同 replay、transition 顺序、编码、结果和最终 digest。

## 8. 发现并修复的既有编码问题

新增的合法 `BuyReserved → PAYMENT → CHOOSE_NOBLE` 集成测试发现：支付后卡已经离开保留区，但 Observation 仍为 CHOOSE_NOBLE 的 pending context 填充历史 `payment_target`，使 3B 编码器无法再恢复其原保留来源。

最小修复是让 `PendingObservation.payment_target` 只在实际 `DecisionPhase.PAYMENT` 中生成。CHOOSE_NOBLE 仍保留 pending main action 和 eligible nobles，3B schema 与 4A 网络维度不变，正式状态机和规则没有修改。失败路径已有稳定回归测试。

## 9. 核心测试结果

新增 14 个 4B 测试，覆盖：

- MAIN_ACTION → PAYMENT，以及 PAYMENT → CHOOSE_NOBLE 的三个同座位独立 transition；
- 对手 MAIN_ACTION/PAYMENT 多阶段自动推进，不向 learning trajectory 写入 opponent Observation；
- 对手最终回合反超时，learning player 上一 transition 得到 `-1`，并分别验证 learning player 为座位 0 和 1；
- 两个学习座位正式获胜均得到 `+1`；购买目标选择为 0 奖励，正式支付结算后才得到胜利奖励；
- 分数和已购卡数完全相同的 final-round 平局，以及双边 forced-skip 平局，均为 `reward=0, terminated=true`；
- 单合法候选仍形成 transition，forced skip 仅出现在 replay 自动事件中；
- 非法 candidate index、stale decision 明确失败并保存现场；
- 小 watchdog 在同学习方真实后继边界截断，reward 为 0，replay 可恢复同一 bootstrap Observation；
- learning/opponent Policy 都只收到 DecisionRequest；
- 两个 learning seat 的完整随机 episode 可复现，且不受 PyTorch global RNG 影响；
- 多个真实 4B decision 经 3B collate 后送入 4A ActorCritic，ragged candidate mask、logits/value shape 和 Categorical 采样正确；采样索引映射回原 action 后可由正式引擎执行。

全部旧测试、4A 测试和 4B 测试通过，Python 编译检查通过。

## 10. 尚未实现

GAE、discounted return、PPO ratio/clipping、entropy/value loss、optimizer、gradient update、reward shaping、rollout buffer、对手池、自我对弈、多进程 sampler、CUDA inference server、训练 checkpoint 和任何长训练均未实现。工作包 4C 尚未开始。
