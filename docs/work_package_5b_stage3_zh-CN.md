# 工作包 5B Stage 3：historical opponent pool infrastructure

## 范围

本阶段只实现对手池基础设施和小规模正确性验证，没有启动 self-play 训练，也没有制定最终历史模型保留算法。

`OpponentPool` 支持：

- 每个 rollout 开始时由当前 learner 深拷贝得到的 frozen latest snapshot；
- 从训练 checkpoint 加载的 historical PPO snapshot；
- ScorePolicy、DevelopmentPolicy 和可选 RandomPolicy；
- 最多 12 个 historical checkpoint；
- historical manifest 原子保存/恢复；
- 独立、版本化的 pool RNG 状态保存/恢复；
- 按 ID 加载特定 checkpoint，或从 historical 集合中确定性抽样。

## 冻结与信息边界

latest 使用参数独立的 `deepcopy`；historical 则新建 ActorCritic 并严格加载 checkpoint `state_dict`。两者都设为 `eval()` 且所有参数 `requires_grad=False`，不共享 learner 的可变参数，也不进入当前 PPO gradient。

神经对手仅通过 `GreedyActorCriticPolicy.choose(DecisionRequest)` 接收 Observation 和合法候选，没有 GameState 入口。`LearningEnvironment` 仍只保留固定 learning player 的 transition，opponent 决策只推进权威引擎。每局创建的 policy 对象持有固定 snapshot，不会在局中切换版本。

## checkpoint 和 manifest 验证

加入 historical pool 时立即校验：

- training checkpoint format version；
- rules version；
- feature schema version；
- ActorCritic model config 和严格 `state_dict`；
- training step。

manifest 记录 opponent ID、kind、checkpoint path/ID、training step、schema、model config、加入原因、加入 step/时间和 pool RNG snapshot。恢复时重新检查 manifest 与实际 checkpoint 是否一致；不兼容值明确失败，不做 fallback。latest 的 manifest metadata 可与恢复后的当前 learner 一起重建冻结快照。

## 验证

单元/集成测试覆盖：

- learner 修改后 latest snapshot 参数不变；
- latest manifest metadata 恢复；
- 两个 historical checkpoint 加入、列出、manifest round-trip；
- pool RNG 恢复后后续抽样序列完全一致；
- 外部固定 seed 对 historical 选择可重现；
- schema 不兼容 checkpoint 明确拒绝；
- historical neural opponent 完成整局且不进入 learning trajectory；
- 两个神经 checkpoint 成对换先手对局。

另外使用真实 `step_000100005.pt` 和 `step_000157351.pt` 做了 2 对/4 局 infrastructure smoke：4 局全部自然结束，engineering truncation、IllegalAction 和 invariant failure 均为 0。此结果不作为棋力评测。

针对性 pool/checkpoint/training-environment 回归为 `37 passed, 1 skipped`；完成 Stage 3 后的全量回归为 `186 passed, 2 skipped`。本阶段未启动 Stage 4 self-play。
