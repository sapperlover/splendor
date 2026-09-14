# 工作包 3A：基线策略与统一对战评测

## 1. 完成范围

本工作包固化了统一 `Policy.choose(DecisionRequest) -> Action` 边界，整理现有 `RandomPolicy`，新增固定的 `ScorePolicy` 和 `DevelopmentPolicy`，并实现所有后续策略可复用的 `run_match` 与 paired-seed evaluation runner。现有规则状态机、合法动作枚举、终局和 forced-skip 规则未修改。

策略只读取 `DecisionRequest.observation` 和 `DecisionRequest.legal_actions`，只对已有候选打分。启发式模块不导入 `GameState`、engine、rules、replay 或 RNG，也不重新判断动作是否合法。forced skip 仍是引擎自动事件，不存在策略可选的 Pass/Skip。

## 2. 三种 baseline

### RandomPolicy

- 在当前 `legal_actions` 上均匀随机选择；
- 使用自有的 `splitmix64-v1` RNG，与游戏 RNG 分离；
- 显式接受 policy seed，同 seed 和同一 `DecisionRequest` 序列得到同一动作序列；
- 不包含任何启发式。

### ScorePolicy

`score-v1` 是确定性的短视直接得分基线：

- 可购买行动有明确的高优先级，卡牌分值是主要区分项；
- 无法购买时，按当前可见高分卡的颜色缺口评分拿 token；
- 公开保留按分值、缺口和对手公开分数威胁评分；盲保留只使用等级；
- PAYMENT 仅在合法支付中权衡黄金与未来可见需求；RETURN_TOKENS 优先保留黄金和目标颜色；
- CHOOSE_NOBLE 按贵族分值后使用稳定公开语义打破平分。

### DevelopmentPolicy

`development-v1` 是与 ScorePolicy 公式不同的发展引擎基线：

- 前期奖励低总费用、低等级、补齐弱色以及对场上贵族仍有用的 bonus；
- 卡牌分值权重为 `250 + 280 * min(current_score, 15)`，因此后期自然提高直接 prestige 的重要性；
- 拿 token 面向当前最有发展价值的最多 5 张可见目标；盲保留前期偏低等级、后期偏高等级；
- PAYMENT/RETURN_TOKENS 优先保留开发目标所需颜色及黄金；
- CHOOSE_NOBLE 在分值之外偏向取走对手公开 bonus 距离较近的贵族。

## 3. 冻结权重

ScorePolicy `score-v1`：

| 参数 | 值 | 参数 | 值 |
|---|---:|---|---:|
| buy_base | 100000 | buy_point | 10000 |
| buy_printed_cost | -25 | reserve_market_base | 22000 |
| reserve_point | 4500 | reserve_shortage | -100 |
| reserve_threat_point | 250 | reserve_deck_base | 12000 |
| reserve_deck_tier | 1000 | take_base | 30000 |
| take_color_value | 100 | payment_gold_cost | 500 |
| payment_color_cost | 10 | return_gold_cost | 1000 |
| return_color_cost | 20 | noble_point | 1000 |

DevelopmentPolicy `development-v1`：

| 参数 | 值 | 参数 | 值 |
|---|---:|---|---:|
| buy_base | 60000 | point_early | 250 |
| point_per_current_score | 280 | noble_bonus | 2200 |
| balance_bonus | 500 | cheap_card | 350 |
| early_low_tier | 120 | reserve_market_base | 28000 |
| reserve_value_percent | 55 | reserve_opponent_threat | 350 |
| reserve_deck_base | 15000 | reserve_deck_stage | 800 |
| take_base | 25000 | take_color_value | 120 |
| payment_gold_cost | 700 | payment_color_cost | 12 |
| return_gold_cost | 1200 | return_color_cost | 25 |
| noble_point | 1000 | noble_denial | 400 |

这些参数只存在冻结的 `ScorePolicyWeights` 和 `DevelopmentPolicyWeights` 中，没有散落到规则引擎。短基准证明两者均非明显荒谬后未再调参。

## 4. Deterministic tie-breaking

候选先按稳定公开语义排序，然后取最高启发分；同分保留语义顺序的第一项。稳定键只使用：

- 动作类型；
- 公开或当前玩家已知的卡牌等级、bonus、分值、费用及最终稳定 ID；
- 盲保留的等级，不使用将抽到的卡 ID；
- 拿取颜色、六色支付/归还向量；
- 公开贵族属性及 ID。

不使用市场槽位、集合原排列顺序、未来牌堆或对手盲保留身份。

## 5. Match runner

`run_match(policy_a, policy_b, game_seed=..., policy_seeds=(...), first_player=...)` 复用 2D 的唯一完整对局推进循环。Policy A 固定映射到引擎座位 0，Policy B 映射到座位 1；`first_player` 只指定本局先手座位。

`MatchResult` 包含 pair/game 标识、游戏与策略 seed、策略名、先手、运行状态、正式终局结果、A 视角胜/平/负、座位胜者、分数、已购发展卡数、双方完成回合、策略决策数、forced skip 数、完整 replay、最终状态 digest 与异常现场/错误信息。

工程 watchdog 仍只分类为 `engineering_truncation`，不生成游戏结果；任何策略非法返回仍由正式 engine apply 拒绝并保存现场。

## 6. Paired evaluation runner

`evaluate_policies(policy_a_spec, policy_b_spec, game_seeds, ...)` 对每个 seed 运行两局：

1. A 为座位 0 且先手；
2. A 仍为座位 0，B 为座位 1 且先手。

两局使用相同的 game seed 和各策略 seed，仅修改 `first_player`；因此初始洗牌/贵族配置的 RNG 语义相同，但策略分歧后不强制共享后续摸牌历史。每个 `MatchResult` 保留 `pair_id`、`game_in_pair` 和 game seed，不会把两局当成无关样本。

汇总包含总 pair/局数、有效正式结束、A 胜/平/负、A 先手与后手子统计、`(win + 0.5 * draw) / valid_games`、平均/中位已完成玩家回合、平均决策数、forced skip、工程截断、异常、IllegalAction 和 invariant failure。成对 bootstrap CI 本阶段未实现，但所需 pair 结构已完整保留。

CLI 入口：

```bash
PYTHONPATH=src python -m splendor.evaluate \
  --player-a score \
  --player-b development \
  --seed-start 0 \
  --seed-count 200
```

`--player-a/--player-b` 支持 `random`、`score`、`development`，因此可直接运行要求的四种组合。`--report` 可保存机器可读 JSON，其中每个 pair 保留两局的结果、seed、replay header/事件数及最终状态 digest。

## 7. 短基准结果

开发 sanity 集固定为 game seed `20000..20099`，policy seed base `40000`。每组 100 个 pair，即 200 局；三组均 200/200 正式结束，engineering truncation、异常、IllegalAction 和 invariant failure 全部为 0。

| A vs B | A 胜 | 平 | A 负 | A 计分 | A 先手计分 | A 后手计分 | 平均玩家回合 | 平均决策 | forced skip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Score vs Random | 199 | 0 | 1 | 99.5% | 100% | 99% | 61.25 | 100.320 | 0 |
| Development vs Random | 200 | 0 | 0 | 100% | 100% | 100% | 63.02 | 103.105 | 1 |
| Score vs Development | 94 | 2 | 104 | 47.5% | 48.5% | 46.5% | 60.85 | 100.610 | 0 |

机器可读结果：

- [`../artifacts/work_package_3a/score-vs-random-100-pairs.json`](../artifacts/work_package_3a/score-vs-random-100-pairs.json)
- [`../artifacts/work_package_3a/development-vs-random-100-pairs.json`](../artifacts/work_package_3a/development-vs-random-100-pairs.json)
- [`../artifacts/work_package_3a/score-vs-development-100-pairs.json`](../artifacts/work_package_3a/score-vs-development-100-pairs.json)

这是开发 sanity benchmark，不是独立测试集、棋力承诺或调参依据。两种启发式都明显不是随机行为；二者对战较接近，且定向状态测试中存在稳定的不同类型选择，因此没有为追求某个预设胜率继续调权重。

## 8. 测试与已知局限

新增测试覆盖：

- 两种启发式的 request-only 签名和私有模块隔离；
- 修改对手未知盲保留真实身份不改变两个策略的选择；
- Random/Score/Development 在 MAIN_ACTION、PAYMENT、RETURN_TOKENS、CHOOSE_NOBLE 均只返回候选成员；forced skip 不在候选中；
- RandomPolicy seed 可复现，启发式平分处理不依赖候选输入顺序；
- Score 在明显高分可购卡中优先高分目标，Development 前期偏向低费用/贵族 bonus，两者在固定合法状态中选择不同；
- 同 game/policy seed 完整 match 的结果、replay 和最终 digest 可复现；
- 同 seed 的两局正确交换先后手并保留 pair_id，胜平负和子统计不重不漏；
- Random/Score/Development 多组完整对局的 replay 和现有 2D 恢复/不变式回归继续通过。

本阶段未实现 paired bootstrap 95% CI，也未实现 Observation 数值/tensor 编码、action feature、Deep Sets、Actor/Critic、PyTorch、PPO、行为克隆、opponent pool、搜索、GUI 或任何训练。
