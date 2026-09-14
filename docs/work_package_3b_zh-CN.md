# 工作包 3B：Observation 与合法候选的数值编码

## 1. 完成范围与边界

本工作包新增纯 NumPy 的 `encode_decision(DecisionRequest) -> EncodedDecision` 和 `collate_encoded_decisions(...) -> EncodedDecisionBatch`。编码器只导入公开 action/contract/model DTO，不导入 `GameState`、engine、rules、RNG 或 replay，不枚举、筛选或修改合法动作。

特征 schema 版本为：

```text
splendor-observation-action-v1
```

正式规则引擎、Observation 构造、3A 的 Random/Score/Development 策略与评测逻辑均未修改。本阶段没有 PyTorch、网络、logit、softmax 或训练实现。

## 2. 固定顺序与总维度

- 普通颜色：`white, blue, green, red, black`
- token 颜色：`white, blue, green, red, black, gold`
- 等级：`1, 2, 3`
- 决策阶段：`MAIN_ACTION, PAYMENT, RETURN_TOKENS, CHOOSE_NOBLE`
- action type：`TakeDistinct, TakePair, ReserveMarket, ReserveDeck, BuyMarket, BuyReserved, Pay, ReturnTokens, ClaimNoble`
- source：`market, deck`

| 特征 | 维度 | 构成 |
|---|---:|---|
| card | 29 | valid 1 + known 1 + tier 3 + bonus 5 + source 2 + points 1 + cost 5 + discounted need 5 + ordinary deficit 5 + gold-uncovered deficit 1 |
| noble | 17 | valid 1 + points 1 + requirement 5 + self bonus gap 5 + opponent bonus gap 5 |
| global | 148 | bank 6 + deck 3 + self 14 + opponent 14 + public_purchased 90 + first-player 1 + phase 4 + final-round 2 + turn context 4 + forced-skip context 1 + pending context 9 |
| action | 82 | phase 4 + type 9 + source 2 + tier 3 + take 6 + payment 6 + return 6 + target card 29 + target noble 17 |

`global` 的玩家块各包含六色 token、五色 bonus、score、已购发展卡数和贵族数。turn context 包含双方已完成回合数、差值和是否相等；pending context 包含是否存在 pending、导致 pending 的六种主行动 one-hot、待归还 token 数和当前符合贵族数。PAYMENT 的目标卡另用共享 card feature 保存。

## 3. 单个决策的 shape

| 字段 | shape | dtype |
|---|---|---|
| `global_features` | `[148]` | `float32` |
| `market_t1/t2/t3` | 各 `[4, 29]` | `float32` |
| `market_t1/t2/t3_mask` | 各 `[4]` | `bool` |
| `self_reserved` / `opponent_reserved` | 各 `[3, 29]` | `float32` |
| 两个 reserve mask | 各 `[3]` | `bool` |
| `nobles` | `[3, 17]` | `float32` |
| `nobles_mask` | `[3]` | `bool` |
| `pending_card` | `[29]` | `float32` |
| `pending_card_mask` | scalar | `bool` |
| `candidate_features` | `[N, 82]` | `float32` |
| `candidate_mask` | `[N]`（全真） | `bool` |

市场、保留卡和贵族的行只是容器，没有 slot 特征。三层市场始终保持独立集合，未 flatten 成固定位置向量。

## 4. Batch 与候选 padding

单个 `EncodedDecision` 保留实际候选数 `N`，不长期分配 252 行。对 batch size `B`，设该 batch 内最大候选数为 `M`：

```text
candidate_features: [B, M, 82] float32
candidate_mask:     [B, M]     bool
```

只有 batch collator 对不足 `M` 的样本补全零行，并将 mask 设为假。其他固定集合在 batch 中分别为 `[B,4,29]`、`[B,3,29]` 和 `[B,3,17]`。测试已验证单候选、普通多候选，以及真实引擎路径中达到 51 个合法 RETURN_TOKENS 候选的大集合；实现不包含 252 或其他全局固定稠密大小。

## 5. 数值缩放

所有数值只除以固定尺度，不执行 clip。因此超出常用基准的 score、bonus 或回合数会得到大于 1 的值，不会丢失信息。

| 数值 | 除数 |
|---|---:|
| 六色 token | `(4, 4, 4, 4, 4, 5)` |
| 三层 deck count | `(40, 30, 20)` |
| card cost / need / per-color deficit | `7` |
| card points | `5` |
| noble points | `3` |
| noble requirement / gap | `4` |
| 总未被黄金覆盖的购买 gap | `14` |
| player score | `15` |
| player per-color bonus | `18` |
| purchased development count | `90` |
| noble count | `3` |
| completed turns | `40` |
| completed-turn delta | `1` |
| consecutive forced skips | `2` |
| pending tokens to return | `3` |
| pending eligible nobles | `3` |

可见卡牌的辅助特征仅为：

```text
discounted_need[c] = max(0, cost[c] - self_bonus[c])
ordinary_deficit[c] = max(0, discounted_need[c] - self_token[c])
gold_uncovered = max(0, sum(ordinary_deficit) - self_gold)
```

贵族辅助特征是双方 `max(0, requirement - bonus)`。它们全部由 Observation 计算并跟随所属物件，没有 ScorePolicy/DevelopmentPolicy 分数、推荐动作或其他启发式特征。

## 6. Unknown、padding 和隐藏信息

- 已知卡：`valid=1, known=1`，编码 tier、source 和全部公开卡牌属性。
- 对手盲保留：`valid=1, known=0, source=deck`，只有 tier/source 有值；bonus、points、cost 及全部辅助特征为零。
- padding：整行为零且 mask 为假，因此不可能与未知真实保留牌混淆。
- 自己的盲保留从 `self_private.blind_reserved_cards` 编码完整属性；对手不会读取这个私有列表。
- 90 位 `public_purchased` 逐位直接复制 Observation，不读取卡表、真实牌堆或其补集。
- `ReserveDeck` 候选仅编码 tier/source，不查询或预测堆顶。

## 7. Action 映射与 ID 隔离

`candidate_features[i]` 只含公开数值语义。Buy/ReserveMarket 使用可见目标卡的共享 29 维 card feature，ClaimNoble 使用共享 17 维 noble feature；Pay/ReturnTokens 只编码引擎已枚举的实际六色向量。

稳定 card/noble ID 只在编码时用于将候选与当前可见物件对应，不进入 NumPy 特征。`EncodedDecision.candidate_actions` 是索引到原始引擎 action 的非数值映射层；`EncodedDecisionBatch` 故意不包含 action 对象。未来模型只消费 batch 数组，选出候选 index 后再用原单样本的 `candidate_actions[index]` 取回精确执行对象。

编码不包含市场/reserve/noble slot、容器 index、card ID、noble ID、seed、RNG、replay 或 debug 字段。

## 8. 验收测试

新增 19 个 3B 测试，覆盖：

- schema 版本、名称唯一性、固定维度、shape、`float32`/`bool`、无 NaN/Inf、空集合和 padding；
- 单候选、实际多候选、51 个真实合法归还候选与 batch-local padding/mask；
- 全部 9 种 action 及 MAIN_ACTION/PAYMENT/RETURN_TOKENS/CHOOSE_NOBLE 四个真实引擎决策阶段；
- 不 clip 超基准 score/bonus，以及 card need/deficit 和 noble 双方 gap 的精确值；
- 对手盲保留卡与同等级未知牌交换后，Observation、全部数值块、候选特征和 mask 逐元素相同；拥有者的私有编码仍能区分身份；
- 未知未来牌堆重排不改变编码或 `ReserveDeck` 候选；
- 公开保留、盲保留和真正购买三种状态中，仅真正购买设置 `public_purchased`；
- 市场、贵族和双方 reserve 重排后 feature multiset 不变，指向同一物件的 candidate feature 不变；
- 两组语义相同但执行 ID 不同的 card/noble action 特征完全相同；
- self/opponent 资产随观察者交换，全局特征没有 player-0/player-1 座位语义；
- encoder 的 import 边界与 3A heuristic 对 3B 无依赖。

全部旧测试、3B 测试和 Python 编译检查通过。3A 基线的引擎、策略和评测路径未发生行为变化。

## 9. 尚未实现

工作包 4 的 PyTorch tensor 转换、card/noble MLP、Deep Sets pooling、Actor、Critic、logits、softmax、合法候选 mask 的网络应用、rollout、奖励、GAE、PPO、optimizer、检查点、行为克隆和 opponent pool 均未实现。

本阶段在稳定数值 contract 和测试后停止。
