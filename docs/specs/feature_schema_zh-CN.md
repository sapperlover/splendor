# Observation 与合法候选特征契约

Schema：`splendor-observation-action-v1`。入口为 `features.encode_decision` / `collate_encoded_decisions`。编码器只读取公开 DTO，不枚举、筛选或修改合法动作；当前 Python 路径使用向量化 NumPy，native 编码与其做差分核对。

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

只有 batch collator 对不足 `M` 的样本补全零行，并将 mask 设为假。其他固定集合在 batch 中分别为 `[B,4,29]`、`[B,3,29]` 和 `[B,3,17]`。支持单候选、普通多候选和真实大候选集合；实现不包含 252 或其他全局固定稠密大小。

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

稳定 card/noble ID 只在编码时用于将候选与当前可见物件对应，不进入 NumPy 特征。`EncodedDecision.candidate_actions` 是索引到原始引擎 action 的非数值映射层；`EncodedDecisionBatch` 故意不包含 action 对象。模型只消费 batch 数组，选出候选 index 后再用原单样本的 `candidate_actions[index]` 取回精确执行对象。

编码不包含市场/reserve/noble slot、容器 index、card ID、noble ID、seed、RNG、replay 或 debug 字段。


## 回归依据

`tests/test_features_3b.py`、`test_features_golden_3b.py`、`test_features_metamorphic_3b.py` 和 `test_native_batch.py` 分别覆盖数值、冻结 golden、隐藏信息/集合语义和 native 差分。Golden 文件位于 `tests/fixtures/feature_encoding_golden.json`；schema 变化必须同时核对 checkpoint 和自对弈数据兼容性。
