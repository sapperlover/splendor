# 工作包 2B：购买与 PAYMENT 纵向切片

本阶段在工作包 2A 之上实现公开市场牌和自己保留牌的完整购买路径。工作包 1 冻结的数据契约没有改动：购买目标仍由 `BuyMarket(card_id)` / `BuyReserved(card_id)` 表示，`PendingTurn.main_action` 保存目标，实际支付仍由六色 `Pay(TokenVector)` 表示。

## 实际状态路径

```text
MAIN_ACTION -- BuyMarket / BuyReserved --> PAYMENT
    （只保存目标；不扣 token、不移动卡牌、不补市场）

PAYMENT -- Pay --> 扣除实际支付并归还 bank
                --> 目标卡进入当前玩家已购区
                --> 公开市场购买时补同等级牌（空堆不补）
                --> 完成本回合
                --> 下一玩家 MAIN_ACTION
```

`MAIN_ACTION` 只为至少存在一种合法支付的目标生成购买候选。`PAYMENT` 重新根据当前权威状态和待购买目标枚举候选；`GameEngine.apply` 仍只接受当前 `legal_actions` 的精确成员，不修补非法支付。

## 支付集合

规则实现集中在 `splendor.rules.legal_payments_for_card`。加成从玩家已购卡重算，随后逐色计算：

```text
need[c] = max(0, cost[c] - bonus[c])
0 <= g[c] <= need[c]
need[c] - g[c] <= held[c]
sum(g[c]) <= held_gold
ordinary_paid[c] = need[c] - g[c]
```

枚举保留全部合法黄金分配，包括玩家持有对应普通色时自愿改用黄金。动作保存最终交回 bank 的五色普通 token 和黄金总数；相同实际六色支付只出现一次，不同实际支付不会合并。候选按固定六色向量排序，以保证 save/restore 和 replay 的确定性。

## 权威状态与信息边界

- 付款时只移动 token、目标卡和必要的市场补牌；卡牌分数、五色加成及已购数量仍不缓存，始终从 `purchased_card_ids` 和规范数据重算。
- 市场购买与保留牌购买均按稳定卡牌 ID 执行，不依赖市场槽位。
- 购买盲保留牌前，真实卡牌只在拥有者私有观察中出现；购买后才进入双方公开已购卡和 `public_purchased`。
- `Observation` 的 PAYMENT 上下文只包含当前玩家已经知道的目标卡；策略入口仍只有 `DecisionRequest(observation, legal_actions)`，不暴露 `GameState`。
- save 文件保存 PAYMENT 阶段和目标，replay 分别记录同一玩家的目标选择和支付事件，因此一个真实回合中的两次决策可以无损恢复和复演。
- 支付后 token 守恒和卡牌唯一位置由 invariant 从权威持有物检查；PAYMENT invariant 还会重新验证目标仍在正确位置且至少有一个合法支付。

## 测试覆盖

- 独立小规模 exhaustive oracle 对照完整支付集合，并单独覆盖无黄金、部分黄金、全黄金、自愿使用黄金、多种支付、恰好支付及不可购买。
- 购买候选的可支付性过滤，公开市场牌和两种来源的自己保留牌，以及禁止购买对手保留牌。
- 选择目标时无资产变化、付款后 token 回 bank、卡牌移动、市场补牌和空牌堆不补牌。
- token 守恒、卡牌位置唯一、分数/加成由已购卡重算。
- PAYMENT 中间阶段的 Observation、save/restore、继续执行和同 seed replay 一致；同一真实回合的两条事件保持同一行动者。
- 非候选支付立即失败、状态不变并保存可重放现场。

## 工作包 2C 边界（已由 2C 解决）

2B 完成时尚不领取贵族，也不触发或结算终局。购买是第一种可能新产生贵族资格或达到终局分数的行动；当时的 2B 引擎会在该边界明确停止，避免静默跳过规则。

当前 2C 已用统一回合结算接管该边界：从权威加成重算贵族资格、处理单候选自动领取及多候选选择、在领取后检查终局、补齐双方回合数，并生成包含全部平局分支的结果。购买和支付规则没有复制。详见 [`work_package_2c_zh-CN.md`](work_package_2c_zh-CN.md)。
