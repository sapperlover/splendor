# 工作包 2A：非购买动作纵向切片

本阶段实现从可复现开局到非购买主行动、必要归还、换人的完整路径。公共入口是 `GameEngine.new(seed, first_player=...)`、`decision()`、`apply(action)`、`save(path)`、`restore(path)` 和 `replay_game(record)`。

## 实际状态路径

```text
MAIN_ACTION
  ├─ TakeDistinct / TakePair ─┬─ token <= 10 ─> 完成回合 ─> 下一玩家 MAIN_ACTION
  │                           └─ token > 10  ─> RETURN_TOKENS
  ├─ ReserveMarket ─> 公开牌移入保留区 ─> 同层补牌 ─> 尽量拿黄金 ─┤
  └─ ReserveDeck   ─> 同层堆顶移入保留区 ─> 尽量拿黄金 ──────────┤

RETURN_TOKENS ─> 精确归还超额数量 ─> 完成回合 ─> 下一玩家 MAIN_ACTION
```

`TakePair` 对应任务描述中的 TakeSame，`ReserveMarket` 对应 ReservePublic，`ReserveDeck` 对应 ReserveBlind；沿用工作包 1 已冻结的类名。

合法候选只由 `splendor.rules.legal_actions` 产生，`GameEngine.apply` 必须验证动作仍是当前候选成员。购买、PAYMENT、贵族和终局动作会抛出 `UnsupportedRuleError`，非法的已实现动作会抛出 `IllegalActionError`；二者都会先保存完整状态和 replay 前缀。

## 信息与恢复边界

- `decision()` 只返回当前玩家的 `Observation + legal_actions`，不返回 `GameState`。
- 公开保留在对手观察中带完整卡牌；盲保留只带等级、来源和未知标记。拥有者的私有观察包含实际牌。
- `public_purchased` 只从两名玩家的 `purchased_card_ids` 构造，绝不扫描牌堆补集。
- 市场、贵族和可见保留牌按公开字段规范化；盲抽候选只含等级，排序不读取堆顶 ID。
- save 文件同时保存完整状态和 replay。完整状态含牌堆顺序、阶段、`PendingTurn` 及 RNG 快照，可以在 RETURN_TOKENS 中恢复。
- replay 头固定数据指纹、规则版本、RNG 算法、seed 和首玩家；事件固定动作及转换前后状态摘要。

## 明确未实现

工作包 2A 不生成或执行购买动作，不进入 PAYMENT，不领取贵族，不检查或产生终局结果，也没有启发式策略、模型或训练代码。完整基础规则仍需工作包 2 的后续切片完成。

