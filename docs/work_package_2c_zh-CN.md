# 工作包 2C：贵族、完整回合结算与正式终局

本阶段把工作包 2A 的非购买行动和工作包 2B 的购买/PAYMENT 接入同一套回合收尾，使双人基础版对局能够从 seed 开局自然运行到正式终局。工作包 1 冻结的 `GameState`、`PendingTurn`、五个 `DecisionPhase`、动作联合和 `TerminalResult` 已足够表达规则，没有新增状态字段或阶段。

公共引擎的最小 API 调整是：`GameEngine.apply(action)` 在非终局时仍返回下一份 `DecisionRequest`，产生正式终局时直接返回 `TerminalResult`；终局状态不会构造成策略请求。

## 1. 最终状态机路径

```text
MAIN_ACTION
  ├─ BuyMarket / BuyReserved ─> PAYMENT ─> Pay ───────────────┐
  └─ TakeDistinct / TakePair / ReserveMarket / ReserveDeck ───┤
                                                               │
                        token > 10 ─> RETURN_TOKENS ────────────┤
                                                               v
                                               [统一贵族资格检查]
                                  ┌────────────────┼────────────────┐
                                  │                │                │
                               无候选           一个候选          多个候选
                                  │          [自动 ClaimNoble]       v
                                  │                │          CHOOSE_NOBLE
                                  │                │                │
                                  └────────────────┴───────< ClaimNoble
                                                   │
                                                   v
                                      [终局触发检查并完成回合]
                                  ┌────────────────┴────────────────┐
                                  │                                 │
                         尚未满足正式结算                    已触发且回合数相同
                                  v                                 v
                      下一玩家 MAIN_ACTION                      TERMINAL
```

规则层只有一个 `_begin_turn_settlement` 入口。非购买行动在无需归还时进入它；归还完成后进入它；支付完成并取得卡牌后也进入它。贵族取得后直接进入唯一的 `_finish_turn`，不会再次检查贵族，因此同一真实回合至多取得一块。

## 2. 贵族结算

- 资格由 `eligible_noble_ids` 根据当前玩家 `purchased_card_ids` 派生的五色 bonus 与场上贵族要求重新计算；token、黄金和缓存值均不参与。
- 无候选时跳过贵族阶段并继续回合收尾。
- 一个候选时，规则层先形成合法 `ClaimNoble` 候选；引擎从同一 `legal_actions` 结果自动执行，记录 `ActionEvent(automatic=True)`。
- 多个候选时保持当前玩家和真实回合不变，进入 `CHOOSE_NOBLE`；合法集合包含全部符合贵族，玩家必须选其中一个。
- 执行后贵族从 `nobles_in_play` 移到当前玩家 `noble_ids`。分数仍从已购发展卡和已取得贵族重算。

## 3. Final round 判定

终局判断只存在于统一 `_finish_turn`：

1. 当前玩家完成支付/归还/贵族等全部结算；
2. 增加该玩家的 `completed_turns`；
3. 若尚未触发 final round，按包含本回合贵族后的权威分数检查 `score >= end_game_prestige`，首次满足时固定 `final_round_triggered_by`；
4. 已有触发者时绝不覆盖该字段；
5. 只有 `final_round_triggered_by is not None` 且双方 `completed_turns` 相等时生成正式结果，否则切换到下一玩家。

因此先手首次触发后，完成回合数为“先手比后手多一”，后手仍执行本轮最后一个回合；后手首次触发时，增加其完成回合数后双方已经相等，立即正式结算。达到 15 分不会提前停止 PAYMENT、贵族选择或该真实回合的其他结算。

## 4. 胜负和平局

正式结果从最终权威持有物构造：

- `scores` 为发展卡分数加贵族分数；
- `purchased_card_counts` 只取 `len(purchased_card_ids)`，不包含贵族或保留牌；
- 先比较分数；同分时发展卡较少者胜；仍相同则 `winner_ids == (0, 1)`，表示共享胜利。

非终局状态没有 `TerminalResult`。终局 invariant 会重新计算双方分数、发展卡数量、胜者和完成回合数并与结果逐项比较。

## 5. Replay、恢复与异常

- 同一真实回合中的主行动、PAYMENT、RETURN_TOKENS、玩家贵族选择分别保留事件；单贵族自动领取也保留独立自动事件。
- replay 按一次玩家决策及其紧随的自动事件组复演，逐项核对动作、行动者、阶段、自动标记和转换前后状态摘要。
- `CHOOSE_NOBLE` 与 final round 已触发但后手尚未完成的状态均可 save/restore 后继续。
- `GameEngine.apply` 在 `TERMINAL` 中没有合法候选；任何继续行动都会作为非法动作失败并保存完整现场，不添加 `pass` 或静默重开。

## 6. 新增测试和关键边界

- 无贵族、恰好一个贵族自动领取、多个贵族完整选择集合及只取得一个。
- 拿 token、保留和购买三类路径都进入同一贵族检查；贵族分数重算及贵族唯一位置 invariant。
- 先手 15 分等待后手、后手首次 15 分立即结束、超过 15 分、贵族使分数达到 15。
- final round 后手反超获胜，且触发者不被覆盖、轮次不延长。
- 高分胜；同分时发展卡较少者胜；同分同卡数共享胜利；贵族不计发展卡数量。
- CHOOSE_NOBLE 和 final round 中间状态 save/restore 后结果一致。
- 从真实 seed 开局自然运行完整对局，包含自动贵族领取并 replay 到相同 `TERMINAL` 状态。
- terminal 后拒绝行动；全部既有 2A/2B 测试、守恒与信息隔离测试继续通过。

## 7. 留给工作包 2D

基础双人规则引擎和基础 replay 已完整。2D 尚待实现的内容是随机/启发式基线策略、观察与动作的数值编码、统一对局运行器和评测基础设施。策略仍只能接收 `DecisionRequest`；2D 不应复制合法性、支付、贵族或终局规则。

本阶段没有实现启发式 AI、搜索、GUI、PyTorch、PPO 或训练代码。

## 8. 2D 正式规则增补

2D 死锁审计证明了合法可达的“`MAIN_ACTION` 无候选”状态。后续正式裁决在不增加玩家 `PassAction` 的前提下引入引擎 forced skip，并为两次连续 forced skip 增加平局终局。因此，本文中“没有新增状态字段”及“不添加 pass”是 2C 完成时的历史记录；当前规范以规则集 `splendor-base-two-player-v2-forced-skip` 和工作包 1 更新契约为准。
