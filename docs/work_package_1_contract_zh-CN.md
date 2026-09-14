# 工作包 1：双人数据契约与阶段机

本文件冻结工作包 2 使用的最小接口。规则数值仍只来自 `data/` 下三个规范文件。2026-09 的死锁审计后，本契约按正式本地规则最小增补 forced-skip 状态与事件，其余冻结契约不变。

## 1. 边界与权威性

- `splendor._state.GameState` 是完整真相，只由规则引擎和故障回放持有。它包含牌堆真实顺序、盲保留牌身份和 RNG 状态。
- 策略接口只有 `Policy.choose(DecisionRequest) -> Action`。`DecisionRequest` 只含 `Observation` 和本次由规则引擎枚举的 `legal_actions`，不含状态引用、随机源或执行回调。
- 策略必须返回候选元组中的原值。规则引擎在应用前再次按当前状态枚举并验证；策略、编码器和对手不得另写合法性判断。
- `splendor.rules` 保存可复用的规则派生函数。后续合法动作枚举也放在这一规则层，invariant、观察构造和状态转移调用它，不能复制公式。
- 没有玩家可选的 `pass`。`MAIN_ACTION` 的权威合法候选为空时，引擎以内部 forced-skip 事件完成回合；它不进入 `DecisionRequest`。其他非终局阶段无候选仍是引擎异常。

Python 无法阻止恶意代码主动导入私有模块，但公开包不导出 `GameState`，正常策略调用链也永远拿不到它。状态对象和所有跨边界 DTO 均为冻结 dataclass，避免引用被意外修改。

## 2. 最小数据契约

### 完整对局状态（引擎私有）

`GameState` 保存：

- 六色公共供应；三个等级分别保存的有序牌堆（元组末端为堆顶）和无槽位语义的公开市场；
- 两名玩家的 token、已购卡 ID、保留牌真实 ID/来源、贵族 ID；
- 场上贵族与本局未使用贵族；二者连同玩家已取得贵族覆盖全部贵族，便于守恒检查；
- 首玩家、当前玩家、双方已完成回合数、连续 forced skip 计数；
- 当前 `DecisionPhase`、`PendingTurn`、最后一轮触发者和可选的 `TerminalResult`；
- 环境 RNG 的完整版本化状态。

分数、五色加成和已购卡数不在权威状态中重复缓存：分别由已购卡/贵族、已购卡和元组长度派生。

### 玩家公开与私有状态

引擎内的 `PlayerState` 是完整持有物。观察侧拆为：

- `PlayerPublicState`：token、派生加成/分数、已购卡、贵族及保留牌的公开描述；
- `PlayerPrivateState`：仅该玩家盲保留牌的完整 `CardView`；
- `PublicReservedCard`：公开保留含完整卡牌；盲保留只含等级和 `deck` 来源，`card=None`。

市场保留的身份一直公开。盲保留身份不会出现在对手公开状态、候选、日志视图或 90 位已购标记中；当它真正被购买后才进入公开已购集合。

### Observation

`Observation` 使用自己/对手相对视角，包含供应、各等级牌堆数量、三组市场卡集合、场上贵族、自己的公开与私有状态、对手公开状态、双方已购卡联合 90 位布尔标记、双方完成回合数、连续 forced skip 计数、先手/最后一轮信息及阶段上下文。

市场、贵族和保留牌字段虽以 tuple 承载，但语义是集合。观察构造时只能按公开语义（例如公开 ID）规范化；未知牌之间不得按隐藏 ID 排序。`public_purchased` 的索引顺序固定为卡牌 CSV 读取顺序。

### 决策阶段、动作与待处理上下文

`DecisionPhase` 只有五值：`MAIN_ACTION`、`PAYMENT`、`RETURN_TOKENS`、`CHOOSE_NOBLE`、`TERMINAL`。

动作采用小型判别联合，而不是一个充满可空字段的大对象：

- 主行动：`TakeDistinct`、`TakePair`、`ReserveMarket`、`ReserveDeck`、`BuyMarket`、`BuyReserved`；
- 支付：`Pay(TokenVector)`，表示实际交回供应的六色 token。相同实际支付只保留一次；
- 归还：`ReturnTokens(TokenVector)`；
- 贵族：`ClaimNoble(noble_id)`。

forced skip 不在这个动作联合中，也没有 `PassAction`。它只能由引擎在 `MAIN_ACTION` 候选确认为空后内部提交。

卡牌和贵族由稳定 ID 定位，不以市场槽位定位。`ReserveDeck` 只含等级，绝不含即将抽到的牌。`PendingTurn` 保存已选择的主行动，并只在对应阶段保存“应归还数量”或“当前符合的贵族 ID”；支付目标可由买牌主行动无损恢复。`PendingObservation` 只给当前玩家呈现可知目标卡和上下文。

`TakeDistinct.colors` 在构造时按规范五色顺序归一化，因此颜色集合的排列不会产生多个动作身份。内部若使用槽位或索引加速，只能留在引擎内部；公共动作中的卡牌 ID 是玩家已知的稳定语义目标，未来动作编码读取目标卡属性而不编码 ID 或槽位。

### 终局结果

`TerminalResult` 保存双方最终分数、已购发展卡数、胜者 ID 元组、双方完成回合数和 `reason`。`reason=final_round` 时按分数/购卡数判定；`reason=consecutive_forced_skips` 时必为 `(0, 1)` 平局。单元素胜者表示独胜；`payoff(player)` 映射为胜 `+1`、负 `-1`、平局 `0`。工程截断不是终局结果。

## 3. 显式阶段转换

概念转换如下；方括号内步骤由引擎自动执行：

```text
MAIN_ACTION --选择购买目标--> PAYMENT --选择实际支付--+
      |                                               |
      +--拿取/保留------------------------------------+
                                                      v
                                   [应用效果；必要时补市场]
                                                      |
                         token > 10 -----------------> RETURN_TOKENS
                                                      |
                                                      v
                                        [重新计算符合贵族]
                                                      |
                  多个候选 --------------------------> CHOOSE_NOBLE
                  一个候选 --------------------------> [自动取得并记日志]
                  无候选 ----------------------------> [跳过]
                                                      |
                                                      v
                                   [终局触发检查；完成回合]
                              +-----------------------+------------------+
                              |                                          |
                    尚未补齐同轮/未触发                         双方回合数相同且已触发
                              v                                          v
                  下一玩家 MAIN_ACTION                              TERMINAL

MAIN_ACTION --合法候选为空--> [forced skip；记录事件并完成回合]
      |                                      |
      |                              连续计数 == 2
      |                                      v
      +--计数 == 1--> 对手 MAIN_ACTION       TERMINAL（平局）
```

只有一个合法动作的决策阶段允许自动选择，但必须记录 `ActionEvent(automatic=True)`；forced skip 另记录为不携带 `Action` 的 `ForcedSkipEvent`。多个合法后继绝不自动裁剪。`state_machine.allowed_next_phases` 描述“下一个可见决策点”，因此允许跨过被跳过或自动执行的中间阶段。任意正常主行动成功执行时，连续 forced skip 计数在规则层清零。

## 4. RNG、replay 与异常现场

- 环境使用 `splitmix64-v1`，快照是 `{algorithm, state, draws}`；洗牌使用确定性的 Fisher–Yates。环境、策略采样和未来对手抽样必须各用独立实例，后两者不得写入 `GameState`。
- replay 头记录 schema 版本、规则集 ID、三个规范数据文件的 SHA-256 指纹、RNG 算法版本、初始环境 seed 和首玩家。
- 每个 `ActionEvent` 记录序号、行动者、阶段、动作、是否自动执行，以及转换前后完整状态摘要。`ForcedSkipEvent` 记录同样的参与者/阶段/摘要，但 `action=None`。完整 replay 记录正式终局结果。
- 状态存档还必须包含当前牌堆顺序、RNG 快照、阶段、`PendingTurn` 和连续 forced skip 计数，所以可从任意决策阶段或一次 forced skip 后恢复。
- replay 使用 schema `2`、规则集 `splendor-base-two-player-v2-forced-skip`；旧规则或旧 schema 不会被静默升级。
- `InvariantGuard` 在发现首个 invariant 失败时，以原子替换写出 JSON：错误码、尝试动作、replay 前缀、完整私有状态及其摘要，然后原样抛出异常。它不修改状态，也不静默开始新局。

## 5. 当前 invariant

基础检查覆盖：恰好两名玩家；非负整数 token；逐色 token 守恒；每张卡恰处于一个位置；每块贵族恰处于一个位置；卡牌等级与牌堆/市场相符；市场、保留区容量；先后手与完成回合数关系；阶段与 pending 上下文一致；只有当前玩家处于归还阶段时可超 10 token；贵族候选由共享规则函数重算一致；终局触发者确实达到门槛；正常 final-round 终局的双方回合数相同，分数、购卡数和胜者可重算一致；forced-skip 活跃中间状态必须为计数 1 的无 pending `MAIN_ACTION`，对应终局必须为计数 2 的平局。

## 6. 文档歧义记录

规则文件第 68 行的“允许把所有可选颜色各拿 1 枚”单独看可能被理解为可选行为；执行大纲第 51 行明确为“只有一至两色则拿所有非空普通颜色”。实现据此把“拿所有剩余非空色”作为唯一合法的不同色拿取组合。工作包 1 未发现其他会改变数据契约或阶段拓扑的未决歧义。
