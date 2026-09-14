# 工作包 2D：完整规则引擎集成验证报告

## 1. 验收结论

**工作包 2D 已通过验收。** 在正式加入“无合法主行动时 forced skip，双方连续 forced skip 则平局”的本地规则后，seed `0..9999` 的 10,000 局 RandomPolicy 完整对局全部进入正式终局。engineering truncation、IllegalAction、invariant failure、replay mismatch 和 save/restore mismatch 均为 0，且未发现已知隐藏信息泄漏。

机器可读的最终统计见 [`../artifacts/work_package_2d/stress-10000-rules-v2.json`](../artifacts/work_package_2d/stress-10000-rules-v2.json)。规则集为 `splendor-base-two-player-v2-forced-skip`，replay schema 为 `2`。

## 2. Forced skip 的正式实现

- 权威规则层只在 `MAIN_ACTION` 且重新枚举的合法候选为空时允许 forced skip；仍有任何候选时调用会立即失败。
- 没有新增 `PassAction` 或可编码的 skip 动作。`DecisionRequest.legal_actions` 仍只包含真正的玩家选择；无选择时策略和未来模型不会收到请求。
- `GameState.consecutive_forced_skips` 是唯一新增的权威状态。一次 forced skip 仅增加当前玩家已完成回合数、设置连续计数并转交行动权；token、卡牌、贵族、市场、牌堆与 RNG 不变。
- 任意合法正常主行动成功执行时，规则层在进入后续 PAYMENT、RETURN_TOKENS 或回合收尾前清零连续计数。
- 中间无正常主行动的第二次 forced skip 生成 `reason=consecutive_forced_skips` 的正式 `TerminalResult`，`winner_ids=(0, 1)`，不按当前分数判胜。
- forced skip 以 `ForcedSkipEvent(action=None, automatic=True)` 记入 replay；事件携带参与者、阶段和转换前后状态 digest。存档同时保存连续计数、完整状态与 RNG。

## 3. 原 198 个死锁现场复验

对旧验证目录中全部 198 个完整现场重新从 seed 开局，按 replay 顺序重放玩家动作和自动贵族事件。每个历史动作在当时仍属于权威合法集，全程 invariant 成立。复验分类为：

| 原审计类别 | 数量 | v2 结果 |
|---|---:|---|
| A：仅当前玩家无动作 | 182 | 自动记录 1 次 forced skip，对手获得正常 `DecisionRequest` 并继续 |
| B：双方均无动作 | 16 | 自动记录 2 次 forced skip，立即正式平局 |
| C：实现/状态错误 | 0 | 无 |
| D：其他 | 0 | 无 |

代表案例 seed 44 不再产生“非终局无合法动作”异常：当时玩家被强制跳过一次，对手随后完成正常主行动，计数清零，最终按正常 final round 终局。回归测试还固定了较短的 A 类 seed 7155 路径和 B 类 seed 2639 路径。

## 4. 10,000 局压力统计

参数：游戏 seed `0..9999`，首玩家由每局游戏 RNG 确定，双方 RandomPolicy 使用与游戏 RNG 隔离的确定性 seed，工程 watchdog 为 400 个已完成玩家回合。

| 项目 | 数值 |
|---|---:|
| 总局数 | 10,000 |
| 自然/正式结束 | 10,000（100%） |
| forced skip 事件 | 242 |
| 双边连续 forced skip 平局 | 17 |
| engineering truncation | 0 |
| 异常局 | 0 |
| IllegalAction | 0 |
| invariant failure | 0 |
| replay mismatch | 0 / 132 个完整对局样本 |
| save/restore mismatch | 0 / 72 个决策边界样本 |
| 玩家 0 胜 | 4,900 |
| 玩家 1 胜 | 5,067 |
| 平局 | 33（正常完全平手 16，forced-skip 平局 17） |
| 平均已完成玩家回合数 | 74.6217 |
| 中位已完成玩家回合数 | 74 |
| 最大已完成玩家回合数 | 100 |
| 平均策略决策数 | 118.2573 |

阶段事件次数：

| 阶段 | 次数 |
|---|---:|
| MAIN_ACTION | 746,217（其中 242 个为非策略 forced skip） |
| PAYMENT | 339,053 |
| RETURN_TOKENS | 96,518 |
| CHOOSE_NOBLE | 15,357 |

主行动执行次数：

| 主行动 | 次数 |
|---|---:|
| TakeDistinct | 282,984 |
| TakePair | 23,487 |
| ReserveMarket | 80,355 |
| ReserveDeck | 20,096 |
| BuyMarket | 294,870 |
| BuyReserved | 44,183 |

其他统计：贵族取得 15,357 次；final round 触发 9,983 次。正常 final-round 终局中，高分胜 9,835 局、同分后以较少发展卡获胜 132 局、完全平局 16 局；另有 17 局 forced-skip 平局。全部阶段、六类主行动、多支付候选、贵族、forced skip、final round 和 terminal 均出现在 replay 覆盖中。

## 5. Replay、save/restore 与 invariant

- 132 个完整对局重放同时校验数据 fingerprint、规则集、RNG 算法、seed、first player、动作/自动事件序列、每步权威状态 digest 与终局结果，0 mismatch。
- 72 个恢复续跑样本覆盖 `MAIN_ACTION`、`PAYMENT`、`RETURN_TOKENS`、`CHOOSE_NOBLE`、final round pending 和 forced skip pending。恢复后的权威状态、RNG、后续合法候选、replay 和最终结果全部一致，0 mismatch。
- 引擎在初始状态和每次 Action/forced-skip 提交后执行 invariant。token、卡牌和贵族位置守恒；阶段与 pending 一致；分数、bonus、购卡数、贵族资格和终局结果从权威持有物重算。forced-skip 活跃中间状态只能是计数 1 的无 pending `MAIN_ACTION`，对应平局终局必须是计数 2。
- 400 回合阈值仍只在 runner 中用作工程防挂，不进入 `GameState`、不产生 `TerminalResult`、不改变正式胜负。

## 6. 隐藏信息与集合顺序验收

metamorphic 与 contract tests 验证：

- 对手盲保留牌与同等级未知牌交换身份后，观察者的 Observation 和合法候选公开语义不变；持有者自己的私有观察仍可区分真实牌。
- 改变未来未知牌堆顺序不改变当前 Observation 或候选；`ReserveDeck` 只暴露等级。
- 对手公开保留牌持续显示身份；盲保留只显示来源、等级和未知标记。
- `public_purchased` 恰好是双方已购发展卡联合集，不含公开或盲保留，也不由真实牌堆补集推导。
- 同等级市场、贵族和各方保留区的无意义重排不改变 Observation 语义或候选语义；内部 handle 仍命中原物件。
- 候选不暴露市场槽位/索引，排序不使用对手盲牌 ID 或未来牌堆 ID。

Actor/Critic 的置换不变性与神经网络张量编码仍留给模型工作包。

## 7. 有限 engine audit

- `legal_actions` 和 forced-skip 前置条件共用 `splendor.rules` 的同一套规则派生；RandomPolicy 不复制合法性，apply 仍重新枚举并拒绝非法动作。
- `ForcedSkipEvent` 不继承、不冒充 `Action`；公开动作类型、阶段候选映射和 Policy 签名都无 skip。
- PAYMENT、RETURN_TOKENS、CHOOSE_NOBLE、forced skip、统一回合收尾和两种终局原因均由显式状态与 invariant 区分，没有根据缓存或当前分数猜测阶段。
- final round 已触发时，最后应答玩家如果 forced skip，该回合仍被正常补齐，并按原 final-round 胜负规则结算。
- 没有静默重开、随机换动作、放宽 invariant、自动归还 token 或其他 fallback。异常路径仍保存 seed、完整私有状态、RNG、replay 前缀、阶段和当前权威候选后立即失败。
- 游戏随机性仍只来自版本化的 engine RNG；策略和验证抽样 RNG 独立且不写入 `GameState`。

审计未发现新的规则实现错误。此前 2D 验证层已修复“把异常 replay 前缀当成完整对局抽样”和“异常现场沿用上一个决策边界候选”两个验证工具问题。本次的 198 个无动作状态已证实为规则死锁，不是引擎 bug；本次变更是经人工裁决的正式规则增补。

## 8. 测试与后续边界

已运行全部单元/集成/metamorphic tests 和全部 Python 编译检查，并完成上述 10,000 局正式压力验收。新增回归覆盖：单方跳过后对手继续；双方连续跳过平局；有合法动作时绝不跳过；正常主行动清零计数；forced skip 前后 save/restore；replay 精确一致；seed 44 正式终局。

当前已知局限只是明确留给后续工作包的功能：启发式 AI、行为克隆、PyTorch/PPO、神经网络 Observation 张量编码、Actor/Critic 置换不变性、搜索、GUI、联网和 3～4 人模式均未实现。它们不属于工作包 2D，本阶段在完成引擎验收后停止。
