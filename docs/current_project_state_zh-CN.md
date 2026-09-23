# Splendor AI：当前项目快照与导航

> 最后更新：2026-09-22
>
> 里程碑：WP5C / 首次 5M learning-decisions 主训练已完成
>
> 当前阶段：分析 5M 模型、完善搜索效率与深度实验、决定下一阶段实验

本页是仓库的当前状态入口。新的 AI agent 应先读本页，再按任务跳转到对应文档和代码。

## 1. 一页结论

项目目前已经具备：

- 通过系统压力验证的双人 Splendor 基础规则引擎；
- 严格隔离私有信息的 Observation + legal-actions 策略边界；
- 数值编码、Deep-Sets Actor–Critic、GAE/PPO、checkpoint/resume；
- Random/Score/Development 策略、冻结神经对手和历史对手池；
- 单进程 multi-env learner/opponent batching，以及 worker-local CPU 神经对手的 multiprocessing collector；
- paired-seed CPU 多进程评测和可复用 worker 池；
- 一次已完成的 5M 纯 terminal-reward 主训练；
- 基于现有 checkpoint 的信息集安全 top-k / critic alpha-beta 搜索，以及向量化特征编码、state-only 叶节点评估、forced-phase folding 和可恢复深度阶梯；
- 完整的训练后分析导出和本地人机对战网页。

当前 DEV 协议下的最优 checkpoint：

~~~text
runs/5c_main_seed1/checkpoints/local_004150000_lifetime_004662112.pt
~~~

它是 **DEV best**，不是经过独立 FINAL 评测和多训练 seed 复验的“最终最优模型”。训练末尾 checkpoint 也不是 best-dev。

## 2. 范围与不可破坏的契约

### 2.1 项目范围

- 只实现本地规则定义的双人基础版 Splendor；
- 不是 Splendor Duel，不包含 3–4 人、扩展或其他游戏模式；
- 当前已有实验性 top-k / critic alpha-beta 搜索，完成了 500 局 search-vs-greedy 和 1,000 局 folded-depth 阶梯；没有 MCTS，也尚未完成多 checkpoint、多训练 seed 或 FINAL 协议复验。

### 2.2 核心契约

1. **规则只实现一次。** <code>GameEngine/rules</code> 是合法性与状态转移的唯一来源。
2. **Policy 只读 <code>DecisionRequest</code>。** 即 Observation + legal actions，不得接触 <code>GameState</code>、牌堆顺序或对手暗保留牌身份。
3. **<code>legal_actions</code> 是唯一候选集。** Policy 不重写规则；内部执行 handle 与模型可见特征分离。
4. **异常立即失败。** 不静默修复、换动作或重开；保存 seed、RNG、state 和 replay 现场。
5. **规则数据从权威持有物重算。** score、bonus、购卡数和贵族资格不依赖易失真缓存。
6. **隐藏信息隔离是 API 契约。** 不得通过 ID、排序、槽位或牌堆补集泄漏。

### 2.3 本地 forced-skip 补充规则

当 <code>MAIN_ACTION</code> 无合法玩家选择时，引擎自动 forced skip：

- 它不是 <code>PassAction</code>，不进入 <code>DecisionRequest</code>，模型不能主动选择；
- 它不改变游戏资源，只结束当前玩家回合；
- 正常主行动成功后，连续 forced-skip 计数清零；
- 双方连续各 forced skip 一次且中间无正常主行动时，正式平局。

当前版本：

- rules：<code>splendor-base-two-player-v2-forced-skip</code>
- replay schema：<code>2</code>
- RNG：<code>splitmix64-v1</code>
- feature schema：<code>splendor-observation-action-v1</code>

## 3. 信息源优先级

文档看似冲突时，按以下顺序判断并报告：

1. [本地双人规则](splendor_standard_rules_zh-CN.md) 和 <code>data/</code>；
2. 当前 contracts/state/rules/engine 代码及通过的测试；
3. 本页对“当前能力”的索引；
4. [5C 现行协议](work_package_5c_current_protocol_zh-CN.md) 和 [5M 报告](work_package_5c_5m_training_report_zh-CN.md)；
5. 按时间形成的各工作包完成文档；
6. [主工作大纲](two_player_ai_workflow_zh-CN.md) 中尚未被实验确认的规划。

早期工作包里的“尚未实现”只描述当时时点，不代表现状。例如 WP2 对训练未实现的声明，已被 WP4/WP5 后续工作取代。

## 4. 里程碑

| 工作包 | 状态 | 主要交付 |
|---|---|---|
| WP1 | 完成 | 数据契约、状态机、RNG、replay、invariant |
| WP2A–2C | 完成 | 非购买、购买/支付、贵族、终局 |
| WP2D | 完成 | 10,000 局压力测试、deadlock audit、forced skip |
| WP3A | 完成 | Random/Score/Development 与统一评测 |
| WP3B | 完成 | Observation/合法候选数值编码 |
| WP4A–4C | 完成 | Actor–Critic、训练环境、Rollout/GAE/PPO |
| WP5A | 完成 | 固定对手短训练，验证纯 terminal reward 可学 |
| WP5B | 完成 | 混合对手、batched collector、历史池、短 self-play |
| WP5C | 完成 | 5M 主训练、promotion/best-dev、训练后分析 |
| 下一阶段 | 未冻结 | FINAL/多 seed、评测效率、容量/算法实验 |

## 5. 当前系统

### 5.1 规则引擎

显式阶段为 <code>MAIN_ACTION</code>、<code>PAYMENT</code>、<code>RETURN_TOKENS</code>、<code>CHOOSE_NOBLE</code>、<code>TERMINAL</code>。

玩家动作类型：

- <code>TakeDistinct</code>、<code>TakePair</code>；
- <code>ReserveMarket</code>、<code>ReserveDeck</code>；
- <code>BuyMarket</code>、<code>BuyReserved</code>；
- <code>Pay</code>、<code>ReturnTokens</code>、<code>ClaimNoble</code>。

所有主行动汇入同一套回合收尾：必要的支付/归还 → 贵族 → 终局检查 → 切换玩家或 terminal。

WP2D 的 10,000 局验收：

| 指标 | 结果 |
|---|---:|
| 正式终局 | 10,000 / 10,000 |
| engineering truncation | 0 |
| 异常 / IllegalAction | 0 / 0 |
| invariant / replay / save-restore mismatch | 0 / 0 / 0 |
| 已知隐藏信息泄漏 | 0 |
| forced skip / 双边 forced-skip 平局 | 242 / 17 |
| 平均 / 最大真实回合 | 74.6217 / 100 |

详见 [WP2D](work_package_2d_engine_validation_zh-CN.md)。

### 5.2 Observation 与特征

| 对象 | 形状/维度 |
|---|---:|
| global | <code>[148]</code> |
| 每层市场 | <code>[4, 29]</code> + mask |
| 自己/对手保留 | <code>[3, 29]</code> + mask |
| 贵族 | <code>[3, 17]</code> + mask |
| pending card | <code>[29]</code> |
| 每个合法候选 | <code>[82]</code> |
| ragged candidate batch | <code>[B, M, 82]</code> + <code>[B, M]</code> mask |

固定普通色顺序为 white, blue, green, red, black；token 向量再加 gold。

- 自己暗保留可见，对手暗保留不显示身份；
- 对手公开保留牌持续可见；
- <code>public_purchased</code> 只表示双方实际购买的发展卡；
- card ID、槽位/保留索引、seed、RNG、replay 计数不进入模型特征；
- action handle 不进入模型；模型返回 candidate index 后由引擎执行原 handle。
- 当前编码器以 NumPy 对整组 card/noble/candidate 向量化；特征 schema 未变，并由改写前捕获的 golden fixture 锁定跨阶段的 bit-exact 输出。
- critic-only 路径使用 <code>EncodedState</code>/<code>EncodedStateBatch</code>，保留 payment target、tokens-to-return 和 eligible nobles 等 pending-turn 上下文，但不构建合法候选特征。

详见 [WP3B](work_package_3b_zh-CN.md) 和 <code>src/splendor/features.py</code>、<code>observation.py</code>。

### 5.3 模型

当前是 302,626 参数的 Deep-Sets Actor–Critic：

- card encoder：29 → 64 → 64；
- noble encoder：17 → 32 → 32；
- pooled state：570 → 256 → 256；
- action representation：132 维；actor：388 → 128 → 1；
- critic：256 → 128 → 1；
- market tier 和 self/opponent reserve 分组保留，组内 masked-sum pooling；
- actor/critic 共享 card/noble encoder；critic 不看候选集。

训练用 categorical sampling；DEV/FINAL/网页对局中的 neural policy 用 greedy argmax。详见 [WP4A](work_package_4a_zh-CN.md)。

### 5.4 训练语义

一个 learning timestep 是从学习方的一个 <code>DecisionRequest</code> 到同一学习方的下一个 request，或 terminal/truncation。PAYMENT、RETURN_TOKENS 和 CHOOSE_NOBLE 可各自形成 transition；对手 transition 不进入 learner trajectory。

5C 实际配置：

| 项目 | 值 |
|---|---:|
| reward / shaping | terminal +1/0/-1 / alpha=0 |
| gamma / GAE lambda | 1.0 / 0.95 |
| PPO clip / value coefficient | 0.2 / 0.5 |
| max grad norm / target KL | 0.5 / 0.02 |
| epochs / minibatch / rollout | 4 / 512 / 8192 decisions |
| optimizer | Adam |

terminal 不 bootstrap；rollout 边界/工程截断使用同玩家 next observation bootstrap；GAE 不跨 episode。详见 [WP4B](work_package_4b_zh-CN.md)、[WP4C](work_package_4c_zh-CN.md) 和 [5C 协议](work_package_5c_current_protocol_zh-CN.md)。

## 6. 5M 主训练快照

### 6.1 实验身份

| 字段 | 值 |
|---|---|
| run ID | <code>5c_main_seed1</code> |
| 起点 | <code>runs/5b_stage4_selfplay_seed_004/checkpoints/step_000512112.pt</code> |
| run-local / 结束 lifetime decisions | 5,000,000 / 5,512,112 |
| 5C / 结束 lifetime PPO updates | 698 / 763 |
| 训练对局 | 110,344 |
| W/D/L | 62,333 / 633 / 47,378 |
| truncation / NaN/Inf / IllegalAction / invariant failure | 0 / 0 / 0 / 0 |
| resume | 1 次 |
| 墙钟时间 | 48,104.68 s（约 13 h 21 min） |

### 6.2 主要 checkpoint 与 Formal DEV

| checkpoint | run-local | vs Score | vs Development | heuristic score | vs 5C 起点 |
|---|---:|---:|---:|---:|---:|
| 起点 <code>step_000512112.pt</code> | 0 | 41.250% | 42.375% | 41.8125% | 50.000% |
| DEV best <code>local_004150000_lifetime_004662112.pt</code> | 4.15M | 70.375% | 78.625% | 74.5000% | 75.375% |
| latest <code>local_005000000_lifetime_005512112.pt</code> | 5.00M | 69.250% | 75.000% | 72.1250% | 78.750% |

4.60M 候选的 heuristic score 为 74.625%，但 500 paired-seed direct promotion 对 incumbent 仅 51.1%，bootstrap 10%/90% 约为 49.1%/53.05%，因此未晋升。全程共 9 次 best-dev promotion。

### 6.3 训练对手

Phase A（0–1M）：

- 30% frozen latest PPO；
- 20% historical PPO；
- 25% ScorePolicy；
- 25% DevelopmentPolicy。

Phase B（1M–5M）：

- 50% frozen latest PPO；
- 30% historical PPO；
- 10% ScorePolicy；
- 10% DevelopmentPolicy。

Historical 总体上约拆为 10% fixed anchors + 20% dynamic history。<code>frozen latest</code> 在 rollout 采样期间冻结，不与 learner 共享可变参数；每局对手身份和版本固定。

### 6.4 评测与选模

- Quick：每约 50k，Score + Development，每个 50 paired seeds；
- Formal：每约 250k，评估上周期 Quick Top 2 + 周期末 checkpoint；
- 每个 Formal candidate 对 Score、Development、起点 512112 各做 200 paired seeds；
- <code>heuristic_score = 0.5 * Score + 0.5 * Development</code>；
- promotion 按 Formal 排名依次挑战 incumbent，用独立 seeds 和 100,000 次确定性 pair bootstrap；
- 先做 200 pairs，临界时扩至总计 500 pairs，使用 10%/90% 分位点；
- historical panel 只诊断遗忘/cycling/non-transitivity，不参与 best-dev 主排名；
- Quick、Formal、Promotion、Historical、FINAL 使用独立 seed domain；FINAL 在训练期间未使用。

### 6.5 训练后观察

- <code>BuyMarket</code> 从早期约 49% 增至后期约 54%–56%，<code>BuyReserved</code> 小幅上升；
- <code>TakePair</code> 后期几乎消失；<code>ReserveDeck</code> 约在 2M 一度升高，后期降至 1% 以下；
- 平均真实回合从早窗口 60.50 降至末窗口 56.43，learner 平均终局分数从 13.56 增至 14.55；
- entropy 约从 1.498 降至 0.953，explained variance 平均约 0.682；
- observed max KL 为 0.003697，低于 0.02 target；
- 95.69% minibatches 发生 gradient clipping，但未观察到数值不稳定。

这些是相关性观察，不是因果结论。详见 [5M 报告](work_package_5c_5m_training_report_zh-CN.md) 和 <code>artifacts/work_package_5c/</code>。

## 7. 性能与验证成本

- WP5B 单进程 32-env learner batching 在当时基准中从 186.34 提升到 513.69 decisions/s（2.76x）；Stage 4 含未批处理 neural opponent 时约 303.56 decisions/s；
- 首次 5M 的实际加权 rollout 约 270.36 decisions/s，它使用单进程 collector，不包含后来的 P0–P3 优化；
- legacy central-opponent multiprocessing preflight 的最佳值是 8 workers × 4 envs = 315.94 decisions/s，为当时单进程对照的 0.925x；该结论只适用于逐对手动作 Queue 往返的旧 backend；
- P1 已在单进程 collector 中按共享模型合并对手请求，并以 critic-only 路径计算 rollout bootstrap；
- P2 的 worker-local CPU 对手短基准中，4 workers × 8 envs 达到约 600.2 decisions/s，是同轮单进程对照的 1.27x，也是 legacy central 4×8 的 2.21x；8 workers 反而回退；
- P0 支持以 paired seed 为单位的 CPU 多进程评测；新 `prepare` 默认 8 evaluation workers，短搜索基准显示 4–8 之后边际收益很小；
- P3 的 digest、合法动作、fork 和 lean-runner 优化使 20 pairs 浅搜索墙钟缩短约 3.2%，cycle signature 微基准约快 4.06x；
- P0–P3 之后的搜索热路径又增加了整组 NumPy 特征编码、不构建 candidate/actor 的 state-only critic 叶节点、按 policy logit 排序的 fail-soft alpha-beta 和严格 cutoff；关闭 folding 时，新旧搜索选择等价性由固定参考实现回归测试锁定；
- <code>fold_forced_phases</code> 可把只有一个合法动作的 PAYMENT / RETURN_TOKENS / CHOOSE_NOBLE 等链折叠为宏转移，使深度预算只消耗在真实选择上；库层 <code>SearchConfig</code> 默认不折叠，当前搜索评测和深度阶梯默认折叠，<code>--plain-phases</code> 只用于诊断；
- 深度阶梯可在 worker 内每完成一局就 fsync 到 <code>&lt;output&gt;.games/</code>，<code>--resume</code> 只重放未完整的 paired seeds 并保留原 pair id/策略 seed；汇总 JSON 跟踪，逐局 sidecar 由 Git 忽略；
- 5C 约运行 127,800 局 validation：Quick 15,600、Formal 73,200、Promotion 23,000、Historical 16,000；
- 新 P2 路径已取消中央 neural-opponent 逐动作 IPC，但尚未在可用 CUDA 的环境中验证完整长 rollout 收益。

新执行 `train_5c prepare` 生成的默认执行配置为 `multiprocessing_local_opponents`、4 workers × 8 envs、8 evaluation workers。已完成的 `5c_main_seed1` 冻结配置仍是 `single_process`，不应用新默认值重解释旧运行。详见 [P0–P3 性能优化](performance_optimization_p0_p3_zh-CN.md)。

### 7.1 搜索与 folded-depth 实验

固定 5.5M latest checkpoint、top-5、单 determinization、CPU paired-seed 评测：

| 对照 | phase 语义 | paired seeds / games | W/D/L | score rate | paired bootstrap 95% |
|---|---|---:|---:|---:|---:|
| depth-4 search vs raw greedy | plain model decisions | 250 / 500 | 343 / 2 / 155 | 68.8% | 64.5%–73.0% |
| folded depth 5 vs 4 | forced chain 折叠 | 250 / 500 | 296 / 2 / 202 | 59.4% | 55.2%–63.6% |
| folded depth 6 vs 5 | forced chain 折叠 | 250 / 500 | 286 / 6 / 208 | 57.8% | 53.4%–62.1% |

folded-depth 阶梯的两级挑战都高于 54% 晋级阈值，因配置的 <code>max-depth=6</code> 而结束；这表示本轮在深度 6 以内没有观测到收益天花板，不表示 depth 6 已是最优深度。成本同时快速上升：对应轮次的 CPU 平均每次实际搜索约为 depth 4 = 0.47 s、depth 5 = 1.4 s、depth 6 = 4.21 s。汇总交付物为 <code>artifacts/search_5p5m_top5_depth4_500games_cpu_w4.json</code> 和 <code>artifacts/work_package_5c/folded_depth_ladder_5p5m.json</code>。

## 8. 已确认的结论

- 引擎可从 seed 开局运行到正式 terminal，中间阶段可 replay/save/restore。
- Policy API 不暴露 <code>GameState</code>，已有集合置换和隐藏信息 metamorphic tests。
- 纯 terminal reward 在完整游戏中能学习；5C 模型相对起点和固定 heuristic 均明显提高。
- best-dev 与 latest 必须分开；训练末尾不自动是最优模型。
- 历史模型存在非传递性，不能只靠“击败最新自己”证明进步。
- 在本轮固定 checkpoint/top-k/单 determinization 协议下，plain depth-4 搜索相对 raw greedy 有正收益，folded depth 5 相对 4、depth 6 相对 5 也都超过 54% 阶梯阈值。
- 网页直接调用现有引擎，浏览器没有复制规则。

## 9. 尚未确认：不要当作事实

- 未运行独立 FINAL seed domain，DEV 数值不是无偏最终结果。
- 只完成一个 5M 训练 seed，没有多 seed 方差和稳定性证据。
- 尚未确认当前模型容量或 Observation 已成瓶颈；没有受控扩容/特征 ablation。
- <code>TakePair</code> 崩落不等于它在规则上没有价值。
- 5M latest 不能概括为“全面弱于”4.15M best；不同对手上的排序不一致。
- “worker 越多越好”和“单进程永远最好”都不成立；当前数字仍是特定硬件上的短基准。
- reward shaping、lambda=1.0、更大网络、其他 RL 算法和 MCTS 都未做正式对照。
- 搜索结果尚未跨 checkpoint、训练 seed、top-k 或 determinization 配置复验；阶梯在 depth 6 因配置上限停止，尚未测到更深处的收益拐点。

## 10. 已知问题与当前研究问题

### 已知问题

1. Validation 总对局量仍很大；P0 已降低墙钟时间，但 Formal candidate、promotion 和 dynamic historical panel 仍是主要成本。
2. 首次 5M 和 legacy central backend 存在 neural-opponent 小 batch/IPC 瓶颈；P1/P2 已缓解该问题，尚待 CUDA 长 rollout 验收。
3. 环境/编码仍是主要 CPU 成本；P3 已去掉部分重复工作，进一步大改前仍应先 profile。
4. 95.69% minibatches 发生 gradient clipping，值得单变量调查，但目前不是已知数值 bug。
5. Dynamic-history 的训练成绩随对手池变化，不能像固定 heuristic panel 一样直接纵向比较。
6. folded search 每增加一层的 CPU 代价在本轮约增长 3 倍；继续推高深度前应先考虑 batching、树复用或更强的剪枝/缓存。


## 11. 代码导航

| 关注点 | 首要代码 |
|---|---|
| 权威数据结构 | <code>src/splendor/model.py</code>, <code>_state.py</code>, <code>contracts.py</code> |
| 动作/规则/状态机 | <code>actions.py</code>, <code>rules.py</code>, <code>state_machine.py</code> |
| 引擎入口 | <code>engine.py</code> |
| Observation/隐藏信息 | <code>observation.py</code> |
| invariant / RNG / replay | <code>invariants.py</code>, <code>rng.py</code>, <code>replay.py</code> |
| 完整对局/压力测试 | <code>runner.py</code>, <code>validation.py</code> |
| Random/启发式策略 | <code>random_policy.py</code>, <code>heuristic_policy.py</code> |
| 数值编码 | <code>features.py</code> |
| 神经网络/神经策略 | <code>actor_critic.py</code>, <code>neural_policy.py</code> |
| 训练环境/PPO | <code>training_env.py</code>, <code>ppo.py</code> |
| 对手池 | <code>opponent_pool.py</code> |
| 5C 训练/评测 | <code>train_5c.py</code>, <code>evaluate_5c.py</code> |
| multiprocessing | <code>multiprocess_collector.py</code>, <code>benchmark_5c_multiprocess.py</code> |
| 5C 后分析 | <code>analyze_5c.py</code> |
| policy/critic 搜索与深度阶梯 | <code>search_policy.py</code>, <code>evaluate_search.py</code>, <code>ladder_search.py</code> |
| 本地网页 | <code>web_app.py</code>, <code>web/</code> |
| 测试 | <code>tests/</code> |

依赖方向：

~~~text
本地规则/data
  -> contracts + authoritative GameState
  -> rules/state_machine/GameEngine
  -> Observation + legal candidate handles
  -> feature encoding
  -> Policy / Actor–Critic
  -> training/evaluation/web UI
~~~

上层不应反向复制或修改下层规则。

## 12. 文档导航

### 12.1 按任务导航

| 任务 | 建议阅读 |
|---|---|
| 了解全景 | 本页 → [主工作大纲](two_player_ai_workflow_zh-CN.md) |
| 修改/审计规则 | [本地规则](splendor_standard_rules_zh-CN.md) → [WP1](work_package_1_contract_zh-CN.md) → [WP2D](work_package_2d_engine_validation_zh-CN.md) |
| 修改 Observation/特征 | [WP3B](work_package_3b_zh-CN.md) → <code>features.py</code> 和相关测试 |
| 修改模型 | [WP4A](work_package_4a_zh-CN.md) → <code>actor_critic.py</code> |
| 修改 transition/PPO | [WP4B](work_package_4b_zh-CN.md) → [WP4C](work_package_4c_zh-CN.md) |
| 修改 collector/对手池 | [WP5B](work_package_5b_zh-CN.md) → [Stage 2](work_package_5b_stage2_zh-CN.md) → [Stage 3](work_package_5b_stage3_zh-CN.md) → [multiprocessing](work_package_5c_multiprocessing_preflight_zh-CN.md) |
| 优化训练/评测/搜索性能 | [P0–P3 性能优化](performance_optimization_p0_p3_zh-CN.md) → 对应代码与测试 |
| 理解 5C 实际协议 | [5C 现行协议](work_package_5c_current_protocol_zh-CN.md) |
| 分析 5M 结果 | [5M 报告](work_package_5c_5m_training_report_zh-CN.md) → <code>artifacts/work_package_5c/</code> |
| 评测固定深度搜索外挂 | [Policy + Critic 搜索](policy_value_search_zh-CN.md) → <code>evaluate_search.py</code> |
| 运行人机网页 | [README](../README.md) → <code>web_app.py</code> |

### 12.2 工作包时间线

- [WP1：契约](work_package_1_contract_zh-CN.md)
- [WP2A：非购买](work_package_2a_zh-CN.md)
- [WP2B：购买与支付](work_package_2b_zh-CN.md)
- [WP2C：贵族与终局](work_package_2c_zh-CN.md)
- [WP2D：引擎验证](work_package_2d_engine_validation_zh-CN.md)
- [WP3A：基线策略](work_package_3a_zh-CN.md)
- [WP3B：数值编码](work_package_3b_zh-CN.md)
- [WP4A：Actor–Critic](work_package_4a_zh-CN.md)
- [WP4B：训练环境](work_package_4b_zh-CN.md)
- [WP4C：PPO](work_package_4c_zh-CN.md)
- [WP5A：固定对手学习](work_package_5a_zh-CN.md)
- [WP5B：总览](work_package_5b_zh-CN.md)
- [WP5B Stage 2：batched collector](work_package_5b_stage2_zh-CN.md)
- [WP5B Stage 3：historical pool](work_package_5b_stage3_zh-CN.md)
- [WP5B Stage 4：短 self-play](work_package_5b_stage4_zh-CN.md)
- [WP5B Stage 4：paired CI](work_package_5b_stage4_paired_ci_zh-CN.md)
- [WP5C：preflight](work_package_5c_preflight_zh-CN.md)
- [WP5C：multiprocessing preflight](work_package_5c_multiprocessing_preflight_zh-CN.md)
- [WP5C：现行协议](work_package_5c_current_protocol_zh-CN.md)
- [WP5C：5M 训练报告](work_package_5c_5m_training_report_zh-CN.md)
- [P0–P3：训练、评测与搜索性能优化](performance_optimization_p0_p3_zh-CN.md)

## 13. 常用命令

全量测试：

~~~bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q
~~~

2026-09-22 最近一次当前工作树结果：<code>281 passed, 2 skipped</code>。两个 skip 均因当时环境无可用 CUDA。

重建 5C 分析交付物：

~~~bash
conda run -n splendor env PYTHONPATH=src \
  python -m splendor.analyze_5c export \
  --run-root runs/5c_main_seed1 \
  --output-dir artifacts/work_package_5c
~~~

继续可恢复的 folded-depth 阶梯：

~~~bash
conda run -n splendor env PYTHONPATH=src \
  python -u -m splendor.ladder_search \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --output artifacts/work_package_5c/folded_depth_ladder_5p5m.json \
  --start-depth 4 --max-depth 6 --workers 8 --resume
~~~

TensorBoard：

~~~bash
conda run -n splendor tensorboard --logdir runs --port 6006
~~~

人机网页：

~~~bash
conda run -n splendor env PYTHONPATH=src python -m splendor.web_app
~~~

打开 <code>http://127.0.0.1:8765</code>。可选 ScorePolicy、DevelopmentPolicy 或 <code>play_checkpoints/</code> 中的兼容 checkpoint，并设置先/后手与 seed。

## 14. 目录与 Git 约定

| 目录 | 用途 | Git |
|---|---|---|
| <code>src/splendor/</code> | 正式代码 | 跟踪 |
| <code>tests/</code> | 测试 | 跟踪 |
| <code>docs/</code> | 规则、契约、报告 | 跟踪 |
| <code>artifacts/</code> | 值得保留的小型实验交付物 | 汇总默认跟踪；<code>*.games/</code> 逐局 sidecar 忽略 |
| <code>runtime/</code> | PID、status、failure dump 等运行状态 | 忽略 |
| <code>runs/</code> | 训练目录、checkpoint、TensorBoard | 忽略 |
| <code>play_checkpoints/</code> | 本地网页可选 checkpoint | checkpoint 本体忽略 |

详细边界见 <code>artifacts/README.md</code>。大 checkpoint 和 run 不在 Git 中，迁移环境时必须另行备份。

## 15. AI agent 接手检查表

1. 先确认任务属于规则、特征、模型、训练、评测还是 UI。
2. 只读相关权威文档，不把早期文档的历史状态当成现状。
3. 检查 Git 工作树，保留用户已有改动。
4. 若训练正在运行，不修改其 config、checkpoint 或输出目录。
5. 改规则先补失败测试，不在 UI、Policy 或训练环境打规则补丁。
6. 改特征时重新检查 hidden-information 和 permutation semantics。
7. 改训练时区分 learner decision、真实玩家回合、episode 和 PPO update。
8. 报告模型结果时明确 DEV/FINAL、paired seeds、先后手交换、greedy/stochastic 和 checkpoint 身份。
9. 不把本页的研究问题自动转化为未经用户确认的大规模训练。
