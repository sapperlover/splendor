# Policy + Critic 固定深度搜索外挂

> 状态：实现、整局 smoke test、paired CPU 并行和 P3 热路径优化已完成
>
> 基准模型：5C latest，lifetime 5,512,112 decisions
>
> 协议版本：<code>policy-value-search-v1</code>

## 1. 目标

在不修改模型、不重新训练、不复制规则的前提下，为现有 Actor–Critic 增加一个可复用的固定深度搜索策略：

- 当前节点只展开 policy logit 最高的 top-k 合法候选；
- 双方使用同一个冻结 checkpoint 提供 policy prior 和 critic；
- 默认双方均取 top-5；
- 默认搜索 4 ply；
- 非终局叶节点使用 critic；
- 已正式终局的叶节点使用精确的 +1/0/-1；
- 从根玩家视角做 minimax 回传。

这不是 MCTS，也没有训练时梯度、PUCT、访问次数或搜索树跨决策复用。

## 2. 信息安全

搜索策略的公开接口仍是：

~~~python
choose(request: DecisionRequest) -> Action
~~~

它不会接收当前真实对局的 <code>GameState</code>。

因为 Observation 不包含未来牌堆和对手盲保留身份，搜索先仅从当前 <code>DecisionRequest</code> 构造信息集一致的 determinization：

- 当前市场、公开购买牌、公开保留牌固定；
- 自己的盲保留牌按自己的私有 Observation 固定；
- 对手盲保留牌从同等级未知牌中抽样；
- 未来牌堆顺序从剩余未知牌中抽样；
- determinization seed 只由当前公开 request、策略 seed 和样本编号确定；
- 不读取真实牌堆顺序或真实对手暗牌。

同一个 request 和搜索 seed 总是生成相同 determinization，因此该策略仍是确定性的。每棵树内的合法动作和状态转移全部调用现有 <code>GameEngine</code>。

## 3. 搜索语义

默认参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| top-k | 5 | 每个决策节点按 policy logit 展开前 5 个候选 |
| depth | 4 | 4 个模型决策边界 |
| determinizations | 1 | 每次真实决策采样一个信息集状态 |
| backup | minimax | 根玩家节点取 max，对手节点取 min |
| leaf | critic | 转换为根玩家视角 |
| leaf clip | [-1, 1] | 与 terminal reward 尺度一致 |

这里一个 ply 是一个 <code>DecisionRequest</code>，而不是一个完整玩家回合。因此 PAYMENT、RETURN_TOKENS、CHOOSE_NOBLE 也会消耗 ply。这与 PPO 的模型决策语义一致，不会把支付或贵族选择静默固定成某个动作。

同一玩家在连续决策阶段仍取 max；只有 <code>GameEngine.current_player</code> 真正切换后才改为 min。终局价值不调用 critic：

- 根玩家胜：+1；
- 平局：0；
- 根玩家负：-1。

多个 determinization 时，每个根动作先在各自树中 minimax，再对根动作价值取平均。计算成本近似随 determinization 数量线性增长。

## 4. 代码入口

- 搜索策略与 determinization：<code>src/splendor/search_policy.py</code>
- 搜索对原始 greedy policy 的 paired 评测：<code>src/splendor/evaluate_search.py</code>
- paired-seed 多进程调度：<code>src/splendor/evaluation.py</code>
- 受控 fork、合法动作与 digest 缓存：<code>src/splendor/engine.py</code>
- 测试：<code>tests/test_search_policy.py</code>
- 公共类：
  - <code>SearchConfig</code>
  - <code>SearchStats</code>
  - <code>PolicyValueSearchPolicy</code>

评测 JSON 会记录：

- 完整搜索参数；
- checkpoint 和 lifetime step；
- paired W/D/L、先后手表现和 bootstrap 区间；
- 展开节点数、叶节点数、模型 batch/position 数；
- inference 时间和每次搜索决策平均时间。

## 5. 已完成验证

自动测试覆盖：

- 配置参数校验；
- <code>choose</code> 仍只有 <code>DecisionRequest</code> 输入；
- determinization 精确重建根 Observation 和合法动作；
- 对手暗保留保持未知，自己的暗保留保持可见；
- 随机完整对局的多种中间阶段均可重建；
- 固定 seed 下选择确定；
- top-k、深度、batched leaf 数量符合预期；
- 默认 top-5 × 4-ply 返回正式合法动作。

使用 5.5M latest checkpoint 做过一组完整默认参数的 paired smoke：

| 指标 | 结果 |
|---|---:|
| paired seeds / games | 1 / 2 |
| 状态 | 通过 |
| IllegalAction / invariant failure | 0 / 0 |
| CPU elapsed | 25.24 s |
| 搜索策略决策调用 | 87 |
| 实际展开搜索的多候选决策 | 61 |
| 展开节点 | 22,894 |
| critic/policy positions | 22,925 |
| 搜索总耗时 | 24.79 s |

该 2 局结果只证明全流程可运行，没有棋力统计意义。工具环境当时不可见 CUDA，因此这里记录的是 CPU smoke。当前可在“单进程 CUDA”和“paired-seed 多进程 CPU”两种执行模式中选择，需用实际搜索参数做短基准，不预设某一种必然更快。

### 5.1 P0 paired CPU 并行短基准

同一 checkpoint、20 paired seeds / 40 games、<code>top-k=2</code>、<code>depth=2</code> 的单次 CPU 短基准：

| workers | 墙钟时间 |
|---:|---:|
| 2 | 9.340 s |
| 4 | 5.732 s |
| 8 | 5.700 s |
| 12 | 5.544 s |
| 14 | 5.521 s |

4 workers 之后收益已明显收窄。搜索评测建议从 4 或 8 workers 起步；如果电脑同时运行其他应用，应保留 CPU 余量，不建议为了几个百分点的收益直接使用 12–14 workers。

### 5.2 P3 搜索热路径

搜索现在使用受控的 <code>GameEngine.fork()</code>，复用已验证的父 state、replay、digest 和合法动作；每个新子状态仍正常做 invariant check。评测 runner 使用 lean 模式，cycle signature 直接生成稳定摘要。

在上述 4-worker、20-pair 浅搜索基准中，P3 使墙钟从 5.732 s 降到 5.550 s（约 -3.2%），合计搜索时间约 -3.9%，评测结果逐字段一致。详见 [P0–P3 性能优化](performance_optimization_p0_p3_zh-CN.md)。

## 6. 运行命令

### 6.1 快速整局 smoke

~~~bash
conda run -n splendor env PYTHONPATH=src \
  python -m splendor.evaluate_search \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --pairs 1 \
  --top-k 5 \
  --depth 4 \
  --determinizations 1 \
  --device cuda \
  --bootstrap-resamples 1000 \
  --progress-every 1 \
  --output runtime/search_smoke_5p5m_top5_depth4.json
~~~

### 6.2 500 局正式直接对战

250 paired seeds 等于 500 局，并交换先后手：

~~~bash
conda run -n splendor env PYTHONPATH=src \
  python -u -m splendor.evaluate_search \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --pairs 250 \
  --top-k 5 \
  --depth 4 \
  --determinizations 1 \
  --device cuda \
  --progress-every 5 \
  --output artifacts/search_5p5m_top5_depth4_500games.json
~~~

### 6.3 paired-seed CPU 多进程

<code>workers &gt; 1</code> 时必须使用 CPU：

~~~bash
conda run -n splendor env PYTHONPATH=src \
  python -u -m splendor.evaluate_search \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --pairs 250 \
  --top-k 5 \
  --depth 4 \
  --determinizations 1 \
  --device cpu \
  --workers 4 \
  --progress-every 5 \
  --output artifacts/search_5p5m_top5_depth4_500games_cpu_w4.json
~~~

对当前机器建议先在 20–50 pairs 上比较 <code>--workers 4</code> 和 <code>8</code>，再锁定正式评测配置。

### 6.4 1000 局

只需把上一条命令改为：

~~~text
--pairs 500
--output artifacts/search_5p5m_top5_depth4_1000games.json
~~~

### 6.5 参数微调

如果完整评测过慢，建议先按以下顺序缩小：

1. 保持 depth=4，把 top-k 从 5 改为 4；
2. 保持 top-k=5，把 depth 从 4 改为 3；
3. 先用 20–50 paired seeds 比较吞吐和明显劣化，再决定是否跑 250 pairs。

不要通过读取真实 <code>GameState</code> 或真实牌堆来提速。增加 <code>--determinizations</code> 可以降低单样本 determinization 方差，但成本近似线性增加。

## 7. 当前限制

- minimax 假设对手会在其 top-k 中选择对根玩家最不利的动作，可能比原始 greedy 对手更保守；
- 单 determinization 对未来未知牌有采样方差；
- 当前树不跨真实决策复用；
- 规则引擎分支展开仍在 Python CPU 上进行；
- 每层模型位置已批量推理，但不同真实对局尚未合并为同一搜索 batch；P0 是进程级并行，不是跨局模型 batching；
- 尚未完成 500–1000 局棋力验收，因此不能宣称搜索强于原始 policy。
