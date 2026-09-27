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

## 4. 代码入口与评测分层

- 搜索策略与 determinization：`src/splendor/search_policy.py`；评测器通过 `PolicyValueSearchPolicy` 接口调用，不在评测代码里实现搜索算法；
- 固定双方对局、种子/局数/并行配置、逐局保存、恢复与汇总：`src/splendor/evaluation.py`；
- 深度阶梯实验编排：`src/splendor/ladder_search.py`；它只生成每个 rung/block 的双方策略配置，并决定下一阶段；
- `src/splendor/evaluate_search.py` 是旧命令的兼容入口，已弃用；新评测使用 JSON 配置调用 `evaluation.py`；
- 受控 fork、合法动作与 digest 缓存：`src/splendor/engine.py`；
- 搜索测试：`tests/test_search_policy.py`。

评测策略通过 `SerializablePolicySpec` 声明。双方可以选择不同 checkpoint，也可以各自选择 checkpoint greedy 或 checkpoint + search 配置。评测器不检查或重写策略算法，只按统一对局协议调用策略。

搜索统计按每个策略自己的 `top_k` 动态记录 `chosen_rank_1_count` 到 `chosen_rank_K_count`，这些是恰好选中第 N 名的计数。旧的 `chosen_rank_top1/2/3_count` 与 `chosen_rank_beyond3_count` 保留，其中 top-2/top-3 是累计计数。`decision_quality` 汇总同时给出逐 rank 比例；比例分母为 `searchable_decisions`，单合法动作的强制决策不计入 rank 分布。通用评测 JSON 的双方 `policy_a_stats` / `policy_b_stats` 和逐局 sidecar 都保存逐 rank 计数。

单次固定评测写出汇总 JSON 和逐局 sidecar。每个 seat-swapped game 完成后，worker 会向 `pair-XXXXXXX.jsonl` 追加并 `fsync` 一条记录；只有一对的两局都完成后，该 pair 才会跳过续跑。配置指纹包括双方策略、checkpoint 内容摘要、搜索参数、种子、规则边界及 worker/device 配置，恢复时必须匹配。

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

新评测入口使用 JSON 描述固定双方、搜索能力和对局协议。配置文件可以重复用于断点续跑；运行时追加 `--resume`，评测器校验配置后跳过完整 pairs。

### 6.1 深度 6 搜索 vs 无搜索，200 pairs

每个 pair 包含交换先后手的 2 局。下面配置允许两侧使用不同 checkpoint，也支持两侧各自选择 greedy 或 search 策略。

~~~bash
mkdir -p runtime
cat > runtime/search_depth6_vs_raw_200.json <<'JSON'
{
  "policy_a": {
    "kind": "search_checkpoint",
    "name": "policy_value_search_depth6",
    "checkpoint": "runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt",
    "search_config": {
      "top_k": 5,
      "depth": 6,
      "determinizations": 1,
      "seed": 8700000,
      "leaf_value_clip": 1.0,
      "fold_forced_phases": true
    }
  },
  "policy_b": {
    "kind": "greedy_checkpoint",
    "name": "raw_greedy_policy",
    "checkpoint": "runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt"
  },
  "pairs": 200,
  "seed_start": 8900000,
  "policy_seed_base": 8600000,
  "workers": 8,
  "device": "cpu",
  "max_completed_turns": 400,
  "adjudicate_deterministic_policy_cycles": true,
  "bootstrap_resamples": 100000,
  "bootstrap_seed": 8800000,
  "progress_every": 1,
  "summary_every": 25
}
JSON

conda run -n splendor env PYTHONPATH=src \
  python -u -m splendor.evaluation \
  --config runtime/search_depth6_vs_raw_200.json \
  --output artifacts/search_5p5m_top5_folded_depth6_vs_rawgreedy_200pairs_cpu_w8.json
~~~

逐局记录保存在输出旁的 `.games/` 目录。每局完成后立即 fsync；Ctrl-C 会停止 worker 并从已落盘记录重建汇总，未完成的 pair 会在追加 `--resume` 后按原种子重跑：

~~~bash
conda run -n splendor env PYTHONPATH=src \
  python -u -m splendor.evaluation \
  --config runtime/search_depth6_vs_raw_200.json \
  --output artifacts/search_5p5m_top5_folded_depth6_vs_rawgreedy_200pairs_cpu_w8.json \
  --resume
~~~

### 6.2 单 worker CUDA

把配置中的 `workers` 设为 `1`、`device` 设为 `cuda`；多 worker 评测使用 CPU 推理。

### 6.3 旧命令

`python -m splendor.evaluate_search` 暂时保留为兼容入口，会输出弃用提示并调用同一评测 API。新实验请保存 JSON 配置并使用 `python -m splendor.evaluation`。

### 6.4 参数微调

如果完整评测过慢，建议先按以下顺序缩小：

1. 保持 depth=6，把 top-k 从 5 改为 4；
2. 保持 top-k=5，把 depth 从 6 改为 5；
3. 先用 20–50 paired seeds 比较吞吐和明显劣化，再决定是否跑 200 pairs。

不要通过读取真实 `GameState` 或真实牌堆来提速。增加 `determinizations` 可以降低单样本 determinization 方差，但成本近似线性增加。

## 7. 当前限制

- minimax 假设对手会在其 top-k 中选择对根玩家最不利的动作，可能比原始 greedy 对手更保守；
- 单 determinization 对未来未知牌有采样方差；
- 当前树不跨真实决策复用；
- 规则引擎分支展开仍在 Python CPU 上进行；
- 每层模型位置已批量推理，但不同真实对局尚未合并为同一搜索 batch；P0 是进程级并行，不是跨局模型 batching；
- 尚未完成 500–1000 局棋力验收，因此不能宣称搜索强于原始 policy。
