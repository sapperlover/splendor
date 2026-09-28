# 独立 determinization PUCT 搜索（第一步）

## 范围与算法

`PUCTSearchPolicy` 接收正式 `DecisionRequest`，只从该请求重建采样世界。每次搜索生成 `N` 个独立 determinization，各有独立引擎、树和边统计；一个世界内的隐藏牌与牌堆固定。内部节点由模拟引擎提供当前行动者的正常 Observation 和完整合法候选，再由冻结的现有 ActorCritic 同次前向给出 policy 与 value。终局由规则引擎给出精确收益，不做随机 rollout。只在所有世界的根复用同一可见请求的网络先验，搜索结束即丢弃树。

这是不完全信息上的实用近似，不是严格 ISMCTS。各世界内部后续计划可能依赖各自采样的隐藏状态，因此仍可能发生 strategy fusion。模型结构、权重、特征、规则和旧 alpha-beta 搜索均未修改。

`simulations=S` 是跨所有世界的总预算。第 `j` 个世界分得 `S//N + (j < S%N)` 次；根网络推理不占预算，每次 simulation 必须走过一条根边并回传。设第 `j` 棵树根边 `a` 的访问计数为 `n_j[a]`、预算为 `S_j`：

```text
root_visit_counts[a] = sum_j n_j[a]
policy_target[a] = (1/N) * sum_j n_j[a] / S_j
```

因此每个世界等权；当 `S%N != 0` 时，`policy_target` 可能不同于汇总计数的归一化。输出数组均按原始 `request.legal_actions` 索引对齐，包括零访问动作。执行时返回原候选对象。

## PUCT 与转移

边的 `N/W/Q` 全部采用根玩家视角。未访问边 `Q=0`。节点行动者与根玩家相同时 `sign=+1`，否则 `sign=-1`，选择最大：

```text
sign * Q + c_puct * P * sqrt(1 + 节点边访问总数) / (1 + 边访问数)
```

平分先看更高先验，再看更小原候选索引。叶节点 value 先检查 finite、裁剪到 `[-1,1]`，再按叶行动者转换成根视角；同一根视角值沿路径各边回传一次。PAYMENT、RETURN_TOKENS、CHOOSE_NOBLE 等连续同玩家阶段按引擎当前行动者判断，绝不按搜索深度交替翻号。

首次到达非终局新节点时只展开该节点，使用其 policy/value 回传并结束该 simulation；未来再访问才选择它的边。`fold_forced_phases=True` 在模拟引擎中连续执行单合法动作，不改变真实策略的单步输出。单候选根直接返回 target/behavior `[1]`、访问 `[0]`、`searchable=False`，不采样世界也不消耗 simulation。`max_simulation_transitions` 限制单次 simulation 中实际 `engine.apply()` 的次数，含折叠动作；到上限且未终局时用该非终局请求的 critic 截断估值，记录 cap 命中，不把它算作正式平局。

## API 与可复现性

```python
from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy

policy = PUCTSearchPolicy(frozen_model, config=MCTSConfig())
action = policy.choose(request)  # 无噪声、温度 0；last_search_result 可读取
result = policy.search(request, add_root_noise=True, temperature=1.0, sampling_seed=123)
training_target = result.policy_target
behavior = result.behavior_policy
```

默认值：`simulations=256`、`determinizations=2`、`c_puct=1.5`、`seed=0`、`fold_forced_phases=True`、`max_simulation_transitions=512`、`root_noise_fraction=0.25`、`dirichlet_alpha=0.3`。`choose()` 总是确定性评测。显式启用根噪声时，每次搜索只采一个合法动作维度的 Dirichlet 向量，所有世界共享；内部节点不加噪声。世界 seed 只由可见请求、有效策略 seed 和世界编号稳定派生；噪声与温度采样用独立本地随机子流，`sampling_seed` 不改变世界采样。温度 0 以 `policy_target` 最大者行动，平分按无噪声根先验及候选索引；正温度仅把 target 按 `target**(1/temperature)` 变换成 `behavior_policy` 再抽样，训练标签仍是原 target。

`MCTSSearchResult` 包含候选索引、是否可搜索、无噪声/实际根先验、各世界预算和根访问向量、汇总访问、`policy_target`、`behavior_policy`、实际 simulation 数、最大树深和 cap 命中；`to_dict()` 可序列化。`MCTSStats.to_dict()` 只输出可相加的计数与耗时，供评测器逐局累加。模型推理计时含从设备复制结果到 CPU 的同步；本次只有 CPU 实测，CUDA 未验证。

## 统一评测

JSON 使用 `kind="mcts_checkpoint"`，仍通过 `search_config` 配置。旧 `search_checkpoint` 继续表示 alpha-beta。评测每局 policy seed 与配置 seed 按 XOR 合成；配置省略 seed 时按 0。两种示例：

- [`configs/mcts_vs_greedy.json`](../configs/mcts_vs_greedy.json)
- [`configs/mcts_vs_alphabeta.json`](../configs/mcts_vs_alphabeta.json)

从仓库根目录执行：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.evaluation \
  --config configs/mcts_vs_greedy.json \
  --output runtime/mcts_step1/mcts_vs_greedy.json

# 中断后使用同一配置和输出路径
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.evaluation \
  --config configs/mcts_vs_greedy.json \
  --output runtime/mcts_step1/mcts_vs_greedy.json --resume
```

固定评测使用无噪声、温度 0 的 `choose()`，沿用确定性 policy-cycle 裁决。未来带随机噪声或采样的自我对弈不能直接套用此裁决。sidecar、checkpoint 身份、配置指纹、逐局统计与 resume 仍由现有统一评测器管理。

## 验证与结果

2026-09-27，新增及受影响测试覆盖预算、等权聚合、根视角符号、新叶回传、终局与截断、随机流、隐藏信息、四个决策阶段、双 MCTS 串行与双 worker 一致性、sidecar/resume 与指纹拒绝。完整测试和 smoke 结果以本次交付记录为准。

真实 5.5M checkpoint 的 MCTS vs 同 checkpoint greedy 示例执行 2 paired seeds / 4 games，CPU、2 workers、S=256/N=2；输出为 `runtime/mcts_step1/mcts_vs_greedy.json`。4 局均有效，0 异常、0 非法动作、0 工程截断；MCTS 总计 31,744 次 simulation，29,408 次网络 batch，38,868 次模拟状态转移，0 次 cap 命中，累计策略搜索墙钟 45.93 秒（多局、多 worker 的累计值）。另外 `configs/mcts_vs_alphabeta.json` 对旧 folded alpha-beta D4/top5 执行 1 pair / 2 games，输出 `runtime/mcts_step1/mcts_vs_alphabeta.json`，两局均有效。该小样本得分率 75% 只说明评测链路可用，不是棋力提升证据。simulation 数不能与固定深度 alpha-beta 的节点数直接比较。

后续阶段可直接消费 `policy_target` 与终局结果生成监督样本；本阶段没有实现数据集、replay buffer、训练器或大规模 self-play。
