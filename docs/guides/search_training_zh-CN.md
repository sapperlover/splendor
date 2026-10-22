# 搜索预算随机化、Forced Playouts 与 Policy Target Pruning

[37423步起点的4k局／24k池持续迭代](../../artifacts/entity_attention_random_budget_iteration_20261019/README.md)已完成20轮；初版“未启动”状态已过时。2026-10-22另设[第20轮继承续跑分支](../../artifacts/entity_attention_r20_s256n4_noble12_iteration_20261022/README.md)：25%高预算S256N4（总1024）、75%低预算S128N1（总128），新采集用12贵族；继承AdamW/RNG及24k活动池，FP/PTP关闭。

2026-10-19实现。入口是 `selfplay_fast`，以及复用它的 `alphazero_loop` / `hybrid_train` / `pool_iteration` 在线采集；Python批量搜索和native搜索使用相同规则。三个开关默认关闭，原冻结实验配置不改写。没有启动正式训练，也没有棋力收益结论。

[有限验证记录](../../artifacts/search_training_20261019/README.md)：183项通过、5项CUDA条件测试跳过；新版128完整CPU循环采集2局、训练2步及小预算评测完成，另验证混合/全低预算batch与checkpoint恢复。

## 配置

在单次采集配置顶层加入：

```json
{
  "budget_randomization": {"low_simulations": 128, "low_probability": 0.75},
  "search_config": {
    "simulations": 1024,
    "determinizations": 1,
    "forced_playouts": 2.0,
    "policy_target_pruning": true
  }
}
```

这是合并片段，保留原配置其他字段。循环训练时把这两个对象放进原配置的 `selfplay` 中；learner的loss权重、窗口、lr、更新步数不因开关自动变化。`forced_playouts`是系数k，不是额外增加的预算；强制访问占用原S预算。可单开预算随机化，也可单开Forced Playouts；pruning要求k>0。移除 `budget_randomization`、设置k=0和pruning=false恢复旧语义。

完整采集示例：[collection_randomized_budget.json](../../configs/entity_attention_v1/collection_randomized_budget.json)。它使用新版128×2的37423步模型，1000局，沿用既有W8/A1000/B500 GPU采集配置，仅作为待运行示例，不表示已经验证这个配方的吞吐或棋力。机器需具备对应CUDA环境及已构建native扩展。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.selfplay_fast generate \
  --config configs/entity_attention_v1/collection_randomized_budget.json \
  --output runs/entity_attention_randomized_budget

conda run -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast validate \
  --input runs/entity_attention_randomized_budget --replay
```

中断后给原generate命令追加 `--resume`。预算比例、预算数或k/pruning改变后必须新开输出目录。修改现有训练配置后也需使用新的run/output身份，不向旧冻结run注入新语义。

## 预算随机化与监督

每个多候选真实决策独立抽样，包含支付、退币、选贵族阶段；不是每局或每完整回合抽一次。抽样由 `SHA256(["splendor-playout-cap-v1", sampling_seed])` 决定，不消耗世界、噪声、走子或训练RNG，不依赖worker调度。单候选不抽样、不搜索。

| 决策 | 总预算 | 根噪声 / 强制访问 / pruning | policy权重 | value权重 |
| --- | --- | --- | --- | --- |
| 高预算 | `search_config.simulations` | 按原配置启用 | 1 | 1 |
| 低预算 | `low_simulations` | 全部关闭 | 0 | 1 |
| 单候选 | 0 | 关闭 | 0 | 1 |

温度沿用原有按已完成回合数的日程；低预算没有额外温度改动。默认世界数不变，必须满足 `determinizations <= low_simulations < simulations`。可选`budget_randomization.low_determinizations`单独指定低预算世界数；当前要求它不超过高预算世界数，且不超过`low_simulations`。预算字段始终是总次数，各预算在该步实际世界数中尽量均分。例如基础`simulations=1024, determinizations=4`，低预算`low_simulations=128, low_determinizations=1`对应S256N4／S128N1。

以上示例平均每个多候选决策执行352次模拟，是固定S1024的34.375%；只是模拟数的期望值，不是实测吞吐。policy监督密度期望为原来的25%；同样训练batch中有效policy样本会减少，learner仍按有效policy权重之和归一化CE。全低预算batch的policy loss为可微0，value继续训练。共享encoder仍会收到value梯度，不能把它称为“冻结policy”。

相对KataGo论文第3.1节仅记录full-search步骤的做法，本项目**额外保留低预算步骤作value监督**，明确区分样本权重。没有开启Policy Surprise Weighting，也没有改变终局胜负z目标。

## Forced Playouts 与标签裁剪

依据[KataGo 2019论文第3.2节](https://arxiv.org/html/1902.10565v5#S3.SS2)，采用本仓库PUCT公式下的整数访问版本：

1. 每个隐藏世界的根节点独立处理。对已访问过的动作c，当 `N(c) < sqrt(k * P_used(c) * N_root)`，将其selection score置为无穷大。`P_used`是加噪后的根先验；多个动作同时满足时按该先验、候选顺序确定次序。未访问动作仍由普通PUCT开启探索；树内节点不强制。
2. 搜索结束，选访问最多的根动作c*作为保留动作；并列按 `P_used`、候选顺序。固定所有最终Q以及原始 `N_root`，对其他动作最多扣除 `floor(sqrt(k * P_used(c) * N_root))` 次访问，且扣除后普通PUCT仍须严格小于c*的最终普通PUCT。普通PUCT使用原实现的 `Q + c_puct * P * sqrt(1+N_root)/(1+N(c))`。剩余<=1的非保留动作清零，保留动作计数不动。
3. 各世界裁剪后分别归一化，再等权平均形成policy target。原始访问计数与Q累积和不变。实际走子与behavior policy仍按**原始等权访问分布**和原温度生成，pruning不反向改变本次走子。

裁剪上限取论文中的理论 `n_forced`，不是机械减去实际触发次数：好动作即使曾被强制探索，也可能完整保留。该实现不包含新版KataGo的LCB、FPU、动态探索常数等其他搜索方法。

## 数据与复核

开启任一新功能时，采集协议仍为 `mcts-selfplay-v2`，数据schema升级到 `splendor-mcts-selfplay-v3`。模型输入与checkpoint结构不变。v1/v2历史数据仍可读取、混池；默认关闭时仍写旧v2。历史v1采集CLI不支持新功能，启用时明确报错并指向 `selfplay_fast`。

v3增加：

| 字段 | 形状 / dtype | 含义 |
| --- | --- | --- |
| `low_budget` | [T] bool | 本步是否抽中低预算 |
| `per_world_policy_target_counts` | [N,C] int32 | 生成标签的裁剪后访问计数；未裁剪时等于原计数 |
| `per_world_root_value_sums` | [N,C] float64 | 根行动者视角的原始value累积和W，Q=W/N |
| `per_world_forced_playouts` | [N,C] int32 | 各动作实际由强制规则选中的次数 |

`root_used_prior`在v3用float64保存，保证整数裁剪边界可精确复算；其他既有概率仍为float32。`root_visit_counts`、`per_world_root_visit_counts`、`simulation_budgets`始终是实际执行的原统计。`searchable`仅表示多候选，不再等于policy是否监督；以 `policy_weight`为准。metadata的 `search_training`保存基础搜索参数、根噪声开关、预算配置；manifest冻结这些设置及算法版本。

2026-10-22支持低预算世界数更少：数组N仍为高预算世界数，低预算实际世界数写入`valid_world_count`，未使用的预算／访问／target／value sum／forced行全部为零。policy及behavior只对有效世界取等权平均。省略`low_determinizations`时保留旧配置序列化与语义，不改写旧数据。

校验会重算预算抽样、每世界预算、裁剪计数、等权目标和behavior，检查低预算关闭根噪声/强制访问、policy/value权重、正式终局标签。`summary.json`新增 `low_budget_decisions`、`full_budget_decisions`、`forced_playouts`、`pruned_visits`，`policy_samples`只统计有效高预算监督。value分支仍可使用所有决策。

## 有限验证

相关回归命令：

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q \
  tests/test_search_training.py tests/test_mcts_policy.py \
  tests/test_native_search.py tests/test_native_chance_search.py \
  tests/test_native_batch.py tests/test_selfplay.py tests/test_search_accel.py \
  tests/test_alphazero_training.py tests/test_alphazero_loop.py \
  tests/test_hybrid_training.py tests/test_joint_pool_training.py
```

覆盖预算抽样边界及RNG隔离、强制规则的根/已访问限制、好动作保留、整数裁剪与逐次扣除参考的一致性、单/双世界Python/native差分，以及三条CPU采集路径的正式终局归档、checked重放、resume、篡改拒绝、replay入池、混合与全低预算batch的实际训练。小预算smoke只验证工程正确性，不能推出学习或棋力收益。
