# 冻结网络搜索

两种策略都只接收 `DecisionRequest`；采样世界来自可见信息，不复制真实牌堆或对手暗牌。两者均为不完全信息近似，不能视为严格 ISMCTS。原生 PUCT 的构建和差异见 [native 指南](native_search_zh-CN.md)，对战入口见 [评测指南](evaluation_zh-CN.md)。

2026-10-07新增可选的原生采样随机节点/信息历史PUCT，入口为`NativeChancePUCTSearchPolicy`和评测`search_backend="native_chance"`。下面的独立确定化描述仍是默认Python/native行为；随机模式见下节，尚不代表已用于正式自迭代训练。

## PUCT / MCTS

### 范围与算法

`PUCTSearchPolicy` 接收正式 `DecisionRequest`，只从该请求重建采样世界。每次搜索生成 `N` 个独立 determinization，各有独立引擎、树和边统计；一个世界内的隐藏牌与牌堆固定。内部节点由模拟引擎提供当前行动者的正常 Observation 和完整合法候选，再由冻结的现有 ActorCritic 同次前向给出 policy 与 value。终局由规则引擎给出精确收益，不做随机 rollout。只在所有世界的根复用同一可见请求的网络先验，搜索结束即丢弃树。

这是不完全信息上的实用近似，不是严格 ISMCTS。各世界内部后续计划可能依赖各自采样的隐藏状态，因此仍可能发生 strategy fusion。该算法使用冻结 Actor–Critic 与相同特征契约。

`simulations=S` 是跨所有世界的总预算。第 `j` 个世界分得 `S//N + (j < S%N)` 次；根网络推理不占预算，每次 simulation 必须走过一条根边并回传。设第 `j` 棵树根边 `a` 的访问计数为 `n_j[a]`、预算为 `S_j`：

```text
root_visit_counts[a] = sum_j n_j[a]
policy_target[a] = (1/N) * sum_j n_j[a] / S_j
```

因此每个世界等权；当 `S%N != 0` 时，`policy_target` 可能不同于汇总计数的归一化。输出数组均按原始 `request.legal_actions` 索引对齐，包括零访问动作。执行时返回原候选对象。

### PUCT 与转移

边的 `N/W/Q` 全部采用根玩家视角。未访问边 `Q=0`。节点行动者与根玩家相同时 `sign=+1`，否则 `sign=-1`，选择最大：

```text
sign * Q + c_puct * P * sqrt(1 + 节点边访问总数) / (1 + 边访问数)
```

平分先看更高先验，再看更小原候选索引。叶节点 value 先检查 finite、裁剪到 `[-1,1]`，再按叶行动者转换成根视角；同一根视角值沿路径各边回传一次。PAYMENT、RETURN_TOKENS、CHOOSE_NOBLE 等连续同玩家阶段按引擎当前行动者判断，绝不按搜索深度交替翻号。

首次到达非终局新节点时只展开该节点，使用其 policy/value 回传并结束该 simulation；未来再访问才选择它的边。`fold_forced_phases=True` 在模拟引擎中连续执行单合法动作，不改变真实策略的单步输出。单候选根直接返回 target/behavior `[1]`、访问 `[0]`、`searchable=False`，不采样世界也不消耗 simulation。`max_simulation_transitions` 限制单次 simulation 中实际 `engine.apply()` 的次数，含折叠动作；到上限且未终局时用该非终局请求的 critic 截断估值，记录 cap 命中，不把它算作正式平局。

### API 与可复现性

```python
from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy

policy = PUCTSearchPolicy(frozen_model, config=MCTSConfig())
action = policy.choose(request)  # 无噪声、温度 0；last_search_result 可读取
result = policy.search(request, add_root_noise=True, temperature=1.0, sampling_seed=123)
training_target = result.policy_target
behavior = result.behavior_policy
```

默认值：`simulations=256`、`determinizations=2`、`c_puct=1.5`、`seed=0`、`fold_forced_phases=True`、`max_simulation_transitions=512`、`root_noise_fraction=0.25`、`dirichlet_alpha=0.3`。`choose()` 总是确定性评测。显式启用根噪声时，每次搜索只采一个合法动作维度的 Dirichlet 向量，所有世界共享；内部节点不加噪声。世界 seed 只由可见请求、有效策略 seed 和世界编号稳定派生；噪声与温度采样用独立本地随机子流，`sampling_seed` 不改变世界采样。温度 0 以 `policy_target` 最大者行动，平分按无噪声根先验及候选索引；正温度仅把 target 按 `target**(1/temperature)` 变换成 `behavior_policy` 再抽样，训练标签仍是原 target。

`MCTSSearchResult` 包含候选索引、是否可搜索、无噪声/实际根先验、各世界预算和根访问向量、汇总访问、`policy_target`、`behavior_policy`、实际 simulation 数、最大树深和 cap 命中；`to_dict()` 可序列化。`MCTSStats.to_dict()` 只输出可相加的计数与耗时，供评测器逐局累加。模型推理计时含从设备复制结果到 CPU 的同步；设备和 batch 形状会影响 FP32 舍入；逐位复现需要固定后端、模型与执行条件。


## 采样随机节点与信息历史 PUCT（native_chance）

该模式以每个玩家可见的观察和行动历史为决策统计键，使用同一个统计池，不是独立世界访问分布的平均：

1. 每次simulation仅从安全root请求允许的未知牌池重新抽样对手已有暗保留牌，自己已知私有牌保持确定；牌池与保留区无重复。
2. 市场补牌、盲保留抽牌发生时，从该simulation的剩余牌池均匀无放回抽样；未来牌堆没有固定顺序。条件于该次隐藏牌分配，每张剩余牌的抽取概率为`1 / remaining_count`。
3. 玩家决策仍按PUCT选择动作；随机结果按环境概率采样，各结果均回传至同一动作的N/W/Q，不能最大化选择好运结果。保存各观察结果的访问次数用于检查。
4. 模拟携带双方各自的完整搜索内信息历史。公开动作和公开翻牌被双方观察；盲牌仅加入持有者的历史。对手盲牌身份和未来顺序不进入另一玩家的节点键；暗保留购买在支付完成公开购买前也不把身份放入对方历史。决策时只使用行动玩家的历史节点，避免沿不同隐藏世界作出该玩家无法区分的计划。
5. 首次访问新的行动玩家信息节点时，用安全编码的policy/value展开；再次访问共享先验及边统计，状态转移仍从该次simulation的状态执行。终局精确收益和value的根视角回传与原后端一致；transition cap按simulation累计实际动作数计算。

这是均匀隐藏牌信念下的多观察者信息集PUCT近似，没有根据真实对手此前行为拟合贝叶斯后验，也不承诺求出博弈均衡。真实游戏引擎仍使用真实固定牌堆，只有搜索模拟使用随机抽牌。

```python
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeChancePUCTSearchPolicy

policy = NativeChancePUCTSearchPolicy(
    frozen_model, device="cuda",
    config=MCTSConfig(simulations=1024, determinizations=1, seed=123),
)
result = policy.search(request, add_root_noise=False, sampling_seed=456)
```

要求`determinizations=1`，在此表示一个共享统计池，不是一个固定世界。每次simulation都有独立隐藏牌样本和随机抽牌子流；`config.seed`控制整份抽样序列，`sampling_seed`独立控制根噪声和行为抽样。固定权重、请求、seed和执行条件可以重复相同结果。

结果保持`MCTSSearchResult`形状：预算只有一项，`per_world_root_visit_counts`一行保存共享根计数，`policy_target=root_visit_counts/S`。这个字段在此不能解释为固定世界计数。直接作业的`job.core.chance_diagnostics()`可查看root隐藏样本数、随机抽牌数、信息节点命中、各根动作的结果访问数和value sum；`debug_leaf_state`只用于内部验证，包含模拟隐藏信息，不进入策略输入。

固定评测JSON在单个策略上设置`"search_backend": "native_chance"`，另一个策略可继续使用`native`，已支持串行及native_batched调用。2026-10-07独立`selfplay_fast`采集也接通同名选项，要求native执行后端及N=1，包含多worker共享槽路径。默认选项不写入旧配置，保持原序列化／fingerprint；现有hybrid自迭代配置仍使用旧确定化教师。

新增回归为`tests/test_native_chance_search.py`。对200个固定留出请求的同预算A/B/C/D复测见[实验记录](../../artifacts/entity_attention_chance_search_20261007/README.md)；稳定性指标和棋力须分别判断。

后续[同一16500步权重、S1024的100 pair棋力对照](../../artifacts/entity_attention_chance_strength_20261007/README.md)已完成200局自然终局：新搜索得分31.25%，配对95%区间24.5%–38.5%，明显弱于旧native确定化。当前实现目标更稳定但搜索更浅，不建议直接替换正式训练教师。

上述棋力与稳定性结果对应不扩宽版本。2026-10-07又加入下面的可选chance progressive widening，覆盖相关CPU回归、CUDA Graph短对打与多worker采集短窗；随后[冻结教师采集＋短训对照](../../artifacts/entity_attention_chance_training_probe_20261007/README.md)已完成三组各200局完整checked采集与500步固定训练，16主面板与独立复测共2000局已完成，未确认训练棋力收益；持续自迭代未验证，跨eager与compiled也不保证目标逐位一致。旧`reproduce_failure`固定世界分支重放不适用于lazy chance。新失败上下文记录simulation隐藏样本seed和独立chance seed，不能将旧固定牌堆重放当作相同随机轨迹。

### Chance progressive widening（可选）

在`native_chance`策略上另设`chance_widening={"k":1.0,"alpha":0.5}`即可开启；省略则保留原来每次从完整剩余牌池采样的行为。参数独立于`MCTSConfig`，旧self-play配置序列化与恢复fingerprint保持原样。独立`selfplay_fast`配置已接受`search_backend="native_chance"`和`chance_widening`；可选`record_search_diagnostics=true`将每根统计写入游戏metadata，默认关闭，不改变NPZ数组契约。[冻结教师采集＋短训对照](../../artifacts/entity_attention_chance_training_probe_20261007/README.md)记录实际调用链和训练结果。

每个决策动作下面，市场补牌／盲抽的具体chance事件按实际剩余牌池建上下文。该上下文的第N次访问最多保留`min(remaining_cards, max(1, ceil(k*N**alpha)))`种卡牌结果：尚有名额时从完整牌池均匀抽样并收入新的卡牌ID；名额已满时在已有不同卡牌ID中均匀抽样。每张剩余卡的物理概率相同，因此复用阶段按受限支持集上的归一化概率抽取；不按value挑结果，也不按自适应访问数加权。结果仍共同回传到动作N/W/Q。这是有限预算下的支持集近似，不是每次都从完整环境分布独立抽样。

只缓存卡牌ID，不复用完整后继State；规则推进、双方观察历史和该simulation的隐藏状态仍重新执行。上下文包含当前真正可抽的牌池，防止把对手刚重采样到手中的暗牌再次抽出。已有对手暗保留牌仍每simulation重新采样，本版本没有对root belief粒子做扩宽。决策动作集合本身不裁剪。

```python
from splendor.native_policy import NativeChancePUCTSearchPolicy, ChanceWideningConfig

policy = NativeChancePUCTSearchPolicy(
    frozen_model, device="cuda",
    config=MCTSConfig(simulations=1024, determinizations=1, seed=123),
    chance_widening=ChanceWideningConfig(k=0.25, alpha=0.5),
)
```

固定评测JSON在同一策略的`search_backend`／`search_config`旁边加入`"chance_widening":{"k":0.25,"alpha":0.5}`；已接通串行和native_batched路径。要求k>0、0<alpha<1；旧native／Python策略不接受此字段。直接作业的chance diagnostics增加扩宽参数、新抽样数、复用数、上下文数、已收入结果数及根动作各牌池的访问／结果集合，供内部核对，不作为玩家输入。失败上下文使用`sampled-chance-pw-puct-native-v1`。

[初次有限实验与复现命令](../../artifacts/entity_attention_chance_pw_20261007/README.md)：同权重、同200局面、S1024，平均逻辑深度从不扩宽的2.623提高至k1的2.893、k0.25的3.461，旧确定化为4.912。该次k0.25的固定请求耗时约增加16.7%；初次交付未测棋力，后续结果见下一段，不把深度增加直接解释为教师更强。该方法的参考定义见[MCMS论文§2.2](https://www.ijcai.org/papers13/Papers/IJCAI13-093.pdf)。

随后完成[k0.25/α0.5的6400次A/B/C/D复测](../../artifacts/entity_attention_chance_pw_noise_20261007/README.md)：相对不扩宽，D平均深度2.560→3.401、JS0.03072→0.04835、Top-1一致率85.20%→76.43%；相对旧确定化，C/D JS仍分别下降54.6%/43.0%。本轮诊断耗时比不扩宽多10.4%，这些是固定局面的eager测量。[同16500步权重、S1024的100 pair棋力](../../artifacts/entity_attention_chance_pw_strength_20261007/README.md)已完成200局自然终局：CPW为73胜2和125负、37.0%（95%区间30.75%–43.25%），仍明显弱于旧确定化。相对不扩宽31.25%提高5.75个百分点，但配对差值区间[−3.75,+15.00]个百分点，不能确认改善。该棋力实验没有训练更新；后续有限训练对照见上述冻结教师实验。当前不建议直接替换正式教师，持续训练及等墙钟棋力收益未验证。

又完成[同权重CPW对greedy的100 pair直接对照](../../artifacts/entity_attention_chance_pw_greedy_20261007/README.md)：CPW为145胜1和54负，纯胜率72.5%、得分72.75%（配对95%区间66.5%–78.5%），明显强于当前裸网，但此前对旧确定化的劣势仍成立。新增native_batched与greedy混合评测；greedy沿用零模拟的logits argmax，单请求eager推理。35项相关回归及四局真实GPU smoke通过；完整200局自然终局、零工程失败，未更新权重或切换教师。

## Policy / critic 固定深度搜索

### 目标

固定深度策略使用冻结 Actor–Critic 和 Python 规则引擎：

- 当前节点只展开 policy logit 最高的 top-k 合法候选；
- 双方使用同一个冻结 checkpoint 提供 policy prior 和 critic；
- 默认双方均取 top-5；
- 默认搜索 4 ply；
- 非终局叶节点使用 critic；
- 已正式终局的叶节点使用精确的 +1/0/-1；
- 从根玩家视角做 minimax 回传。

这不是 MCTS，也没有训练时梯度、PUCT、访问次数或搜索树跨决策复用。

### 信息安全

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

### 搜索语义

默认参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| top-k | 5 | 每个决策节点按 policy logit 展开前 5 个候选 |
| depth | 4 | 4 个模型决策边界 |
| determinizations | 1 | 每次真实决策采样一个信息集状态 |
| backup | minimax | 根玩家节点取 max，对手节点取 min |
| leaf | critic | 转换为根玩家视角 |
| leaf clip | [-1, 1] | 与 terminal reward 尺度一致 |

默认 `SearchConfig.fold_forced_phases=False` 时，一个 ply 是一个 <code>DecisionRequest</code>，而不是完整玩家回合，因此 PAYMENT、RETURN_TOKENS、CHOOSE_NOBLE 也会消耗 ply。显式启用 folding 后，单合法候选阶段在引擎内折叠，不消耗深度预算；现有阶梯配置与 README 网页示例启用 folding。这与 PPO 的模型决策语义一致，不会把支付或贵族选择静默固定成某个动作。

同一玩家在连续决策阶段仍取 max；只有 <code>GameEngine.current_player</code> 真正切换后才改为 min。终局价值不调用 critic：

- 根玩家胜：+1；
- 平局：0；
- 根玩家负：-1。

多个 determinization 时，每个根动作先在各自树中 minimax，再对根动作价值取平均。计算成本近似随 determinization 数量线性增长。

### 代码入口与评测分层

- 搜索策略与 determinization：`src/splendor/search_policy.py`；评测器和网页对局服务通过 `PolicyValueSearchPolicy` 接口调用，不在调用层实现搜索算法；
- 固定双方对局、种子/局数/并行配置、逐局保存、恢复与汇总：`src/splendor/evaluation.py`；
- 深度阶梯实验编排：`src/splendor/ladder_search.py`；它只生成每个 rung/block 的双方策略配置，并决定下一阶段；
- `src/splendor/evaluate_search.py` 是旧命令的兼容入口，已弃用；新评测使用 JSON 配置调用 `evaluation.py`；
- 受控 fork、合法动作与 digest 缓存：`src/splendor/engine.py`；
- 搜索测试：`tests/test_search_policy.py`。

评测策略通过 `SerializablePolicySpec` 声明。双方可以选择不同 checkpoint，也可以各自选择 checkpoint greedy 或 checkpoint + search 配置。评测器不检查或重写策略算法，只按统一对局协议调用策略。

搜索统计按每个策略自己的 `top_k` 动态记录 `chosen_rank_1_count` 到 `chosen_rank_K_count`，这些是恰好选中第 N 名的计数。旧的 `chosen_rank_top1/2/3_count` 与 `chosen_rank_beyond3_count` 保留，其中 top-2/top-3 是累计计数。`decision_quality` 汇总同时给出逐 rank 比例；比例分母为 `searchable_decisions`，单合法动作的强制决策不计入 rank 分布。通用评测 JSON 的双方 `policy_a_stats` / `policy_b_stats` 和逐局 sidecar 都保存逐 rank 计数。

单次固定评测写出汇总 JSON 和逐局 sidecar。每个 seat-swapped game 完成后，worker 会向 `pair-XXXXXXX.jsonl` 追加并 `fsync` 一条记录；只有一对的两局都完成后，该 pair 才会跳过续跑。配置指纹包括双方策略、checkpoint 内容摘要、搜索参数、种子、规则边界及 worker/device 配置，恢复时必须匹配。


## 回归与实验边界

PUCT：`test_mcts_policy.py`、`test_search_accel.py`、`test_native_search.py`；固定深度：`test_search_policy.py`、`test_search_policy_equivalence.py`、`test_search_policy_hidden_permutation.py`；合批：`test_batched_search.py`。旧 Python 合批的 CPU/GPU 排名只适用于当时实现，不用于判断当前 native 采集配置。现行性能证据见 [吞吐报告](../reports/inference_throughput_zh-CN.md)。
