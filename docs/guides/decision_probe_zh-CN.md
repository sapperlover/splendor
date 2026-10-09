# 固定 256 局面的 checkpoint 决策诊断

入口为 [`splendor.decision_probe`](../../src/splendor/decision_probe.py)。长期测试集固定在 [`bank.json`](../../artifacts/decision_probe_v1_20261009/bank.json)，采集与基线结果见[产物说明](../../artifacts/decision_probe_v1_20261009/README.md)。用途是用较少搜索预算观察 checkpoint 的决策变化，不能把 JSD、动作变化率或 value 漂移换算成胜率或棋力提升。

## 测试集口径

重新运行独立、完整的 checked 引擎对局，没有读取训练 NPZ、训练局面或监督留出集。四种固定启发式组合为 Score/Score、Score/Development、Development/Score、Development/Development。每种组合各贡献 64 个局面，前中期和后期各 32 个；整个集合各阶段 128 个。每个局面来自不同游戏种子，只保留主行动且至少两个候选动作的真实决策，候选顺序原样保存。

阶段使用与既有分阶段搜索实验一致的公开阈值：任一方公开分数 ≥10 为后期，其余为前中期。在预定阶段的合格决策中等概率抽取一个，选位随机种子固定。若某局没有该阶段的合格决策，保留完整排除记录，并用新的独立种子补采；不依据任何 checkpoint 的判断筛选局面。阶段各占一半是诊断配额，整体均值不代表自然对局中的阶段频率。

256 个根节点、完整来源对局、排除对局、配置、种子、源码及 native 扩展哈希均保留。每个根节点都从来源对局前缀 checked replay 核对，完整来源也重放至自然终局。评测只解码保存的 `DecisionRequest`，不会把真实隐藏牌或未来对局轨迹交给搜索。测试局面和所有来源种子长期保留作诊断，不用于训练、教师采集或优化器更新；后续实验避开该种子域。反复用此集合选配置后，其结果属于开发诊断，不再是独立棋力验收。

## 每个新 checkpoint 的一条命令

从仓库根目录执行，替换 checkpoint 和新输出目录。`--resume` 可在第一次启动时带上，中断后原命令续跑。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u -m splendor.decision_probe evaluate \
  --checkpoint runs/你的实验/checkpoints/final.pt \
  --output runtime/decision_probe/你的checkpoint名称 \
  --compare-to artifacts/decision_probe_v1_20261009/baseline_16600_s1024_n1 \
  --resume
```

默认 native S1024/N1、c_puct=1.5、无根噪声、温度 0、强制阶段折叠、单次模拟最多 512 transitions、FP32 eager、TF32 关闭。默认两个固定搜索种子 2026101500/2026101501，每个种子各跑一次完整 S1024/N1，总预算 256×2×1024=524288 simulations。两个种子分别选动作，再平均逐局面诊断指标，不将访问分布先平均后选动作。相同种子在不同 checkpoint 上配对，约束世界抽样带来的比较波动；模型导致搜索路径不同，配对并不能消除全部随机性。

默认自动选择 CUDA/CPU；最多同时保留 64 棵根搜索树，完成后释放，逐根原子保存。不设 RAM 硬上限，不长期积累已完成搜索树。中断最多重算当前尚未落盘的一批；完成的逐根结果会跳过。相同输出目录严格核对 checkpoint 内容、bank、源码、搜索参数、种子、设备和推理配置，禁止把不同任务混进同一结果。锁禁止两个进程同时写同一个目录。

`--compare-to` 也可以指向上一个 checkpoint 的完成目录，观察相邻代际变化。未来输出默认放入忽略的 `runtime/`；值得长期复核的汇总及其配套 `searches.json` 按产物约定另行保留。基线目录内的模型路径是来源记录，离线比较不需要再次加载该模型。

## 指标及解读

`summary.json` 是单个 checkpoint 的绝对诊断，`searches.json` 保存每个局面、每个搜索种子的原始结果，`comparison.json` 是相对 `--compare-to` 的配对诊断。三组汇总为 `all`、`early_mid`、`late`，并保留逐局面数据、mean/median/p90；`defined_values` 是该统计中有定义的观测数，绝对指标通常是根节点数×重复数。

| 指标 | 定义与用途 |
| --- | --- |
| `action_changed` | 温度 0 下搜索实际首选候选索引是否变化；逐根逐种子配对后平均。根节点相同、候选顺序相同才可比较。 |
| `visit_jsd` | 搜索访问目标分布的 Jensen–Shannon divergence，使用自然对数，范围 0–ln2；不是开平方的 JS distance。N>1 使用搜索原生 `policy_target`，即各世界归一化访问分布的均值，不能直接用总访问数除以总预算代替。 |
| `high_visit_rank_inversion` | 参照 checkpoint 中累计覆盖 80% 访问质量的最小高访问动作集合，严格有序动作对在新 checkpoint 中的反转比例。参照并列对排除，新分布并列计半次反转；只有一个动作时没有可排序对，记 null。 |
| `high_visit_mean_rank_displacement` | 同一高访问动作集合在完整候选排序中的平均绝对名次位移，并列使用中位名次。 |
| `top3_overlap` / `top3_order_changed` | 两边前三候选集合交集比例／顺序变化。少于三个候选则取全部；并列按保存的候选顺序排序，接近并列或零访问项应结合 margin 解读。 |
| `baseline_chosen_mass_change` | 新模型分配给参照首选动作的访问概率变化；可发现动作未切换但支持度明显改变。 |
| `mass_on_baseline_top3_change` | 新分布在参照前三动作上的质量增减。 |
| `confident_action_change` / `near_tie_action_change` | 参照访问第一、第二名差距 ≥0.2／<0.05 时的动作变化率；分母随参照 checkpoint 改变，均是描述性分组。 |
| `prior_jsd` / `root_value_change` | 网络根 policy 先验变化，以及根 value 有符号／绝对变化；分辨 policy 与 critic 的变化来源。 |
| 访问熵、有效动作数、top1 质量及 margin | 判断搜索趋于集中、分散，或首选在相近动作间摇摆。 |
| 搜索对先验的动作改写率及 JSD | 判断搜索是否仍在明显改写网络的直接决策。 |
| 平均深度、终局命中比例、transition cap 次数 | 检查前中期／后期搜索是否更深、是否更多模拟直接到达终局、是否有工程截断。 |

每个 checkpoint 额外记录两个搜索种子之间的动作不一致率和访问 JSD（`within_checkpoint_seed_variation`），比较报告同时带上两边结果。若跨 checkpoint 变化与同权重跨种子变化相近，就不能只凭动作变化率声称学到了新策略。该背景波动不能直接相减当作“去噪后的真实变化”。同 checkpoint、同种子的复现用于工程可重复性，不测棋力。

这些数据回答“模型是否改变了决策、在哪个阶段、改了哪些高访问动作、变化来自先验还是 value”，没有优劣方向。较低 JSD 也可能只是停滞，较高动作变化率也可能是退化。该固定集合主要覆盖两类启发式策略形成的局面分布；不能外推为所有对手、所有状态上的泛化或棋力证明。

本版256个根节点没有对手暗保留牌，不能用来验收该类隐藏信息估计。阶段均值和搜索终局命中率可以帮助定位变化，但也不能直接解释搜索对战胜率的原因。已完成的16500→16600示例、搜索随机性和实测成本见[结果说明](../../artifacts/decision_probe_v1_20261009/README.md)。

## 更改搜索口径与离线比较

可用 `--simulations`、`--determinizations`、`--backend native_chance`、`--chance-pw-k`、`--chance-pw-alpha` 改搜索。为新搜索口径重新跑同一参照 checkpoint，并使用新的输出目录，再让新模型与它比较。默认拒绝搜索参数或执行源码不一致的结果；确需比较搜索算法时用 `--allow-search-change`，报告会标明变化，此时不能把差异归因于 checkpoint。仅 Git HEAD 改变而执行文件内容相同不会误判为算法变化。

可选 `--inference-mode compiled --device cuda` 使用原生 CUDA Graph 路径；默认及已保存正式基线使用 eager。设备、批次组成和推理路径不同可能带来浮点差异，报告保留两边配置；对微小差异的判断应保持执行配置一致。`--repeat-seeds` 可以改重复数，但必须为所有待比较模型重新生成同种子参照；单种子时没有搜索种子波动统计。

已经完成的两份结果可以只运行离线比较，无需搜索或 checkpoint 文件：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -m splendor.decision_probe compare \
  --a artifacts/decision_probe_v1_20261009/baseline_16600_s1024_n1 \
  --b runtime/decision_probe/你的checkpoint名称 \
  --output runtime/decision_probe/你的checkpoint名称/comparison.json
```

测试集已经冻结，日常评测无需重采或重复完整游戏审计。需要核对来源时用 `python -m splendor.decision_probe verify`，它会重放并更新 `verification.json`。未来新建测试集必须使用新的目录和新种子；原 bank 不覆盖。
