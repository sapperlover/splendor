# 短期 value 与对手 policy 辅助目标

2026-10-27结果更新：[原六组离线实验及五模型10000 pair复核](../reports/auxiliary_offline_results_20261027_zh-CN.md)均已完成；policy辅助在线分支已提交13轮。下方2026-10-23至26日内容保留工程发布及当时启停记录，当前完成状态以本段及专题结果为准。

2026-10-26新增可组合的[到终局分数归属辅助](score_ownership_zh-CN.md)：双方各预测一级／二级／三级牌及贵族的新增分，共8路。使用score_ownership_weight单独控制，旧配置缺省为0并保持原训练身份；新104k仅分数归属配置使用4 worker／17GiB容量核算，原实验与原始记录保持。

2026-10-25新增[policy辅助最优离线模型→60k在线自迭代命令](../../artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/README.md)：`random_budget_iteration.continuation_auxiliary`显式接通模型头检查、compact缓存标签、replay身份与联合更新。新分支继续对手policy辅助0.15、短期value关闭；原在线配置省略此字段，行为与序列化保持不变，不自动升级历史冻结源码。该分支随后已启动并提交13轮；第12轮完整搜索60.20%，第13轮greedy63.75%。

2026-10-24最新：流式统计、4 worker共享只读供数、传输／同步优化已全部接入队列后两组，本次后两组均18GiB；后三项设为所有新建离线训练默认，缓存依池大小配置。原冻结源码／配置／计划保持，[新执行发布与183项回归、CUDA逐位验证](../../artifacts/offline_training_defaults_20261024/README.md)，[默认规则](offline_execution_zh-CN.md)。本次未启动正式训练；下文默认0／后两组未接入为此前记录。

2026-10-24最新执行更新：当前policy104k组与串行队列已按用户要求停止在115000步，未重启。`run_training_round`改用固定大小的会话累计统计；分离value／辅助路径只传状态和所需标签，指标与任务梯度范数合并回读，更新前非有限检查保留。87项回归及真实CUDA逐位对照通过；独立冻结发布与命令见[流式统计与传输优化](../../artifacts/training_streaming_20261024/README.md)。18GiB／4 worker配置保持，后续两个冻结实验未切换。

2026-10-24新增可选`training_workers`（默认0）。当前104k仅policy组执行改为18GiB和4 worker共享只读供数，独立256MiB验证缓存；主进程保持原抽样顺序和已消费RNG恢复边界。缓存／worker可作为执行参数更改，算法与历史身份保持，详见[实现、配置与验证](../../artifacts/training_prefetch_20261024/README.md)。

2026-10-24支持独立关闭短期value、仅训练对手policy辅助（主终局value仍训练）。[三组后续消融](../../artifacts/entity_attention_auxiliary_queue_20261024/README.md)依次为104k仅policy、104k双辅助、44k仅value，串行跟随当前完整流程。value辅助0时不注册短期头、不生成标签／计算loss和诊断；policy-only为784778参数。`predict_shortterm_value`默认true以加载旧模型；为保留原随机流，初始化消耗原短期头随机数后丢弃关闭头，policy-only与双辅助对手头初始化逐位相同。至少一项辅助必须为正，原目标公式和默认权重不变。

2026-10-23用户已授权[第21–37轮104k局仅短期value正式流程](../../artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023/README.md)，准备、验证与启动身份见该专题；下方两份R21–R31预设仍为原实现交付记录。新实验采用最新轻量抽样记录、5%完整主KL+MSE早停选点，并顺序对raw R37执行greedy1000pair／搜索A500pair。

2026-10-23实现，入口为 `splendor.offline_continuation`。两个任务只使用历史轨迹生成训练标签，共享原 attention 编码器；原主 policy 和终局 value 继续训练。现有在线配置未切换，新配置提供“仅短期 value”和“两项辅助”两组。旧短期进展任务仍关闭，已撤回的购买结算特征没有恢复。

## 短期 value

尺度为 **2、6、14个玩家回合的平均前视距离**。一次玩家回合指任一玩家完成一次主行动及其支付／退币／贵族等子决策；双方各行动一次计2回合。支付等子决策不另加回合。

设 `v_t` 是第t个主行动根节点的搜索value，`z`为终局胜负（胜1、和0、负−1），终局位于T。对每个H：

```text
lambda_H = H / (H + 1)                 # 2/3、6/7、14/15
y_H(T) = z
y_H(t) = (1-lambda_H) v_t + lambda_H y_H(t+1)
       = (1-lambda_H) Σ[k=0..T-t-1] lambda_H^k v_(t+k)
         + lambda_H^(T-t) z
```

包含当前根节点；终局吸收剩余尾部权重；没有硬截断。H是几何分布的平均前视偏移，不是半衰期，也不是“恰好第H回合的结果”。

公式依据[KataGo方法说明](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/docs/KataGoMethods.md#short-term-value-and-score-targets)与[同版本训练写入源码](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/cpp/dataio/trainingwrite.cpp#L411)：`nowFactor=1/(1+H)`，最后一个value取全部剩余权重。本仓库预测标量胜负期望，采用三路tanh输出和平均MSE；没有引入围棋的分数／所有权目标。

根value从v3原始 `per_world_root_value_sums` 与 `per_world_root_visit_counts` 还原：先对每个有效世界求 `ΣW/ΣN`，再对世界等权平均。不能用剪枝后的policy计数，也不能按世界访问量重新加权。计算EMA时先统一为seat0视角，最后转回样本玩家视角，避免双方交替时符号错误。

只监督主行动样本。v1/v2缺W统计时明确报错。单候选动作未搜索、自动跳过造成缺回合时，缺口及之前的EMA均不可精确还原，显式屏蔽；保留缺口后的连续有效后缀，不用终局或网络输出补造标签。日志记录 `shortterm_positions`、`shortterm_eligible_positions`、`shortterm_coverage` 及三个尺度的MSE。覆盖率分母是主行动的value权重，完整留出上的覆盖率可用于判断池是否适合该辅助。

## 对手 policy 的两层预测

按用户确认，目标是**对手下一次主行动的搜索分布**，不是整回合所有子决策的混合。当前主行动和支付等状态均可作输入，定位未来第一次对手主行动；若该目标的policy权重为0，当前样本也屏蔽，不能跳过去改取更后面的高预算行动。终局前无后续对手主行动时屏蔽。

第一层4类softmax，使用软标签CE：

| 类别 | 从未来搜索policy聚合的动作 |
| --- | --- |
| buy | buy_market、buy_reserved |
| take | take_distinct、take_pair |
| reserve | reserve_market、reserve_deck |
| other | pay、return_tokens、claim_noble；正常主行动标签为0 |

第二层为购买与保留各自独立的牌softmax。两路使用相同的当前可见牌集合：市场12槽、己方保留3槽、对手保留3槽、pending1槽，另加一个“不可见牌”类，共20槽。不存在、unknown占位及重复牌槽均屏蔽；pending与市场／保留重复时保留第一个可见槽。根据tier、奖励色、分值、原始费用匹配同一张牌，忽略来源和随观察玩家变化的支付缺口；当前标准牌库中这些属性唯一标识卡牌。

未来补入市场的牌、对手暗保留牌和ReserveDeck不能对应当前可见牌时，概率合入不可见类。不输入未来牌面、不根据未来合法动作制作输入mask。当前暂不可购买／保留但可见的牌仍参与预测，因为当前行动可能改变对手下一次主行动时的可用集合。

记聚合后的类型概率为 `q_buy,q_take,q_reserve,q_other`，牌级联合质量为 `q_buy(c),q_reserve(c)`，其和分别等于 `q_buy,q_reserve`。购买和保留的条件分布为 `p_buy(c),p_reserve(c)`：

```text
L_opponent = CE(q_type, p_type)
             - Σ_c q_buy(c) log p_buy(c)
             - Σ_c q_reserve(c) log p_reserve(c)
```

因此牌级CE自动按对应行动概率加权，没有购买／保留质量时该分支无梯度。两路不会与take／other混做一个softmax；若需读取无条件预测概率，分别取 `p_type(buy)*p_buy(c)`、`p_type(reserve)*p_reserve(c)`。对手标签采用归档的 `policy_target`，继承其中的世界等权与剪枝语义，乘未来该主行动的 `policy_weight` 及当前 `value_weight`。设计与[KataGo的下一手policy训练目标](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/cpp/dataio/trainingwrite.cpp#L1172)同旨，动作分层和不可见牌归类是本仓库适配。

## 训练与验证诊断（2026-10-23追加）

对手统计每次训练更新、固定子集验证及完整留出验证均记录。记当前value权重为 `v_i`，未来对手主行动的policy权重为 `a_i`，`w_i=v_i*a_i`。覆盖率使用是否存在可用标签的指示量；损失与牌质量继续使用实际policy权重：

| 日志字段 | 定义 |
| --- | --- |
| `opponent_label_coverage` | `Σ v_i * 1[a_i>0] / Σ v_i`；分母包含终局前无对手主行动、以及未来policy权重为0的当前样本 |
| `opponent_label_positions`、`opponent_eligible_positions` | 上述分子／分母；均以当前value权重计数 |
| `opponent_positions` | `Σ w_i`，实际训练损失分母；policy权重不是0/1时与label_positions不同 |
| `opponent_buy_mass`、`opponent_reserve_mass` | `M_buy=Σ w_i q_buy(i)`、`M_reserve=Σ w_i q_reserve(i)` |
| `opponent_buy_unknown_mass`、`opponent_reserve_unknown_mass` | `U_buy=Σ w_i q_buy(unknown,i)`，保留同理 |
| `opponent_buy_unknown_ratio`、`opponent_reserve_unknown_ratio` | `U_buy/M_buy`、`U_reserve/M_reserve` |
| `opponent_unknown_ratio` | `(U_buy+U_reserve)/(M_buy+M_reserve)` |
| `opponent_buy_conditional_ce`、`opponent_reserve_conditional_ce` | 各自原始CE总和除以 `M_buy`／`M_reserve`，包含未知牌类 |

原 `opponent_buy_ce`／`opponent_reserve_ce` 仍除以 `Σ w_i`，用于原训练loss；新条件CE只作诊断。所有比例在完整集合的原始总和上相除，不平均各batch的比例或条件CE。分母为0时新比例／条件CE输出JSON `null`，同时保留质量为0的字段；原训练loss保持可微零值。

共享编码器梯度默认训练总步数1及每100步记录，配置顶层 `encoder_grad_norm_every_steps` 可改为1以逐步记录，0关闭梯度诊断。恢复按累计 `total_train_step` 继续采样节奏。频率属于记录设置，不参与训练身份，可在恢复时更改，也允许旧run未保存该字段；原始config.json保留初始设置，实际请求的设置写入每个会话的source identity，其余训练配置仍严格冻结。统计包括状态数值／牌／贵族编码器、角色embedding、attention blocks及启用时的market_relation_encoder；不包括动作编码／cross-attention、主policy/value readout与辅助头。

`encoder_grad_norm_<task>` 是对应归一化loss对这些参数的、全局裁剪前的L2范数；`encoder_grad_norm_<task>_weighted` 再乘该任务在总loss中的系数。任务名与系数如下：

| task | 系数 |
| --- | --- |
| `policy`、`value` | learner的policy_loss_weight、value_loss_weight |
| `shortterm_h2`、`shortterm_h6`、`shortterm_h14` | shortterm_value_weight / 3 |
| `shortterm_value` | shortterm_value_weight；三个尺度平均loss的梯度 |
| `opponent_action`、`opponent_buy`、`opponent_reserve` | opponent_policy_weight；购买／保留使用原训练的联合质量CE |
| `opponent_policy` | opponent_policy_weight；三个对手分支之和的梯度 |

`encoder_grad_norm_total` 从实际联合反向后的编码器`.grad`读取，仍在全局裁剪之前。各任务梯度方向不同，不能用分支范数相加代替该总范数；合计任务与其分支也不能重复相加。梯度诊断通过额外 `autograd.grad` 求导、保留计算图，不累积到训练`.grad`；最终仍执行原总loss反向、一次全局裁剪和一次AdamW更新。采样点有额外反向计算开销，验证与搜索不求这些梯度。未采样的训练行不含这些梯度字段。

追加验证与原始结果见[诊断记录](../../artifacts/continuation_auxiliary_20261023/diagnostics/README.md)。

## 模型、训练与配置

架构身份 `entity_attention_continuation_v1`。短期value头沿用原value读取方式；对手类型头读取状态，牌头读取对应牌token及global／双方玩家token，输出购买／保留两路logits，不增加attention层。128×2关系模型参数：原583682，仅短期value650757，两项辅助851853。普通forward、value-only与原生搜索均不执行辅助头；额外训练头增加训练计算和checkpoint大小。

标签在每局首次载入compact cache时生成，纳入缓存字节记账；原NPZ、主标签、split及抽样RNG规则保持。policy高预算和value全预算各B256，辅助在value batch上训练并复用其状态编码；一次AdamW更新、一次全局梯度裁剪。默认新增loss为 `0.25*mean(MSE_H) + 0.15*L_opponent`，权重是起始配置，未证明最优。仅短期配置将 `opponent_policy_weight=0`，不注册对手头、不生成其标签。

两份配置使用现有第21–31轮44000局manifest（绝对round_index63–73，配置1基64–74），原始整局train/validation切分；从零初始化原主干，独立辅助seed、新AdamW、lr1e-4、value曝光上限12次、每5%完整留出及三次严格上升早停。按完整留出 **主policy KL＋主终局value MSE** 选择整个checkpoint，辅助loss只作诊断，不加入选点指标。配置／协议／模型头身份均受恢复检查；不能把旧run直接当成新辅助run恢复。

```bash
# 仅短期value；新建独立输出目录。真正训练时去掉 --prepare-only。
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.offline_continuation \
  --config configs/entity_attention_v1/offline_shortterm_value_r21_r31_20261023.json \
  --output runs/shortterm_value_r21_r31_20261023 --prepare-only

# 同时开启两项；同一命令支持中断恢复。
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.offline_continuation \
  --config configs/entity_attention_v1/offline_continuation_r21_r31_20261023.json \
  --output runs/continuation_r21_r31_20261023 --prepare-only

# 训练完成后在原命令添加 --select-only 选点。
```

准备阶段冻结数据身份与计划，不扫描全池标签覆盖；首次训练／验证载入各局时生成标签。两份R21–R31预设首次交付时正式训练尚未启动，只完成[有限工程验证](../../artifacts/continuation_auxiliary_20261023/README.md)；随后R21–R37仅短期value实验独立验证并曾启动，现按用户更正停止，只提供手动命令，见上方专题。工程短测不能判断两个任务的棋力收益。最初实现入口是离线训练；2026-10-25已为显式配置continuation_auxiliary的新在线分支接通调用链，旧在线配置不自动启用。
