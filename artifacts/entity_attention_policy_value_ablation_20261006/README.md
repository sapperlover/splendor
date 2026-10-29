# Entity attention：固定数据 policy / value 消融（2026-10-06）

2026-10-23归档位置更新：本专题关联`runs/entity_attention_policy_value_ablation_20261006`中的4个抽样账本／逐步日志已按用户授权外置至[当日T7归档](../repository_archive_20261023/README.md)，原路径保留相对链接。复算这些原件前需解包；复训前将需写入的文件恢复为原位真实文件。历史指标、正式结果及来源身份不改写。

本实验检查：同一模型在同一批自对弈标签上，去掉 value 更新、或阻断 value 向共享 trunk 的梯度，是否改善 policy 拟合及实际棋力。正式自迭代仍停在第 50 新轮 / 14500 步；本次没有采集新数据、恢复正式训练或改动正式 learner。

三主分支各 500 步、17 个棋力面板 / **2700 局全部有效**。另做了 256 局面的小样本拟合、policy-only 同数据延长到 2000 步及组合模型数值复核。原始证据见 [fitting.json](fitting.json)、[analysis.json](analysis.json)；完整主协议见 [specification.json](specification.json)。

## 结论与适用范围

**本实验中，联合训练 500 步的主要损伤落在 value 函数及其搜索效果上，而非已经证实 policy 被共享 trunk 毁坏。** 普通联合分支 MCTS 对同一 PPO 的得分从起点 62.50% 降到 37.75%，两者各 200 局；配对下降 24.75 个百分点，95% 区间 `[-34.00, -15.50]`。其 greedy 得分仍接近起点。保留联合训练后的 policy，仅换回起点的完整 value 函数，在共同 50 种子上从 43.00% 恢复到 65.00%，恢复 22.00 个百分点，区间 `[+11.00, +33.00]`；同组起点为 67.00%。

去掉 value 更新或阻断其到 trunk 的梯度，都避免了这次联合 500 步的大幅搜索退化，但尚未证明超过起点。移除 value 反传能改善训练集 policy 拟合，却没有显示相应的留出或裸网棋力增益；不能因此宣布彻底拆成两套网络就会提升长期训练。

更充分地拟合当前标签也没有自动带来泛化：只训 policy 延长到 2000 步，训练 KL 继续降到 0.3214，留出 KL 的点估计反而升到 0.4423。256 局面可以拟合到 KL 0.0345，说明这条训练调用链能记住小样本；它并未解决大池拟合或棋力提升问题。

这是一份 **单起点、单数据/训练抽样种子、最新一轮 175 训练局、固定教师标签** 的诊断。不能从中认定原 3000 局历史窗口过长，也不能认定提高 policy 新数据占比会提升棋力。原正式第 50 轮使用更广历史池 / 100 步，数据重复强度不同；本报告不将最新-only / 500 步的损伤套用到所有正式在线轮次。

下一步建议保留有限对照：policy 可按先前候选方案逐步提高最新份额，但 **value 先保持较广历史的均匀采样，避免同步大幅提高最新份额**；另设阻断 value 到 trunk 梯度的对照。继续采用每轮 100 步的有限阶段观察，不把小池 KL 降到很低当成目标。分离采样、长期 detach 或完整双网络均尚未实现和验收；本次没有启动任何新正式分支，也不建议长期冻结旧 value 不再学习。

## 棋力结果

表中百分比为对 PPO 得分，包含和局半分。greedy 每节点均 200 局；MCTS 的局数注明在单元格内。两列中的 PPO 分别使用 greedy / MCTS，因此只能比较各模型相对同模式起点，不能用两列之差衡量搜索增益。

| 分支 / 更新步 | greedy 得分（200 局） | S1024/N1 MCTS 得分 |
|---|---:|---:|
| 起点 0 | 67.25% | 62.50%（200 局） |
| joint 100 | 66.25% | 59.00%（100 局） |
| joint 500 | 66.50% | 37.75%（200 局） |
| policy_only 100 | 68.50% | 59.50%（100 局） |
| policy_only 500 | 64.50% | 62.50%（200 局） |
| value_detached 100 | 66.50% | 68.00%（100 局） |
| value_detached 500 | 63.50% | 67.50%（100 局） |

相对同模式起点的配对差值（单位：百分点，未做多重比较校正）：

| 节点 | greedy 差值 / 95% 区间（100 种子） | MCTS 差值 / 95% 区间 |
|---|---:|---:|
| joint 100 | −1.00 / [−9.00, +7.00] | −8.00 / [−19.00, +3.00]（50 种子） |
| joint 500 | −0.75 / [−10.00, +8.25] | −24.75 / [−34.00, −15.50]（100 种子） |
| policy_only 100 | +1.25 / [−7.00, +9.25] | −7.50 / [−17.50, +3.00]（50 种子） |
| policy_only 500 | −2.75 / [−12.25, +6.50] | 0.00 / [−9.25, +9.50]（100 种子） |
| value_detached 100 | −0.75 / [−9.75, +8.00] | +1.00 / [−10.00, +12.00]（50 种子） |
| value_detached 500 | −3.75 / [−13.50, +5.75] | +0.50 / [−10.50, +11.50]（50 种子） |

所有 greedy 差值区间跨过 0。不能仅以 policy_only 100 的 68.50% 或 detached 100 的搜索 68.00% 选定“最佳更新配置”。MCTS 的其他关键比较：

| 配对比较 | 共同种子数 | 左 / 右得分 | 差值及 95% 区间（百分点） |
|---|---:|---:|---:|
| joint 500 − joint 100 | 50 | 43.00% / 59.00% | −16.00 / [−28.00, −4.00] |
| policy_only 500 − joint 500 | 100 | 62.50% / 37.75% | +24.75 / [+15.25, +34.25] |
| detached 500 − joint 500 | 50 | 67.50% / 43.00% | +24.50 / [+12.00, +36.50] |
| detached 500 − policy_only 500 | 50 | 67.50% / 53.50% | +14.00 / [+0.50, +28.00] |
| detached 500 − detached 100 | 50 | 67.50% / 68.00% | −0.50 / [−10.00, +9.50] |
| 最新-only joint 100 − 原正式第 50 轮 | 50 | 59.00% / 56.00% | +3.00 / [−9.00, +15.00] |

detached 相对 policy_only 的 500 步优势只在本次 50 种子、未校正区间上出现，不能据此宣布长期优于 policy_only；二者都未确认超过起点。原正式第 50 轮是此前补测的外部参考，其完整 200 局得分为 61.75%，共同前 50 种子为 56.00%；这不是本次三分支同 batch 的消融。对比其整体均值与本次 100 局候选均值会得到误导的差值。

固定 **完整起点 value** 的三个面板均为同样的 50 种子 / 100 局：

| 500 步 policy 来源 | 使用自身 value（共同 100 局） | 换为起点 value | 换 value 的差值 / 95% 区间（百分点） |
|---|---:|---:|---:|
| joint | 43.00% | 65.00% | +22.00 / [+11.00, +33.00] |
| policy_only | 53.50% | 63.50% | +10.00 / [−0.50, +21.00] |
| value_detached | 67.50% | 61.50% | −6.00 / [−18.50, +6.50] |

同组起点为 67.00%。三个固定 value 模型相对起点的区间均跨过 0；在 value 相同时，policy_only − joint 为 −1.50 个百分点 `[-12.50, +9.50]`，detached − joint 为 −3.50 `[-15.00, +8.50]`，也没有确认 policy 消融带来搜索增益。此对照有力支持本次 joint 的 value 函数是重要损伤来源，但不排除 policy 改变、policy/value 交互或其他起点上的不同结果。

## 固定条件

- 起点：正式第 50 新轮的 `round-049/checkpoints/initial.pt`，累计 14400 步。其模型权重与该轮实际采集教师 `round-048/checkpoints/final.pt` 相同；两者 checkpoint 文件哈希不同，不能以文件字节相同表述。
- 数据：仅该轮采集的 200 局 S1024/N1 自对弈数据，固定为 175 训练局 / 14427 局面、25 留出局 / 2051 局面。policy 权重非零的局面分别为 11856 / 1698。按整局划分，三个分支的划分相同。数据内的 `data_round=58` 是继承源轮编号后的采集索引，对应当前分支第 50 新轮，不是本次又训练到第 58 新轮。
- 所有分支使用 **100% 本轮数据**，不混入历史池或离线数据。本实验不是此前最新数据占 25% 的续训方案，也不能直接判断 3000 局历史窗口的长短。
- 每个主分支训练 500 步，batch 256，观察 0 / 100 / 300 / 500 步，累计步数分别为 14400 / 14500 / 14700 / 14900。500 步共 128000 次局面抽样，约为 14427 训练局面的 8.87 倍；有放回抽样，不代表完整遍历 8.87 个 epoch。
- AdamW 初始状态、模型参数、Torch RNG、数据抽样 RNG 全部从同一 checkpoint 恢复；学习率与 weight decay 均为 `1e-4`，全局梯度裁剪阈值 1。三分支逐步核对实际 batch 引用及最终抽样 RNG，500 个 batch 完全相同。
- 固定教师标签不刷新。采集教师带根噪声（fraction 0.25 / alpha 0.3），前 10 个双方完成回合使用温度 1，之后温度 0；棋力评测关闭根噪声，温度 0。

| 分支 | policy 更新 | value 更新 | 共享 trunk 接受 value 梯度 |
|---|---|---|---|
| `joint` | 正常 | 正常 | 是 |
| `policy_only` | 正常 | critic 参数冻结，梯度为 None | 否 |
| `value_detached` | 正常 | critic 在 detached 状态表征上训练 | 否 |

`policy_only` 的 critic 不接受 AdamW 旧动量或 weight decay 更新，但其输入来自训练中的 trunk，所以 **value 函数仍可能变化**。`value_detached` 保留原全局梯度裁剪；critic head 的梯度仍可改变 policy 的裁剪系数，因此它只消除了 value 到 trunk 的直接反传，没有完全分离两个优化过程。

全部实现限定在本目录的诊断脚本。`policy_only` checkpoint 的 loss weight 0 用来记录实际实验目标，可用于本次只读 frozen-model 评测；现行正式 `LearnerConfig` 仍要求正权重，不应直接将这些诊断 checkpoint 当作正式续训入口。

## 固定数据拟合

KL 是对有 policy 标签的局面计算 `KL(教师搜索分布 || 裸网分布)`，即交叉熵减标签熵；value 为全部有效 value 标签上的 MSE。两个指标均使用对应权重的分母。每个观察节点遍历全部训练集和全部留出集，表中不是训练末尾一个随机 batch 的 loss。

| 分支 / 更新步 | 训练 policy KL | 留出 policy KL | 训练 value MSE | 留出 value MSE |
|---|---:|---:|---:|---:|
| 起点 0 | 0.4258 | 0.4262 | 0.8204 | 0.7619 |
| joint 100 | 0.4096 | 0.4179 | 0.7144 | 0.7660 |
| joint 300 | 0.4095 | 0.4232 | 0.4830 | 0.8726 |
| joint 500 | 0.4014 | 0.4278 | 0.3609 | 0.9613 |
| policy_only 100 | 0.4035 | 0.4212 | 0.8503 | 0.8059 |
| policy_only 300 | 0.4075 | 0.4400 | 0.8225 | 0.7744 |
| policy_only 500 | 0.3835 | 0.4274 | 0.8091 | 0.7611 |
| value_detached 100 | 0.4015 | 0.4179 | 0.7766 | 0.7581 |
| value_detached 300 | 0.4080 | 0.4403 | 0.7458 | 0.7629 |
| value_detached 500 | 0.3830 | 0.4264 | 0.7107 | 0.7827 |

去掉 value 到 trunk 的更新后，训练集 policy KL 更低，但 500 步留出 KL 三者都接近起点。联合训练的 value 训练误差大幅降低、留出误差升高，呈现过拟合迹象。不能将单个 25 局留出集的变化直接推广为普遍退化：按整局配对 bootstrap 10000 次，joint 500 相对起点的留出 MSE 差为 `+0.1994`，95% 区间 `[-0.0606, +0.4679]`，跨过 0；policy KL 差为 `+0.0015`，区间 `[-0.0072, +0.0106]`。

联合 / policy_only / detached 500 步的留出 value 函数相对起点 RMS 变化分别为 `0.5836 / 0.1403 / 0.2606`。这些是预测函数变化量，不是 value MSE。完整整局重采样结果和原始每局统计见 [holdout_uncertainty.json](holdout_uncertainty.json)。

搜索在 `native_search.cpp` 的 `SearchJob::submit` 中把网络 value 裁剪到 `[-1, 1]`，补充统计如下（同一 25 局留出集，描述性点估计）：

| 节点 | 原始 value MSE | 搜索裁剪后 MSE | 越界局面比例 | 原始预测标准差 | 裁剪预测与结果相关系数 |
|---|---:|---:|---:|---:|---:|
| 起点 | 0.7619 | 0.7619 | 0.63% | 0.4369 | 0.5076 |
| joint 500 | 0.9613 | 0.9360 | 20.87% | 0.7508 | 0.3941 |
| policy_only 500 | 0.7611 | 0.7611 | 0.05% | 0.4222 | 0.4951 |
| value_detached 500 | 0.7827 | 0.7813 | 4.39% | 0.5099 | 0.4696 |

联合训练的 value 变化不只是少数输出越界；裁剪后的误差仍升高，预测幅度增大、相关性降低。相关系数与符号正确率也不是搜索棋力本身，尤其不能代表尚未覆盖的搜索叶局面。原始统计与 checkpoint / native 源码哈希见 [value_profile.json](value_profile.json)。

## 梯度与小样本拟合探针

同一组 8 个固定 batch 的起点共享 trunk 梯度，policy 与 value 的平均余弦为 `+0.8609`，8 个都同向；两者平均范数分别为 `2.62 / 12.07`。等权 loss 不代表等量梯度。此处没有观察到起点普遍方向冲突，却观察到 value 梯度范数较大。后续节点的余弦有正有负，不能仅凭少数探针将长期自迭代退化归因为梯度冲突。

探针在每个节点计算的是假设同时使用两项原始 loss 时的梯度；`policy_only` 和 `value_detached` 的 value 探针 **不是该分支实际反传到 trunk 的梯度**。实际训练梯度裁剪次数及所有节点探针见 [analysis.json](analysis.json) 和 [fitting.json](fitting.json)。

三主分支均 500 / 500 步触发全局裁剪。不能把仅损失权重等于 1 或起点两项梯度同向，理解为 value 更新对 policy / 表征影响很小；也不能只凭裁剪次数认定其是唯一原因。

另做一个探索性拟合检查：从训练集选 256 个互异且有 policy 标签的局面，仍从同一起点恢复，只训 policy，batch 64 / 1000 步。固定子集 KL 从 `0.4858 → 0.2189 → 0.0571 → 0.0345`（0 / 100 / 500 / 1000 步），而原 25 局留出 KL 为 `0.4262 → 0.5428 → 0.5888 → 0.6081`。

这证明当前调用链能拟合这 256 个固定标签，并不证明模型能充分表示全部 14427 局面，也不证明记住标签会提升棋力。它在查看主训练结果后添加，预算、batch 和抽样方式不同，不作为三分支公平比较或挑选棋力模型的依据。原始引用、配置、版本与曲线见 [capacity_probe.json](capacity_probe.json)。

为了检查全池是否仅仅 500 步训练不足，再从主 policy-only 500 步 checkpoint 的模型、AdamW 与 RNG 原样接续，继续同一 175 训练局 / batch 256，到 1000 / 2000 步。critic 始终冻结，主三分支产物不覆盖。

| policy-only 更新步 | 全训练集 KL | 原 25 局留出 KL |
|---|---:|---:|
| 500 | 0.3835 | 0.4274 |
| 1000 | 0.3598 | 0.4311 |
| 2000 | 0.3214 | 0.4423 |

新增 1500 步及节点诊断耗时 31.31 秒，确实继续降低训练误差，但没有留出改善。2000 步的全池 KL 仍远高于 256 局面探针，所以不能据此声称模型已充分学会整个教师分布，也不能仅凭这一点区分容量、优化和标签异质性。该延长检查在看过 500 步结果后添加，未补测 1000 / 2000 步棋力，不与其他分支进行等预算比较。实际抽样引用、父 checkpoint、两份新 checkpoint、命令 / 环境 / 源码身份见 [policy_only_longer.json](policy_only_longer.json)。

## 棋力协议

17 个主面板，共 2700 局，全部完成且有效；附加 S8 两局只验证组合模型调用链，不计入主棋力。

- 固定对手：同一个 PPO 5.5M **latest** checkpoint，SHA-256 `009d17d2e9bc0d879142d7ca3ff1777a6f8caa9dda9243c6a2d5eb3538b5d5b0`。候选起点及三个分支 100 / 500 步均评测裸网 greedy 和 MCTS。
- greedy：候选和 PPO 都不搜索，每节点 100 个种子换座 / 200 局。
- MCTS：两侧都用 S1024/N1，`c_puct=1.5`、温度 0、无根噪声，原生 batched 搜索，FP32 CUDA compiled / reused。起点、joint 500、policy_only 500 各 100 种子 / 200 局，其余四节点各 50 种子 / 100 局。
- 三个 500 步候选各增加 50 种子 / 100 局 MCTS：policy 使用训练后网络，**完整 value 函数（含原 trunk）固定为起点网络**。这项对照避免把“冻结 value head”误当成“冻结 value 函数”。
- 100 个种子为 `13500000..13500099`，50 种子面板取前 50 个。配对比较只使用双方共同种子，整对重采样 10000 次，种子 `13530000`。50 种子候选相对起点的差值应与起点前 50 种子相比，不能直接减起点全 100 种子均值。
- 得分为 `(赢局 + 0.5 × 和局) / 有效局数`。这是重复 DEV 面板，不是独立 FINAL；相同种子保证配对的初始条件，双方行动不同后游戏轨迹仍会不同。

**两个评测模式中的 PPO 也分别采用 greedy / MCTS，因此不能直接将同一候选的两列得分差解释成搜索增益。** 可比较各分支相对同模式起点的变化。多节点查看带来选择偏差，本报告给出的配对区间没有进行多重比较校正。

组合模型的数值与 compiled state-only value 路径独立验证见 [compiled_composition_verification.json](compiled_composition_verification.json)。必须用本目录 `evaluate.py` 执行固定 value 面板：现行仓库评测 CLI 不读取这些配置的组合模型 metadata，直接套 stock CLI 会变成使用候选自身 value 的普通评测。

## 验证与成本

CPU smoke 中自定义 joint 单步与现行 `AlphazeroLearner.train_step` 的模型参数差为 0；policy-only critic 梯度为 None 且参数不变；detached value 对共享 trunk 的直接梯度为 None；组合模型 eager 的 policy / value 数值与各来源网络完全相同。独立 compiled 校验最大误差：policy `6.44e-6`、value 和 state-only value 均 `1.98e-6`。compiled API 保留候选 bucket padding，复核比较实际请求的 35 列（返回 bucket 48 列）；首次复核脚本未裁去额外列导致 shape 断言失败，已按现行 API 契约修正，主棋力调用链未因此失败。

2700 个主对战均无 exception、非法动作、invariant failure 或工程截断。联合 500 步普通搜索面板为 75 赢 / 124 负 / 1 和，没有 policy cycle 和局，下降主要体现为负局增加。完整验证还核对 210 项冻结正式输入、全部 batch ledger、每个候选模型与结果身份、正式源码 / native 扩展及训练输入保全。

三个主分支各 500 步并含节点诊断的循环耗时分别为 14.54 / 14.26 / 14.25 秒，不含所有进程准备成本；17 个棋力面板记录的评测耗时合计 3929.18 秒（65.49 分钟）。这些是本机此次测量，包含已存在编译 cache 的不确定冷热状态，不能当作纯训练或纯推理性能基线。RTX 4050 Laptop 6GB、i5-13500H、PyTorch 2.14.0+cu130 / Python 3.12.14；细节以各 recording 原始环境为准。

## 产物与复核

- 协议、输入哈希及逐局划分：[specification.json](specification.json)。
- 同 batch 训练曲线、实际 checkpoint 身份：[fitting.json](fitting.json)；训练命令、环境、dirty HEAD / 源码及本地 native 扩展哈希：[training_recording.json](training_recording.json)。实验时 HEAD 为 `0a176b1a269a9850f51e7350e475a6c801f82644`，工作区已有未提交改动，因此不能只靠 HEAD 复现，完整源码快照保存在本地 `runtime/`。
- 各棋力原始汇总：`evaluations/*.json`；独立配置：`configs/*.json`；逐面板命令与环境：`recording/*.json`。
- 汇总与共同种子配对比较：[analysis.json](analysis.json)；完整交付检查：[verification.json](verification.json)；CPU 回归及梯度阻断检查：[smoke.json](smoke.json)。
- `partial_analysis.json` 是运行中保留的中间快照，最终结论以完整 `analysis.json` 为准；不作为完成棋力证据。
- 本地模型、逐步训练日志和冻结抽样引用：`runs/entity_attention_policy_value_ablation_20261006/`（按仓库约定忽略）；可恢复评测 sidecar 和源码快照：`runtime/entity_attention_policy_value_ablation_20261006/`（忽略）。本次保留全部正式模型、数据和已有产物。

从仓库根目录执行，统计与完整校验不会重跑棋力：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_policy_value_ablation_20261006/analyze.py
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_policy_value_ablation_20261006/verify.py
```

如果本地面板已完整完成，下列命令只检查并跳过；缺失面板会实际运行评测，未完成面板从 sidecar 恢复：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src TORCHINDUCTOR_COMPILE_THREADS=2 python -u artifacts/entity_attention_policy_value_ablation_20261006/run_suite.py
```

训练脚本创建新诊断输出，不用于在已有结果上续训。原配置、原数据、源码哈希、抽样状态和 checkpoint 必须与协议一致才能复核。新机器迁移需另外保全被忽略的 `runs/` 和 `runtime/`。
