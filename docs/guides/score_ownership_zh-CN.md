# 到终局的分数归属辅助目标

2026-10-26新增用户请求的[policy＋得分归属104k离线脚本](../../artifacts/entity_attention_128x2_policy_score_r21_r37_20261026/README.md)：policy辅助0.15＋本目标0.25，短期value关闭，其余沿用原policy104k实际配置，18GiB／4共享worker、lr1e-4。从零训练，默认最低主KL＋MSE模型对raw R37 greedy1000pair后搜索A500pair；`train`仅训练与选点。当前实现（包含支付样本与最新验证诊断）已独立冻结并有限验证，policy＋得分组合已完成训练／原对战及后续10000 pair复核；原greedy／S1024N1为53.475%／56.80%，新greedy／S128N1为55.080%／57.013%，见[完成汇总](../reports/auxiliary_offline_results_20261027_zh-CN.md)。

2026-10-26实现。任务监督当前位置之后、沿归档对局实际续弈获得的分数，按玩家与来源拆成8路回归，共享原attention状态编码器。输出顺序固定为：

| 输出序号 | 玩家视角 | 新增分来源 |
| --- | --- | --- |
| 0–3 | 当前决策玩家self | 一级牌、二级牌、三级牌、贵族 |
| 4–7 | opponent | 一级牌、二级牌、三级牌、贵族 |

设当前位置s，终局T，`score[p,c]`为玩家p已获得的来源c的分数，则标签为 `y[p,c](s)=score[p,c](T)-score[p,c](s)`。已经获得的分数不计入；不取双方差值，不按胜负改符号，不做时间折扣。每方四项之和严格等于该方终局分数减当前位置分数。各路对应的实现字段为 `self_tier1/self_tier2/self_tier3/self_noble/opponent_tier1/opponent_tier2/opponent_tier3/opponent_noble`。

## 标签的准确时点

标签起点为归档的**决策前状态**，包含当前动作及其后续产生的增量：

- 主行动选择购买和付款前，都计入即将买到的牌分；Pay结算之后不再计入该牌分。
- 选贵族前包含本次贵族分；ClaimNoble结算之后不再计入。
- 只有一名可选贵族时，引擎自动领取，归档可直接从付款／拿取／保留／退币跳至下一决策或终局；同一步中的卡牌分和自动贵族分分别归类。
- 保留牌只在未来真正购买结算时贡献对应等级的牌分。零分牌的新增分为0。
- 最后一个决策即使是付款或领取贵族，仍从terminal_result补上末步得分。
- 自动forced skip不产生分数；没有后续得分时8路标签均为0。主行动、支付、退币和选贵族状态均可监督。

依据[现行规则](../splendor_standard_rules_zh-CN.md)、`rules._apply_payment/_apply_claim_noble`及`GameEngine._advance_automatic_transitions`，支付结算和贵族结算增加分数。实现先把每个快照的self/opponent分数恢复成固定seat0/seat1视角，用相邻快照及终局分数求每步增量；支付时按pending_card的等级分配该牌分数，其余增量分配到贵族一类，包含折叠到同一转换中的自动领取。最后反向累计未来增量并转回当前样本视角。无需重放或搜索，生成每局标签的复杂度为O(决策数)。

检查包括分数不能倒退、只有行动玩家能获得该步增量、支付卡必须可见且等级唯一、扣除卡牌分后的增量只能为0或规范牌库中一名贵族的分值、非末步的贵族分须与公开贵族数量变化一致、8路增量与终局总分差一致。违反时明确报错，不补造标签。自然终局归档只需现有状态快照、pending牌面、player_id和terminal_result，不依赖v3根W、访问数或trace；单候选根没搜索及forced skip造成的回合时钟空隙不会屏蔽本目标。原始NPZ和主policy/value标签保持。

## 模型与损失

沿用 `entity_attention_continuation_v1`，增加可选 `predict_score_ownership` 和8输出 `score_ownership_head`，采用原value读取方式。预测为线性回归值，训练标签除以15，不裁剪分数或预测；读取自然分数预测时乘15。标签在cache中保存为int16，传输时仍是8个精确分数；仅在loss处转换和缩放。

```text
L_ownership = Σ_i value_weight_i * mean_8((prediction_i - target_i/15)^2)
              / Σ_i value_weight_i
L_total = 原主任务loss + score_ownership_weight * L_ownership + 其他已启用辅助loss
```

仅启用本目标时配置为：

```json
"continuation_auxiliary": {
  "shortterm_value_weight": 0.0,
  "opponent_policy_weight": 0.0,
  "score_ownership_weight": 0.25,
  "protocol": "continuation-ema-h2-h6-h14-opponent-main-v1"
}
```

标签实现版本为 `remaining-score-by-player-source-v1`；continuation的外层protocol保持，新增正权重与模型头标志冻结任务身份。0.25是可调整的起始权重，未验证最优。可与短期value／对手policy组合；关闭本目标时不生成标签、不注册模型头，并保留旧continuation训练身份和已有头的初始化。旧checkpoint缺省的predict_score_ownership为false，可严格加载。未给现有实验配置添加本任务。

普通forward、value-only和原生搜索不执行该辅助头；训练在value/辅助分支复用状态编码及state-only传输，不多做一遍encoder。128×2关系主模型583682参数，仅本目标651402参数，增加67720参数。

每步及固定／完整验证记录8路自然单位MSE（分²）／MAE（分），另记平均 `score_ownership_loss`、`score_ownership_mse_points`、`score_ownership_mae_points`、有效value权重 `score_ownership_positions`。共享编码器梯度沿用首步及每100步采样，记录8个分支及合计的加权前／后范数；每路系数为score_ownership_weight/8，合计为score_ownership_weight。标量沿用批量回读。零权重batch的辅助loss可微为0。选点与早停仍使用完整留出的主policy KL＋主终局value MSE。

## 验证诊断（2026-10-26补充）

按用户最终要求，**支付状态继续参与本辅助训练**，原标签、样本池、抽样顺序、损失、权重及其他目标保持原样。以下信息只在固定子集和完整留出验证中增加；不添加逐步诊断日志，不额外读取训练池、不额外执行网络前向，也不参与选点／早停。既有保存的验证结果不自动回填，新增字段从下一次实际验证开始产生。原任务protocol、checkpoint和replay身份保持。

每路新增 `score_ownership_<输出名>_target_mean_points`、`prediction_mean_points`、`target_zero_fraction`，分别表示带value权重的标签均值、预测均值和零标签比例；沿用原8路MSE／MAE，帮助识别只学到均值或大量零标签的情况。

两种MSE基线均以自然分数为单位，并给出各路和八路平均：

- `zero_mse_points`：始终预测0的MSE，即 `E[y²]`。
- `constant_mse_points`：在**当前完整评估集合**上，每路始终预测该路均值的最优常数MSE，即 `E[y²]-E[y]²`。这是评估集合的常数基线，不是训练集拟合的均值，也不影响模型训练；同一固定面板的标签及权重不变时，该基线跨checkpoint固定。
- `constant_skill`：八路平均的 `1-MSE_model/MSE_constant`；0表示与常数基线相当，正数表示更好，负数表示更差。常数误差为0时记JSON `null`，不制造比值。

模型与基线都按相同有效value权重统计；所有原始分子、目标一／二阶矩和权重分母先跨batch累计，再求均值及方差，不平均batch均值或batch内方差。诊断使用float64矩，GPU标量合并为一次回读。未改变原训练loss和梯度的计算精度与算式。

分组统计提供 `score_ownership_<分组>_{positions,mse_points,mae_points,zero_mse_points,constant_mse_points,constant_skill}`；每组常数基线使用该组自己的各路均值：

| 分组 | 定义 |
| --- | --- |
| `phase_main_action`／`phase_payment`／`phase_return_tokens`／`phase_choose_noble` | 按当前决策阶段分别统计，保留全部支付样本 |
| `non_payment` | 仅为诊断汇总排除支付阶段，便于与全体误差比较，不筛除训练样本 |
| `stage_early` | 当前双方最高分小于5 |
| `stage_middle` | 当前双方最高分至少5、但小于10 |
| `stage_late` | 当前双方最高分至少10 |

前中后期按当前公开分数划分，不使用未来终局或实际剩余回合。`positions`是有效value权重和；空组权重为0、误差与基线为JSON `null`。总体、阶段和前中后期是不同切片，不可重复相加。验证与原训练逐位一致性的有限回归见[诊断交付](../../artifacts/score_ownership_20261026/diagnostics/README.md)。

## 新配置与运行

[`offline_score_ownership_r21_r37_20261026.json`](../../configs/entity_attention_v1/offline_score_ownership_r21_r37_20261026.json)采用现有R21–R37共104000局、原整局split与原主标签：93537局训练／10463局验证，共7478428／835367个决策；从零attention128、两路B256、新AdamW、lr1e-4、最多12次value曝光／350552步、每5%完整留出／三次严格上升早停。只启用本辅助，未启动正式训练。

按[离线执行默认](offline_execution_zh-CN.md)，使用流式会话累计、training_workers=4共享只读供数、state-only传输及批量标量回读。缓存设17408MiB（17GiB）：同104k池先前仅policy的完整arena实测17869150492字节（约16.64GiB）；本目标每位置16字节、单个标签数组，少于旧对手标签的180字节和4个数组，对齐开销也不增，因此该旧arena值是本配置的容量上界，低于17GiB。这是容量核算，未重新预热整个104k池，不是进程树总内存上限。

```bash
# 只准备身份与计划，不预热全池、不训练。
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.offline_continuation \
  --config configs/entity_attention_v1/offline_score_ownership_r21_r37_20261026.json \
  --output runs/score_ownership_r21_r37_20261026 --prepare-only

# 正式训练使用同命令去掉 --prepare-only；恢复也使用同命令。
# 完成后添加 --select-only，以主KL＋MSE选择完整checkpoint。
```

上述入口直接使用当前工作区实现；实际训练时source/session记录当前配置、源码和执行参数。历史冻结发布保持原版本。本次完成新配置的prepare-only及[有限工程验证](../../artifacts/score_ownership_20261026/README.md)，没有正式训练／棋力结论。
