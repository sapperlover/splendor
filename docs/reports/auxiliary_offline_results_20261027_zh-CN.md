# 近期辅助目标离线实验完成结果（2026-10-27）

本页补齐此前停留在准备、等待或中断时点的文档状态，直接整理各实验现有summary／results，不重跑实验或重新选点。

## 原训练与原两组对战

六组训练、早停选点及原对战均已完成。前五组使用同一原R21–R37的104k池；最后一组是原R21–R31的44k池。KL／MSE对应各自选中节点。对手均为raw R37；原greedy每组1000 pair，原搜索为双方S1024N1（每世界1024×1，总1024）每组500 pair。

| 实验 | 停训步 | 选中步 | 完整留出KL | 完整留出MSE | 原Greedy | 原S1024N1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [仅短期value](../../artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023/results.md) | 192804 | 140221 | 0.208836 | 0.736733 | 53.025% | 51.400% |
| [仅对手policy](../../artifacts/entity_attention_128x2_opponent_policy_r21_r37_20261024/results.md) | 192804 | 140221 | 0.205370 | 0.735986 | 54.150% | 55.550% |
| [value＋policy](../../artifacts/entity_attention_128x2_continuation_r21_r37_20261024/results.md) | 192804 | 140221 | 0.206134 | 0.733559 | 53.275% | 53.100% |
| [policy＋得分](../../artifacts/entity_attention_128x2_policy_score_r21_r37_20261026/results.md) | 192804 | 140221 | 0.205600 | 0.734649 | 53.475% | 56.800% |
| [无辅助](../../artifacts/entity_attention_128x2_noaux_r21_r37_20261026/results.md) | 210332 | 122694 | 0.212048 | 0.736596 | 51.075% | 57.500% |
| [44k仅短期value](../../artifacts/entity_attention_128x2_shortterm_value_r21_r31_vs_r37_20261024/results.md) | 96611 | 59453 | 0.227784 | 0.769171 | 49.375% | 42.950% |

104k几种目标的主KL／MSE接近；44k仅value的原搜索结果为42.95%，明显低于这些104k模型。小池与大池同时改变了训练数据，单凭这组结果不区分收益来源。

## 新种子的五模型大面板

[十组评测](../../artifacts/offline5_vs_r37_10k_20261027/README.md)全部完成，共100000 pair／200000局。固定上述原选点；每模型greedy及双方S128N1各10000 pair，S128N1是每世界128×1，总128。完整置信区间见[原结果表](../../artifacts/offline5_vs_r37_10k_20261027/results.md)。

| 原104k模型 | Greedy，10000 pair | 双方S128N1，10000 pair |
| --- | ---: | ---: |
| 仅短期value | 54.050% | 56.290% |
| value＋policy | 55.390% | 56.975% |
| 仅对手policy | 54.722% | 57.098% |
| 无辅助 | 53.265% | 56.700% |
| policy＋得分 | 55.080% | 57.013% |

Greedy中辅助组点估计都高于无辅助，双辅助与policy＋得分略高；S128N1五组集中在56.29%–57.10%，没有明显拉开。无辅助继续离线训练也能超过R37，当前结果不能把继续训练的收益全部归于辅助目标。原500 pair搜索使用S1024N1，新大面板使用S128N1，二者分别解释。

## 在线与其他已完成复核

- [policy辅助在线分支](../../artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/README.md)已提交13轮；第12轮greedy66.075%、S1024N1搜索60.20%（500 pair已完成），第13轮greedy63.75%。此前“第12轮973/1000局”已过时；第14轮尚无完成训练记录。既有平台分析作为当时快照保留。
- [第12轮对旧混合R20的greedy](../../artifacts/entity_attention_local12_vs_mixed_r20_greedy_20261026/README.md)已完成1000 pair，得分70.35%。
- [原人类棋谱Greedy复评](../../artifacts/bga_927500015_current_greedy_20261026/README.md)及[第12轮完整搜索矩阵](../../artifacts/bga_927500015_local12_analysis_20261026/comparison.md)已完成：主行动Greedy首选吻合由旧模型26/56降到第12轮22/56，完整搜索8组中7组首选吻合下降。
- [2000个多支付局面对照](../../artifacts/payment_search_matrix_local12_2k_20261026/README.md)六搜索、两greedy均已完成，搜索共12000根；首选及前三花金倾向见专题。
