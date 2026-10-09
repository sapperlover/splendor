# 多搜索平均标签：固定数据短训对照

2026-10-09工程后续：[行为搜索复用M4连续分支](../../docs/guides/entity_attention_avg4_iteration_zh-CN.md)已实现，后续完成12轮／17700步；计划16轮未完成，稳定棋力收益未确认；其行为＋3份总4倍与本报告额外四份＋原行为总约5倍属于不同生成口径。下文“未接入”保留当时版本范围，历史测量与结论不变。

2026-10-07，报告复核修订。对冻结 native 教师采集的200局数据重放每个可搜索决策，用完整S1024/N1预算额外独立搜索4次（world与根噪声seed都变化），平均访问分布作为policy标签。两组训练从相同模型、AdamW状态与RNG开始，采用相同抽样账本、value标签、批次、优化器配置和步数；改变的训练输入只有policy target。每次搜索预算保持1024，总标签预算增加到4096，不能称为等总预算、只改变标签方差或保持平均教师强度不变。

**结果：平均标签学生相对原单次标签学生的均值拟合与直接对打优势成立；相对训练起点的净提升尚未确认。**100步时，平均组对独立8次搜索经验均值的KL比单次组低0.01885，按77个来源对局cluster bootstrap的95%区间为[−0.03413,−0.00465]；直接对打119胜、4和、77负，得分60.5%，100 pair区间[54.25%,66.75%]。到起点的KL变化只有−0.00334，区间[−0.01796,+0.01166]；对起点搜索得分47%、greedy49%，均未确认超过起点。结果支持把平均标签作为减少短训回退的改进方向，不证明长期自迭代收益。后续按此方向推进，不追加fresh-M1对照，也不把它作为必要前置条件。

身份修正：旧PPO5.5M的[S1024/N1对同权重greedy得分92%](../mcts_50pair_sweep/05_s1024_n1_vs_greedy.json)不是本次attention起点的结果，不能据此量化当前教师的搜索优势。

## 设计与实际调用

- 数据与起点：`runs/entity_attention_chance_training_probe_20261007/native`的200局、16500步checkpoint，SHA256 `dbfbe066…`。13,243个可搜索决策：train11,709、validation1,534；训练只从train抽样。
- 重标注：checked回放compact trace，在各根上运行`NativePUCTSearchPolicy`，每份S1024/N1、c_puct1.5、fold forced、cap512、root noise fraction0.25／α0.3。world base `16940000+rep`、noise base `16950000+rep`，实际随机流还由公开请求哈希派生；平均四份归一化访问分布，不改变存档行为或value标签。原存储单次结果不计入这四份。
- 训练：从相同起点恢复完整AdamW/RNG，逐步复现原native臂的`batch_references.npy`，每步policy/value各256样本，clip各1.0、lr1e-4、value detach，100／300／500步保存。控制臂模型参数和AdamW状态完全复现（模型max_abs_diff=0）。
- 共享trunk由policy更新，critic输入随之改变；因此平均臂的后续梯度、优化器状态、critic参数与value函数也会不同。相同优化配置和value输入不等于相同优化轨迹或固定value函数。

## 标签统计

全部13,243个决策等权。两个独立单次搜索之间JS0.07874／Top-1一致率67.71%；存储标签与新单次标签JS0.07838。量级相近支持随机化口径相近，不单凭JS宣布同分布。

将四份结果拆成两组、每组平均两次后，组间JS为0.04937、Top-1一致率74.35%，较单次JS低约37%、一致率高6.6pp。**这是M2对M2的稳定性诊断，未直接测M4对M4的JS，也未测M2学生棋力。**M4学生的训练收益如下。

## 独立经验均值拟合

[目标噪声实验](../entity_attention_target_noise_20261007/README.md)的200个留出根、77个来源对局，D组8次搜索均值作为参考；它是随机教师的经验均值，不是已知最优策略。

| 学生 | 100步 | 300步 | 500步 |
| --- | ---: | ---: | ---: |
| 起点 | 0.26987 | — | — |
| 单次标签 | 0.28538 | 0.29042 | 0.29047 |
| 四次平均 | **0.26653** | **0.27004** | **0.27291** |

复核改按来源对局cluster bootstrap，10,000次、seed16980000，局面等权；原逐根bootstrap保留在原`analysis.json`，下表与报告采用修订结果[review_statistics.json](review_statistics.json)。这些区间条件于本次训练的模型和8份参考目标，不覆盖训练seed与参考搜索seed的全部不确定性。

| 步数 | 平均−单次KL及95%区间 | 平均−起点KL及95%区间 |
| --- | --- | --- |
| 100 | −0.01885 [−0.03413,−0.00465] | −0.00334 [−0.01796,+0.01166] |
| 300 | −0.02038 [−0.03078,−0.01029] | +0.00017 [−0.01641,+0.01707] |
| 500 | −0.01756 [−0.03413,−0.00314] | +0.00304 [−0.01445,+0.02067] |

相对单次标签的三个区间均低于0，相对起点均跨0。Top-1对经验均值的一致率约63%–65.5%，不能仅由比例相近宣布选招完全相同。

## 棋力与value诊断

| 面板 | pair数 | 得分／胜和负 | 95%配对区间 | 对照口径 |
| --- | ---: | --- | --- | --- |
| avg4-100 vs native100 | 100 | **60.5%／119、4、77** | [54.25%,66.75%] | 不同学生权重，双方旧native S1024/N1 |
| avg4-100 vs冻结起点 | 50 | 47.0%／46、2、52 | [37.0%,56.5%] | 单次组45%；配对差+2pp [−12,+15.5] |
| avg4-100 greedy vs冻结greedy | 50 | 49.0%／49、0、51 | [39%,59%] | 单次组49%；配对差0pp [−14,+14] |
| avg4-300 vs冻结起点 | 50 | 46.0%／46、0、54 | [36%,55%] | 与单次500步42%步数不同，不能作为同更新量收益 |

四个面板共500局全部自然终局，零工程失败与policy cycle draw；逐局记录、模型哈希、种子、席位和S1024预算复核一致。直接对打是这对已训练学生的有效相对棋力证据；greedy未检出差异，不能据此断言收益只发生在搜索路径。

用旧参考对局的终局标签重算value MSE：

| 步数 | 单次组 | 平均组 | 起点 |
| --- | ---: | ---: | ---: |
| 100 | 0.72227 | 0.74721 | 0.73626 |
| 300 | 0.80508 | 0.77835 | — |
| 500 | 0.80772 | 0.78524 | — |

100步平均−单次MSE为+0.02493，聚类区间[+0.00509,+0.04434]。历史终局标签不是学生更新后的反事实回报，因此不直接证明棋力损伤；它表明相同value标签与批次没有固定value函数，本次未分离policy与value对棋力的贡献。

## 成本、解释与边界

- 每次完整S1024搜索保留原有限预算深度，区别于先前在固定总S下增加N、拆分单树预算的实验；平均输出及总搜索计算量确实改变，不能宣称平均教师质量已被固定。
- 平均分布是当前世界采样器与根噪声机制下教师行为分布的经验均值，不等于对真实动作价值求期望或最优信念策略。世界抽样与根噪声同时被平均，未隔离两者的训练贡献。
- 本次额外四次重标注：13,243×4×1024＝54,243,328次模拟，约28分钟；原采集另有13,560,832次模拟、约7分钟。从零复现合计约五倍单次模拟量、约35分钟。生产若复用行为搜索作为M份之一，理论模拟量约M倍，但标签构成与本次四份重新生成不同；真实墙钟须实测，不按M严格线性承诺。
- 一个起点、一个采集／训练seed、固定200局、最多500步；未验证持续自迭代、M剂量曲线、等搜索预算的数据量选择。value head在两组均正常更新、trunk梯度由policy提供；未验证value梯度进入trunk的联合训练是否同样受益。
- 执行记录：重标注汇总及首次训练遇到读取标签的内存问题，随后修正读取方式，完整标签已落盘，控制臂和输入哈希复核通过。执行时生产源码/native快照已保存；实验脚本自身没有冻结执行前版本，不能补称为已冻结。修订spec另记录当前复核版本和产物身份，不替代缺失的执行前脚本快照。

## 产物与复核

`specification.json`保留冻结输入、搜索参数和生产源码身份，修订区记录报告版本；`labels_summary.json`、`fit.json`、`control_reproduction.json`、`analysis.json`、`verification.json`保留原结果。新`review_statistics.py/json`复算对局聚类KL、历史终局MSE，以及正式分头训练20轮的同轮留出前后变化。长期自迭代的改进分析见[整合报告第九节](../../docs/reports/entity_attention_search_teachers_zh-CN.md#九如何改善当前自迭代先改变标签再判断剩余限制)。

重型标签在ignored `runtime/entity_attention_target_averaging_20261007/labels.npz`（13,243×4×47 float64，SHA `b9ad4f73…`）；分片在同目录`label_parts/`，模型在ignored `runs/entity_attention_target_averaging_20261007/`。迁移需要另行备份这些文件。

```bash
# CPU重算修订统计；覆盖review_statistics.json，无采集、训练或棋力赛
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/entity_attention_target_averaging_20261007/review_statistics.py

# 原始汇总／审计重算，覆盖各自同名JSON
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/entity_attention_target_averaging_20261007/experiment.py summarize
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/entity_attention_target_averaging_20261007/experiment.py analyze
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/entity_attention_target_averaging_20261007/experiment.py verify

# 原fit会运行GPU裸网前向并覆盖fit.json，无搜索或训练
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/entity_attention_target_averaging_20261007/experiment.py fit
```

`relabel/train/evaluate`分别会执行GPU搜索、训练、棋力赛，已完成目录不要重复覆盖。这些命令对应2026-10-07固定数据对照，不会自动启用连续M4。后续独立行为复用M4分支已完成12轮，见[M4指南](../../docs/guides/entity_attention_avg4_iteration_zh-CN.md)；原单次标签run的冻结配置不因后续实验而改变。
