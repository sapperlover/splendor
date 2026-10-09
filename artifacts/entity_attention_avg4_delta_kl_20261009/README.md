# 行为+M4 分支：独立 M4 配对 ΔKL 诊断（2026-10-09）

用已有的 M4 自迭代数据（`runs/entity_attention_avg4_behavior_20261009/seed1_avg4`），在第 4/8/12 个新轮（`round-003/007/011`）上分别抽取训练局面和留出局面，对**完全相同的局面**重新独立运行 4 次完整 S1024/N1 搜索并取平均，得到 fresh M4 标签；再用该轮训练前（`initial.pt`）和训练后（`final.pt`）的学生网络计算 `KL(π || p)`，比较同一标签下的配对变化 `ΔKL = KL_after − KL_before`。所有比较都固定同一标签，不直接比较不同标签的绝对 KL。

入口为 [run.py](run.py)，汇总为 [summary.json](summary.json)，逐局面原始结果与搜索分块在忽略的 `runtime/entity_attention_avg4_delta_kl_20261009/`。

## 数据与口径

| 轮 | 训练步 | 教师（=训练前学生） | before → after |
| --- | ---: | --- | --- |
| r004 | 16800 → 16900 | `round-002/checkpoints/final.pt` | `round-003/initial.pt` → `final.pt` |
| r008 | 17200 → 17300 | `round-006/checkpoints/final.pt` | `round-007/initial.pt` → `final.pt` |
| r012 | 17600 → 17700 | `round-010/checkpoints/final.pt` | `round-011/initial.pt` → `final.pt` |

before 与教师逐张量一致（`model_state_sha256` 相同）；每个采样局面的教师、根噪声比例 0.25、c_puct 1.5、S1024/N1、transition cap 512 与原采集一致。fresh M4 使用全新种子：world base `26700000`、noise base `26800000`（原存储标签用 `26300000/26400000`），每根跑 4 次独立搜索后平均。训练集为每轮 256 个该轮 train 分片的可搜索局面；留出集为整局划分的 validation 分片中每轮抽取 256 个局面，分别来自 19/23/21 个对局，未评估完整留出集。补充集 `train_ledger` 从该轮 `usage/policy_recent` 账本中抽取实际被 policy 采样到的当前轮局面（去重后随机取 256 个）。

均匀 Train 样本并非纯未见局面：r004/r008/r012 分别有 95/96/95 个局面出现在本轮 policy 账本，共 286/768（37.2%）。该组只能用于评估该轮 Train 分布，不能称为“其余未抽中的局面”。

每轮当前轮账本采样量：r004 6391 次 / 4199 个可搜索唯一局面；r008 6415 / 4129；r012 6284 / 4100（另有约 1000 个被抽到的强制单候选局面，无 policy 权重，已排除）。

## 结果

`fresh` 是 fresh M4 标签，`stored` 是该轮实际训练用的 behavior+3 平均标签；两者都是同一标签下的配对 ΔKL。负值表示训练后更接近教师目标。区间为按来源对局 10,000 次 cluster bootstrap 的 95% 区间；重复抽中对局的次数完整保留。合并三轮时在各轮内独立重采样，固定原各轮局面权重（当前每组各轮等权），区间以这三个固定训练前后 checkpoint 和已保存标签为条件。

2026-10-09 修订：原 `np.isin` bootstrap 丢失重复抽中对局的权重，旧区间失效，点估计和原始搜索仍保留有效。旧报告与汇总保存在 `runtime/entity_attention_avg4_delta_kl_20261009/historical_README_bootstrap_isin_v1.md` 和 `historical_summary_bootstrap_isin_v1.json`；下表使用修正后的 v2 汇总。

| 集合 | fresh ΔKL | stored ΔKL | 位置数 |
| --- | ---: | ---: | ---: |
| 训练（该轮 train 分片均匀抽样） | **+0.00860** [+0.00189, +0.01510] | +0.00778 [+0.00104, +0.01461] | 768 |
| 训练-账本（实际抽中，去重抽样） | −0.00367 [−0.01003, +0.00264] | −0.00528 [−0.01182, +0.00103] | 768 |
| 留出（整局划分后抽取局面） | **+0.00813** [+0.00246, +0.01408] | +0.00805 [+0.00187, +0.01469] | 768 |
| 同一 Train 分片从未抽中 | +0.00415 [−0.00154, +0.00980] | +0.00352 [−0.00217, +0.00923] | 768 |

分轮结果（fresh ΔKL）：

| 轮 | 训练 | 训练-账本 | 留出 |
| --- | ---: | ---: | ---: |
| r004 | +0.00904 [−0.00228, +0.02019] | −0.00502 [−0.01520, +0.00484] | +0.01479 [+0.00359, +0.02535] |
| r008 | +0.00027 [−0.00863, +0.00938] | −0.00473 [−0.01370, +0.00435] | +0.00321 [−0.00324, +0.00978] |
| r012 | +0.01648 [+0.00320, +0.02988] | −0.00128 [−0.01519, +0.01202] | +0.00639 [−0.00455, +0.01916] |

学生熵（训练前 → 训练后）与目标熵：

| 轮 | 训练集学生熵 | 留出集学生熵 | fresh 标签熵 |
| --- | ---: | ---: | ---: |
| r004 | 1.736 → 1.762 | 1.733 → 1.751 | 1.510 / 1.521（训练/留出） |
| r008 | 1.731 → 1.791 | 1.755 → 1.814 | 1.505 / 1.534 |
| r012 | 1.862 → 1.886 | 1.823 → 1.851 | 1.546 / 1.522 |

其他诊断（含新增未抽中组）：单个 fresh 搜索之间的平均 pairwise JS 为 0.08060；fresh M4 均值与 stored M4 标签的平均 JS 为 0.03017。两者量级说明独立 M4 标签之间确实有可观抽样差，但本分析比较的是固定标签下的前后差。

## 结论

1. **账本组只有改善的点估计，尚无明确证据**：fresh −0.00367、stored −0.00528 的修正区间都跨 0，撤回原“更新确实拟合了 minibatch”的显著性结论；该组也是去重局面均匀抽样，而非按 minibatch 实际重复次数加权的完整训练目标。
2. **当前轮整体 Train 分布平均净退化**：三轮合并的 fresh/stored ΔKL 均约 +0.008，区间高于 0；fresh 下 53.5% 局面 KL 上升，不能称为“绝大部分”。这一组混有实际训练过的局面，不能单独证明未抽中局面的退化。
3. **抽样留出分布平均净退化**：fresh +0.00813、stored +0.00805 的合并区间均高于 0，fresh 下 56.9% 局面 KL 上升。分轮看，只有 r004 的 fresh 留出区间完全高于 0；不能宣称三个轮次均显著退化。
4. **stored 标签偶然性不足以解释全部退化**：独立 fresh M4 下仍见平均净退化。留出上 fresh−stored 的 ΔKL 差约 +0.00008，区间 [−0.00316, +0.00334]；未发现明确差异不等于证明两种标签等价，更不能排除标签质量或搜索噪声的影响。
5. **更新动力学、历史目标混合和覆盖量仍待因果对照**：熵上升是观测，不足以证明步长过大；账本组熵同样上升而 KL 点估计下降。此前低学习率/裁剪实验提示更小有效更新值得测试，但本实验未改变这些变量，不能确定当前 M4 分支的根因。

## 同一 Train 分片从未抽中的局面

新增 `train_unseen`：在每轮当前数据的 Train 可搜索局面中，排除本轮全部 100 步 `usage/policy_recent` 账本出现过的 `(game_id, decision_index)`，从余集不放回均匀抽取 256 个。与已有三组使用相同教师、before/after、4×S1024/N1、根噪声和 fresh 种子规则；独立抽样流为 `unseen_sample`，不按搜索结果或 KL 选择局面。冻结的来源哈希、账本哈希和具体局面见 [unseen_specification.json](unseen_specification.json)。已完成全部 768 个局面、3072 次完整搜索、3,145,728 次模拟，结果已写入 [summary.json](summary.json)。没有重新训练模型。

| 轮 | Train 可搜索局面 | 本轮抽中过的唯一局面 | 未抽中过的候选池 | 抽样局面 / 来源对局 |
| --- | ---: | ---: | ---: | ---: |
| r004 | 11904 | 4199 | 7705 | 256 / 138 |
| r008 | 11435 | 4129 | 7306 | 256 / 139 |
| r012 | 11802 | 4100 | 7702 | 256 / 135 |

新标签 KL 的前后变化：

| 轮 | before | after | ΔKL | 对局聚类 95% 区间 |
| --- | ---: | ---: | ---: | --- |
| r004 | 0.27021 | 0.27790 | +0.00769 | [+0.00006, +0.01539] |
| r008 | 0.32463 | 0.33073 | +0.00610 | [−0.00385, +0.01554] |
| r012 | 0.26107 | 0.25973 | −0.00134 | [−0.01324, +0.01051] |
| 合并 | 0.28530 | 0.28945 | +0.00415 | [−0.00154, +0.00980] |

合并点估计变差，但区间跨 0；不能确认未抽中组稳定退化，也没有确认改善。r004 下限仅约 +0.00006，在多个探索性比较中不视为独立稳定结论；r008/r012 区间均跨 0，r012 点估计略改善。fresh 下 55.9% 局面 KL 上升，top-1 与 fresh 教师的一致率 69.01% → 66.54%（描述性变化，未检验显著性）。stored 的合并 ΔKL +0.00352 也跨 0。因此撤回“只改善抽中局面、未抽中局面稳定退化”的完整解释链；当前证据更准确地表述为：未确认对同轮未抽中局面的泛化收益，原均匀 Train/留出样本存在平均净退化，具体原因尚待相同账本下的变量对照。

“从未抽中”限于本轮 policy 更新中的局面身份，不要求该对局的其他局面从未训练，也不排除 value 分支见过该局面或其他对局具有相同观测。原始结果为 `positions_unseen.jsonl`、`fresh_targets_unseen.npz` 和 `search_unseen/`，均在同一 runtime 下；搜索分块原子写入，重跑同一命令复用已完成分块。

必要验证见 [verification.json](verification.json)：四项回归通过，独立目录 CPU smoke 通过；所有新局面不在该轮 policy 账本中，全部搜索预算和均值符合要求，before/after/教师哈希未变，重建汇总保留四组并复现结果。34 个不同集合间同种子重复根中，33 个访问计数完全相同；一个根的一次搜索相差一个访问计数，已如实记录。不同批次下出现的这一个访问差异未单独定位原因，不宣称新重搜索逐位复现已有标签。

## 边界

- 单起点、单训练 seed、三个轮次；每集合每轮 256 个局面，区间按来源对局聚类，只覆盖本次抽样。
- `train_ledger` 只含当前轮被抽到的局面；该轮 policy 窗口也包含此前三轮，因此实际更新目标混有更老教师的目标，本分析测的是“当前轮目标是否被改善”。
- 账本采样去重后每轮约 4.1–4.2k 个可搜索位置，约占该轮 train 分片可搜索位置的一部分；train 集合与 train_ledger 集合的绝对 KL 基线不同（约 0.26–0.31），因此只比较同一集合内的 ΔKL。
- 未运行棋力面板；KL 与熵不是胜率或 Elo。

## 复核

```bash
# 采样 + 主集搜索/评测（每轮 256 train + 256 validation，每根 4×S1024）
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u artifacts/entity_attention_avg4_delta_kl_20261009/run.py all --device cuda
# 账本实际训练局面补充集
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u artifacts/entity_attention_avg4_delta_kl_20261009/run.py ledger --device cuda
# 只重建 summary（读取已落盘的 positions*.jsonl）
conda run -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_avg4_delta_kl_20261009/run.py analyze
# 同一 Train 分片从未被本轮 policy 抽中的局面：启动或续跑
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u artifacts/entity_attention_avg4_delta_kl_20261009/run.py unseen --device cuda
# 有限回归与隔离的 CPU smoke（不会覆盖正式搜索或汇总）
conda run -n splendor env PYTHONPATH=src python -m pytest -q \
  artifacts/entity_attention_avg4_delta_kl_20261009/test_run.py
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/entity_attention_avg4_delta_kl_20261009/run.py smoke --limit 2 --simulations 16
```

主集与两种补充集合计约 12.58M 次模拟（主集 6.29M、账本集 3.15M、新增未抽中集 3.15M）。分块 NPZ、逐局面 JSONL、fresh_targets 在 `runtime/entity_attention_avg4_delta_kl_20261009/`；原身份与种子在 [specification.json](specification.json)，新增抽样和来源在 [unseen_specification.json](unseen_specification.json)。旧 specification 是原搜索准备时快照，保留当时的脚本哈希；当前修订代码身份见 summary 和 verification，不能把旧准备哈希当成当前统计实现的哈希。
