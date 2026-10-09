# 固定 256 个独立真实局面的决策诊断

2026-10-09完成测试集、两份 S1024/N1 checkpoint 搜索及配对比较。实现和日常命令见[指南](../../docs/guides/decision_probe_zh-CN.md)，入口为 [`src/splendor/decision_probe.py`](../../src/splendor/decision_probe.py)。结论只关于搜索决策变化，没有对局胜率、Elo、正确动作标签或棋力提升结论。

## 固定测试集

- [`bank.json`](bank.json)：256 个根节点，每个来自不同的独立来源游戏，主行动、至少两个候选。SHA-256：`b4937c360a2908e72bc320b3fe27b9df90d2567a8c4676812c114c06afd9e9ac`。
- [`source_games.jsonl`](source_games.jsonl)：256 局完整 compact action trace、源策略、种子和选位信息。没有读取训练或监督留出局面。
- [`excluded_source_games.jsonl`](excluded_source_games.jsonl)：游戏种子 2026100939 的 Development/Development 对局，没有符合后期配额的多候选主行动；保留完整对局并用游戏种子 2026101195 补采。总共实际进行了 257 局。
- [`verification.json`](verification.json)：256 个来源对局和排除对局完整 checked replay，256 个决策前缀均精确复原保存的请求。

四种组合 Score/Score、Score/Development、Development/Score、Development/Development 各 64 个根节点，每种组合前中期／后期各 32。公开最高分 ≥10 为后期，两阶段各 128；每局在指定阶段的合格决策中等概率选一个。采集程序不加载任何 checkpoint，不按网络质量或搜索分歧筛选局面。

游戏种子起点 2026100900，阶段不合格时源游戏编号加 256，因此实际使用 2026100900–2026101155 中除 2026100939 外的种子，外加 2026101195。后续训练／采集保留并避开 2026100900–2026101299 这一游戏种子域；选位种子起点 2026101300、搜索重复种子 2026101500/2026101501 不作游戏种子使用。测试集和来源游戏不用于训练。

| 来源描述 | 前中期 | 后期 | 整体 |
| --- | ---: | ---: | ---: |
| 根节点数 | 128 | 128 | 256 |
| 平均双方已完成回合数 | 26.95 | 57.31 | 42.13 |
| 平均候选数 | 21.24 | 30.23 | 25.74 |
| 已触发最后一轮的根节点数 | 0 | 12 | 12 |

候选数范围 2–38；当前行动者为 player0/player1 的根节点数 124/132。**256 个根节点中对手暗保留牌数均为 0**：本集覆盖固定启发式策略形成的真实局面，但不覆盖对手暗保留牌推断场景，不能据此验收隐藏信息估计能力。未来如扩展其他来源分布应新建独立版本，不覆盖本集。未知后续抽牌仍由搜索重采样，搜索种子波动统计并不因此消失。

一次性采集命令（已执行；日常评测无需再采）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -u -m splendor.decision_probe build \
  --bank artifacts/decision_probe_v1_20261009/bank.json --resume
```

## 已完成搜索与实际比较

共同协议：native、每次 S1024/N1、c_puct1.5、无根噪声、温度0、forced folding、transition cap512；固定两个搜索种子，各自完整搜索、配对各自的首选动作，不先平均分布后选动作。每个 checkpoint 共 512 次根搜索、524288 simulations。CUDA FP32 eager、TF32 关闭、active roots64、唯一进程；没有进行训练或完整棋力对局。

| checkpoint | 来源模型 SHA-256 | 原始逐根数据／摘要 |
| --- | --- | --- |
| 16500 | `dbfbe066144a485d15ec633aa87c85ac27864f142183725b86562e4ba44ed2c4` | [searches](reference_16500_s1024_n1/searches.json)、[summary](reference_16500_s1024_n1/summary.json) |
| 原16600 | `25247adec9a3e6be8e027f48e0036141657cfdfb7c33f57cab335c522a7139df` | [searches](baseline_16600_s1024_n1/searches.json)、[summary](baseline_16600_s1024_n1/summary.json) |

两份 summary 保存完整 checkpoint 来源路径、命令、bank/逐根数据/代码/native 扩展哈希、搜索配置及环境。模型原件仍在 `runs/`，没有纳管。后续新 checkpoint 默认与原16600结果比较，也可改成最近一次完成的结果。

16500 为参照、16600 为新模型的配对结果见[比较报告](baseline_16600_s1024_n1/comparison.json)；[`comparison_16500_to_16600.json`](comparison_16500_to_16600.json) 是同一结果的独立离线命令输出，数值相同。

| 指标（两个配对种子平均） | 整体 | 前中期 | 后期 |
| --- | ---: | ---: | ---: |
| 首选动作变化率 | 16.21% | 20.70% | 11.72% |
| 根访问 JSD（自然对数） | 0.02131 | 0.02894 | 0.01367 |
| policy 先验 JSD | 0.00470 | 0.00709 | 0.00232 |
| 高访问有序动作对反转率 | 13.61% | 17.53% | 9.46% |
| 高访问动作平均绝对名次变化 | 1.000 | 0.885 | 1.115 |
| 根 value 平均绝对变化 | 0.07464 | 0.12149 | 0.02778 |
| 前三动作集合重合比例 | 84.31% | 80.86% | 87.76% |

动作变化率为 83/512 次同局面同种子的比较，不是 83 个不同游戏。逐根平均同样给出 16.21%；两个种子各自结果保留在报告中。高访问集合是参照分布累计覆盖80%质量的最小前缀；仅一个动作或全部并列时的反转率没有定义，汇总只取有定义的根节点，分母可查看 `defined_values`。

参照首两名访问概率差 ≥0.2 的 234 次决策中，13 次改选（5.56%）；差 <0.05 的 162 次中，56 次改选（34.57%）。因此总体动作变化包含较多相近候选间的切换，应结合 margin、JSD、高访问排序和分阶段指标观察。前中期变化更大是此测试集上的事实，不能据此断言决策变好。

| 同 checkpoint、不同搜索种子 | 整体 | 前中期 | 后期 |
| --- | ---: | ---: | ---: |
| 16500 动作不一致率 | 14.45% | 17.19% | 11.72% |
| 16600 动作不一致率 | 14.06% | 16.41% | 11.72% |
| 16500 根访问 JSD | 0.02993 | 0.03659 | 0.02328 |
| 16600 根访问 JSD | 0.02767 | 0.03238 | 0.02295 |

跨 checkpoint 的动作变化率与同权重跨种子的波动量级接近。配对搜索种子使世界抽样更可比，但搜索路径会随网络改变；这里既不能把16.21%全算作策略收益，也不能将14%左右的波动直接相减。没有正确动作参照或实际对局，本结果不判断净收益／退化。

原16600搜索改写网络先验首选的比例：整体30.86%、前中期39.06%、后期22.66%。其模拟终局命中比例：整体29.11%、前中期0.42%、后期57.80%；平均树深分别3.46／4.20／2.72。后期确实更多模拟直接到达终局，但这不是搜索胜率收益归因实验。两 checkpoint 均无 transition cap 命中。

在 RTX4050 LaptopGPU 上，这次原16600搜索主体及汇总保存耗时17.48秒，16500为18.04秒；不含首次进程导入、checkpoint加载和根 value 前向。16600进程峰值RSS1,116,820KiB，约1.07GiB；这是单次运行的成本记录，不保证其他checkpoint、硬件或更高预算相同。

## 日常使用与验证

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u -m splendor.decision_probe evaluate \
  --checkpoint runs/你的实验/checkpoints/final.pt \
  --output runtime/decision_probe/你的checkpoint名称 \
  --compare-to artifacts/decision_probe_v1_20261009/baseline_16600_s1024_n1 \
  --resume
```

自动设备选择；日常命令不加载参照 checkpoint、不重搜参照、不审计完整来源游戏。输出 `summary.json`、`searches.json`、`comparison.json`；可恢复分片在被忽略的 `work.games/`。中断后原命令续跑，身份变化必须新建输出目录。若更改 S/N/backend 或执行源码，先为新口径重跑参照，不能把搜索口径差异当成模型变化。

六项有限测试通过（`conda run --no-capture-output -n splendor env PYTHONPATH=src python -m pytest -q tests/test_decision_probe.py`）：真实来源重放和阶段配额、自然阶段不合格游戏保留及补采、JSD与排序并列口径、native不等世界预算目标及配对身份、逐根中断续跑、native_chance/CPW入口。正式256根×两种子搜索已在两 checkpoint 上完成，一条命令的缓存恢复及自动比较已实际调用；来源与预算复核见 [`validation.json`](validation.json)。未在本次执行 compiled 推理对照、棋力赛或训练。
