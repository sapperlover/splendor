# S256N4 vs S1024N1 vs greedy：固定局面教师比较（2026-10-22）

状态：只读复用 [128 固定局面搜索稳定性](../entity_attention_r20_search_stability_128_20261021/README.md) 的 `raw_searches.jsonl.gz`，没有启动 GPU 搜索或训练，不干扰正在运行的训练。

## 数据范围

- 96 个 baseline 根 + 32 个 high-disagreement 筛选根，共 128 个固定真实局面；
- 每个方法 8 个搜索 seed；greedy 每根固定一次；
- S256N4 与 S1024N1 的总预算都是 1024；greedy 零搜索；
- 这里只比较标签行为，不是棋力赛。S256N4/S1024N1 对共同 greedy 的既有分数为 77.20% / 78.05%，整 pair 差约 −0.85pp，区间跨 0，不能据此选教师。

## 主要结果

所有比例按根等权；“stable”指 8 个 seed 的首选动作完全一致。

| 指标 | S256N4 | S1024N1 |
|---|---:|---:|
| 全 8 seed stable 率（baseline） | **70.8%** | 36.5% |
| 全 8 seed stable 率（high-disagreement） | **43.8%** | 0.0% |
| mode 首选与 greedy 相同（baseline） | **76.0%** | 61.5% |
| mode 首选与 greedy 相同（all） | **73.4%** | 58.6% |
| 平均 policy 在 greedy 动作上的质量（baseline） | **0.466** | 0.450 |
| 平均 policy 到 greedy one-hot 的 JS（baseline） | **0.255** | 0.272 |
| stable 且与 greedy 相同（baseline） | 55/96 | 30/96 |
| stable 但与 greedy 不同（baseline） | **13/96** | 5/96 |
| stable 但与 greedy 不同（high-disagreement） | **2/32** | 0/32 |
| 不稳定且与 greedy 不同（baseline） | 28.1% | **60.4%** |

补充：现有 canonical 稳定性表的跨 seed 两两首选一致率为 S256N4 **88.39%**、S1024N1 **66.52%**；全 8 seed 完全一致更严格，因此上表 stable 率更低，但结论方向相同。

## 解释

S256N4 的“稳定”**大部分是更贴合 greedy，而不是稳定地给出不同招法**：

- baseline 68 个 stable 根里，55 个选 greedy，13 个选不同动作；
- high-disagreement 14 个 stable 根里，12 个选 greedy，2 个选不同动作；
- 它的 greedy 首选匹配率、greedy policy 质量都更高，到 greedy 的 JS 更低；但政策分布差异不大，因此它不是简单的 greedy 复制。

S1024N1 的不稳定部分反而更多是“偏离 greedy”的来源：

- baseline 只有 36.5% 全 8 seed 稳定；
- 不稳定根中 60.4% 与 greedy 不同，而 S256N4 只有 28.1%；
- 它有更多探索性，但标签噪声更大。

## 教师选择

若只能选一个：**S256N4 是更稳的默认教师**，胜率与 S1024N1 基本持平，标签稳定性和高预算参考吻合度略好；但它会更多强化 greedy 的招法，探索性较弱。

若目标是增加探索/避免 greedy 坍缩：**S1024N1 更合适**，代价是标签不稳定、在 high-disagreement 根上 0% 全 8 seed 稳定。

更稳妥的方案不是二选一，而是 **S256N4 为主、少量保留 S1024N1 或高分歧根**；最终应由 student 蒸馏/在线 A/B 决定，当前标签统计不足以证明某个教师最终更强。

## 产物

- 机器可读汇总：[results/summary.json](results/summary.json)
- 复算脚本：[scripts/teacher_compare.py](scripts/teacher_compare.py)
- 原始逐根/逐 seed 搜索：[../entity_attention_r20_search_stability_128_20261021/raw_searches.jsonl.gz](../entity_attention_r20_search_stability_128_20261021/raw_searches.jsonl.gz)
