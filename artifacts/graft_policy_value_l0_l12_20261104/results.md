# L12 与 L0/L12 policy・value 嫁接对 raw R37（双方 S128N1）

每组 2000 pair／4000 局，同开局换先后手；开局为 L0 参照面板前 2000 个种子。

| 面板 | policy | value | 完成pair | 得分率 | 95%整pair区间 |
| --- | --- | --- | ---: | ---: | --- |
| L0 参照（08_search_policy_vs_R37，10000 pair） | L0 | L0 | 10000/10000 | 57.098% | [56.420%, 57.783%] |
| 01_search_L12_vs_R37 | L12 | L12 | 2000/2000 | 64.550% | [63.087%, 66.025%] |
| 02_search_C1_policyL12_valueL0_vs_R37 | L12 | L0 | 2000/2000 | 62.550% | [61.087%, 64.075%] |
| 03_search_C2_policyL0_valueL12_vs_R37 | L0 | L12 | 2000/2000 | 61.687% | [60.175%, 63.138%] |

| 分解（百分点） | 估计 | 95%区间 | 方法 |
| --- | ---: | --- | --- |
| total: L12 - L0 | +7.452 | [+5.811, +9.094] | candidate whole-pair bootstrap + 10k L0 CI, normal combination |
| policy: C1 - L0 | +5.452 | [+3.811, +7.094] | candidate whole-pair bootstrap + 10k L0 CI, normal combination |
| value: C2 - L0 | +4.590 | [+2.977, +6.203] | candidate whole-pair bootstrap + 10k L0 CI, normal combination |
| interaction: L12 - C1 - C2 + L0 | -2.590 | [-5.079, -0.101] | candidate whole-pair bootstrap + 10k L0 CI, normal combination |
| value given new policy: L12 - C1 | +2.000 | [-0.013, +3.975] | paired whole-pair bootstrap, shared seeds |
| policy given new value: L12 - C2 | +2.863 | [+0.900, +4.825] | paired whole-pair bootstrap, shared seeds |
| policy graft vs value graft: C1 - C2 | +0.863 | [-1.200, +2.900] | paired whole-pair bootstrap, shared seeds |

双方 S128N1（每世界128×1世界＝总128），配置A，无根噪声，T0，12贵族；对手 raw R37。
与 L0 的差值为非配对正态合成（L0 用其 10k 面板区间）；三组候选之间共用开局，按整 pair 配对 bootstrap。
单一训练轨迹的推理期嫁接，不训练；未完成面板不作结论。
