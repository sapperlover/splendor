# 固定对战队列结果

状态：`complete`；完成 13/13 组。

每组200 pair／400局，和棋半分；CI按整pair bootstrap。Greedy只取决于policy。未完成组不报告最终分数。

| 组 | A | B | 完整pair | A胜／和／负 | A得分 | 95% CI | 循环裁和 |
| --- | --- | --- | ---: | --- | ---: | --- | ---: |
| [S01_P12V4_vs_P4V4](panels/S01_P12V4_vs_P4V4.json) | P12V4 | P4V4 | 200/200 | 218/7/175 | 55.375% | [50.750%,60.000%] | 0 |
| [S02_P4V12_vs_theta5000](panels/S02_P4V12_vs_theta5000.json) | P4V12 | theta5000 | 200/200 | 228/5/167 | 57.625% | [52.500%,62.625%] | 0 |
| [S03_P8V4_vs_theta5000](panels/S03_P8V4_vs_theta5000.json) | P8V4 | theta5000 | 200/200 | 366/0/34 | 91.500% | [88.750%,94.250%] | 0 |
| [G01_P4V4_vs_theta5000](panels/G01_P4V4_vs_theta5000.json) | P4V4 | theta5000 | 200/200 | 291/4/105 | 73.250% | [68.750%,77.625%] | 0 |
| [G02_P8V8_vs_theta5000](panels/G02_P8V8_vs_theta5000.json) | P8V8 | theta5000 | 200/200 | 310/0/90 | 77.500% | [73.250%,81.500%] | 0 |
| [G03_P12V12_vs_theta5000](panels/G03_P12V12_vs_theta5000.json) | P12V12 | theta5000 | 200/200 | 321/1/78 | 80.375% | [76.500%,84.250%] | 0 |
| [G04_offline9500_vs_theta5000](panels/G04_offline9500_vs_theta5000.json) | offline9500 | theta5000 | 200/200 | 194/1/205 | 48.625% | [43.500%,53.625%] | 0 |
| [G05_hybrid_r6_vs_theta5000](panels/G05_hybrid_r6_vs_theta5000.json) | hybrid_r6 | theta5000 | 200/200 | 209/2/189 | 52.500% | [47.625%,57.375%] | 0 |
| [G06_hybrid_r9_vs_theta5000](panels/G06_hybrid_r9_vs_theta5000.json) | hybrid_r9 | theta5000 | 200/200 | 213/2/185 | 53.500% | [48.500%,58.625%] | 0 |
| [G07_P12V12_vs_P4V4](panels/G07_P12V12_vs_P4V4.json) | P12V12 | P4V4 | 200/200 | 241/1/158 | 60.375% | [55.750%,65.000%] | 0 |
| [G08_P8V8_vs_P4V4](panels/G08_P8V8_vs_P4V4.json) | P8V8 | P4V4 | 200/200 | 221/2/177 | 55.500% | [50.625%,60.375%] | 0 |
| [G09_P12V12_vs_P8V8](panels/G09_P12V12_vs_P8V8.json) | P12V12 | P8V8 | 200/200 | 221/2/177 | 55.500% | [51.375%,59.625%] | 0 |
| [G10_hybrid_r9_vs_hybrid_r6](panels/G10_hybrid_r9_vs_hybrid_r6.json) | hybrid_r9 | hybrid_r6 | 200/200 | 220/3/177 | 55.375% | [50.375%,60.500%] | 0 |

## 对θ5000的同种子搜索得分差

差值为新面板减旧面板，对手均为θ5000。S01直接互打单独见上表。

| 新面板 | 旧面板 | 得分差（百分点） | 配对差95% CI（百分点） |
| --- | --- | ---: | --- |
| S02_P4V12_vs_theta5000 | early_step000034398_vs_theta5000 | -28.500 | [-34.378,-22.625] |
| S02_P4V12_vs_theta5000 | final103194_vs_theta5000 | +2.125 | [-4.125,+8.375] |
| S02_P4V12_vs_theta5000 | final_policy_early_value_vs_theta5000 | -34.625 | [-39.875,-29.375] |
| S03_P8V4_vs_theta5000 | early_step000034398_vs_theta5000 | +5.375 | [+1.375,+9.375] |
| S03_P8V4_vs_theta5000 | final103194_vs_theta5000 | +36.000 | [+30.500,+41.378] |
| S03_P8V4_vs_theta5000 | final_policy_early_value_vs_theta5000 | -0.750 | [-4.000,+2.625] |

固定训练轨迹与对手的探索性结果；θ5000搜索对照复用DEV，未作多重比较校正。新开局互打独立于旧DEV及原训练池，但没有覆盖训练seed变异。
