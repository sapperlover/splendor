# R43：64根配对动作排序诊断

根部无噪声S1024/N1；实际访问首选为A。第一组32根A对访问次选；第二组28根A对Policy Greedy首选，加4根同选控制。每根先32份同世界配对，95%后验区间跨0者补到128；后续双方只用R43 Greedy。

完成：{'roots': 64, 'distinct_action_comparisons': 60, 'same_action_controls': 4, 'expanded_roots': 50, 'world_pairs': 6848, 'arms': 13696, 'checked_python_replays': 128, 'censored_arms': 0, 'unresolved_distinct_roots': 40}。

| 面板 | 方法 | 有效根 | 经验排序准确率 | 已分辨根准确率 | 差值相关 | 错选损失z | 后验期望损失z |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | V_same_search_world | 60 | 0.5083 | 0.6000 | -0.1785 | 0.1117 | 0.1153 |
| all | root_Q | 60 | 0.4750 | 0.5000 | 0.1453 | 0.1128 | 0.1161 |
| all | MCTS_visit_choice | 60 | 0.5583 | 0.6500 | -0.0020 | 0.0754 | 0.0813 |
| all | V_mean_paired_worlds | 60 | 0.5417 | 0.6500 | -0.1737 | 0.1010 | 0.1053 |
| all | PolicyGreedy_when_in_pair | 60 | 0.4917 | 0.5000 | -0.0552 | 0.1292 | 0.1325 |
| search_top2 | V_same_search_world | 32 | 0.4844 | 0.6667 | 0.1934 | 0.0723 | 0.0807 |
| search_top2 | root_Q | 32 | 0.3906 | 0.3333 | 0.0409 | 0.1370 | 0.1408 |
| search_top2 | MCTS_visit_choice | 32 | 0.5469 | 0.6667 | 0.0241 | 0.0669 | 0.0754 |
| search_top2 | V_mean_paired_worlds | 32 | 0.4844 | 0.6667 | 0.2053 | 0.0723 | 0.0807 |
| search_top2 | PolicyGreedy_when_in_pair | 32 | 0.5469 | 0.6667 | — | 0.0669 | 0.0754 |
| search_vs_policy | V_same_search_world | 28 | 0.5357 | 0.5455 | -0.5499 | 0.1568 | 0.1549 |
| search_vs_policy | root_Q | 28 | 0.5714 | 0.6364 | 0.2897 | 0.0851 | 0.0879 |
| search_vs_policy | MCTS_visit_choice | 28 | 0.5714 | 0.6364 | 0.0033 | 0.0851 | 0.0879 |
| search_vs_policy | V_mean_paired_worlds | 28 | 0.6071 | 0.6364 | -0.5418 | 0.1339 | 0.1334 |
| search_vs_policy | PolicyGreedy_when_in_pair | 28 | 0.4286 | 0.3636 | — | 0.2003 | 0.1977 |

四个同选对照不计排序、相关与损失；未分辨局面不当作已知真值。固定32份完整对照、后验期望正确率、按根bootstrap区间、逐根差值及补样记录见summary.json。经验错选损失为两动作间基于MC均值的损失，后验期望损失考虑剩余不确定性，均非最优博弈遗憾值。z损失乘50为续局得分率百分点。

子节点V主指标来自原S1024同一隐藏世界，按root玩家转换符号，并与native一致裁剪到[-1,1]；终局直接用真实payoff。V_mean_paired_worlds为额外平均世界诊断，其预测与MC目标共享抽样，只作辅助。MCTS的差值相关使用访问份额差，不是价值差。根Q是单世界搜索树自身策略的估计，而参考值来自Greedy续局；二者差异可能包含续局策略不一致，不能仅凭排序误差归罪value或搜索。

隐藏世界采样沿用现有均匀重建器，非历史行为后验。本样本按搜索冲突分组，不是总体棋力评测。非终局工程上限不算和棋，若存在会单独列出并给收益差的最坏情况范围。
