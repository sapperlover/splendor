# 低学习率选中节点对R31：1000pair greedy

2026-10-22按用户追加授权完成1000pair／2000局，候选在对战前固定为本段新增4000步、累计63453步的完整checkpoint，SHA `f3b0dd6a6e9e0241ccc646a0dd00f5519399f989bacf84db5264aef00614fdf6`。这是完整留出KL＋MSE最低节点（0.217213／0.759487），不是本段10000步终点，也未拼接policy/value或按对战重新选点。

| 对手 | 候选胜／和／负 | 候选得分 | 95%整pair区间 |
| --- | --- | ---: | --- |
| R31 final | 992／22／986 | **50.150%** | **[48.025%,52.300%]** |

双方确定性greedy，12贵族；CPU8worker，耗时109.19秒。开局44000000–44000999，每个交换先后手，和棋半分；整pair bootstrap10000次，seed44020000。异常、非法行动、工程截断、不变量失败及确定性循环判和均0。

## 与降学习率前的共同对手比较

复用父模型59453步对同一R31、相同规则／冻结评测源码／开局面板的已有结果49.525%，不额外运行其他对战。新模型减父模型的共同对手得分差为 **+0.625个百分点，95%整pair差值区间[-2.051,+3.326]个百分点**（10000次，seed44020001）。该差值不是候选与父模型直接互打的胜率。

本面板未确认新候选强于R31，也未确认降学习率带来greedy棋力提升；跨0／50%的区间不能证明无收益或模型等效。复用已查看开局，单一训练seed，未作多重比较校正。未测搜索和value棋力。

## 证据与复现

- [完整配置](evaluation_configs/greedy_vs_R31.json)、[启动前计划](greedy_vs_R31_plan.json)、[实际命令及执行记录](greedy_vs_R31_execution.json)。
- [原始1000组配对结果](panels/greedy_vs_R31.json)、[汇总和核验](greedy_vs_R31_summary.json)：已核对双方checkpoint／规则／配置SHA、全部交换先后手、逐局实际零搜索、有效局数、得分区间与共同对手差值。
- 原训练曲线和早停选点见[results.md](results.md)，没有恢复训练或运行其他面板。

从仓库根目录复现或恢复该面板，已保存逐局sidecar时使用`--resume`：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=runtime/entity_attention_128x2_offline_r21_r31_20261022/source/src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m splendor.evaluation \
  --config artifacts/entity_attention_128x2_r21_r31_lr2e5_20261022/evaluation_configs/greedy_vs_R31.json \
  --output artifacts/entity_attention_128x2_r21_r31_lr2e5_20261022/panels/greedy_vs_R31.json \
  --progress-every 50 --summary-every 50 --resume
```
