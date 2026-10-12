# 第29轮终点：BCD分头在线池趋势实验

三分支独立从**原第29轮结束**出发，使用已经采好的**第30轮1000局及其在线历史**。加载原第30轮`initial.pt`，它保存第29轮终点相同权重及第30轮原AdamW／Torch／采样RNG；起点global58898。不是从第30轮终点重复训练，也不采新一轮。

| 分支 | Policy池 | Value池 |
| --- | --- | --- |
| B | 第28–30轮，3k局 | 第25–30轮，6k局 |
| C | 第25–30轮，6k局 | 第22–30轮，9k局 |
| D | 第28–30轮，3k局 | 第22–30轮，9k局 |

全部沿用lr2e-5、1000步、**每个头batch256**、AdamW原状态、loss权重1:1、共享encoder正常联合反传、全参数统一clip1。不同池的两个256局面批次只产生一次优化器更新；每步共512个带头身份的抽样引用，不是每头batch512。未采用旧split训练的value detach／分头梯度裁剪。A为原配置6k/6k已完成第30轮，只复用现有指标与greedy结果，不重训或重测A。

数据全部来自原无限自迭代的S1024/N1 attention自对弈：`round-012`至`round-020/selfplay`，对应绝对第22–30轮。`dataset.json`由原`round-017`与`round-020/replay_snapshot.json`去重合并，保留原整局train/validation划分。训练只抽Train局面，池内按局面均匀有放回；3k／6k／9k指含留出的来源总局数，实际Train数量及每代决策数见`plan.json`。不使用最初大离线池，不追加或重算标签。

两头从同一原始每步RNG状态分叉抽样，持久化的RNG按原6k池推进。这样同池时两头抽样相同，B的value与C的policy逐步保持原A序列；B/D的policy和C/D的value也相同。标签池未改变的头不会因另一个头的抽样消耗改变而换批次。恢复只需原checkpoint中的RNG；引用按policy256在前、value256在后记录。

指标节点0、100、250、500、1000步。共同9代每代2048个固定Train局面，最近6代直接继承原第30轮引用；Validation使用9代全部原留出对局的全部局面。每代与前中期／后期KL、MSE、teacher top1一致率、value偏差和相关系数均保留；窗口比较使用相同范围和权重。0步公共指标计算一次，三个分支引用；不做大量0步自对战。

每分支终点进行**对P4V4的200 pair greedy**，沿用原第30轮对P4V4的相同开局与交换先后手，8 CPU worker、每worker1线程。Greedy不使用value进行动作选择。计算配对95% CI，以及相对已有A在同面板上的配对得分差。此为重复使用的DEV趋势筛查，不启动任何搜索对战。顺序B训练→B greedy→C训练→C greedy→D训练→D greedy→汇总，子进程逐阶段退出以释放缓存和GPU内存。

从仓库根目录运行（或中断后原命令续跑）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u artifacts/entity_attention_r30_head_pools_20261011/run.py run
```

每100步保存checkpoint，250步另存；中断后从最近完整checkpoint恢复AdamW、模型、Torch和采样状态，丢弃未提交训练行；greedy按已完成pair恢复。不会修改主训练配置、主进程或教师。模型与完整抽样账本在`runs/entity_attention_r30_head_pools_20261011/`，运行日志和状态在同名`runtime/`，配置、指标和结果在本目录。

[结果](results.md)、[机器汇总](summary.json)、[逐代与阶段矩阵](generation_stage_matrix.csv)。可单独运行`run.py analyze`刷新已完成部分。不得把低KL／MSE或greedy趋势直接当成带搜索棋力和长期自迭代改善的证明。

本次已全部完成，主要解读见[趋势结论](conclusions.md)。对实际第29轮起点的greedy差值另复用已有同面板对局计算，原始结果见[起点对照](greedy_start_comparison.json)，复算入口为`compare_start.py`，不增加对战。
