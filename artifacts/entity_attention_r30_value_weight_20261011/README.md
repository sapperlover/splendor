# 第29轮起点：value loss权重0.25与0

最新状态见[结果表](results.md)，完整统计见[summary.json](summary.json)。复算汇总：`conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_r30_value_weight_20261011/analyze.py`。

两条**独立分支**均从绝对第29轮终点（global 58898）出发，恢复原第30轮开训checkpoint的模型、AdamW及RNG；分别将value loss权重从1改为0.25、0。执行顺序：0.25训练及对战完成后，再训练及对战0。权重0分支不接续0.25分支。

固定原第25–30轮6000局在线窗口，保留原整局Train/Validation划分、均匀Train局面有放回采样；无离线历史池。lr=2e-5，batch=256，各训练1000步，policy权重1、全局梯度裁剪1，其余不变。每50步保存完整状态，可中断续跑。

训练前与100/250/500/1000步记录各代固定Train子集（每代2048局面）及整个6000局窗口的完整Validation KL/MSE，包含前中期／后期拆分（任一玩家分数达到10为后期）。最新第30轮留出共124局、9994局面，其中policy有效8283局面。原权重1第30轮的对应诊断和对战直接复用，在`baseline/`保留必要原始结果副本。

两个分支终点分别对固定P4V4做200 pair greedy与200 pair S1024/N1搜索，同原第30轮的开局、先后手交换、搜索种子和执行设置。greedy种子30800000–30800199；搜索30900000–30900199。和棋计半分；以整pair bootstrap区间及相对权重1同面板得分差判断。固定DEV面板用于诊断，结果不能直接当成独立新面板确认或训练seed复现。

**权重0取消value-loss梯度，不冻结任何参数。** 既有AdamW动量、weight decay以及policy驱动的共享encoder变化仍会改变value；保留它们是为了只改一个变量。降低value权重也会改变联合梯度的全局裁剪比例；记录原始梯度范数、裁剪频率及参数变化，避免将结果误解为纯critic参数效应。

参数漂移统计中，`critic`为完整value head，`actor`只指最后的policy MLP；runner的`shared_encoder`集合名实际涵盖其余参数，包含共享state encoder与policy action-query模块，不能解释为纯共享encoder范数。

一次性运行／中断后同命令续跑：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u artifacts/entity_attention_r30_value_weight_20261011/pipeline.py
```

配置、冻结数据索引与代码身份见`plan.json`、`replay_snapshot.json`、`execution.json`；各分支下保存拟合原始统计、对战逐pair结果及验证。模型与采样账本位于`runs/entity_attention_r30_value_weight_20261011/`；运行状态／源码快照位于`runtime/entity_attention_r30_value_weight_20261011/`。每个训练／评测阶段使用独立子进程，阶段完成即释放缓存；不改动主自迭代配置和进程。

`pipeline.py`在训练与对战链路退出后自动运行`analyze.py`，更新结果表与汇总；未完成时如实标为incomplete。本次已启动的runner另附`pipeline.py --attach`收尾进程，它阻塞等待runner文件锁释放，不轮询训练或对战进度。
