# 第九轮固定窗口：500 / 1000 / 2000 / 5000步

后续已补齐100／250步，并完成较小学习率、9k／3k／1k在线池及batch单变量对照，见[后续完成结论](../entity_attention_r9_ofat_20261010/analysis.md)。后续同时补算最新代固定Train，明确了value长训的训练—留出分化；原有四点结果不变。

2026-10-10已完成。500／1000／2000／5000步的第九轮完整留出KL分别为0.437501／0.438483／0.443908／0.436822，MSE为0.715736／0.726227／0.772683／0.829076。1000步相对500步两项差值区间均跨0；2000步两项均变差；5000步KL的微小下降区间跨0，MSE上升0.113341，整局配对95% CI为[+0.059843,+0.170553]。不支持通过简单延长本轮联合训练改善最新一代留出拟合，也不能据此确定最优步数或棋力变化。

新增4500步耗时218.55秒，含训练中三个节点的拟合评估；四次完整留出评估合计3.39秒。[验证记录](verification.json)确认原500步的模型、AdamW和全部RNG状态逐位保留，并核对1,152,000条新增抽样引用，均属于原Train分片；留出引用抽中数为0。逐局分子／分母复算与汇总一致，四个节点的模型身份、步数与配置一致。

用户确认从第八轮结束模型（38,398步）出发，用已采好的第九轮1000局，保留原第4–9轮6000局活动窗口，只改变训练步数。前500步直接复用原第九轮模型（38,898步），继续其AdamW、采样RNG和Torch RNG训练到总1000、2000、5000步；这些是同一轨迹的嵌套检查点。学习率、权重衰减、loss权重、梯度裁剪、batch256与原有效配置相同，没有新采集或教师升级。

训练只抽取窗口内原Train对局。评估完整第九轮原Validation：100局、8174个value局面、6688个有效policy局面，不重新划分、不抽小样。KL为原行为搜索标签与学生分布的KL(教师∥学生)，MSE使用原终局value标签。复用现行`pool_fitting.evaluate`指标实现，保留逐局分子／分母，并对各点相对500步的差值做10,000次整局配对bootstrap。

固定模型、窗口、原split、优化器配置及数据身份见[plan.json](plan.json)；运行源码与环境见[execution.json](execution.json)。[结果表](results.md)和[汇总](summary.json)每个指定节点自动更新；完整指标在`fitting/`，checkpoint、逐步训练日志和抽样账本在忽略的`runs/entity_attention_r9_step_extension_20261010/`，控制台日志与源码快照在`runtime/entity_attention_r9_step_extension_20261010/`。

从仓库根目录启动或中断续跑：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u artifacts/entity_attention_r9_step_extension_20261010/run.py run
```

每50步保存完整优化器与RNG检查点。Ctrl+C后重新执行同一命令，从最近完整检查点继续；未提交部分重新训练。`run.py analyze`只复算统计。原实验与正在运行的组合对战不修改、不停止。

这些结果反映固定标签与固定DEV上的拟合，不构成棋力测试，也未覆盖训练seed差异；不能单凭留出指标确定最佳自迭代步数。
