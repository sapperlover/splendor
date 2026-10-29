# 停写冗余抽样记录与500步checkpoint

2026-10-23按用户授权：停写当前训练的policy/value两份`.bin`和两份`.steps.jsonl`，在线保存周期由50步改为500步，七个固定诊断点保持`0,100,250,500,1000,2000,3000`。详细记录语义见[训练产物指南](../../../docs/guides/training_artifacts_zh-CN.md)。

`ReferenceWriter`增加不持久化明细的模式，保留原batch摘要及小型身份JSON；`HybridReferenceWriter`默认使用该模式，近期离线单路目标训练同样关闭明细。主指标仍逐步记录，恢复不复制完整主指标到会话文件。旧显式审计、历史产物与T7归档保留。

在线周期点与固定点取并集，完整一轮保存9个不同训练节点，另保留`final.pt`，共10个`.pt`文件；会话限步会额外保存会话终点。解除“诊断点必须整除保存周期”的校验，恢复只放开保存周期差异，学习率等算法设置仍需一致。

实际启动脚本使用独立`runtime/entity_attention_r31_10k_p60k_iteration_20261023/source_recording_checkpoint_20261023`。以原六轮调度快照为底，只应用[五个文件的本次补丁](patches.json)，没有带入工作区其他模型／辅助头改动；完整文件身份见[runtime_source.json](runtime_source.json)。原冻结源码、原始run配置、已完成模型／面板及state均不改写。后续运行会记录新会话／轮次配置。

发布准备入口为`artifacts/entity_attention_r31_10k_p60k_iteration_20261023/prepare_recording_checkpoint.py`；旧`prepare.py`、`prepare_fixed_holdout.py`及`prepare_six_round_holdout.py`保留历史身份，不用于更新当前发布。

工作区[86项回归](workspace_tests.xml)和实际冻结源码[37项回归](runtime_tests.xml)全部通过，使用微型数据、mock采集与有限CPU smoke；覆盖不生成四份明细文件、主日志batch摘要连续、改变周期后的精确恢复、周期点与七诊断点并集、全量留出与轮间切换。[实际脚本只读预检](input_check.json)通过；[已完成目标恢复](noop_resume.json)返回0、state SHA不变，仅复用42个已完成留出结果，未执行正式采集、续训或棋力评测。历史372个checkpoint及原始run配置保持，完整核对见[verification.json](verification.json)。工程验证不代表新的棋力结果，不清理旧checkpoint。

验证命令：从仓库根目录使用`conda run -n splendor env PYTHONPATH=src python -m pytest -q`运行六个相关测试文件；冻结源码使用`PYTHONPATH=runtime/entity_attention_r31_10k_p60k_iteration_20261023/source_recording_checkpoint_20261023/src`运行driver、warmup、mixed-budget及断点记录回归。实际入口为原脚本的`--check`和`--rounds 6`，后者只用于已完成6轮的空跑恢复检查，不能将其改成更大轮数当成只读命令。
