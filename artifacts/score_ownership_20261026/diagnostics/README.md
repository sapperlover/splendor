# 未来得分辅助目标：只补验证诊断

2026-10-26按用户最终更正完成：保留全部支付状态，只补充固定子集及完整留出的诊断。原8路标签、训练样本／权重、loss算式、模型和优化器、抽样顺序、其他辅助目标及主KL＋MSE选点保持。原任务protocol及checkpoint/replay身份保持，既有冻结源码和历史产物未覆盖；新验证字段由当前源码入口执行时产生。

新增信息见[字段和口径](../../../docs/guides/score_ownership_zh-CN.md#验证诊断2026-10-26补充)：各路标签／预测均值、零标签比例、零预测与当前完整评估集合的最优常数MSE基线；四个决策阶段、非支付汇总和当前最高分分段（<5、5–9、≥10）的MSE／MAE／基线／相对常数基线表现。最优常数使用评估集合各路均值，不称为训练集均值；同一面板跨checkpoint固定。空组及零常数方差的比值为JSON null。

实现只在验证时累计原始加权和、float64目标矩，再统一求比例和方差；GPU标量合并回读。不新增训练逐步字段，不改缓存／worker供数、不额外扫描池或执行网络前向。阶段切片中的“非支付”仅是分析视图，不删除支付训练样本。

验证结果：

- [tests.xml](tests.xml)：61项相关回归通过，包含不等权／不等batch大小的总体常数方差、零基线、分段边界、空组、支付状态梯度仍非零、诊断无梯度及不消耗Torch RNG。
- 微型实际离线链路：关闭诊断的串行训练，对照开启诊断的4 worker共享池＋中断恢复；model、AdamW、采样RNG、Torch RNG以及全部既有完整留出指标逐位一致。验证24个样本包含10个支付样本，训练仍记录24个有效value／分数标签；固定和完整留出都有新增字段，逐步日志没有详细诊断字段。
- [real_validation.json](real_validation.json)：从R21–R37每轮随机抽1局，共17局／1356个决策；8路标签逐项匹配checked引擎的牌／贵族归属重放。用初次工程smoke的已有checkpoint执行实际evaluate入口，1356个value和分数样本均保留，其中支付392个、非支付964个；模型、优化器和Torch RNG保持。该checkpoint只用于工程检查，没有正式模型收益结论，也没有执行新训练或搜索。
- [baseline_identity.json](baseline_identity.json)记录本次修改前的标签、损失及梯度任务函数SHA；有限验证核对这些函数原文未变。当前源码／测试身份与初次产物保全检查见[verification.json](verification.json)，本次聚焦变更见[implementation.patch](implementation.patch)。

复核命令（仓库根目录，conda splendor）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_score_ownership.py tests/test_continuation_auxiliary.py tests/test_continuation_diagnostics.py \
  tests/test_policy_aux_online_iteration.py tests/test_training_prefetch.py \
  --junitxml=artifacts/score_ownership_20261026/diagnostics/tests.xml

conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/score_ownership_20261026/diagnostics/validate.py
```

第二条依赖本机原104k局manifest／抽中的17份NPZ及`runtime/score_ownership_20261026/smoke/smoke_checkpoint.pt`；输入位置和SHA保存在real_validation.json。该样本混合原train/validation来源，明确只作工程检查，没有拟合参数或正式留出结论。详细运行日志与本次修改前的聚焦源码快照在`runtime/score_ownership_diagnostics_20261026/`。CPU验证通过；本次没有执行CUDA训练或正式棋力实验。
