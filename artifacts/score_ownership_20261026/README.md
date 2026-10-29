# 分数归属辅助目标工程验证

2026-10-26实现。[任务、时点、8输出、归一化和命令](../../docs/guides/score_ownership_zh-CN.md)。本目录只保存有限工程验证，不代表模型或棋力收益。

同日追加[验证诊断](diagnostics/README.md)：各路标签／预测均值、零标签比例、零预测和评估集合最优常数MSE基线，以及决策阶段、前中后期误差。按用户最终要求保留支付样本，原训练标签、损失、抽样和其他目标不变；原始`tests.xml`、`smoke.json`、`verification.json`保留初次实现时的源码身份和结果，当前补充的验证另存diagnostics，不覆盖历史证据。

新增测试覆盖：仅统计未来新增分、玩家相对视角、三个等级及双方贵族、零分保留牌的购买、支付／贵族结算时点、末步及中途自动单贵族领取、forced skip／平局、损坏数据明确失败、15分缩放与8路平均MSE、梯度、旧模型／旧身份兼容以及三辅助组合。独立的微型离线链路比较串行训练与4 worker共享池中断恢复，检查model／AdamW／采样RNG逐位一致、分路指标、梯度采样点和主KL＋MSE选点；源码继续使用流式统计、state-only传输和批量标量回读。

`tests.xml`保存131项通过的相关回归；`resume_rng.xml`单独核验4 worker恢复与串行训练的Torch RNG，连同model／AdamW／采样RNG逐位一致。`smoke.json`保存3局真实历史数据的路径／SHA、源码与配置SHA、一轮CPU更新、严格重载及3个native S8N1根（每根总8，共24次模拟）。共243个决策的8路标签逐项与checked引擎重放后的牌／贵族ID归属一致，包含末次付款后自动领取贵族；新标签生成本身不需要trace或重放。hook确认搜索未执行辅助头，升级base权重后主policy/value输出逐位一致。仅本辅助的模型651402参数，未进行CUDA训练测速。

新104k配置已完成prepare-only，见`verification.json`中的计划快照。默认training_workers=4、17GiB共享训练区；同池已测旧policy arena17869150492字节是本目标的保守容量上界，核算依据见专题。未预热完整104k池，也未启动正式训练。

复现短验证：

```bash
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_score_ownership.py tests/test_continuation_auxiliary.py tests/test_continuation_diagnostics.py \
  tests/test_training_prefetch.py tests/test_entity_attention.py tests/test_policy_aux_online_iteration.py \
  tests/test_alphazero_training.py tests/test_offline_mixed_budget.py tests/test_offline_early_stopping.py \
  --junitxml=artifacts/score_ownership_20261026/tests.xml

conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/validate_continuation_auxiliary.py \
  --config configs/entity_attention_v1/offline_score_ownership_r21_r37_20261026.json \
  --checkpoint runs/entity_attention_r20_s256n4_noble12_iteration_20261022/rounds/round-030/checkpoints/final.pt \
  --output runtime/score_ownership_20261026/smoke --artifacts artifacts/score_ownership_20261026
```
