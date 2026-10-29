# 共享编码器梯度与对手标签诊断

2026-10-23追加实现。[字段与公式](../../../docs/guides/continuation_auxiliary_zh-CN.md#训练与验证诊断2026-10-23追加)规定梯度边界、权重、采样频率及质量分母。原目录的139项测试、smoke与prepare记录保持首次交付的源码／配置身份，本目录保存追加后的结果。

新增主policy、终局value、H2／H6／H14、短期value合计、对手类型／购买／保留与对手合计的编码器梯度L2范数，分别记录加权前／后值；实际联合反向的编码器总范数在裁剪前读取。默认第1步及每100步记录，`encoder_grad_norm_every_steps=1`逐步记录，0关闭；两个离线配置均显式设100。该记录频率可在恢复时更改，旧run缺该字段仍可恢复；训练身份不变，每会话source identity保存实际请求值。

对手覆盖率按当前value权重计数有可用标签的比例。未知牌比例分别按购买／保留标签质量归一化，另记合并比例；条件CE也分别除以购买／保留总质量。验证先聚合原始总和再相除，零分母为null。原训练loss与标签协议保持。

`tests.xml`保存156项通过、0失败／跳过的相关回归与新增分析测试。新增测试用解析梯度核对L2／loss系数和未使用参数，用分数policy／value权重核对覆盖率、未知质量、条件CE及跨batch聚合；3次连续联合更新在诊断开关两侧的参数、裁剪后梯度、AdamW状态与RNG逐位一致。另覆盖零辅助质量、共享参数边界、跨诊断采样点的中断恢复及记录频率更改／旧字段缺失的恢复。

`smoke.json`使用原R31权重、相同3局历史数据，执行一次带全部诊断的CPU更新，严格重载并完成3个S8N1原生根（每根总8，共24次模拟）。数值仅为工程验证，不代表棋力或全池估计；源码与输入SHA随记录保存。正式训练及新增CUDA测速均未启动。

该次抽样value batch16条中，对手有效标签3条（覆盖18.75%）；购买／保留条件CE为3.04420／2.83336，未知牌质量比例分别0%／49.13%，合并5.26%。加权后的编码器范数：主policy8.16157、主value17.38713、短期value合计0.49411、对手policy合计0.41045；实际编码器联合范数19.50201。上述数字只描述这一次更新，不能据此判断损失权重或棋力。

```bash
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_continuation_auxiliary.py tests/test_continuation_diagnostics.py \
  tests/test_entity_attention.py tests/test_entity_attention_relations.py tests/test_attention_first_round.py \
  tests/test_alphazero_training.py tests/test_alphazero_diagnostics.py tests/test_offline_scratch.py \
  tests/test_offline_mixed_budget.py tests/test_offline_early_stopping.py tests/test_value_auxiliary.py \
  tests/test_joint_pool_training.py --junitxml=artifacts/continuation_auxiliary_20261023/diagnostics/tests.xml

conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/validate_continuation_auxiliary.py \
  --config configs/entity_attention_v1/offline_continuation_r21_r31_20261023.json \
  --checkpoint runs/entity_attention_r20_s256n4_noble12_iteration_20261022/rounds/round-030/checkpoints/final.pt \
  --output runtime/continuation_auxiliary_20261023/diagnostics \
  --artifacts artifacts/continuation_auxiliary_20261023/diagnostics
```
