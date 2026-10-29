# 短期 value 与对手 policy 工程验证

同日追加的[共享编码器梯度／对手覆盖率／未知牌比例与条件CE诊断](diagnostics/README.md)使用独立结果目录；下述139项及首次smoke／prepare原件保留首次交付身份。现行字段与配置以[专题指南](../../docs/guides/continuation_auxiliary_zh-CN.md)为准。

2026-10-23交付，含[目标公式、可见性、权重及运行命令](../../docs/guides/continuation_auxiliary_zh-CN.md)。本目录只证明实现及调用链通过有限验证，没有正式训练／棋力结论。

`tests.xml`保存139项通过、0失败／跳过的相关回归结果，覆盖EMA当前项／终局尾项／世界等权／玩家视角、缺口屏蔽、软标签聚合、未知牌与重复槽、两路牌CE权重、辅助梯度、基线输出不变、严格模型加载、离线断点精确恢复及主指标选点。

`verification.json`记录Git HEAD、源码SHA及两份配置在真实44000局池上的prepare-only结果：39629局train／4371局validation，3170796／349435个决策，预算各148632步；数据digest与固定验证引用一致。准备检查未执行正式训练，也没有扫描全池标签覆盖。

`smoke.json`保存3局真实R31源路径／SHA、代码SHA、配置、base checkpoint SHA、一轮CPU更新及3个原生搜索根。源为原池最后3个train对局，非随机统计样本：共257个决策、174个主行动，137个主行动的短期标签有效。中间一局turn36只有一个候选、valid_world_count=0，造成该局turn0–36的37个标签屏蔽，后缀23个有效；另两局各54／60个全部有效。这个78.74%的覆盖率不能外推全池。

原R31权重升级后主policy/value输出逐位一致；有限更新使用8条policy与16条value样本，两项辅助均有有效监督。更新后AlphaZero checkpoint严格重载，3个不同开局的native S8N1（每根总8，共24次模拟）全部完成；hook确认普通搜索未调用辅助头。base583682参数，两项辅助851853参数。只验证CPU/native路径，未做新增辅助的CUDA训练测速。

复现命令（从仓库根目录执行；smoke的checkpoint写入忽略的runtime）：

```bash
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_continuation_auxiliary.py tests/test_entity_attention.py tests/test_entity_attention_relations.py \
  tests/test_attention_first_round.py tests/test_alphazero_training.py tests/test_offline_scratch.py \
  tests/test_offline_mixed_budget.py tests/test_offline_early_stopping.py tests/test_value_auxiliary.py \
  tests/test_joint_pool_training.py --junitxml=artifacts/continuation_auxiliary_20261023/tests.xml

conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/validate_continuation_auxiliary.py \
  --config configs/entity_attention_v1/offline_continuation_r21_r31_20261023.json \
  --checkpoint runs/entity_attention_r20_s256n4_noble12_iteration_20261022/rounds/round-030/checkpoints/final.pt \
  --output runtime/continuation_auxiliary_20261023 --artifacts artifacts/continuation_auxiliary_20261023
```

旧进展辅助与已撤回购买特征的历史记录保留；本次没有切换现行在线配置、启动长训练或改写历史NPZ。
