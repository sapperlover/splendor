# Splendor 基础版规则数据

- 中文规则与引擎规格：[`docs/splendor_standard_rules_zh-CN.md`](docs/splendor_standard_rules_zh-CN.md)
- 90 张发展卡：[`data/splendor_cards.csv`](data/splendor_cards.csv)
- 10 块贵族：[`data/splendor_nobles.csv`](data/splendor_nobles.csv)
- token、人数与终局常量：[`data/splendor_config.json`](data/splendor_config.json)
- 工作包 1 数据契约与阶段机：[`docs/work_package_1_contract_zh-CN.md`](docs/work_package_1_contract_zh-CN.md)
- 工作包 2A 非购买动作切片：[`docs/work_package_2a_zh-CN.md`](docs/work_package_2a_zh-CN.md)
- 工作包 2B 购买与支付切片：[`docs/work_package_2b_zh-CN.md`](docs/work_package_2b_zh-CN.md)
- 工作包 2C 贵族、完整回合与终局：[`docs/work_package_2c_zh-CN.md`](docs/work_package_2c_zh-CN.md)
- 工作包 2D 引擎集成验证报告：[`docs/work_package_2d_engine_validation_zh-CN.md`](docs/work_package_2d_engine_validation_zh-CN.md)
- 工作包 3A 基线策略与评测框架：[`docs/work_package_3a_zh-CN.md`](docs/work_package_3a_zh-CN.md)
- 工作包 3B Observation 与合法候选数值编码：[`docs/work_package_3b_zh-CN.md`](docs/work_package_3b_zh-CN.md)
- 工作包 4A PyTorch Deep-Sets Actor–Critic：[`docs/work_package_4a_zh-CN.md`](docs/work_package_4a_zh-CN.md)
- 工作包 4B 单学习方环境与 transition：[`docs/work_package_4b_zh-CN.md`](docs/work_package_4b_zh-CN.md)
- 工作包 4C Rollout、GAE 与 PPO 学习闭环：[`docs/work_package_4c_zh-CN.md`](docs/work_package_4c_zh-CN.md)
- 工作包 5A 固定 Random 对手短训练与学习验证：[`docs/work_package_5a_zh-CN.md`](docs/work_package_5a_zh-CN.md)
- 工作包 5B Stage 1–4 总览：[`docs/work_package_5b_zh-CN.md`](docs/work_package_5b_zh-CN.md)
- 工作包 5B Stage 2 batched collector：[`docs/work_package_5b_stage2_zh-CN.md`](docs/work_package_5b_stage2_zh-CN.md)
- 工作包 5B Stage 3 opponent pool：[`docs/work_package_5b_stage3_zh-CN.md`](docs/work_package_5b_stage3_zh-CN.md)
- 工作包 5B Stage 4 短历史池 self-play：[`docs/work_package_5b_stage4_zh-CN.md`](docs/work_package_5b_stage4_zh-CN.md)
- Stage 4 扩展成对评测与 bootstrap CI：[`docs/work_package_5b_stage4_paired_ci_zh-CN.md`](docs/work_package_5b_stage4_paired_ci_zh-CN.md)
- 工作包 5C 5M 主训练配置与 preflight：[`docs/work_package_5c_preflight_zh-CN.md`](docs/work_package_5c_preflight_zh-CN.md)

当前代码已完成工作包 1、2A～2D、3A、3B、4A～4C、5A，以及 5B Stage 1–4。Stage 4 best-dev checkpoint 为 `step_000512112.pt`。5C 的 5M 主训练配置、真实 CLI、独立实验目录与 GPU preflight 已完成；正式 5M 训练尚未启动，等待人工执行冻结命令。

在项目 Conda 环境运行测试：`conda run -n splendor python -m pytest`

从仓库根目录运行基线评测：`PYTHONPATH=src python -m splendor.evaluate --player-a score --player-b development --seed-start 0 --seed-count 100`

运行 5A 固定 Random 对手训练：

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.train_5a \
  --opponent random --reward terminal --total-learning-steps 100000 \
  --device cuda --training-seed 1 --output-dir runs/5a_random_seed_001
```

查看 TensorBoard：`conda run -n splendor tensorboard --logdir runs`

数据范围仅限 2–4 人基础标准版，不含扩展、《璀璨宝石：对决》或漫威版。
