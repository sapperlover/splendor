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
- 5C 现行评测、promotion、训练对手与历史池协议：[`docs/work_package_5c_current_protocol_zh-CN.md`](docs/work_package_5c_current_protocol_zh-CN.md)
- 5C 首次 5M 主训练完成报告：[`docs/work_package_5c_5m_training_report_zh-CN.md`](docs/work_package_5c_5m_training_report_zh-CN.md)

当前代码已完成工作包 1、2A～2D、3A、3B、4A～4C、5A、5B Stage 1–4，以及
5C 首次 5M learning-decisions 主训练。该 run 的 DEV 最优 checkpoint 为
`local_004150000_lifetime_004662112.pt`；详细结果、训练节点和局限见上方 5C 完成报告。

目录约定：[`artifacts/`](artifacts/README.md) 只放值得随仓库保留的实验交付物，默认纳入 Git；
`runtime/` 放日志、断点进度、失败现场和其他程序运行状态，整体忽略。专用训练运行目录
`runs/` 同样不进入 Git。

在项目 Conda 环境运行测试：`conda run -n splendor python -m pytest`

从仓库根目录运行基线评测：`PYTHONPATH=src python -m splendor.evaluate --player-a score --player-b development --seed-start 0 --seed-count 100`

运行 5A 固定 Random 对手训练：

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.train_5a \
  --opponent random --reward terminal --total-learning-steps 100000 \
  --device cuda --training-seed 1 --output-dir runs/5a_random_seed_001
```

查看 TensorBoard：`conda run -n splendor tensorboard --logdir runs`

## 本地人机对局网页

从仓库根目录启动本地服务：

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.web_app
```

然后在浏览器打开 `http://127.0.0.1:8765`。开局前可选择
`ScorePolicy` / `DevelopmentPolicy` / [`play_checkpoints`](play_checkpoints) 中的 PPO
checkpoint、自己先后手和 seed。PPO 对手使用 CPU greedy 推理，
不会访问训练中的模型或 GPU。浏览器仅接收
当前玩家的 Observation 派生视图与规则引擎枚举的合法动作；所有规则
判定和状态转移仍由 `GameEngine` 完成。

数据范围仅限 2–4 人基础标准版，不含扩展、《璀璨宝石：对决》或漫威版。
