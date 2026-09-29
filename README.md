# Splendor 双人 AI

- 项目当前快照与导航（AI agent 首先阅读）：[`docs/current_project_state_zh-CN.md`](docs/current_project_state_zh-CN.md)
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
- Policy + Critic 固定深度搜索外挂：[`docs/policy_value_search_zh-CN.md`](docs/policy_value_search_zh-CN.md)
- 独立 determinization PUCT/MCTS 搜索与统一评测：[`docs/puct_mcts_search_zh-CN.md`](docs/puct_mcts_search_zh-CN.md)
- MCTS 自我对弈数据采集、归档与读取：[`docs/mcts_selfplay_data_zh-CN.md`](docs/mcts_selfplay_data_zh-CN.md)
- MCTS 四组 50 pairs 顺序评测脚本与输出约定：[`docs/mcts_50pair_sweep_zh-CN.md`](docs/mcts_50pair_sweep_zh-CN.md)
- 冻结搜索与自博弈采集加速：[`docs/search_throughput_optimization_zh-CN.md`](docs/search_throughput_optimization_zh-CN.md)
- P0–P3 训练、评测与搜索性能优化：[`docs/performance_optimization_p0_p3_zh-CN.md`](docs/performance_optimization_p0_p3_zh-CN.md)
- AlphaZero 式多轮搜索-训练循环：[`docs/alphazero_training_loop_zh-CN.md`](docs/alphazero_training_loop_zh-CN.md)

当前代码已完成工作包 1、2A～2D、3A、3B、4A～4C、5A、5B Stage 1–4，以及
5C 首次 5M learning-decisions 主训练。该 run 的 DEV 最优 checkpoint 为
`local_004150000_lifetime_004662112.pt`；详细结果、训练节点和局限见上方 5C 完成报告。

AlphaZero 式循环（MCTS 采集 → 跨代 replay → 监督 policy/value 训练 → 固定评测）
已实现并跑通微型 smoke，生产规模长训练尚未执行；启动、恢复与产物约定见上方
AlphaZero 文档：`configs/alphazero_loop.json`、`python -m splendor.alphazero_loop --resume`。

目录约定：[`artifacts/`](artifacts/README.md) 只放值得随仓库保留的实验交付物，默认纳入 Git；
`runtime/` 放日志、断点进度、失败现场和其他程序运行状态，整体忽略。专用训练运行目录
`runs/` 同样不进入 Git。

在项目 Conda 环境运行测试：`conda run -n splendor python -m pytest`

新建 5C 配置默认使用 4 workers × 8 envs 的 worker-local CPU 神经对手，并使用 8 个 CPU evaluation workers。已完成的 5M run 仍保留其原始 `single_process` 冻结配置；详见上方 P0–P3 文档。

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
greedy checkpoint、自己先后手和 seed。也可在启动服务时配置一个搜索 checkpoint，
让它作为额外对手出现在列表中：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -u -m splendor.web_app \
  --search-checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --search-depth 6 --search-top-k 8 --search-determinizations 1 \
  --search-seed 9200000 --search-leaf-value-clip 1.0 \
  --search-fold-forced-phases
```

搜索配置在服务端创建 `PolicyValueSearchPolicy`，仅在当前玩家 Observation 与合法动作上工作；
搜索模型使用 CPU 推理。未指定搜索 checkpoint 时，普通 PPO checkpoint 仍使用 CPU greedy 推理。
浏览器仅接收当前玩家的 Observation 派生视图与规则引擎枚举的合法动作；所有规则
判定和状态转移仍由 `GameEngine` 完成。

当前引擎和 AI 范围仅限双人基础标准版，不含 3–4 人、扩展、《璀璨宝石：对决》或漫威版。
