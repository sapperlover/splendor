# Splendor 双人 AI

双人基础版规则引擎、Deep-Sets／实体注意力 Actor–Critic、PPO、冻结网络搜索与 AlphaZero 式训练，含 C++ 原生后端和本地人机网页。规则范围为双人基础版。

从 [文档导航](docs/index.md) 查找操作说明；[当前状态](docs/current_project_state_zh-CN.md) 区分完成、未完成与历史方案；[实验索引](artifacts/experiment_index.json) 定位原始证据。

较早 M4 分支的结论见 [M4 与固定数据诊断完整结论](docs/reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md)。M4 分支已完成12轮／17700步，固定目标可以拟合，稳定留出收益和棋力收益尚未确认；较优步数、学习率和历史配额仍未确定。运行与恢复见 [M4 指南](docs/guides/entity_attention_avg4_iteration_zh-CN.md)，行为变化诊断见 [固定256局面测试集](docs/guides/decision_probe_zh-CN.md)。

最新 [P4起点九轮自迭代完整报告](docs/reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)：9轮1000局／500步、同预算纯旧池对照和全部评测已完成；终点S1024/N1对P4／旧池对照／P12＋V4的200pair得分分别58.750%／61.625%／58.125%，区间均高于50%。旧池loss退化与棋力提升并存，value收益不均衡；限定单训练seed和固定DEV面板，未确定最优参数或教师换代的独立贡献。[流程与恢复命令](docs/guides/entity_attention_p4_pool_iteration_zh-CN.md)保留。

## 环境与使用

从原第9轮终点38898步继续的[长期纯在线续训入口与命令](docs/guides/entity_attention_p4_pool_forever_plan_20261011_zh-CN.md)已准备：继承6000局窗口，1000局／1000步／lr2e-5，每轮1000步完整模型无条件升级，支持同一命令中断续跑；正式任务尚未启动。

从仓库根目录执行，使用 Conda `splendor`，设置 `PYTHONPATH=src`。依赖见 `pyproject.toml`，C++ 扩展构建见 [原生后端指南](docs/guides/native_search_zh-CN.md)。

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q tests/test_rules_and_phases.py
conda run -n splendor env PYTHONPATH=src python -m splendor.evaluate \
  --player-a score --player-b development --seed-start 0 --seed-count 100
conda run -n splendor env PYTHONPATH=src python -m splendor.web_app
```

网页默认地址为 `http://127.0.0.1:8765`。支持 Score／Development、`play_checkpoints/` 中的 greedy、固定深度搜索及原生 MCTS；MCTS 可在开局及对局中调整 S/N 预算。含模型局面评估条、逐步悔棋和顶部临时支付栏。第24轮模型启动命令见 [评测与网页指南](docs/guides/evaluation_zh-CN.md)。网页模型使用 CPU 单线程，可与训练同时运行，浏览器只接收当前玩家视图和合法动作。

训练入口见 [AZ 循环](docs/guides/alphazero_zh-CN.md)、[离线学习](docs/guides/entity_attention_offline_zh-CN.md) 和 [PPO 协议](docs/specs/ppo_training_protocol_zh-CN.md)。checkpoint 和训练 NPZ 不随 Git 分发；历史命令需要对应本地数据和模型。

## 仓库分工

| 目录 | 内容与入口 |
| --- | --- |
| `src/splendor/`、`tests/`、`data/` | 实现、回归、卡牌与常量；[标准规则](docs/splendor_standard_rules_zh-CN.md) |
| `docs/specs/`、`docs/guides/`、`docs/reports/` | 契约、操作说明、可复核结论；[导航](docs/index.md) |
| `docs/archive/`、`configs/archive/` | 被替代的设计与旧准备说明；[历史导航](docs/archive/README.md) |
| `configs/`、`scripts/` | [配置导航](configs/README.md)、[工具导航](scripts/README.md) |
| `artifacts/` | 长期实验交付物、必要原始统计；[产物约定](artifacts/README.md) |
| `runs/` | 训练数据、checkpoint、TensorBoard；Git 忽略 |
| `runtime/` | 日志、缓存、运行状态、失败现场、本地保全；Git 忽略 |

实验目录保留原身份，运行与恢复路径保持稳定。历史资料、移动清单与本地保全位置见 [本次整理记录](artifacts/repository_cleanup_20261009/README.md)。
