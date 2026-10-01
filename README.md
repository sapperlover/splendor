# Splendor 双人 AI

双人基础版规则引擎、Deep-Sets Actor–Critic、PPO、冻结网络搜索和 AlphaZero 式训练循环，含 C++ 原生后端与本地人机网页。玩法仅实现双人基础版；不含 3–4 人、扩展或 Duel。

- [当前状态](docs/current_project_state_zh-CN.md) 与 [文档导航](docs/index.md)。
- [标准规则](docs/splendor_standard_rules_zh-CN.md)；卡牌、贵族与常量位于 `data/`。
- [实验索引](artifacts/experiment_index.json)、[产物约定](artifacts/README.md)、[配置导航](configs/README.md)。

PPO seed1 已完成 5M learning decisions；其 DEV best 为 `local_004150000_lifetime_004662112.pt`，latest 为 `local_005000000_lifetime_005512112.pt`。AZ seed1 已完成 5 轮 × 500 局及 θ1–θ5 扩展对照。native CUDA pool 已实现并保存性能对照；窗口与整轮收益分别解读。具体结论和未验证范围见文档导航中的报告。

## 环境与验证

从仓库根目录执行，使用 Conda `splendor` 并设置 `PYTHONPATH=src`。依赖见 `pyproject.toml`；native 扩展构建见 [指南](docs/guides/native_search_zh-CN.md)。

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q tests/test_rules_and_phases.py
conda run -n splendor env PYTHONPATH=src python -m splendor.evaluate \
  --player-a score --player-b development --seed-start 0 --seed-count 100
```

不以旧报告的测试数量代表当前工作树。训练与搜索命令见 [AZ 指南](docs/guides/alphazero_zh-CN.md)、[PPO 协议](docs/specs/ppo_training_protocol_zh-CN.md)、[评测指南](docs/guides/evaluation_zh-CN.md)。checkpoint 和训练 NPZ 不随 Git 分发，需要另行备份或提供。

## 本地人机网页

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.web_app
```

浏览器打开 `http://127.0.0.1:8765`。支持 Score/Development、`play_checkpoints/` 中的 PPO greedy，以及启动时指定的固定深度搜索对手：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.web_app \
  --search-checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --search-depth 6 --search-top-k 8 --search-determinizations 1 \
  --search-seed 9200000 --search-leaf-value-clip 1.0 --search-fold-forced-phases
```

网页模型使用 CPU；浏览器只接收当前玩家视图和合法动作。规则仍在服务端引擎执行。

## 仓库分工

| 目录 | 用途 |
| --- | --- |
| `src/splendor/`、`tests/`、`data/` | 源码、回归、规范数据 |
| `docs/specs/`、`docs/guides/`、`docs/reports/` | 契约、操作、可复核结论 |
| `configs/`、`scripts/` | 稳定配置入口与运行/分析工具 |
| `artifacts/` | 长期实验交付物；相邻 `*.games/` 忽略 |
| `runs/` | 训练 NPZ、checkpoint、TensorBoard，忽略 |
| `runtime/` | 日志、运行状态、失败现场、生成配置及历史归档，忽略 |

旧工作包/计划原文已完整本地打包到 `runtime/document_archive/2026-10-01/`，不继续留在 docs。原实验与恢复目录保持原位，失效测量有哈希清单和替代依据。
