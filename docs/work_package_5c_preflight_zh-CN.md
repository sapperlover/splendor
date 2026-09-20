# 工作包 5C：5M 主训练配置冻结与 Preflight

## 1. Preflight 结论

**PASS**。

本轮只冻结第一条 5C 主实验、补齐运行/恢复/评测入口并完成短 preflight，**没有启动 5M 正式训练，也没有访问 FINAL seed**。

随后完成了正式训练前的 multiprocessing collector preflight。`8 workers × 4 envs` 是多进程方案中最快的拓扑，但平均 `315.94 decisions/s` 低于原单进程的 `341.63 decisions/s`（`0.925x`），所以正式配置保持单进程 32 logical env。完整结果见 `docs/work_package_5c_multiprocessing_preflight_zh-CN.md`。5M 仍未启动。

run-format v4 GPU preflight 在 RTX 4050 Laptop GPU 上使用 32 个逻辑环境，完成 32 局 latest/historical neural-opponent 对局和 1,492 个学习样本，做了一次小型 PPO update。验证了：

- `step_000512112.pt` 存在，feature schema、rules version、model config 匹配；
- Adam 有 20 组可恢复 state，checkpoint 可完整加载；
- active environments = 32，frozen latest 与 learner 参数存储独立，PPO update 后位级不变；
- latest 和 historical 神经对手都完成了真实对局；
- run-local `1,234,567` 处 LR 为 `8.2716062e-5`，entropy coefficient 为 `0.0080246928`；
- 完整 5C checkpoint 恢复了 model、Adam、schedule position、sampler RNG、mixture RNG、pool manifest/RNG 和 collector next-episode index；
- TensorBoard writer 可正常生成 event；
- TRAIN / Quick / Formal / Promotion / Historical / FINAL / preflight seed domains 不相交。

Preflight 原始结果保存在 `runs/5c_main_seed1/preflight.json`。
全量回归、Python 编译检查和 `git diff --check` 作为每次协议更新的验收项。两个 CUDA 条件测试在普通工具沙箱中会跳过；宿主 GPU preflight 需要实际运行 CUDA 路径。

## 2. 冻结的正式配置

| 项目 | 固定值 |
|---|---:|
| learner 起点 | `step_000512112.pt` |
| run-local 训练样本 | `5,000,000` |
| lifetime 起点 | `512,112` |
| training seed | `1` |
| device / precision | CUDA / FP32 |
| collector | 单进程（multiprocessing preflight 未显示吞吐收益） |
| 最佳 multiprocessing 拓扑（未启用） | `8 workers / 4 envs per worker` |
| active environments | `32` |
| rollout / minibatch | `8192 / 512` |
| PPO epochs | `4` |
| gamma / GAE lambda | `1.0 / 0.95` |
| clip / value coefficient | `0.2 / 0.5` |
| max grad norm / target KL | `0.5 / 0.02` |
| optimizer | Adam，保持现有 betas/eps，weight decay `0` |
| reward | 纯 terminal `+1/-1/0`，shaping alpha `0` |
| LR | run-local `0..5M`：`1e-4 -> 3e-5` 线性下降 |
| entropy | run-local `0..5M`：`0.01 -> 0.002` 线性下降 |
| 常规 checkpoint / quick DEV | 每 `100k` / `50k` |
| formal DEV | 每 `250k` |
| quick / formal pairs | `50 / 200` |
| FINAL minimum pairs | `500` |

LR 和 entropy 的自变量是 **run-local learning-decision samples**，checkpoint 保存当前 schedule position 和值。resume 会从该 position 继续，不会重置为起点。

正式样本预算精确为 5,000,000。为使 checkpoint 仍保持在完整 episode 边界，checkpoint/eval 时已开始的对局会运行至结束，但边界尾段不加入 PPO rollout，不计入 5M、LR/entropy schedule 或 50k/250k 计数器，并以 `boundary_drain_decisions_not_trained` 显式记录。

## 3. Opponent schedule、历史池与评测协议

现行 5C run-format v4 的完整协议见 [work_package_5c_current_protocol_zh-CN.md](work_package_5c_current_protocol_zh-CN.md)。本报告只保留运行与 preflight 说明。

训练总体比例仍为：

- Phase A（0～1M）：30% latest、20% historical、25% Score、25% Development；
- Phase B（1M～5M）：50% latest、30% historical、10% Score、10% Development。

active historical pool 改为 2 个 fixed anchors（157351、512112）加 10 个 dynamic slots。dynamic 非空时，historical bucket 内按 1/3 fixed、2/3 dynamic 抽样；为空时全部从 fixed 抽样。新 global best 必定进入 dynamic pool；Formal winner 与现存所有 dynamic checkpoint 都相距至少 500k 时也进入。满池使用确定性时间间距淘汰规则。

## 4. DEV、promotion 与 best-dev

- Quick：ScorePolicy、DevelopmentPolicy，各 50 paired seeds，不更新 best；
- Formal fixed：周期内 Quick Top 2 加末尾 checkpoint；每个候选对 Score、Development、512112 各跑 200 paired seeds；
- 主指标：`0.5 * Score + 0.5 * Development`；
- promotion：Formal candidates 按主指标降序，所有不低于 incumbent Formal 分数 0.01 以上的候选按序挑战，首个成功者晋升；每次先 200 pairs，bootstrap 10th/90th percentile 跨阈值时扩至总计 500 pairs；
- paired bootstrap 固定 100,000 次，使用按 cycle/candidate/stage 派生并写入结果的确定性 seed；
- Secondary Historical：Formal winner 对当前 active pool 全成员各跑 50 pairs，只作遗忘/cycling 诊断；
- Quick、Formal、Promotion、Historical、FINAL 使用互不相交的 seed domain。
- 所有 neural 评测策略使用 greedy argmax；训练 learner 和 neural opponent 使用 stochastic categorical sampling。

FINAL 在训练过程中不会自动运行。

## 5. 直接复制的命令

所有命令都在仓库根目录执行。

### A. 环境/版本检查

```bash
conda run -n splendor python -c "import platform, torch; print('Python:', platform.python_version()); print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

### B. 全量测试

```bash
conda run -n splendor python -m pytest
```

### C. 正式 5M 训练

如果 `runs/5c_main_seed1/` 已被删除或移走，先重新生成一次冻结配置：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.train_5c prepare \
  --run-root runs/5c_main_seed1
```

然后只对这个新目录执行一次 `train`：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.train_5c train \
  --config runs/5c_main_seed1/config.json \
  2>&1 | tee -a runs/5c_main_seed1/logs/train.log
```

`train` 会先对 step 0 运行初始 Formal fixed baseline：3 个对手 × 200 paired seeds × 2 个先后手，共 1,200 局；随后对初始两个 active historical anchors 各跑 50 pairs，共 200 局。这期间 learning-decision step 仍为 0，但不是死锁。标准输出会立即写入 `training_process_started`、`evaluation_panel_started` 和每个对手的 started/completed 事件，并在每个对手内大约每 10% paired seeds 输出一次 progress。

### D. 从 latest 恢复

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.train_5c resume \
  --config runs/5c_main_seed1/config.json \
  2>&1 | tee -a runs/5c_main_seed1/logs/train.log
```

`resume` 自动读取 `runs/5c_main_seed1/checkpoints/latest.json`，不需要人工猜测最后一个 step 文件名。

如果在 step-0 formal DEV 中断，`resume` 会识别“checkpoint 已写入但初始基线尚未完成”，重新完成该基线后才进入采样；不会把这种中间 checkpoint 误当成已完成的 step 0。

### E. 指定 checkpoint 的 Quick DEV

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.evaluate_5c \
  --config runs/5c_main_seed1/config.json \
  --checkpoint runs/5b_stage4_selfplay_seed_004/checkpoints/step_000512112.pt \
  --kind quick --pairs 50 --device cuda \
  --output runs/5c_main_seed1/eval/quick/manual_step_000512112.json
```

### F. 指定 checkpoint 的 Formal DEV

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.evaluate_5c \
  --config runs/5c_main_seed1/config.json \
  --checkpoint runs/5b_stage4_selfplay_seed_004/checkpoints/step_000512112.pt \
  --kind formal --pairs 200 --device cuda \
  --output runs/5c_main_seed1/eval/formal/manual_step_000512112.json
```

### G. 训练完成后对 best-dev 做 FINAL 评测

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.evaluate_5c \
  --config runs/5c_main_seed1/config.json \
  --checkpoint-manifest runs/5c_main_seed1/checkpoints/best_dev.json \
  --kind final --pairs 500 --device cuda \
  --output runs/5c_main_seed1/final/best_dev_final_500_pairs.json
```

### H. TensorBoard

```bash
conda run -n splendor tensorboard --logdir runs/5c_main_seed1/tensorboard --port 6006
```

## 6. 结果路径

| 用途 | 精确路径/解析方式 |
|---|---|
| Run root | `runs/5c_main_seed1/` |
| 冻结配置 | `runs/5c_main_seed1/config.json` |
| 准备元数据 | `runs/5c_main_seed1/metadata.json` |
| compact status | `runs/5c_main_seed1/status.json` |
| preflight | `runs/5c_main_seed1/preflight.json` |
| Metrics | `runs/5c_main_seed1/metrics.csv` 和 `metrics.jsonl`（训练后生成） |
| TensorBoard | `runs/5c_main_seed1/tensorboard/` |
| Training log | `runs/5c_main_seed1/logs/train.log` |
| Checkpoints | `runs/5c_main_seed1/checkpoints/` |
| Latest 解析 | `runs/5c_main_seed1/checkpoints/latest.json` 的 `checkpoint` 字段 |
| Best-dev 解析 | `runs/5c_main_seed1/checkpoints/best_dev.json` 的 `new_best_checkpoint` 字段 |
| Checkpoint 清单 | `runs/5c_main_seed1/checkpoints/manifest.jsonl` |
| Quick DEV | `runs/5c_main_seed1/eval/quick/` |
| Formal DEV | `runs/5c_main_seed1/eval/formal/` |
| Promotion | `runs/5c_main_seed1/eval/promotion/` |
| Historical diagnostic | `runs/5c_main_seed1/eval/historical/` |
| FINAL | `runs/5c_main_seed1/final/best_dev_final_500_pairs.json` |
| 初始历史池 | `runs/5c_main_seed1/historical_pool/initial.json` |
| 各 checkpoint 历史池 | `runs/5c_main_seed1/historical_pool/local_*_lifetime_*.json` |
| 历史池变更记录 | `runs/5c_main_seed1/historical_pool/events.jsonl` |
| Crash/failure dump | `runs/5c_main_seed1/failures/` 和 `eval/failures/` |
| 最终摘要 | `runs/5c_main_seed1/summary.json`（训练完成后生成） |

当前 run root 只处于 `prepared` 状态；`latest.json`、`best_dev.json`、metrics、正式 DEV 和 FINAL 结果尚未生成，这是预期状态。
