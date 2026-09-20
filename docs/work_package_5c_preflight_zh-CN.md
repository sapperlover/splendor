# 工作包 5C：5M 主训练配置冻结与 Preflight

## 1. Preflight 结论

**PASS**。

本轮只冻结第一条 5C 主实验、补齐运行/恢复/评测入口并完成短 preflight，**没有启动 5M 正式训练，也没有访问 FINAL seed**。

随后完成了正式训练前的 multiprocessing collector preflight。`8 workers × 4 envs` 是多进程方案中最快的拓扑，但平均 `315.94 decisions/s` 低于原单进程的 `341.63 decisions/s`（`0.925x`），所以正式配置保持单进程 32 logical env。完整结果见 `docs/work_package_5c_multiprocessing_preflight_zh-CN.md`。5M 仍未启动。

GPU preflight 在 RTX 4050 Laptop GPU 上使用 32 个逻辑环境，完成 32 局 latest/historical neural-opponent 对局和 1,512 个学习样本，做了一次小型 PPO update。验证了：

- `step_000512112.pt` 存在，feature schema、rules version、model config 匹配；
- Adam 有 20 组可恢复 state，checkpoint 可完整加载；
- active environments = 32，frozen latest 与 learner 参数存储独立，PPO update 后位级不变；
- latest 和 historical 神经对手都完成了真实对局；
- run-local `1,234,567` 处 LR 为 `8.2716062e-5`，entropy coefficient 为 `0.0080246928`；
- 完整 5C checkpoint 恢复了 model、Adam、schedule position、sampler RNG、mixture RNG、pool manifest/RNG 和 collector next-episode index；
- TensorBoard writer 可正常生成 event；
- TRAIN / DEV / FINAL / preflight 四个 seed domain 不相交。

Preflight 原始结果保存在 `runs/5c_main_seed1/preflight.json`。
最终全量回归为 `202 passed, 2 skipped`，Python 编译检查和 `git diff --check` 通过。两个 skip 是普通工具沙箱不可见 CUDA 时的条件跳过；上述宿主 GPU preflight 已实际运行 CUDA 路径。

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
| checkpoint / quick DEV | 每 `100k` |
| formal DEV | 每 `250k` |
| quick / formal pairs | `50 / 200` |
| FINAL minimum pairs | `500` |

LR 和 entropy 的自变量是 **run-local learning-decision samples**，checkpoint 保存当前 schedule position 和值。resume 会从该 position 继续，不会重置为起点。

正式样本预算精确为 5,000,000。为使 checkpoint 仍保持在完整 episode 边界，checkpoint/eval 时已开始的对局会运行至结束，但边界尾段不加入 PPO rollout，并以 `boundary_drain_decisions_not_trained` 显式记录。

## 3. Opponent schedule 和历史池

Phase A（run-local `0..<1,000,000`）：

- latest 30%
- historical 20%
- ScorePolicy 25%
- DevelopmentPolicy 25%

Phase B（run-local `1,000,000..5,000,000`）：

- latest 50%
- historical 30%
- ScorePolicy 10%
- DevelopmentPolicy 10%

切换边界会先结束现有 episode；旧 episode 不会中途更换对手。RandomPolicy 只出现在 DEV/FINAL，不进入训练 mixture。

初始历史池固定 8 个时期分散的模型：`100005`、`157351`、`206542`、`255701`、`304858`、`354015`、`403183`、`512112`。容量为 12，均匀抽样。每次 formal DEV 会考虑新 best-dev，仅在至少间隔 250k 且真正更新 best-dev 时加入，不会机械地每 250k 添加。初始 8 个快照始终受保护；池满后只在 4 个动态 best-dev 槽内轮换，不会 FIFO 丢掉早期覆盖。

## 4. DEV 与 best-dev

Quick DEV 使用固定 50 paired DEV seeds，对手为 Random、Score、Development、403183 和 512112，只记录健康状态，不更新 best-dev。

Formal DEV 使用固定 200 paired DEV seeds，在 quick panel 上加入 historical 157351 和 304858。best-dev 主依据仍为：

`0.2 * Random + 0.4 * Score + 0.4 * Development`

当 composite 直接提高至少 0.01 时更新；当增益在 `(0, 0.01)` 内时，使用同 seed 的 100,000 次 paired bootstrap，只有 95% CI 下界大于 0 才更新。每次考虑都写入 `best_dev_history.jsonl`。

FINAL 使用与 TRAIN/DEV 不重叠的独立 seed domain，CLI 拒绝少于 500 paired seeds。训练过程不自动运行 FINAL。

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

`train` 会先对 step 0 运行 formal DEV：7 个对手 × 200 paired seeds × 2 个先后手，共 2,800 局。这期间 learning-decision step 仍为 0，但不是死锁。标准输出会立即写入 `training_process_started`、`evaluation_panel_started` 和每个对手的 started/completed 事件，并在每个对手内大约每 10% paired seeds 输出一次 progress；`status.json` 会同步显示 `phase=formal_evaluation` 及当前对手/已完成 paired seeds，TensorBoard 会立即写入 `system/process_alive` 和 `system/formal_evaluation_progress`。

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
| FINAL | `runs/5c_main_seed1/final/best_dev_final_500_pairs.json` |
| 初始历史池 | `runs/5c_main_seed1/historical_pool/initial.json` |
| 各 checkpoint 历史池 | `runs/5c_main_seed1/historical_pool/local_*_lifetime_*.json` |
| 历史池变更记录 | `runs/5c_main_seed1/historical_pool/events.jsonl` |
| Crash/failure dump | `runs/5c_main_seed1/failures/` 和 `eval/failures/` |
| 最终摘要 | `runs/5c_main_seed1/summary.json`（训练完成后生成） |

当前 run root 只处于 `prepared` 状态；`latest.json`、`best_dev.json`、metrics、正式 DEV 和 FINAL 结果尚未生成，这是预期状态。
