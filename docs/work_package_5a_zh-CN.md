# 工作包 5A：固定 Random 对手短训练与学习验证

## 1. 验收结论

5A 分类为 **A：明确学习**。随机初始模型在独立 DEV-EVAL 上对 RandomPolicy 的 score rate 为 `0.005`；训练到约 25k 学习方决策后上升到 `0.955`，最终 100,005 步的 200 对/400 局关键评测为 `388/0/12`，score rate `0.970`。中间小样本评测有局部回落，但整体提升明确，且没有系统正确性异常。

本工作包仅训练当前 ActorCritic + PPO，固定对手仅为 RandomPolicy，奖励仅为正式终局 `win=+1 / loss=-1 / draw=0`，reward shaping alpha 为 0。

## 2. 实现范围

新增 `splendor.train_5a` 作为单进程训练 CLI，支持总学习决策数、训练 seed、device、rollout 大小、评测/检查点间隔、评测对数、watchdog、恢复 checkpoint 和输出目录参数。入口会拒绝非 Random 对手和非零 shaping，不能意外开启后续工作包功能。

每局由相互独立的确定性 RNG stream 产生 game seed、学习策略 seed、RandomPolicy seed、learning-player 座位和首玩家。学习方和对手在一局内固定。策略仍只能通过 `DecisionRequest -> EncodedDecision`看到 Observation 和引擎枚举的合法候选；评测适配器使用 `model.eval()` 和 greedy argmax，评测前后会检查参数未变。

训练目录同时写入：

- `config.json`：完整训练、PPO、模型和 runtime 配置；
- `metrics.csv` 和 `metrics.jsonl`：每个 update/评测点的结构化原始记录；
- `tensorboard/`：按 cumulative learning-player decisions 记录 `eval/ppo/rollout/behavior/system` scalar；
- `checkpoints/`：模型、Adam、计数、全部训练 RNG 和 5A 进度；
- `eval/`：包含逐局结果的成对换先手评测记录；
- `logs/`：工程截断或异常的可重放现场。

新运行会拒绝覆盖非空目录。TensorBoard 和结构化 writer 在正常退出或异常退出时都会 flush/close。运行产物位于已忽略的 `runs/`，不提交大 checkpoint 或 event 文件。

## 3. 实验配置与环境

| 项目 | 实际值 |
|---|---:|
| 设备 | NVIDIA GeForce RTX 4050 Laptop GPU / CUDA |
| Python | 3.12.14 |
| PyTorch / CUDA runtime | 2.14.0+cu130 / 13.0 |
| 实验基线 git commit | `26f1f72fb1cccf2b3c374bbfff92d52b1f6b70bc` (`dirty=true`，即包含本工作包实现) |
| rules version | `splendor-base-two-player-v2-forced-skip` |
| feature schema | `splendor-observation-action-v1` |
| RNG algorithm | `splitmix64-v1` |
| 模型参数量 | 302,626 |
| training seed | 1 |
| 总 learning-player decisions | 100,005 |
| PPO updates | 14 |
| rollout target | 8,192 |
| 训练完整对局 | 1,816 |
| wall-clock | 764.10 s（含全部中间/关键评测与两个进程） |

PPO 使用 4C 冻结的默认值：`lr=3e-4`、`clip=0.2`、`value_coef=0.5`、`entropy_coef=0.01`、`max_grad_norm=0.5`、`epochs=4`、`minibatch=512`、`target_kl=0.02`、`gamma=1.0`、`lambda=0.95`。

TRAIN game seed 使用非负域，本次实际为 `10,000,000..10,001,815`。每局的 learning-policy/opponent/seat/first-player stream 分别使用 `10^15 / 2×10^15 / 3×10^15 / 4×10^15 + game seed`。DEV-EVAL 使用负数域；中间固定 50 对为 `-10,000,001..-10,000,050`，最终 200 对为 `-10,000,001..-10,000,200`。训练样本不会使用 DEV seed。

## 4. DEV-EVAL 学习曲线

中间 DEV 为 50 对/100 局；100,005 步 key 评测为 200 对/400 局。`first/second` 是 PPO 执先/执后的 score rate。由于 rollout 与 checkpoint 只在完整 episode 边界截断，评测步数可略高于目标点。

| step | pairs/games | W/D/L | score rate | first | second | mean/median turns |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50/100 | 0/1/99 | 0.005 | 0.010 | 0.000 | 80.18 / 80 |
| 24,580 | 50/100 | 95/1/4 | 0.955 | 0.920 | 0.990 | 65.59 / 66 |
| 40,988 | 50/100 | 98/1/1 | 0.985 | 0.980 | 0.990 | 66.09 / 66 |
| 50,036 | 50/100 | 99/1/0 | 0.995 | 0.990 | 1.000 | 65.71 / 66 |
| 66,475 | 50/100 | 96/0/4 | 0.960 | 0.960 | 0.960 | 66.38 / 66 |
| 82,873 | 50/100 | 98/1/1 | 0.985 | 0.970 | 1.000 | 66.68 / 66 |
| 100,005 DEV | 50/100 | 95/0/5 | 0.950 | 0.920 | 0.980 | 66.32 / 66 |
| 100,005 key | 200/400 | 388/0/12 | 0.970 | 0.980 | 0.960 | 65.79 / 65 |

按固定小 DEV 集的 score rate，本次最好 checkpoint 是 `step_000050036.pt` (`0.995`)。最终 checkpoint 是 `step_000100005.pt`，其更大 key 集 score rate 为 `0.970`。中间回落在 100 局样本上可见，本工作包没有据此调参。

## 5. PPO 与行为诊断

- entropy 从首个 update 的 `2.1081` 平滑下降到 `1.6545`，最低未接近 0；
- approximate KL 全程最大 `0.00382`，低于 `target_kl=0.02`，无 KL early stop；
- value loss 从 `0.1168` 降至 `0.0038`，中间无发散；
- explained variance 前 13 个 update 大致位于 `0.69..0.89`，最后一个仅 748 样本的尾 batch 为 `0.371`；
- clip fraction 最大 `0.0448`，所有 loss、gradient 和 advantage/return 都通过 finite 检查。

1,816 局训练局中，学习方为 `1517/3/296`（训练胜率仅作诊断），座位 0/1 分别为 890/926 局。学习方 MAIN_ACTION 累计分布为：

| action | count | fraction |
|---|---:|---:|
| TakeDistinct | 19,571 | 30.88% |
| TakePair | 2,025 | 3.19% |
| ReserveMarket | 7,636 | 12.05% |
| ReserveDeck | 946 | 1.49% |
| BuyMarket | 29,708 | 46.87% |
| BuyReserved | 3,498 | 5.52% |

其他决策阶段计数为 PAYMENT `33,206`、RETURN_TOKENS `3,230`、CHOOSE_NOBLE `185`。每局平均购买 `18.29`、保留 `4.73`、拿 token `11.89`、取得贵族 `1.10`，学习方最终平均分 `15.37`；平均完整玩家回合数 `69.83`。策略没有退化成只拿 token、从不买牌或只保留的循环。

## 6. 真实 checkpoint 退出/恢复

第一个训练进程从 step 0 运行到 `50,036`（7 次 update），保存 `step_000050036.pt` 后正常退出。第二个独立进程加载该 checkpoint，恢复 model、Adam、global step、update count、RNG 和训练进度，从 update 8 继续到 step `100,005` / update 14。

checkpoint 仅在完整 episode 边界保存，因此不需要丢弃或伪造局中环境状态。恢复后第一个 update 为 step `58,228` / update 8，训练计数、曲线、optimizer 和独立 RNG stream 正常连续。这是真实进程退出与重启，不是单测中的序列化往返。

## 7. 吞吐与资源

| 项目 | 实测 |
|---|---:|
| rollout 加权吞吐 | 188.81 learning decisions/s |
| rollout 平均对局吞吐 | 3.43 games/s |
| 全流程吞吐（含评测） | 130.88 decisions/s |
| rollout 总时间 | 529.65 s |
| 环境+合法动作+编码 | 185.49 s |
| model sampling/inference | 344.16 s |
| PPO update | 17.47 s |
| evaluation | 215.83 s |
| CUDA peak allocated | 158.09 MiB |
| CPU max RSS（粗略） | 1,927.30 MiB |

当前最大耗时为单决策 model sampling/inference，其次是环境+编码；PPO update 不是本次实验的主要瓶颈。5A 没有因此引入多进程、inference server、AMP 或其他优化。

## 8. 正确性与已知局限

训练累计 forced skip `28`，engineering truncation `0`。全部评测点均为 IllegalAction `0`、invariant failure `0`、exception `0`、engineering truncation `0`。最终 400 局关键评测有 forced skip `10`。未观察到 NaN/Inf、candidate mapping 错误、对局长度异常增长或策略无意义循环。

新增 6 个 5A 测试，覆盖 train/DEV seed 隔离与座位变化、禁用未来训练功能、greedy 冻结评测、整局对手/学习座位固定、terminal/truncation 统计分离，以及包含 step-0 评测和真实 checkpoint load 的小型 CLI 恢复。4C 既有测试继续覆盖 rollout 采样期参数冻结。最终编译检查通过，全量测试为 `172 passed, 2 skipped`；两个 skip 均是当前工具沙箱不可见 CUDA 时的条件跳过，而本次实际 CUDA 训练已覆盖同一前向/更新路径。

在确认对 Random 存在明确学习后，又对最终 `step_000100005.pt` 做了可选的两组只评测。两组都使用相同的 200 个 DEV paired seeds（400 局）、冻结 greedy PPO 与换先后手协议：

| opponent | W/D/L | score rate | PPO first | PPO second | mean/median turns |
|---|---:|---:|---:|---:|---:|
| ScorePolicy | 79/1/320 | 19.875% | 24.00% | 15.75% | 60.94 / 62 |
| DevelopmentPolicy | 42/1/357 | 10.625% | 13.00% | 8.25% | 62.84 / 62 |

这 800 局全部自然结束，engineering truncation、IllegalAction、invariant failure 和 exception 均为 0；模型参数在评测前后严格一致。结果表明 5A 策略已学会稳定击败 Random，但棋力仍明显低于两个手写启发式，其中 DevelopmentPolicy 更强。

已知局限：中间 DEV 仅 50 对，存在可见的样本噪声；实验仅有一个 training seed，因此证明了本次确定性实验的学习方向，不是多 seed 统计稳定性。`best_checkpoint` 使用 50 对小 DEV 集选择，不应将 `0.995` 解读为长期泛化保证。

## 9. 运行方式与未实现内容

从仓库根目录运行：

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.train_5a \
  --total-learning-steps 100000 \
  --training-seed 1 \
  --opponent random \
  --reward terminal \
  --device cuda \
  --rollout-size 8192 \
  --eval-interval 20000 \
  --checkpoint-interval 50000 \
  --eval-pairs 50 \
  --final-eval-pairs 200 \
  --output-dir runs/5a_random_seed_001
```

恢复时增加 `--resume-checkpoint runs/5a_random_seed_001/checkpoints/<checkpoint>.pt`。TensorBoard 启动方式：

```bash
conda run -n splendor tensorboard --logdir runs
```

本工作包没有实现 historical/self-play opponent pool、Score/Development 混合训练、reward shaping、behavior cloning、多进程 sampler、GPU inference server、`torch.compile`、AMP、分布式训练、search、超参搜索或百万级长训练。未开始工作包 5B，等待人工确认本学习曲线。
