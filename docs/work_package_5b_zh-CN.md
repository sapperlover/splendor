# 工作包 5B：Stage 1–3 验收报告

## 1. 当前结论

Stage 1 的固定混合对手实验已完成 500,005 个累计 learning decisions（其中 5B 新采样 400,000）。相对起点 checkpoint，PPO 对 ScorePolicy 和 DevelopmentPolicy 都有明确提升，对 RandomPolicy 未发生灾难性遗忘。开发综合指标最佳的 checkpoint 是 `step_000403183.pt`，综合分 `0.5205`。

Stage 2 的 multi-env batched collector 已实现并通过独立吞吐验收；32 个逻辑环境实测达到 serial 的 `2.76×` decisions/s。Stage 3 的 frozen latest/historical/fixed opponent pool、manifest 和可恢复 RNG 也已实现，并用真实 checkpoint 完成了成对换先手对局测试。

Stage 4 self-play 训练按要求没有启动。下一步需要人工确认起始 checkpoint；本报告不自行选择 self-play 的起点、learning rate 或对手比例。

## 2. Stage 1 固定混合对手训练

训练对手严格按 episode 抽样：RandomPolicy 20%、ScorePolicy 40%、DevelopmentPolicy 40%。每局的 opponent 类型、参数和 RNG 整局固定，奖励仍只有正式终局 `+1/-1/0`，`alpha=0`、`gamma=1`、`lambda=0.95`，其余 PPO 参数保持 5A 配置。

实验从 5A `step_000100005.pt` 开始。按用户确认的节奏，checkpoint 约每 50k 保存，后半段约每 100k 评测，并在 500k 结束时评测。所有评测使用固定 200 paired seeds/400 局、交换先后手，评测 seed 不进入训练。固定开发综合指标为：

`0.2 * Random score + 0.4 * ScorePolicy score + 0.4 * DevelopmentPolicy score`

该指标只用于 checkpoint selection，不进入游戏 reward。

### DEV 面板

| step | Random W/D/L | Random score | Score W/D/L | Score score | Development W/D/L | Development score | composite |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,005 | 388/1/11 | 0.9713 | 73/0/327 | 0.1825 | 48/1/351 | 0.1213 | 0.3158 |
| 157,351 | 388/5/7 | 0.9763 | 161/1/238 | 0.4038 | 131/4/265 | 0.3325 | 0.4898 |
| 206,542 | 371/5/24 | 0.9338 | 150/3/247 | 0.3788 | 131/13/256 | 0.3438 | 0.4758 |
| 304,858 | 384/1/15 | 0.9613 | 154/1/245 | 0.3863 | 110/7/283 | 0.2838 | 0.4603 |
| **403,183** | **397/2/1** | **0.9950** | **166/2/232** | **0.4175** | **154/1/245** | **0.3863** | **0.5205** |
| 500,005 | 386/2/12 | 0.9675 | 153/3/244 | 0.3863 | 136/6/258 | 0.3475 | 0.4870 |

最佳 checkpoint 上，PPO 执先/执后 score rate 分别为：Random `0.990/1.000`，Score `0.4175/0.4175`，Development `0.450/0.3225`。三个面板的平均完整玩家回合数分别为 `62.49/59.85/61.96`。

起点到最佳点，ScorePolicy 从 `0.1825` 升至 `0.4175`，DevelopmentPolicy 从 `0.1213` 升至 `0.3863`；最终点略有回落，但仍明显高于起点。Random 全部检查点保持在 `0.9338..0.9950`。

### 训练运行与行为

- 8,418 局完整训练对局，learner W/D/L = `3612/11/4795`（只作采样 sanity check）；
- 对手局数 Random/Score/Development = `1706/3345/3367`，实际比例约 `20.27%/39.74%/40.00%`；
- MAIN_ACTION 为 259,109 次：TakeDistinct 91,891（35.46%）、TakePair 5,802（2.24%）、ReserveMarket 27,293（10.53%）、ReserveDeck 4,680（1.81%）、BuyMarket 121,688（46.96%）、BuyReserved 7,755（2.99%）；
- PAYMENT/RETURN_TOKENS/CHOOSE_NOBLE 决策数为 `129443/11181/267`，取得贵族 3,438 块，forced skip 37 次；
- 63 次 PPO update；最后一次 update 的 entropy `1.3778`、approx KL `0.00247`、clip fraction `0.03375`、value loss `0.09964`、explained variance `0.8963`，无 KL early stop；
- training engineering truncation = 0；全部 DEV 面板的 IllegalAction、invariant failure、exception 和 engineering truncation 均为 0；无 NaN/Inf。

完成运行的最终 checkpoint 为 `runs/5b_fixed_mix_seed_002_resume_206542/checkpoints/step_000500005.pt`，自动选中的最佳 checkpoint 为同目录下的 `step_000403183.pt`。启用降频评测后的接受运行从 `step_000206542.pt` 精确恢复，此后未改动配置、checkpoint 或输出目录。

## 3. Stage 2 吞吐结果

详细设计、语义测试和原始基准见 `docs/work_package_5b_stage2_zh-CN.md`。在 RTX 4050 Laptop GPU 上，每个配置采集 8,192 decisions：

| collector | decisions/s | games/s | mean inference batch | speedup |
|---|---:|---:|---:|---:|
| serial | 186.34 | 3.87 | 1.00 | 1.00× |
| batched-1 | 186.06 | 3.86 | 1.00 | 1.00× |
| batched-8 | 431.98 | 8.86 | 7.92 | 2.32× |
| batched-16 | 480.94 | 9.57 | 15.84 | 2.58× |
| batched-32 | 513.69 | 9.97 | 31.69 | 2.76× |

32 环境是本次短基准中的最快配置；16 到 32 仍有收益，但瓶颈已主要在环境/编码。本阶段没有引入 AMP、`torch.compile`、自定义 CUDA 或分布式框架。

## 4. Stage 3 opponent pool

详细契约和验证见 `docs/work_package_5b_stage3_zh-CN.md`。当前能力包括：

- 参数独立、`eval()`、`requires_grad=False` 的 latest 快照；
- 严格校验 checkpoint format/rules/schema/model config 的 historical 模型；
- ScorePolicy、DevelopmentPolicy 和 RandomPolicy；
- 容量上限 12、add/list/sample/load；
- manifest 原子保存/恢复，pool RNG 恢复后后续抽样序列一致；
- 神经对手只通过 `DecisionRequest` 决策，整局版本固定，对手 transition 不写入 learner trajectory。

真实 `step_000100005.pt` 与 `step_000157351.pt` 完成 2 对/4 局换先手 infrastructure smoke：4 局全部自然结束，IllegalAction、invariant failure 和 engineering truncation 均为 0。这不是棋力实验。

## 5. 测试和产物

- Stage 2 针对性测试：`5 passed`；
- Stage 3 pool/checkpoint/training-environment 回归：`37 passed, 1 skipped`；
- 最终全量回归：`186 passed, 2 skipped`；
- Stage 2 原始基准：`runs/5b_infrastructure/batched_benchmark_step_000500005.json`；
- Stage 3 真实 checkpoint smoke：`runs/5b_infrastructure/pool_validation/100005_vs_157351.json`。

## 6. 对 5B 关键问题的回答

1. **PPO 在 ScorePolicy / DevelopmentPolicy 上是否继续学习？** 是。最佳点相对起点分别从 `0.1825/0.1213` 提升到 `0.4175/0.3863`。
2. **是否遗忘 Random 基本能力？** 没有，所有检查点的 Random score rate 均不低于 `0.9338`。
3. **batched sampler 提高多少？** 32 环境从 `186.34` 提升到 `513.69 decisions/s`，为 `2.76×`。
4. **frozen latest neural opponent 是否正确工作？** 是，快照与 learner 参数独立且不参与梯度。
5. **historical pool 是否能保存、抽样、恢复并完整对局？** 是，单元、集成和真实 checkpoint smoke 均通过。
6. **小规模 self-play 是否稳定？** 未验证；根据本次明确限制，Stage 4 未启动。
7. **当前最佳 checkpoint 的综合表现？** `step_000403183.pt`，Random/Score/Development score rate 为 `0.9950/0.4175/0.3863`，composite `0.5205`。

下一步是由人工根据上述 DEV 面板确认 self-play 起点和配置，然后才能进入 Stage 4。
