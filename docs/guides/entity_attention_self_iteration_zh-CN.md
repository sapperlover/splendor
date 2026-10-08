# 新模型自迭代：10×1000

2026-10-04，按本次用户要求沿用旧10×1000训练策略，使用新模型采集、新模型训练。配置交付时正式运行未启动；用户随后已运行，2026-10-05的loop_state记录轮0–2完成、下一轮采集曾中断。有限验证见 [原始结果与摘要](../../artifacts/entity_attention_10x1000_seed1/README.md)。

| 项目 | 设置 |
| --- | --- |
| 配置 / 输出 | `configs/entity_attention_v1/online_10x1000.json` / `runs/entity_attention_10x1000_seed1` |
| 起点 | `runs/entity_attention_offline/checkpoints/final.pt`，entity_attention_v1 |
| 规模 | 10轮，每轮1000个完整新对局 |
| 采集调度 | native_cuda_pool，8个搜索进程，共1000活跃局，每进程125局；GPU batch上限500、micro64、slot1、wait5ms |
| 搜索与探索 | S1024/N1、c_puct1.5、fold forced、单模拟最多512 transitions；采集根噪声0.25/alpha0.3，前10个双方合计回合温度1、之后0 |
| 推理 | CUDA FP32、compiled候选档位16/32/48/64、reused native transport、actor_projection=false、fail_fast |
| learner | batch256/history128，AdamW lr=1e-4、wd=1e-4，policy/value权重1/1、grad clip1 |
| 每轮步数 | `ceil(4 × 当轮新train局面数 / 256)`，不设置正式步数上限；首轮全current，之后current/history各128 |
| replay / 留出 | 最近6000局、4 GiB LRU，整局按10%留出；新建在线池，不导入离线旧模型数据 |
| 状态与记录 | 首轮只热启动权重，新建optimizer且累计步数从0开始；后续轮延续权重、optimizer和采样状态；每100步及轮末保存checkpoint，保留现有诊断/资源/引用记录 |
| 每轮搜索评测 | `mcts_vs_initial`：新一代与固定起点双方S1024/N1、根噪声0、温度0，50 pairs/100局；native_batched active100/B64 |
| 每轮裸网评测 | `greedy_vs_initial`：新一代与固定起点双方greedy，50 pairs/100局 |

1000是同时进入调度的棋局数，500是一次GPU前向的局面数上限，二者独立。首次采集将全部1000局交给8个搜索进程，没有“前500局全部结束后才安排后500局”的屏障。具体推理请求仍按microbatch导出。

两组评测始终使用本次离线final作为对手，不随父代改变；每组交换先后手，跨轮复用固定DEV种子，并各做10000次paired bootstrap。评测是趋势观察，训练后的合法模型直接进入下一轮采集，没有按胜率拒绝更新的门槛。首轮采集用起点，第二轮起使用上一轮训练后的新模型；这与固定评测对手的选择相互独立。

从仓库根目录首次启动：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.alphazero_loop \
  --config configs/entity_attention_v1/online_10x1000.json \
  --output runs/entity_attention_10x1000_seed1 --progress-format human
```

中断恢复使用同一配置和输出目录，加 `--resume`：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.alphazero_loop \
  --config configs/entity_attention_v1/online_10x1000.json \
  --output runs/entity_attention_10x1000_seed1 --progress-format human --resume
```

每轮结果位于 `rounds/round-XXX/evaluation/mcts_vs_initial.json` 和 `greedy_vs_initial.json`；模型位于 `rounds/round-XXX/checkpoints/final.pt`。终端、loop_state、TensorBoard与统计导出使用一致的面板名称。源码仍支持历史 `mcts_vs_parent` 配置；搜索评测的父代/起点两种面板必须选择其一。

已通过11个循环相关CPU用例、2个调度/历史配置检查，以及真实RTX4050 GPU两轮短测（每轮4局、2步、S16/N1、每种评测1 pair）。调度短窗口实际确认8×125局同时进入调度；GPU短测确认第二轮采集/训练继承上一代、optimizer累计4步，两轮评测均使用固定起点，四个面板均完整通过。短测不证明棋力或1000活跃局的提速倍率。

`online_smoke.json` 是本次有限验证配置；`online_3x200.json` 保留此前建议，当前正式任务以 `online_10x1000.json` 为准，不修改已有运行的冻结配置。

2026-10-05新增[固定形状、kernel启动与独立双缓冲实测](../reports/entity_attention_execution_speed_zh-CN.md)。现有run的冻结配置继续使用原执行设置。新采集预设`collection_static_graph.json`采用shared_slots=2、固定地址CUDA Graph和owner单缓冲；双缓冲额外收益未稳定，未作为推荐默认。这些采集执行设置不改变learner或轮间评测；不得把新配置直接用于原run的`--resume`。
