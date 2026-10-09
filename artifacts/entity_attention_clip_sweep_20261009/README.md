# policy 组 clip 阈值回放探针（2026-10-09）

在 16500 步冻结起点上逐条重放分头训练第 20 新轮（round-019）已记录的抽样账本、模型与 AdamW/RNG 状态，只改变 **policy 组**（共享 trunk + policy head）的 `max_grad_norm`；critic head 的裁剪固定 1.0，学习率取 2.5e-5 或 1e-4。每个节点先用 100 步账本，再用独立冻结的 50 步扩展账本延伸到 125/150 步。入口是 [probe.py](probe.py)，复用了 [formal_lr_step 探针](../entity_attention_formal_lr_step_20261007/experiment.py)的账本、恢复与诊断函数；没有修改任何其他文件。

## 口径

- 起点 `runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1/rounds/round-019/checkpoints/final.pt`，16500 步，SHA-256 `dbfbe066…4ed2c4`。
- 主账本为 round-019 已提交的 `usage/{policy_recent,value_recent,value_history}/*.bin`（digest `b57b2011…634dc`，100 步）；扩展账本 seed 17030000、50 步，沿用原探针冻结的文件。
- policy clip ∈ {0.5, 1.0, 2.0, 4.0, 关闭（1e9）} × lr ∈ {2.5e-5, 1e-4}，共 10 条臂；value clip 固定 1.0。
- 诊断：stored-target recent/current KL、8 次搜索经验均值 bank KL、与起点的 policy 变化 KL/top-1、policy/critic 参数位移（L2、RMS、max）；没有棋力面板。

对照复现：`clip1` 两条臂与既有 lr×步数探针在 recent KL、bank KL、变化 KL 上逐位一致（`abs_diff = 0.0`），扩展节点 125/150 也与原结果一致。

## 结果

`recent KL` 与 `bank KL` 越低越好；`L2@150` 是从起点到 150 步检查点的 policy 组参数位移。起点（16500）为 recent 0.42109、bank 0.26987。

| 臂 | recent@100 | recent@150 | bank@100 | bank@150 | L2@150 | 起点→150 policy KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lr1e-4 clip0.5 | 0.41621 | 0.40896 | 0.26786 | 0.27077 | 0.260 | 0.02029 |
| lr1e-4 clip1 | 0.41977 | 0.41696 | 0.26565 | 0.27758 | 0.386 | 0.02400 |
| lr1e-4 clip2 | 0.42608 | 0.42097 | 0.29508 | 0.28620 | 0.586 | 0.02284 |
| lr1e-4 clip4 | 0.42608 | 0.42515 | 0.29463 | 0.29384 | 0.668 | 0.03103 |
| lr1e-4 关闭 | 0.42582 | 0.42765 | 0.29609 | 0.29569 | 0.684 | 0.03261 |
| lr2.5e-5 clip0.5 | 0.40692 | 0.40540 | 0.26442 | 0.26546 | 0.104 | 0.01546 |
| lr2.5e-5 clip1 | 0.40882 | 0.40525 | 0.26598 | 0.26683 | 0.147 | 0.01519 |
| lr2.5e-5 clip2 | 0.41247 | 0.40675 | 0.26652 | 0.26693 | 0.203 | 0.01794 |
| lr2.5e-5 clip4 | 0.41256 | 0.40735 | 0.26501 | 0.26728 | 0.212 | 0.01804 |
| lr2.5e-5 关闭 | 0.41253 | 0.40735 | 0.26502 | 0.26728 | 0.212 | 0.01804 |

每步裁剪尺度（`min(1, clip / 原范数)`，100 步均值）：lr1e-4 为 0.227/0.406/0.786/0.965/1.000，lr2.5e-5 为 0.266/0.506/0.888/0.998/1.000。`clip4` 与关闭在两条 lr 上几乎相同，说明原始范数极少超过 4。

## 结论

- **clip1 不是抑制学习的限制。** 在 lr1e-4 下放宽到 2/4/关闭，参数位移从 0.386 增到 0.586–0.684，但 recent KL 与 bank KL 都变差（150 步 bank 0.2776 → 0.2862/0.2938/0.2957，且高于起点 0.2699）。该账本内更大的位移没有带来更好的KL，不能把这一现象直接归因为方向错误。
- **方向反而是“有效步长还要更小”。** lr1e-4 收紧到 clip0.5 后，150 步 recent 0.40896、bank 0.27077，均好于 clip1；参数位移约小 33%。但 100 步时 clip0.5 的 bank KL 0.26786 略差于 clip1 的 0.26565，单账本内不是全面占优。
- **lr2.5e-5 对阈值不敏感。** 各 clip 的 recent KL 在 0.4054–0.4074、bank 在 0.2655–0.2673，位移随裁剪尺度近似线性（0.104→0.212）；在本账本和节点上，较低lr的拟合对阈值较不敏感，尚不能外推为所有数据下步长足够小。
- 结合原 lr 探针：固定账本上“更小有效步长 + 更低 lr”是最稳的方向；“clip1 每步都触发”是阈值落在正常范数分布中间的观测，不等于更新被压坏，更不是瓶颈所在。

## 范围与复核

- 单起点、单主账本加单扩展账本、单训练种子；KL/位移不是棋力，未运行 50-pair 或更大强度面板。
- bank KL 的参照是 16500 模型 8 次搜索经验均值目标，不是新搜索标签；value clip 固定 1.0，且 value 范数基本不触发。
- `clip0.5` 的相对优势目前只是同账本内的诊断信号，若考虑改正式配置，应先换独立账本/种子复现，再评估棋力面板。

```bash
# 主扫描（10 臂 × 100 步）与扩展（+50 步）
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u artifacts/entity_attention_clip_sweep_20261009/probe.py run
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u artifacts/entity_attention_clip_sweep_20261009/probe.py extend
# 只重建汇总（不加载模型）
conda run -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_clip_sweep_20261009/probe.py summarize
# CPU 两步 smoke
conda run -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_clip_sweep_20261009/probe.py smoke
```

`arm_*.json`/`ext_*.json` 与 [summary.json](summary.json) 保存在本目录；逐步骤 metrics、节点 checkpoint、raw 诊断放在忽略的 `runs/entity_attention_clip_sweep_20261009/` 与 `runtime/entity_attention_clip_sweep_20261009/`。
