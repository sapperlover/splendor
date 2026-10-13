# R43 起点：S4096/N1，policy 2k、value 6k

用户于 2026-10-13 指定：value 保持最近 6000 局，policy 缩至最近 2000 局，其他训练参数不变。当前为单分支连续自迭代实验；运行状态见 `session-*.json`、[逐轮汇总](summary.json)，完成的训练与评测见自动生成的 `results.md`。尚无本分支棋力结论。

起点固定原绝对 R43、72898 步，来自父 run 的本地第 34 轮 `round-033/checkpoints/final.pt`，SHA `b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14`。完整继承模型、AdamW、采样 RNG、6k 在线窗口、整局 Train/validation 划分和固定诊断引用。原 run 保持原处，新输出为 `runs/entity_attention_r43_s4096_p2k_v6k_20261013/`。

## 冻结设置

| 项目 | 本分支 |
| --- | --- |
| 新采集 | S4096/N1，每轮 1000 局；root noise 0.25、c_puct 1.5、温度日程及执行拓扑沿用原配置 |
| policy 池 | 最近 2000 个完整对局中的 Train 局面，按局面均匀有放回抽样 |
| value 池 | 最近 6000 个完整对局中的 Train 局面，按局面均匀有放回抽样 |
| 更新 | 每轮 1000 步；每头 B256；lr 2e-5；AdamW weight_decay 1e-4；CE:MSE=1:1；统一 global clip=1 |
| 梯度 | 两头均回传共享 encoder；一次总损失、统一裁剪、一次 AdamW 更新 |
| 诊断 | 0/100/250/500/1000 步；每代固定 2048 Train 局面、完整整局留出；继承旧代固定引用 |
| 对战 | 保留原 P4V4 固定对手、种子与预算：每轮 greedy 200 pair；绝对轮号逢 3 的 S1024/N1 200 pair |
| 本次会话 | 从启动起连续运行最多 8 小时；到时 SIGTERM，最多再留 120 秒退出；中途已提交对局/checkpoint/pair 可恢复 |

执行环境沿用原运行命令的 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=TORCHINDUCTOR_COMPILE_THREADS=1`。首次启动漏设编译线程限制，观察到临时编译进程造成额外内存占用，在尚无新对局提交时仅停止本分支、恢复这些环境变量并续跑；原定北京时间 2026-10-13 09:20 左右的截止时间不延长。首次会话记录和日志保留，原因见 `runtime/.../launch.json`。

配置见 [`hybrid_r43_s4096_p2k_v6k_20261013.json`](../../configs/entity_attention_v1/hybrid_r43_s4096_p2k_v6k_20261013.json)，驱动计划见 [`r43_s4096_p2k_v6k_20261013.json`](../../configs/entity_attention_v1/r43_s4096_p2k_v6k_20261013.json)。源 checkpoint 已记录有效 lr=2e-5，因此新配置的 `offline.learner.learning_rate` 写为 2e-5 以满足 fork 身份校验；它与原主线实际学习率相同。

池容量以完整局计数，包含约 10% 留出；留出不参与抽样。第一轮 R44 的 policy 窗口是旧 R43 的 1k 局 S1024＋新 R44 的 1k 局 S4096；从 R45 训练开始，policy 2k 全部来自 S4096。value 需累计 6 轮新采集才全部替换。局长相近时，最新一轮在 policy 的抽样占比约 1/2，原 6k 策略约 1/6；实际占比和曝光在每轮 `replay_snapshot.json`／`source_usage.json` 中记录。

复用 [R30 分池联合训练](../entity_attention_r30_head_pools_20261011/run.py) 的抽样方法：每头从同一个步首 RNG 状态抽样，只用 value 的 6k 抽样推进持久 RNG。这样缩 policy 池不改变 value 的抽样流。每头有效 batch 仍为 256；两个批次合计处理 512 行，训练日志 `batch_size=512`，`policy_batch_size=value_batch_size=256`。这是分池监督，仍联合更新。历史 policy 标签保留原采集教师，不做全池最新教师重标。

## 旧实验依据与判读

[第 30 轮 B=policy 3k/value 6k](../entity_attention_r30_head_pools_20261011/conclusions.md) 的最新代 Train KL 为 0.416350，原 6k/6k 为 0.422533；最新代留出 KL 0.415640 对 0.415682，几乎持平。Greedy 83.25% 对 81.75%，差值 95% 区间 [-3,+6]pp，未确认棋力改善。2k 尚未做过同口径长期复核。因此本次是提高近期曝光的尝试，不能把此前结果描述为已验证小池更强。

本分支同时改了采集预算和 policy 池，结果只能判断这组组合是否推进，不能单独归因于其中一项。优先检查最新代 Train/留出 KL、value 留出 MSE，再看完成的同预算对战；诊断仍覆盖整个 6k 窗口，policy 的旧代指标需结合实际训练窗口解释。N1 仍只抽一个隐藏世界，S4096 不增加世界数，本分支未额外加入跨世界平均标签。

## 运行与恢复

仓库根目录执行；同一命令自动恢复缺失对局、最近 checkpoint 与已提交评测 pair。再次执行会开始新的 8 小时会话，不能与当前会话并行启动。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u \
  artifacts/entity_attention_r43_s4096_p2k_v6k_20261013/run.py --max-seconds 28800

conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.pool_iteration status \
  --config configs/entity_attention_v1/r43_s4096_p2k_v6k_20261013.json
```

会话时间限制与 driver 的持续运行计划分开记录。`summary.json` 的 `in_progress` 表示没有整个实验的固定轮数终点；是否仍有进程运行，应结合 `session-*.json` 和 `runtime/entity_attention_r43_s4096_p2k_v6k_20261013/launch.json` 判断。运行日志位于该 runtime 目录。校验与测试见 `verification.json`、`tests.xml`、`gpu_training_smoke.json`，源码差异见 `implementation.patch`；正式采集自身的 manifest 和 `runs/.../source/` 保存配置指纹、版本与 dirty 源码身份。

## R47 终点恢复 policy 6k 的续训命令（已准备，未启动）

本次 8 小时会话已完成 R44–R47，共 4000 局／4000 步，于北京时间 2026-10-13 09:20 左右到时停止。用户随后指定把这 4k 局作为历史，policy 恢复 6k，前面补旧 S1024 数据，再继续自迭代。新配置从原 R47 的 76898 步终点 fork（父分支本地第 4 轮），完整保留模型、AdamW、采样 RNG、留出划分和诊断引用；仅把 `policy_replay_max_games` 从 2000 改为 6000，其余采集、训练及评测参数沿用。

起始历史窗口 R42–R47 为 2k S1024＋4k S4096，policy/value 都使用完整 6k 中的 Train 局面。仍先采集再训练：R48 采集 1k 后淘汰最旧 R42 的 1k，首个训练窗口为 1k S1024＋5k S4096；R49 再完全替换为 S4096。历史标签保留原采集教师，不重标。新配置与父 checkpoint／6k 数据来源均已通过只读预检，两个头的 Train 池一致；没有启动本次续训。

新输出为 `runs/entity_attention_r47_s4096_p6k_v6k_20261013/`，不会覆盖本分支。仓库根目录执行以下命令持续运行；同一命令支持中断后恢复，本命令没有 8 小时自动停止限制。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u -m splendor.pool_iteration run \
  --config configs/entity_attention_v1/r47_s4096_p6k_v6k_20261013.json
```
