# B → C → D 连续自迭代趋势实验

2026-10-08，按最新要求精简。当前入口是 [iteration_suite.py](../../src/splendor/iteration_suite.py)，配置是 [iteration_trend_20261008.json](../../configs/entity_attention_v1/iteration_trend_20261008.json)。已完成B前两轮与首次100pair评测，得分51.25%（95%区间44.5%–58.0%），未确认净收益；[原始核验](../../artifacts/entity_attention_iteration_trend_seed1_20261008/first_evaluation_review.json)。C/D尚未运行。旧四组对照套件及其有限验证保留为历史记录，当前入口不再执行旧协议。

## 配置和顺序

从共同的 **16500 步冻结起点** fork，恢复同一模型、AdamW moments 和采样 RNG。父 run 为 `runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1`，fork 边界为第 20 新轮；起点 SHA-256 为 `dbfbe066144a485d15ec633aa87c85ac27864f142183725b86562e4ba44ed2c4`。

| 顺序 | 组 | 每轮新局 | 更新步数 | AdamW lr | Policy 数据 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | B：steps50 | 200 | 50 | 1e-4 | 最近 4 轮 |
| 2 | C：lr_low | 200 | 100 | 2.5e-5 | 最近 4 轮 |
| 3 | D：fresh50 | 200 | 50 | 1e-4 | 仅最新 1 轮 |

每组只保留 **seed1 一条轨迹、16 轮**，去掉 seed2 的第二次重复。**完整完成 B 的训练和起点评测后才开始 C，完成 C 后才开始 D**。三个组仍从同一冻结起点分别 fork，不继承上一组的训练结果。

三组共同使用 value 3000 局 FIFO、value batch 128 近 4 轮 + 128 更早局面、policy batch256、value detach、分头 clip1.0、FP32、旧 native S1024/N1 教师及原噪声/温度日程。D 只改变 policy 窗口；C 改变同一 AdamW 所有参数组的 lr，包含 critic。父 run、历史 NPZ 和旧实验记录保持原路径。

本轮不安排 A 训练或 A 相关比较，不安排 PPO、0 轮自测、组间直接对打或额外棋力确认阶段。

## 观察趋势

每条轨迹在 **第 2、4、8、12、16 轮** 对冻结起点测 **100 pair（200 局换座对局）**，各节点使用相同的初局、policy 和搜索种子。一个 pair 是同一初局的两次换座，不把两局当成两个独立样本。

每个节点后更新 `analysis.json`，保存 seed1 的得分与 95% CI，以及相邻节点的共同种子配对得分差与区间。区间在每条固定训练轨迹内重采样 paired seed，不当作训练随机种子总体的置信区间。没有实际 0 轮测量，不虚构 0 轮结果。

每轮沿用训练器现有的 `validation_before/after.json`、`training_metrics.jsonl`、`source_usage.json`，观察近期/当前留出 policy KL、value MSE 和样本暴露。调度脚本只读取现成诊断，不再扫描引用账本、重新训练验证模型或额外复核全部历史结果。

第 4 新轮时 P4 窗口全部来自新分支，第 15 新轮时原 3000 局 FIFO 全部替换。早期节点看初始变化，后期节点看持续改善、平台或退化。100 pair 仍可能看不清小差异；主要看这条轨迹的多节点走势；当前没有跨训练种子的重复性证据，不按单次最高分选择“最优”。本轮没有自动的显著优胜标记，也不据此单独认定 B/C 优于未重跑的 A。

## 速度和资源

总计 **48 个训练轮、9600 新局、3200 更新步骤、15 个评测面板 / 1500 pair / 3000 评测局**。依据既有 200 局采集约 422 秒、S1024 100 局评测约 320 秒的记录，采集约 5.6 小时、评测约 2.7 小时；加上训练、进程启动和诊断，粗估 **8–12 小时**，实际取决于功耗、温度和桌面负载。这不是新脚本全程的实测耗时。

用户实际首次100pair/200局评测阶段约18分钟，慢于上面的历史估算；8–12小时仅保留为准备时粗估，整套实际耗时尚未测出。

当前执行配置采用 8 search workers、活动局上限200、microbatch64、GPU batch≤200、Graph 桶 `[16,32,64,128,200]`、shared_slots2、owner 单缓冲。评测为 native_batched、64 活动局、batch64、FP32 eager。

**尚无“速度最优”的证据**。shared_slots2 的收益有特定开局短窗口证据；200 桶的历史分析主要是减少 padding 的算术推算。此前未在这个起点、完整 200 局和同一资源约束下完成执行参数对照，eager64 评测也未证明最快。当前设置沿用已有优化和正式采集的批次参数，不增加新的速度实验。

保留每轮独立子进程、4 GiB replay 缓存和 user systemd scope 的 **MemoryHigh14 GiB / MemoryMax16 GiB / SwapMax0**。系统可用 RAM 低于 3 GiB 时停止本次阶段；CUDA OOM 或其他实际错误也会明确报错。父进程中断后清理自己的 scope，避免后台 worker 残留。已移除磁盘容量门槛、预留空间判断及磁盘轮询。

## 启动与续跑

从仓库根目录执行；首次启动和中断续跑使用同一命令：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  python -u -m splendor.iteration_suite \
  --config configs/entity_attention_v1/iteration_trend_20261008.json
```

Ctrl+C 后等待本次进程退出，再执行原命令。采集从已保存单局继续，训练恢复已提交 checkpoint、Adam/RNG，评测从完整换座 pair 继续。若训练轮已提交但完成标记未写入，训练器会识别已完成轮，不再重训；若评测完成但分析尚未刷新，下次运行补分析。重复启动已完成套件直接退出。

启动只做配置、CUDA 和内存限额检查；可以加 `--prepare-only` 只检查环境，不进行训练或评测。首次保存配置、命令种子和入口代码身份；恢复时保持实验配置一致，不再对全仓库源码及所有已完成 checkpoint 逐项重做哈希审计。训练器自身的数据/断点恢复约束保留。

日志、完成标记和调度状态在 `runtime/entity_attention_iteration_trend_seed1_20261008/`；训练模型和 NPZ 在 `runs/entity_attention_iteration_trend_seed1_20261008/`；配置快照、环境记录、原始评测汇总、资源阶段记录和趋势汇总在 `artifacts/entity_attention_iteration_trend_seed1_20261008/`。重型模型、NPZ 和相邻 `*.games/` sidecar 仍按现行忽略规则保存。修改实验参数需换输出目录，避免混入已有轨迹。

精简版的必要验证记录见 [交付说明](../../artifacts/entity_attention_iteration_trend_seed1_20261008/README.md)。此前双种子版本只完成准备和短验证，原manifest及记录保留在 `artifacts/entity_attention_iteration_trend_20261008/`。旧 [四组方案](../../artifacts/entity_attention_iteration_suite_20261008/README.md)及 [历史 CUDA smoke](../../artifacts/entity_attention_iteration_suite_validation_v2_20261008/README.md)只作为各自版本的工程记录。
