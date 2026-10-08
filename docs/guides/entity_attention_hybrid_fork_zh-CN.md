# 从第九轮创建 S512/N2 与 S1024/N2 训练分支

2026-10-05。入口 `python -m splendor.hybrid_train --fork-from <hybrid-run> --fork-round <轮次>` 从已完成的父run边界创建新run。第九轮起点为9500步；模型、AdamW moments/步数、训练采样及Torch随机状态接续，不重新离线拟合。父run的配置、数据、checkpoint和进程不修改。

教师选择依据是[有限S/N筛选](../../artifacts/entity_attention_search_screen_20261005/README.md)，新计划的长期训练收益尚未验证。

2026-10-06更新：用户决定回到原第九轮、S1024/N1，只缩小采集/更新周期；最新准备方案为[200局/100步、在线窗口最多3000局/最近15个采集教师](entity_attention_hybrid_s1024_n1_zh-CN.md)。下面的N2配置和替换表保留为此前分支的实际设置，不再代表最新推荐。本次检查S512/N2至少已完成33新轮/12800步；已完成的新轮10/20/30面板得分为68.0%/63.5%/62.5%，各100局，不能单独归因于N或新局数。

本机 S512/N2 分支曾发生退出故障：2026-10-05 第18个新轮次、11300步已完整保存，第19轮开始采集时发生系统 RAM OOM。内核明确记录杀死主进程 PID 435906；旧孤儿进程、训练缓存及其他 Python 进程共同占用内存。本次已清理16个无主进程并修复 owner 死亡后的 worker 退出，见[诊断与清理记录](../../artifacts/entity_attention_worker_cleanup_20261005/README.md)。冻结配置仍为离线/在线各8 GiB数组缓存，该预算不包含 metadata、Python/PyTorch、编译进程等开销；退出修复本身不降低正常训练内存，不能据此保证长期不再 OOM。

## 两组配置

| 参数 | S512/N2 | S1024/N2 |
| --- | --- | --- |
| 配置 | `configs/entity_attention_v1/hybrid_fork_s512_n2.json` | `configs/entity_attention_v1/hybrid_fork_s1024_n2.json` |
| 总模拟 / 世界数 | 512 / 2，每世界256 | 1024 / 2，每世界512 |
| 每轮采集 / 更新 | 100局 / 100步 | 相同 |
| 在线replay | 最近2000个完整对局，含原split验证局 | 相同 |
| learner | batch256、AdamW lr=1e-4、weight_decay=1e-4、policy/value=1/1、clip1 | 相同 |
| 采集执行 | native_cuda_pool，1 owner、8 search workers、active100、batch上限128、microbatch64 | 相同 |
| Graph / 槽 | 固定地址native-cudagraph，B[16,32,64,128]、C[16,32,48,64]、shared_slots=2、owner pipeline_slots=1 | 相同 |
| 推理 / 探索 | FP32、actor_projection=false、root noise fraction=.25、前10回合温度1之后0 | 相同 |
| 留出 / 缓存 | 新对局按整局10%验证；原离线/在线LRU各8 GiB，合计16 GiB按需使用 | 相同 |
| 棋力评测 | 每10个新轮次，50 paired seeds / 100局，双方统一S1024/N1，对固定theta_5000 | 相同 |

除run_id和采集S外，两份配置相同。每个分支先继承父run第8～9轮最近2000局N1数据；NPZ通过原路径引用，模型与小型冻结元数据复制到新run。源runs数据目录仍须保留。每轮采用最新模型，不设置胜率接纳、回滚或早停。

## 数据逐步替换

加入每轮100局新N2数据时，按原采集轮号、game_id顺序移出最旧100局，原始文件不删除；train/validation原split不变，不按胜负筛选。在线batch在池内所有train positions上均匀有放回采样，不是各局等权，也不强制每batch的新旧局数。

| 分支新轮次 | 累计新局数 | 在线池继承N1 / 新N2完整局数 | 每batch原离线 / 在线positions |
| --- | --- | --- | --- |
| 1 | 100 | 1900 / 100 | 64 / 192 |
| 5 | 500 | 1500 / 500 | 64 / 192 |
| 10 | 1000 | 1000 / 1000 | 64 / 192 |
| 11 | 1100 | 900 / 1100 | 32 / 224 |
| 15 | 1500 | 500 / 1500 | 32 / 224 |
| 20 | 2000 | 0 / 2000 | 32 / 224 |
| 21～60 | 2100～6000 | 0 / 最近2000局N2 | 32 / 224 |
| 61及以后 | 6100及以后 | 0 / 最近2000局N2 | 0 / 256 |

**第20个新轮次后在线池全部为N2；第61个新轮次起训练batch才全部来自N2在线池。** 原离线十轮数据与在线池分开，其配额只是归零，不重复5000步离线阶段。日程保留旧计划剩余1000局的64配额、再5000局的32配额，换算成100局小轮次，避免轮次缩小后十倍加速离线退出。

N2在online batch中的期望比例为 `新分支train positions / 在线总train positions`，在整个batch中还乘以 `在线配额 / 256`。由于局长和整局留出，5%新局数不是恰好5%训练样本。`replay_snapshot.json.replacement` 保存实际池组成和位置比例；每步 `training_metrics.jsonl.source_positions` 区分 `offline / inherited_online / branch_online`，`source_usage.json.online_replacement.actual_draws` 汇总实际抽样次数。它们只是记录，不会拒绝更新。

## 启动与恢复

从仓库根目录，在各自未使用的新输出目录启动。两条命令显式选择父run第9轮，即 `rounds/round-008/checkpoints/final.pt`；父run后来继续训练也不会改变选择。

S512/N2：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_fork_s512_n2.json \
  --fork-from runs/entity_attention_hybrid_static_graph_seed1 --fork-round 9 \
  --output runs/entity_attention_hybrid_s512_n2_from_round9_seed1 \
  --forever
```

S1024/N2：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_fork_s1024_n2.json \
  --fork-from runs/entity_attention_hybrid_static_graph_seed1 --fork-round 9 \
  --output runs/entity_attention_hybrid_s1024_n2_from_round9_seed1 \
  --forever
```

本机单GPU，比较采集成本时分别运行，避免争抢GPU影响结果。先做有限对照可将 `--forever` 换成 `--online-rounds 20`：每组完成2000新局/2000更新，从9500到11500步，在新轮10、20评测。

以后恢复，移除fork参数，改为resume：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_fork_s512_n2.json \
  --output runs/entity_attention_hybrid_s512_n2_from_round9_seed1 \
  --resume --forever
```

另一组使用对应配置和输出路径。`--online-rounds`是分支自创建后完成轮数的绝对目标；完成20轮后再次指定20不会追加20轮。新run的算法配置仍冻结；2026-10-06起仅允许恢复时调整采集`max_batch_size`和`batch_buckets`，已有采集轮保持该轮冻结尺寸，详见[S1024/N1 Graph调整说明](entity_attention_hybrid_s1024_n1_zh-CN.md)。初始化要求learner配置一致，防止悄悄改变优化器、学习率或损失权重。

`--check-config` 加相同fork参数可只读检查父模型、离线数据、继承游戏和日程。`--online-rounds 0` 加fork参数可以只初始化新目录，不采集、不更新；之后使用resume。省略fork-round默认选择父run最新已完成轮，正式对照建议显式固定9。

## 编号、恢复与验证范围

分支日志及目录从新轮1 / round-000开始；第一轮NPZ的数据轮号为9，learner轮号为10，累计训练步数从9500开始。继承数据轮号7、8与新数据9及以后不冲突，训练采样RNG接续；采集/划分种子按接续数据轮号派生，两组种子日程相同。分支 `hybrid_state.json.completed_rounds` 记录自己的新轮次。

`fork_manifest.json` 保存父模型、父配置/快照、继承NPZ哈希、数据轮偏移和来源；`checkpoints/fork_initial.pt` 保留父checkpoint全部内容，`checkpoints/theta_5000.pt` 保留共同对手。原离线固定验证和父run首轮在线固定验证复制到新目录，退出训练窗口后仍可诊断。模型、优化器和随机状态按既有checkpoint机制恢复，已提交更新和完整采集不会重复做。

入口交付时的有限验证见[分支入口交付](../../artifacts/entity_attention_hybrid_fork_20261005/README.md)：相关CPU回归、真实9500步checkpoint的两组只初始化调用链。CPU回归覆盖实际N2采集/训练、优化器接续、逐参数/状态/RNG一致的中断恢复、数据键隔离、N1完全退出、父run不变、原留出集保留、父run部分淘汰数据不重新进入。当时未启动正式分支长训；后来用户启动的 S512/N2 长训状态见本文开头及退出清理记录，S1024/N2 未发现正式运行产物。
