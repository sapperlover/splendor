# 2026-10-19 离线训练冷产物

后续位置更新（2026-10-21）：本批当时留在本机的24个关键模型中，23个后来进入[20261021归档](../../artifacts/repository_archive_20261021/README.md)，只有37423起点仍留本机；下文“本机保留”是当时记录。需要旧run的全部模型和账本时，按各批manifest恢复20261019与20261021两个包，不能只解压旧包就假定关键模型仍为本机实体。本批原始清单和包未改动。

本目录集中保存7个离线run的冷产物：**361个真实文件、7,551,001,104字节（7.0324GiB）**，其中334个中间checkpoint、27个抽样账本／逐步指标文件。使用同一文件系统rename，原`runs/`路径保留相对符号链接。没有压缩或删除文件，同盘移动尚未释放空间。

| 来源run | checkpoint | 账本／指标文件 | 原文件GiB |
| --- | ---: | ---: | ---: |
| `entity_attention_256x4_aux_r24_r43_20261014` | 53 | 8 | 2.721 |
| `entity_attention_128x2_relations_no_aux_r24_r43_20261019` | 87 | 3 | 1.282 |
| `entity_attention_r24_history_pool_20261012` | 51 | 2 | 0.748 |
| `entity_attention_r43_history_pool_20261012` | 51 | 2 | 0.748 |
| `entity_attention_r24_r43_aux_ablation_20261019` | 46 | 6 | 0.757 |
| `entity_attention_r24_r43_policy_surprise_offline_20261013` | 23 | 3 | 0.386 |
| `entity_attention_r24_r43_value_auxiliary_offline_20261013` | 23 | 3 | 0.391 |

## 目录与用途

```text
20261019/
  README.md
  manifest.json                 # 原路径、包内路径、SHA、原inode/size/mtime、保留模型与训练池
  verification.json             # 实际迁移与保留项核验
  SHA256SUMS                    # 相对本目录的361个真实文件校验值
  metadata/experiment_context.json  # 配置、结果文字和来源身份快照
  runs/<原run>/...              # 真实checkpoint／账本／逐步指标，保持来源结构
  runtime/maintenance_before/   # 本次改动前的文档、旧索引与工作树保全
```

checkpoint保留完整权重、优化器及随机状态，账本用于精确抽样／训练复核。它们仍有历史证据价值；外置仅改变保存位置。这是从原run拆出的冷产物包，不包含训练池、关键模型或全部历史源码快照；完整复核需结合主仓库的对应版本和保留产物。metadata中的维护时工作树信息不能替代实验执行时的源码身份。

## 本机保留模型

- 256×4双辅助：initial、30619（value较优）、34021（已测中期）、37423（已完成点中KL＋MSE较优）、44227（最后提交）。
- 新版128×2：initial、选中的37423、final。
- R24／R43历史池：各initial、选中24388、final、best_validation_kl。
- 普通／仅回合消融，以及旧PSW／双辅助：各分支initial、final。

共24个模型。验证曲线、完整留出、选点、配置和两份共72000局原始数据仍在原处；逐文件保留身份在manifest中。原实验结果没有重新测量。

## 校验、外置与恢复

从本目录执行只读校验：

```bash
sha256sum -c SHA256SUMS
```

该命令只校验清单中的361个冷产物，不检查主仓库保留模型／训练池，也不把维护快照计为新的实验原始结果。后者的迁移时核验见verification；说明、配置与结果快照的SHA另见metadata。

外置时复制／移动整个本目录的真实内容，保留来源结构和文件时间。目录内的`runs/`是普通目录，源run处才是兼容链接；不要仅复制那些链接。外置目标完成校验后，再按需要处理本地真实副本。

恢复旧路径访问：把本目录放回仓库`archive/20261019/`，或在该路径建立指向外置20261019目录的链接。如此原run中的相对链接即可恢复；外置介质未连接时这些冷文件不可读。只读复核前先校验对应批次。如需恢复训练，先按下述方法撤回该run的迁移，恢复原位真实文件后再执行原命令，避免截断／追加账本时改写冻结归档。

完全撤回迁移时，对manifest的每个entry依次核对source仍为指向destination的本次链接，删除该单个链接，将对应真实文件rename回source；不要递归删除run或归档目录。跨文件系统恢复改用保留时间精度的复制并核对SHA、size、mtime_ns，原inode无法跨文件系统保留。冻结训练清单不应为迁就搬移而改写。

20261012旧归档已由用户另行外置，不在本包中，本次没有重新校验旧外部内容。

本批重型内容已外置至 `/media/lc/T7/splendor/archive-20261019.tar.zst`。恢复旧路径访问前，从仓库根目录执行：

```bash
tar --zstd -xf "/media/lc/T7/splendor/archive-20261019.tar.zst" -C "/home/lc/下载/splendor"
```
