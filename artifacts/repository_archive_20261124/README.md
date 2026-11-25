# 2026-11-24：四类冷产物T7归档

状态：**流式压缩、全包逐文件校验、本机释放与最新λ在线输入检查均已完成**。仅处理用户授权的四类，共93655个原件。本机原路径保留相对恢复链接；没有清理缓存、旧轮内诊断点、原始评测或网站记录。

| 授权范围 | 文件数 | 压缩前分配空间GiB |
| --- | ---: | ---: |
| MC普通checkpoint | 34 | 0.2249 |
| λ普通checkpoint | 40 | 0.2646 |
| 已完成λ主日志及旧policy在线13轮主日志 | 14 | 0.4664 |
| 三条旧在线分支回放快照 | 30 | 0.6567 |
| 旧R21–R37共享104k池的Train整局 | 93537 | 5.3683 |

旧在线三条来源为 `entity_attention_r20_s256n4_noble12_iteration_20261022`、`entity_attention_r31_10k_p60k_iteration_20261023`、`entity_attention_policy140221_r31_r37_p60k_iteration_20261025`。只取其已提交轮次中属于各自目录的30份 `replay_snapshot.json`；父分支继承的更早快照没有扩入本批。

## 外盘身份与释放结果

- 包：`/media/lc/T7/splendor/archive-20261124.tar.zst`。
- 大小：5283377829字节（4.9205GiB）。旁有同名 `.sha256` 文件。
- SHA256：`ba900bf1562e024aca3a06bb2a418ab6c753132055a328c7a11f78f54f5bd2e0`。
- 原件分配空间6.9809GiB；恢复链接0.3573GiB；扣除链接净释放**6.6236GiB**。释放阶段观测可用空间增加6.5792GiB，包含并发系统写入和目录/账本开销差异。
- GNU tar POSIX PAX保留纳秒mtime，`zstd -T2 -3`直接流式写入T7；本机没有暂存完整包。
- 全流逐一校验普通文件的路径、大小、SHA256及纳秒mtime，校验通过后才将原件原子替换为相对恢复链接。释放前后再次核对包SHA及全部链接。

## 保留与恢复边界

最新 `lambda161924_fpu_p60k_iteration_20261124` 的历史60k、首轮新局、标签、状态和模型保持本机。历史60k来自旧policy分支第8–13轮，共53980个Train和6020个validation，与本批旧104k Train没有交集。全部原有固定留出保持；旧104k中的10463个validation约0.6002GiB未外置。

433个保留模型、94001个保留NPZ和113970个源码/配置/索引/证据文件在释放前后通过身份复核。最新λ在线实际冻结入口 `--check` 前后均通过，结果一致。MC70842、λ161924、原R37、网站固定模型及完整验证节点、initial/final、恢复点均保留。

MC仍为172044步的incomplete状态，旧104k低学习率实验仍中断于228000步；本次归档不代表训练完成。两者的主 `training_metrics.jsonl`、progress和恢复点保持真实文件。MC使用近期60k；旧104k实验若要续训，应先恢复本批93537个Train。旧policy初始64k也引用其中部分旧训练局，旧在线分支检查/恢复及快照复核需先恢复所需文件；最新λ在线分支没有这个依赖。

## 恢复

完整恢复本批原件到链接目标：

```bash
tar --zstd -xf '/media/lc/T7/splendor/archive-20261124.tar.zst' -C '/home/lc/下载/splendor'
cd '/home/lc/下载/splendor/archive/20261124'
sha256sum -c SHA256SUMS
```

恢复后旧路径的相对链接即可读取。只恢复旧104k训练输入：

```bash
tar --zstd -xf '/media/lc/T7/splendor/archive-20261124.tar.zst' -C '/home/lc/下载/splendor' --wildcards \
  'archive/20261124/runs/entity_attention_r20_s256n4_noble12_iteration_20261022/rounds/*/selfplay/games/*.npz' \
  'archive/20261124/runs/entity_attention_r31_10k_p60k_iteration_20261023/rounds/*/selfplay/games/*.npz'
```

包内这两个来源仅含本次选中的Train，不会覆盖仍在原位的validation。其他部分以 `files.jsonl.gz` 的 `archive_member` 为tar成员名。需要继续写入的日志或模型，应使用 `cp -p --remove-destination <恢复后的归档目标> <原路径>` 恢复为真实文件并保留mtime，避免经链接修改冻结归档。

## 记录与校验

[manifest.json](manifest.json)、[scope_summary.json](scope_summary.json)保存范围、数量、字节数及清单SHA；[external_verification.json](external_verification.json)、[completion.json](completion.json)、[final_verification.json](final_verification.json)保存校验与实际释放结果；[归档前](new_online_check_before.json)与[归档后](new_online_check_after.json)记录最新λ入口预检。[stream_restore_smoke.json](stream_restore_smoke.json)记录中文路径、纳秒mtime与相对链接恢复测试。

[run.py](run.py)为本批执行入口，沿用11月1日的流式打包与校验机制；批次已完成，不要重跑或覆盖外盘包。`files.jsonl.gz`、`retained_models.jsonl.gz`、`protected_games.jsonl.gz`、`guards.jsonl.gz`、`package_inventory.jsonl.gz`为本地大型明细；前四份亦随包保存。`archive/20261124/SHA256SUMS`及`moves.jsonl`为恢复账本，不进入Git。包内manifest是打包前冻结计划，最终状态以本专题completion为准。

[documentation_changes.json](documentation_changes.json)记录本次相关文档更新前后SHA；历史实验原件和结论未重写，既有未提交工作保留。
