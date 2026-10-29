# 2026-10-25：四组离线普通模型／逐步日志与原80k局池归档

状态：**流式归档、全包校验和本机释放均完成**。外盘包9487466722字节（8.8359GiB）；净释放11870924800字节（11.0557GiB）。完整结果见[completion.json](completion.json)及[最终核验](final_verification.json)。

用户本次只授权四组已完成辅助目标离线实验的大部分普通checkpoint、各自逐步`training_metrics.jsonl`，以及原80000局历史池。共80676个原件：672个普通模型、4份逐步主日志、80000个采集NPZ；原分配空间12201373696字节，原始数据12033997922字节。不含旧R31→R37在线324个普通模型，也不含父分支42142个可选冷对局。

| 离线实验 | 普通checkpoint | 同批日志 |
| --- | ---: | ---: |
| 104k仅短期value | 192 | 1 |
| 104k仅对手policy | 192 | 1 |
| 104k双辅助 | 192 | 1 |
| 44k仅短期value | 96 | 1 |

上述四组的全部完整留出百分比节点、预先选中的最优模型、initial/final保留，共54个模型；正式对战原始结果、完整留出指标、选择结果、配置、源码及数据清单保留。当前policy140221起点、raw R37锚点和在线所有恢复节点保留。全仓其余632个真实模型、6148个元数据／源码／结果文件及164462个其他真实采集NPZ以完整SHA和身份指纹保护。

当前在线依赖的初始R31–R37全部64000局、最新分支58320个已落盘对局均留本机；它们与所选原80k池没有重叠。后续启动仍检查完整初始64k，不能只按当前活动池淘汰状态搬走这些文件。

外盘目标为`/media/lc/T7/splendor/archive-20261025.tar.zst`，旁有同名`.sha256`。沿用既有归档格式：GNU tar POSIX PAX保留纳秒mtime，zstd两线程级别3，直接流式写T7；不在本机暂存完整包。压缩后完整解压流逐文件核对类型、路径、大小、SHA256及纳秒mtime，再次核验源文件和保护集后，才原子替换本机原件为相对链接。归档与释放期间持有受影响实验和当前在线入口的锁。

本机只保存恢复元数据；链接指向`archive/20261025/runs/...`。挂载T7不会自动解开包，因此已外置冷文件暂不可通过链接读取。恢复全部本批原件：

```bash
tar --zstd -xf '/media/lc/T7/splendor/archive-20261025.tar.zst' -C '/home/lc/下载/splendor'
cd /home/lc/下载/splendor/archive/20261025
sha256sum -c SHA256SUMS
```

原80k池的离线复训、全量复核以及旧80k在线分支续跑，须先恢复所需NPZ。四组离线的普通节点或逐步曲线分析须恢复对应checkpoint或日志。要继续写入的模型／日志，应按`files.jsonl.gz`复制回原位真实文件并恢复mtime，避免通过链接修改冻结归档；最新在线分支所需输入无需恢复本包。

- [manifest.json](manifest.json)：范围、原分配空间和保护集摘要。
- [files.jsonl.gz](files.jsonl.gz)：每个原件的路径、SHA、大小、纳秒mtime、inode和包内路径。
- [retained_models.jsonl.gz](retained_models.jsonl.gz)、[guards.jsonl.gz](guards.jsonl.gz)：保留模型及恢复／结果依赖的原始身份。
- [package_inventory.jsonl.gz](package_inventory.jsonl.gz)：包内全部普通文件的预期身份。
- [scope_summary.json](scope_summary.json)：逐实验、逐类别占用与数量。
- [stream_restore_smoke.json](stream_restore_smoke.json)：三个合成文件的PAX/zstd、纳秒mtime、原子链接替换及GNU tar恢复验证；实际原件另做全量校验。
- [run.py](run.py)：`plan`冻结，`finish`流式导出、全量校验及释放；拒绝覆盖已有外盘包。此批完成后不要重复执行。

`archive/20261025/SHA256SUMS`和`moves.jsonl`为本地展开恢复清单，逐文件长期记录使用本目录压缩JSONL；本次不调整Git忽略规则，不提交或推送。

外盘包SHA256：`8cdcad685d3a2730db02012dab065696b480247052f1e7fc2055ebada0f08afb`。80680个包内普通文件均通过SHA／大小／纳秒mtime验证；80676个原路径链接逐一核对。54个离线关键模型再次核验，旧在线324个普通模型保留；当前在线启动输入检查前后相同。

[外盘校验](external_verification.json) · [在线输入检查](online_input_check_after.json) · [文档修订记录](documentation_changes.json)
