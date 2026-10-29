# 2026-10-23：离线中间模型、冷账本与旧局池T7归档

状态：**流式归档、全量外盘校验及本机释放均完成**。外盘包5810422036字节（5.8104GB），原路径相对链接77272个，净释放13.9211GB（12.9650GiB）。完成记录见[completion.json](completion.json)。

本批仅包含用户明确选择的五类，共77272个原始文件，实际数据14077945331字节，本机分配空间14237585408字节（14.2376GB）。不搬移当前在线324个普通checkpoint，不清理缓存，不改忽略规则，不提交或推送。

| 范围 | 文件数 | 原分配空间GB |
| --- | ---: | ---: |
| 80k attention离线普通checkpoint | 135 | 0.9587 |
| 44k attention离线普通checkpoint | 83 | 0.5893 |
| 低学习率续训普通checkpoint | 5 | 0.0355 |
| 已结束轮次／实验的抽样账本与逐步训练日志 | 1011 | 9.0375 |
| 旧R43主线及P4、hybrid等旧NPZ | 76038 | 3.6167 |

保留原8万局、父分支第21–31轮全部44000局及未完成轮2142局、当前分支60000局，共186142个采集NPZ。当前R22–R31历史输入与六轮固定留出仍可直接读取。所有未选中的512个真实checkpoint、229个配置／state／dataset／当前源码及索引文件均以SHA保护；包括当前全部372个模型、三组离线全量留出节点／选中模型／最新恢复点、原R20／R31／R43锚点及既定MLP／Cross64保留模型。历史报告、正式结果和配置源码不迁移。

外盘目标为`/media/lc/T7/splendor/archive-20261023.tar.zst`，旁边保存同名`.sha256`；不覆盖20261012／19／21／22包。以GNU tar POSIX pax保留纳秒mtime，zstd两线程级别3，直接流式写T7，不在本机暂存完整压缩包。压缩后完整解压流逐文件检查普通文件类型、大小、SHA256、纳秒mtime；再次核对本机原件和保护集后，才将原路径原子替换为相对链接。整个阶段持有受影响入口的目录锁和现有运行锁。

本批不去重，两个头的账本均保留原字节与身份。77272条逐路径记录使用gzip JSONL压缩保存，避免大型展开JSON进入仓库；不改变归档的原路径布局或数据字节：

- [manifest.json](manifest.json)：范围、状态、保护集摘要和压缩记录身份。
- [files.jsonl.gz](files.jsonl.gz)：每个原件的路径、类别、大小、分配空间、纳秒mtime、inode、SHA和包内路径。
- [retained_models.jsonl.gz](retained_models.jsonl.gz)、[guards.jsonl.gz](guards.jsonl.gz)：本机保留模型及恢复依赖身份。
- [package_inventory.jsonl.gz](package_inventory.jsonl.gz)：包内完整普通文件清单，包含恢复元数据。
- [stream_restore_smoke.json](stream_restore_smoke.json)：三文件短验证，覆盖runs/runtime包内路径、纳秒时间、外置链接和实际解包恢复；真实保护集另做全量校验。
- [run.py](run.py)：`plan`／`export`／`release`三阶段，拒绝覆盖已有外盘包。

本机`archive/20261023/`只保留恢复元数据；原路径链接指向`archive/20261023/runs/...`或`archive/20261023/runtime/...`。未解包时冷原件不可读，T7挂载本身不会自动展开归档。恢复方式从仓库根目录执行：

```bash
tar --zstd -xf '/media/lc/T7/splendor/archive-20261023.tar.zst' -C '/home/lc/下载/splendor'
cd archive/20261023
sha256sum -c SHA256SUMS
```

旧局池相关的离线复训／完整诊断、旧逐步轨迹复核需要先恢复本包。继续写入的模型或账本应按`files.jsonl.gz`复制回原位真实文件，并恢复`mtime_ns`，避免通过兼容链接修改冻结归档。当前在线继续运行所需输入及原80k离线原局池留在本机；本次不恢复训练或启动棋力评测。

包SHA256：`4b9c795c627794b911e1e77e21afc55cb24433e44d8671b3260318aa6236b532`。外盘77276个普通文件均通过SHA、大小与纳秒mtime校验；其中77272个原件和4个恢复元数据。保护集186142个采集NPZ、512个保留模型及229个配置／state／索引／源码身份释放前后相同。

[外盘校验](external_verification.json) · [导出记录](export_state.json) · [在线恢复输入检查](online_input_check.json)
