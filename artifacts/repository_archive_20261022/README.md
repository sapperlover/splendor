# 2026-10-22：第21–30轮及Cross64离线冷产物归档

状态：**归档、完整外盘校验及本地释放均已完成**，见[completion.json](completion.json)。外盘包为`/media/lc/T7/splendor/archive-20261022.tar.zst`，2447304521字节（2.4473GB），不覆盖20261012／20261019／20261021旧包。旁边有同名`.sha256`；压缩包SHA256为`1e61748d5030717f3fbb2ae236c3a7230c48df265c490fe80b2beedcf193c7e9`。

用户随后明确要求“21–31轮的采集对局都留下”，最终清单**不包含任何NPZ或selfplay采集文件**。最初含淘汰训练局的草案只做过清点，未压缩、未移动或删除原件；撤回记录见[plan_revision.json](plan_revision.json)。

## 范围与保留

- 新在线分支`entity_attention_r20_s256n4_noble12_iteration_20261022`第21–30轮，对应物理`round-020`至`round-029`：归档中间模型及较大的逐步训练／抽样账本，10个`final.pt`全部原位保留。第31轮及第32轮未完成采集不作迁移。
- Cross64／角色std0.05离线run：保留`initial.pt`及13528、27056、40584、54112、67640、81168步全部6个留出节点，包含value最低、KL＋MSE最低及最后恢复点；归档81个普通恢复模型及较大的训练／抽样账本。固定与全量留出指标、dataset、输入清单、配置和源码快照留本机。
- **所有第21–31轮44000个采集NPZ全部保留本机**，包括已淘汰的训练局及所有留出；第32轮2142个已完成NPZ、原80000局NPZ亦原位保留。当前活动池仍是第26–31轮24000局，不改变池选择或标签。
- 当前R31最终模型、R20／R43锚点、旧attention离线模型及其他实验不迁移；没有删除编译缓存，没有启动训练或棋力评测。

最终计划涉及291个checkpoint和55个账本文件：其中291个独立模型、43个独立账本为外盘真实原件；另外12份字节相同的逐步账本按SHA去重为相对别名（在线两头账本10份，离线两头账本2份）。12个重复文件均有明确canonical路径；不以“看起来没用”为由丢弃唯一原始数据。逐路径SHA／大小／mtime／inode及类别见[manifest.json](manifest.json)。

## 校验与空间

以GNU tar POSIX pax＋zstd双线程级别3直接流式写T7，不在本机暂存同一压缩包。外盘解压流必须逐文件核验SHA、大小及纳秒mtime，且包内真实原件不能是链接；另保存压缩包SHA及外盘同名`.sha256`。完整校验和压缩包SHA复查通过后，才将本地冷文件原路径原子替换为相对链接。操作期间持有在线及离线入口使用的锁。

所有80000个原始NPZ、44000个新分支NPZ及2142个未完成轮次的已采NPZ均逐文件计算SHA／元数据的有序身份摘要，释放前后全部一致；最新在线state、活动池快照和最新模型亦核验通过。外盘337个普通文件（334个真实冷原件＋3份元数据）全部通过SHA／大小／纳秒mtime校验。移除原分配空间4326412288字节，兼容链接占1368064字节，**净释放4325044224字节（4.3250GB／4.0280GiB）**；系统可用空间同期增加4324388864字节，差异含清单等元数据占用。17个范围内保留模型SHA一致。

外盘逐文件校验见[external_verification.json](external_verification.json)，压缩过程见[export_state.json](export_state.json)，包内清单见[package_inventory.json](package_inventory.json)。执行程序[run.py](run.py)，分`plan`／`export`／`release`三阶段，拒绝覆盖既有外盘包。

## 恢复

从仓库根目录恢复冷原件，使原路径链接重新可读：

```bash
tar --zstd -xf '/media/lc/T7/splendor/archive-20261022.tar.zst' -C '/home/lc/下载/splendor'
cd archive/20261022
sha256sum -c SHA256SUMS
```

原路径兼容链接指向`archive/20261022/runs/...`；未解压时冷文件不可读，关键模型和采集NPZ不受影响。在线继续运行所需的最新模型、活动池及原始输入均保留，不依赖本包。

如需复算旧轮逐步训练、抽样轨迹或恢复该离线训练，先解压本包；会继续写入的账本／模型应按manifest复制回原路径的真实文件，并按`mtime_ns`恢复时间，避免通过兼容链接修改冻结归档。去重别名也可从其`identical_to`路径复制为独立文件，再按原mtime恢复。包内manifest为打包前冻结计划，外置完成状态以本专题manifest和completion为准；SHA256SUMS核验的是唯一真实原件，别名身份见manifest。

文档修订前后SHA及受影响索引的旧记录见[documentation_changes.json](documentation_changes.json)。实验原始测量、配置、种子、来源身份和历史结论未改写；没有改忽略规则、提交或推送。
