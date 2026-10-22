# 2026-10-21 checkpoint去重与T7外置归档

用户明确授权后已完成：**161个字节相同副本去重，1885个冷模型外置**。本次移除本地模型原分配空间 **12.8474GB（11.9650GiB）**；加此前编译缓存及96个MLP恢复点清理，共约16.28GB。当前attention训练、8万局NPZ、所有账本及本机保留模型未迁移。

外盘包：`/media/lc/T7/splendor/archive-20261021.tar.zst`，**10,795,534,732字节（10.7955GB）**。旁边保存同名`.sha256`文件；压缩包SHA256为：

`7f61d4ebaca016342d0872fdcea2e8e60f11d2fb68171c49873f5bfcaf4c7016`

## 包内内容与保存方式

- 809个旧普通恢复模型＋1076个历史评测／EMA等模型，合计1885个；来源是上一次冻结的checkpoint审计清单，本次没有顺带归档后续新增模型。
- 包中模型均为真实文件，未用指向仓库的链接替代数据；另含9个清单／说明文件，共1894个普通文件。GNU tar POSIX pax保留模型纳秒mtime，zstd双线程级别3，直接流式写T7，不在本机额外生成压缩包。
- 161个副本经SHA核对后改为指向指定原件的相对链接。去重保持模型字节；别名的inode／跟随链接后的mtime归属原件，原别名mtime记录在manifest。
- 同盘rename集中到`archive/20261021/`，原模型路径保留相对链接；外盘完整流验证与压缩包SHA复查通过后，才删除本地1885个归档实体。本地只保留轻量元数据，不另存同一份压缩包。
- 20261012／20261019旧包与原982个归档链接未改动。1885个新冷模型路径需解压本包后才能读取；161个去重别名中部分指向这些冷模型，也同样需要解压。

## 复核结果

完整外盘流逐文件SHA、模型size及纳秒mtime全部通过；确认1885份模型均为真实文件。保留模型84份的SHA在释放本地实体前核验，MLP的9个指定节点、R20／R43及37423锚点仍在本机。当前attention整个run排除在所有迁移／去重目标之外，归档后原PID12682仍在运行。
额外核对8万局原NPZ的size/mtime及输入manifest SHA全部一致；核验时attention已提交86000步，现有checkpoint均为原位真实文件。详见[verification.json](verification.json)、[外盘校验](external_verification.json)和[完成记录](completion.json)。

## 恢复与查看

在仓库根目录恢复冷模型，使既有兼容链接可读：

```bash
tar --zstd -xf '/media/lc/T7/splendor/archive-20261021.tar.zst' -C '/home/lc/下载/splendor'
cd archive/20261021
sha256sum -c SHA256SUMS
```

解压的是archive子目录，不覆盖正在运行的attention目录。归档内manifest是打包时的冻结状态，外置完成状态以本专题manifest、completion及外盘校验记录为准。

如需恢复某个旧训练，先按manifest将该run的模型恢复为原位真实可写文件；不要通过归档链接写入冻结模型／账本。若该run的其他冷节点或账本在20261019包，还需恢复相应旧包。去重别名如需恢复独立原文件，复制指定原件并按记录还原mtime，不移动其他模型所依赖的原件。

**20261019当时保留的24个关键模型中，23个已进入本次20261021包；37423起点仍留本机。** 两批包互补，不能只解压旧包就假定当时的全部保留模型仍是本机实体。

EMA第12–19轮160个间隔50步节点＋8个轮起点，以及9个EMA导出模型均完整保存在本包；未删除来源轨迹。原规则／训练版本以对应实验的冻结来源记录为准，manifest里的git_head只描述本次维护操作版本。

## 逐文件清单

原路径、包内路径、SHA、size、mtime、原inode及类别见[manifest.json](manifest.json)；实际去重／移动日志见[moves.jsonl](moves.jsonl)。包内全部1894文件的预期值见[package_inventory.json](package_inventory.json)。压缩及恢复前验证状态见[export_state.json](export_state.json)。操作程序见[run.py](run.py)，按stage／export／release分阶段执行，拒绝覆盖已有批次或T7文件。

没有改模型结构、训练参数、数据、标签、实验结论或忽略规则；没有启动额外训练／棋力评测，没有提交或推送。
