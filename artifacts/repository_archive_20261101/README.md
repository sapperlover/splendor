# 2026-11-01：普通模型、已淘汰训练局与评测明细T7归档

状态：**流式归档、全包校验、本机释放和输入复核均完成**。外盘包11755850563字节（10.9485GiB）；原件分配空间14.6318GiB，恢复链接占0.8947GiB，净释放13.7371GiB。释放阶段观测可用空间增加13.6236GiB，差额包含本机清单／日志等开销。磁盘可用空间由约4GB升至18GB。

按用户确认的2026-11-01盘点清单外置**234543个原件**。未扩大到编译缓存、旧共享104k局池、最近160局网站记录或其他文件；未启动训练、采集或棋力评测，未修改忽略规则、提交或推送。

| 范围 | 文件数 | 压缩前分配空间GiB |
| --- | ---: | ---: |
| 104k policy＋得分普通checkpoint | 192 | 1.8516 |
| 104k无辅助普通checkpoint | 210 | 1.3890 |
| R37／104k／lr1e-4普通checkpoint | 105 | 0.6945 |
| R37／104k／lr2e-5较早普通checkpoint | 227 | 1.5015 |
| 旧R31→R37在线普通checkpoint | 324 | 2.1427 |
| 新policy在线1500、2500步checkpoint | 26 | 0.2309 |
| 三组已结束离线逐步training_metrics.jsonl | 3 | 1.1717 |
| 新policy在线第1–7轮已淘汰训练局 | 63038 | 3.5924 |
| 评测panels／逐局sidecar／完整留出原始明细 | 170323 | 1.4269 |
| runtime工程验证／smoke模型 | 95 | 0.6306 |

所有普通checkpoint范围均排除了initial/final、完整留出百分比节点、原选中最优点、七个固定诊断点及当前恢复点。新policy在线26个模型仅是13个已完成轮各自的1500／2500步周期点；固定诊断七节点和final保持本机真实文件。

63038个NPZ仅来自新policy在线`round-000..006`的train split，已退出最新60k活动池；所有既有固定留出、初始64k、活动60k和`round-013`未完成采集均保留。原R21–R37共享104k池未外置，原路径、字节、split与mtime保持。

R37／104k／lr2e-5训练仍是**中断于228000步的incomplete状态**；本次归档完成不代表训练完成。`step-000228000.pt`、`step-000140221.pt`及全部完整留出节点、主`training_metrics.jsonl`、progress、配置、AdamW/RNG恢复信息继续保留；其227个较早普通模型外置。归档的三份主日志属于policy＋得分、无辅助和lr1e-4三组已结束训练。

## 外盘位置与校验

- 包：`/media/lc/T7/splendor/archive-20261101.tar.zst`。
- 包SHA256：`c16e8d2b032ff59ae52cbd677618af6f14995211ea7c79dfbd1ebfcd7ed045fe`，同目录有`.sha256`文件。
- GNU tar POSIX PAX保留纳秒mtime，`zstd -T2 -3`直接流式写T7，本机不暂存完整包。
- 全量解压流逐一验证234553个普通文件（234543个原件＋10个恢复元数据）的类型、路径、大小、SHA256与纳秒mtime。外盘包SHA在释放前、释放后再次核对。
- 本机原件再次核对SHA／inode／大小／mtime后，原子替换为相对恢复链接；234543条链接均逐一验证。
- 350个保留模型、177530个保留NPZ及105543个配置／源码／索引／报告／近期网站证据文件通过释放后身份复核。新在线实际冻结启动入口的`--check`前后一致，旧R31入口`--check`也通过。

当前网站固定模型仍为新policy在线`round-008/checkpoints/final.pt`，SHA `852cb0420e94d77c146bc2330600f48d52f638e4cca96a5c1578e9dfe54940ec`。所有保留模型与游戏数据保持原身份；本次文档修订另记于`documentation_changes.json`。

## 恢复与后续使用

挂载T7本身不会解开压缩包，冷文件原路径的链接在解包前不可读。恢复全部本批原件：

```bash
tar --zstd -xf '/media/lc/T7/splendor/archive-20261101.tar.zst' -C '/home/lc/下载/splendor'
cd /home/lc/下载/splendor/archive/20261101
sha256sum -c SHA256SUMS
```

部分恢复时，以`files.jsonl.gz`的`archive_member`作为tar成员名；目录成员需使用`--wildcards 'archive/20261101/artifacts/<实验>/panels/*'`覆盖逐局sidecar。本包恢复到仓库ext4文件系统，保留纳秒mtime。

要继续写入归档过的日志或模型，先把解包后的真实文件复制回原位并恢复mtime，例如`cp -p --remove-destination archive/20261101/<原相对路径> <原相对路径>`；避免通过链接修改冻结归档。当前新在线和低学习率228000步恢复入口的必要输入未外置，两个在线只读输入检查已验证；旧历史节点的复算／复训按实际涉及范围恢复。

评测的紧凑summary、results和配置继续在原专题阅读；完整pairs数组、逐局面明细及`.json.games/`在本包中。五模型10k及140221对R37两组10k的原始面板也已外置；调用依赖原始面板的`status`、重建汇总或评测恢复前，先恢复相应JSON及完整sidecar目录，不能把原始文件暂不可读解释为实验未完成。

## 记录与边界

- [manifest.json](manifest.json)、[scope_summary.json](scope_summary.json)：包级身份与逐类数量／空间。
- [external_verification.json](external_verification.json)、[completion.json](completion.json)、[final_verification.json](final_verification.json)：全流校验、实际释放和收尾复核。
- [online_input_check_before.json](online_input_check_before.json)、[online_input_check_after.json](online_input_check_after.json)、[old_online_input_check_after.json](old_online_input_check_after.json)：实际启动入口只读检查。
- [stream_restore_smoke.json](stream_restore_smoke.json)：三个合成UTF8路径的打包、纳秒mtime及链接恢复验证。
- [run.py](run.py)、[execution.json](execution.json)：本批执行实现、授权范围和命令。批次已完成，不要重跑或覆盖已有外盘包。
- 本地`files.jsonl.gz`、`retained_models.jsonl.gz`、`protected_games.jsonl.gz`、`guards.jsonl.gz`及原批准清单随包保存。包内`manifest.json`是打包前冻结计划；本目录manifest和completion记录最终状态。`package_identity.json`的manifest SHA对应包内／`archive/20261101/manifest.json`的冻结版本。
- `archive/20261101/SHA256SUMS`、`moves.jsonl`是本地展开恢复账本；逐文件清单和原批准`approved_candidates.json.gz`均不作为Git交付物。后者未被现有通配忽略规则覆盖，应保持本地；本次按仓库约定未调整`.gitignore`。

当前空间为操作后快照。原先20261025等批次中“旧在线324个普通节点及全部新局留本机”仅对应当时；本次上述模型与63038个淘汰训练局的位置以本批清单为准。历史实验原始记录和结论未重写。
