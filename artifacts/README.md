# 长期实验交付物

2026-10-27[近期实验完成汇总](../docs/reports/auxiliary_offline_results_20261027_zh-CN.md)：六组原离线训练及对战、五模型十组10000 pair复核均已完成；在线已提交13轮，第12轮搜索也已完整结束。下方工程发布“未启动”等表述按其日期解释。

2026-10-25已完成[四组离线普通模型／逐步日志和原80k局池归档](repository_archive_20261025/README.md)：672个普通checkpoint、4份主日志、80000个NPZ，包8.84GiB，净释放11.06GiB。最优／完整留出／initial/final模型和当前在线输入保留；原80k离线复训或完整复核须先恢复本包。下方较早“原80k留本机”是当时记录，当前位置以本批清单为准。

[2026-10-23短期value与对手policy工程验证](continuation_auxiliary_20261023/README.md)保存回归JUnit、3局标签覆盖、一次CPU更新及3个S8N1原生根，含代码与输入SHA；没有正式训练／棋力结论。

2026-10-23后续训练的[记录规则](../docs/guides/training_artifacts_zh-CN.md)：当前在线／近期离线默认不再产生policy/value两份`.bin`和两份`.steps.jsonl`，主指标仍逐步保存；R31在线500步恢复点＋七个固定诊断点。历史原件和上述归档保持，既有冻结实验不自动套用新源码；[交付](entity_attention_r31_10k_p60k_iteration_20261023/recording_checkpoint_20261023/README.md)。

2026-10-23已完成[五类冷产物T7迁移](repository_archive_20261023/README.md)：223个离线普通checkpoint、1011个抽样账本／逐步日志、76038个旧NPZ，流式包5.8104GB，完整SHA／纳秒mtime校验后净释放13.9211GB。原80k局、父分支46142局、当前60k局及所有未选模型保留；旧局池复训／完整诊断需先恢复归档。本次未启动训练或棋力评测。

本目录中的源码、冻结协议、配置、紧凑汇总和报告纳入Git；原始评测、诊断数组、逐行记录和大型数据清单在本地或外部介质保存，不随Git分发。2026-10-27已按[Git产物整理记录](repository_git_cleanup_20261027/README.md)停止跟踪一批历史原件，原路径与字节保持，清单记录SHA和摘要位置；这次整理没有将它们迁移至T7。阅读入口为[实验导航](index.md)，机器可读身份与SHA见[实验索引](experiment_index.json)，按 valid、historical_valid、invalid_or_superseded、incomplete、unverified 明确适用范围；有效不等于“已证明棋力提升”。

2026-10-22按授权完成[第21–30轮及Cross64离线冷产物T7归档](repository_archive_20261022/README.md)：291个模型＋43个唯一账本外置、12个相同账本去重，包`archive-20261022.tar.zst`约2.45GB，净释放4.3250GB。**第21–31轮全部44000个采集对局、第32轮已采对局、原80k局均保留本机**；final及离线关键节点不动，冷路径恢复方式见专题。

2026-10-21先按授权清理5组缓存和96个MLP普通恢复模型，释放3.4335GB；随后按新授权完成[161个副本去重及1885个模型T7外置](repository_archive_20261021/README.md)，再释放12.8474GB。包名`archive-20261021.tar.zst`，逐文件SHA和纳秒mtime已核验；当前训练、数据、账本和保留模型不动。[原占用审计](checkpoint_storage_audit_20261021/README.md)保留为操作前快照，具体当前位置依归档manifest，外部实体不计为本地可释放空间。

日志、失败现场、cache、分片运行状态和可恢复 `*.games/` sidecar 放在忽略的 `runtime/` 或相邻被忽略的 sidecar 目录。训练 NPZ、checkpoint、TensorBoard 放在忽略的 `runs/`。新报告引用的必要汇总不能只存于 runtime；重型 sidecar/模型不因此纳管。

失效或被替代的原始测量在本地保留，随仓库只保留失效原因、原/新位置、哈希与替代依据；历史有效实验也按下述边界保存原件和Git摘要，限定版本。未完成评测summary可在原恢复路径保留并明确incomplete，不能当成完成结果。这两类与有效基线分开索引。

既有 `work_package_*` 产物目录是历史实验身份，暂保留路径以兼容 manifest 和工具。docs 中的工作包原文已本地打包，不作为现行说明。归档位于 `runtime/document_archive/2026-10-01/`，被 Git 忽略；迁移环境需要另行备份。

[2026-10-01 保全清单](retention_manifest_2026-10-01.json) 记录新增原件副本的来源、字节数与 SHA-256。副本 JSON 内保留测量时绝对路径，不能据此认定复制后的目录缺失；外部清单定位长期副本。不用当前代码版本或硬件补填缺失的历史信息。

## 原始证据与目录维护

复算所需原始证据应保全，但保全不等于进入Git。诊断NPZ、逐局面／逐root结果、完整pairs数组、JSONL曲线、来源局清单、抽样引用和原始CSV默认保存在本地或归档；Git保留聚合指标、结论、输入身份和恢复位置。源码patch、小型测试fixture、精选统计表和最终图表可以纳管；`data/`内正式CSV输入不受原始CSV规则影响。`oom_kernel.txt`作为既有OOM事件的必要摘录继续保留；小型冻结验收结果可以保留，常规stdout、JUnit和资源采样无需随每次运行纳管。

新的临时日志、会话资源采样和生成中间件写`runtime/`；训练数据、checkpoint与逐步主指标写`runs/`。完成实验的紧凑summary、配对统计的聚合结论及身份写回对应实验目录，完整逐pair／逐root数组单独保存。summary／plan／manifest的文件名不构成纳管理由：混入大型明细时另存紧凑摘要，原件保持字节与SHA。目录名是实验身份，不按是否热门或文件数量合并；重复root bank可以分别保留身份记录，原件不必重复纳管。

外部数据登记至少包含相对路径、字节数、SHA-256和实际存储位置；已归档时再登记包名、包SHA、包内路径和恢复命令。本机保留但尚未归档的文件须明确标注，不能假定在T7。大留出manifest等运行输入停止跟踪后，新机器须先恢复原件，紧凑摘要不能作为运行输入。历史JSON的SHA依赖不因停止跟踪而重写。

新产物超过1MB默认审查其用途；持续增长的逐行明细无论大小都默认不入Git。维护源码、协议和小摘要，避免直接`git add artifacts/`把新生成的大JSON纳入。现有冻结入口仍可能在artifacts原路径写原始结果，按忽略规则本地保存；不为调整Git范围重写冻结源码或中断实验。

[本次整理清单](repository_cleanup_20261009/manifest.json)记录路径、原SHA、文档修订前后SHA与本地保全位置。历史JSON里的绝对路径／配置路径／源码hash代表当时身份，迁移后依清单定位，不用当前文件hash回填历史测量。`evidence_hashes`表示当前阅读／复核入口的内容，旧值在整理清单中保留。

## 冷产物位置登记（2026-10-19）

[本次集中归档](repository_archive_20261019/README.md)将7个离线run的334个中间checkpoint及27个账本／逐步指标移入`archive/20261019/`，原路径保留相对链接；24个关键模型及两份共72000局训练池仍在原位。重型文件仍受现有嵌套`runs/`忽略规则覆盖，未改`.gitignore`。

位置、外置状态与恢复入口见[归档登记](../archive/README.md)。20261012旧归档已由用户外置，当前未挂载，具体外部位置未提供；旧manifest和结论索引继续保留，不能把旧链接当作当前可读原始数据。外置新归档时携带真实目录和校验清单；Git中的记录不包含重型原件。

历史源码SHA与当前阅读证据区分：索引中`historical_source_hashes`保存原记录，不随当前源码演进刷新；`evidence_hashes`用于当前可读交付物。文档与派生状态修订的前后SHA及旧索引值见[本次文档修订清单](repository_archive_20261019/documentation_changes.json)。
