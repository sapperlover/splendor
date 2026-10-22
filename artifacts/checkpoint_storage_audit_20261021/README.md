# Checkpoint占用梳理与已授权清理（2026-10-21）

后续已按用户明确授权执行：161个相同副本完成去重，809个旧恢复点＋1076个历史／EMA模型共1885个已打包至`/media/lc/T7/splendor/archive-20261021.tar.zst`，完整SHA与mtime校验后释放本地实体，本阶段再释放12.8474GB。[完成状态与恢复方法](../repository_archive_20261021/README.md)。下文分级是操作前冻结快照，候选清单不再代表本地真实文件；原审计／首次清理记录保留不改。

本次按用户明确授权，仅删除5组已结束实验的compile_cache/triton_cache（10个目录），以及停止MLP运行中96个普通恢复checkpoint。其余模型、数据、账本、原始实验结果和正在运行的attention均未删除或迁移。

已移除原分配空间 **3.4335GB（3.1977GiB）**；其中缓存2.4418GB、MLP恢复点0.9917GB。当前训练继续写文件，因此前后整个仓库du差值不等于本次删除量。逐文件大小／时间／SHA及目录清单见[cleanup.json](cleanup.json)，删除脚本见[cleanup_authorized.py](cleanup_authorized.py)；该脚本完成后拒绝重复执行。

MLP保留9个模型：initial、13528、27056、40584、54112、67640、81168、94695、97000。七个留出节点和最新可恢复97000的SHA与原停止记录相符，删除后再次核验全部9个保留文件未变。删除非关键恢复点不改变其原已测指标或棋力结果。

## 其余checkpoint分级

清单时间：2026-10-21T22:37:15.811674+08:00。检查runs/runtime/play_checkpoints中的真实.pt/.pth，不加载模型、不运行训练或评测；共2130个、13.4344GB。该时间点attention仍在运行，所有现有及后续本轮模型均应保留；其checkpoint数量会继续增长。

| 建议 | 数量 | 分配空间GB | 清单 |
| --- | ---: | ---: | --- |
| 保留本机 | 84 | 0.5870 | [keep_local.jsonl](keep_local.jsonl) |
| 可去重副本 | 161 | 1.0289 | [deduplicate_candidates.jsonl](deduplicate_candidates.jsonl) |
| 旧普通恢复节点 | 809 | 5.2207 | [ordinary_step_candidates.jsonl](ordinary_step_candidates.jsonl) |
| 建议外置保留 | 1076 | 6.5979 | [externalize_candidates.jsonl](externalize_candidates.jsonl) |

- **可去重副本**：161个，约1.0289GB；已逐字节SHA确认与另一个本地文件完全相同。每项列出`identical_to`。建议保留指定原件，将副本原路径替换为相对链接后释放实体副本，避免动态路径失效；本次未执行。只计可去重项，其他仍被直接引用的同内容别名没有一并算作可删。
- **旧普通恢复节点**：809个，约5.2207GB；不属于活跃attention、保留MLP、显式锚点／选中模型或EMA来源，未发现字面量配置／报告消费者。删掉不会改变当前attention的输入，但会丢失旧轮内模型轨迹、旧中断点和临时复核能力。脚本可动态拼接路径，所以“未发现引用”不是“永远无用途”；默认建议先外置，确定不需要这些能力后才删除。
- **建议外置保留**：历史被引用模型、历史边界／导出、EMA来源与少量工程模型，共6.5979GB。外置时保持来源结构、核对SHA和mtime，并在原路径保留兼容链接；若要恢复旧训练，应先恢复原位真实可写文件。只有移到另一块盘并处理本地实体副本才释放本盘空间，本次没有新建归档或迁移。
- **保留本机**：当前attention整个run、已明确保留的9个MLP模型，以及15个现有配置／选中模型或R20/R43/37423锚点。详细理由及引用者都写入清单；仅有模型留在本机不代表可以删除其NPZ池、配置、账本或冻结源码。

## 空间最多的来源

以下均是checkpoint空间，不含NPZ／JSON／账本；其中“普通恢复点”与上方809项是同一批，不能再次相加。

| 来源run | checkpoint总GB | 旧普通恢复点GB | 说明 |
| --- | ---: | ---: | --- |
| `entity_attention_37423_random_budget_p6k_v24k_20261019` | 3.1245 | 1.5337 | 8万局源分支；保留R20，其余分级外置，NPZ原局不动 |
| `entity_attention_r9_online_lr2e5_1000_forever_20261011` | 1.3360 | 0.6421 | 保留清单内锚点，其余按逐文件分级处理 |
| `entity_attention_p4_pool_iteration_20261010` | 1.3045 | 1.0903 | 保留清单内锚点，其余按逐文件分级处理 |
| `entity_attention_hybrid_s1024_n1_from_round9_seed1` | 0.9867 | 0.0000 | 保留清单内锚点，其余按逐文件分级处理 |
| `entity_attention_hybrid_static_graph_seed1` | 0.7550 | 0.4997 | 保留清单内锚点，其余按逐文件分级处理 |
| `entity_attention_r43_s4096_p2k_v6k_20261013` | 0.5843 | 0.4934 | 保留清单内锚点，其余按逐文件分级处理 |

**EMA来源单独保护**：第12–19轮160个间隔50步模型，加8个轮起点，共168个、1.1932GB。它们来自原EMA冻结plan，不能因为17–19轮结果已完成就丢弃；可以带清单外置。现有9个EMA导出只是最终推理权重，不能替代来源模型、AdamW或训练轨迹。

另发现982个既有checkpoint链接，当前介质未挂载／目标不可读；本次不改这些链接、不补造它们的SHA、不把外部实体算作本地可释放空间。详见[existing_links.json](existing_links.json)。

## 复核依据

完整逐文件数据见[checkpoints.jsonl](checkpoints.jsonl)，汇总见[inventory_summary.json](inventory_summary.json)，相同SHA分组见[duplicate_groups.json](duplicate_groups.json)。清单记录路径、分配大小、mtime、inode、SHA、字面量消费者、分类理由和去重原件；引用扫描覆盖artifacts/configs/docs/scripts/src/README及run级config/selection/progress/state。分类不承诺发现所有动态拼接消费者。

脚本[audit.py](audit.py)仅扫描、哈希并生成本目录清单，正式训练仍由用户原进程执行；重新审计会更新本目录快照，不能把随训练增长的新模型误当作旧普通恢复点。没有额外删除、搬移、压缩、修改忽略规则、提交或推送。
