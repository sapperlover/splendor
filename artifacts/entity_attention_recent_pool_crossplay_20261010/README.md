# 大池训练的分支组合搜索与 greedy 对战

2026-10-10。固定三组搜索与十组 greedy，各 **200 pair／400 局**，按下表顺序串行执行；总计 **2,600 pair／5,200 局**。状态和已完成结果以自动更新的 [results.md](results.md)、[summary.json](summary.json) 为准，冻结日程见 [plan.json](plan.json)。本轮只评测，不训练，不重复此前已完成的 P12＋V4 对 θ5000 面板。

P4／P8／P12 是同一大池离线训练轨迹平均抽样4／8／12次，即 **34,398／68,796／103,194步**。P表示policy函数，V表示value函数，各自包括来源模型的完整encoder。组合网络分别计算，MCTS取对应policy先验和value评估；新组合与来源的权重逐张量核对。前轮依据见[大池报告](../entity_attention_recent_pool_20261009/README.md)。

## 搜索队列

| 顺序 | A | B | 目的 |
| --- | --- | --- | --- |
| S01 | P12＋V4 | P4＋V4 | 固定V4，直接检验最终policy是否更强；使用新开局，补充此前对共同对手θ5000的得分差 |
| S02 | P4＋V12 | θ5000 | 与已有P4＋V4、P12＋V12、P12＋V4共同构成policy/value的2×2组合，观察晚期value及交互 |
| S03 | P8＋V4 | θ5000 | 固定V4，比较policy第4／8／12次平均抽样的搜索收益 |

双方 native **S1024/N1**、c_puct=1.5、无根噪声、温度0、fold forced phases、每次模拟transition上限512；CUDA compiled、fail_fast、actor_projection=false、reused transport、一个推理owner、100个活动对局、max batch64、候选bucket16/32/48/64。配置沿用前轮已验证设置。双网络增加推理计算量，比较控制模拟次数。

S01采用 **30500000～30500199** 新开局，policy seed base30510000、bootstrap seed30520000；S02/S03沿用 **14900000～14900199** 的DEV开局及旧policy/search种子，支持与三个已有θ5000面板计算整pair配对差。两批开局均与整个29,000局原池无种子重合。新开局直接互打与复用DEV对手得分差分别解释，均不覆盖训练seed不确定性。

## Greedy 队列

使用标准argmax policy、互换先后手，各组200pair；**greedy不使用value**，所以P12＋V4与P12＋V12归为同一个P12，不重复跑value变体。这里G组名称中的P4V4/P8V8/P12V12仅标识所载入的完整源checkpoint，棋力由其policy决定。

| 顺序 | 对战 |
| --- | --- |
| G01～G03 | P4／P8／P12 分别 vs θ5000 |
| G04～G06 | 旧离线9500／hybrid第6轮／hybrid第9轮 分别 vs θ5000 |
| G07～G09 | P12 vs P4、P8 vs P4、P12 vs P8 |
| G10 | hybrid第9轮 vs 第6轮 |

所有greedy组使用同一批30500000～30500199新开局；CPU **8 worker，每个Torch线程数1**，通过既有pair并行评测器执行。生产native_batched的greedy路径仍是逐局前向，因此此处采用CPU对局并行。CPU worker数按本次16逻辑CPU、约31GiB内存设备设置，未宣称跨硬件最优。每组独立子进程，完成后退出，再启动下一组，避免长期累计模型／编译图／worker；没有新增硬内存限制。

## 执行、查看与恢复

首次运行先做有限验证：两个新组合实际native/CUDA compiled搜索（S16/N1、2pair），以及生产CPU8 greedy（2pair）；随后重开已完成的小面板，确认恢复保留同一批已完成对局。smoke留在runtime，排除于正式统计；摘要与配置见[smoke.json](smoke.json)。不做大量0轮自测或额外完整留出扫描。模型／配置身份、源码及native扩展SHA、命令和环境见[execution.json](execution.json)及对应runtime会话快照。

从仓库根目录执行，**同一命令完成整条队列并支持恢复**：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  python -u artifacts/entity_attention_recent_pool_crossplay_20261010/run.py run
```

后台任务已经启动时不要再执行第二份；目录锁拒绝重复runner。Ctrl+C或对runner发送SIGTERM会转发SIGINT给当前评测子进程，使生产评测器保存逐局记录和汇总；恢复时跳过完整面板和已完成pair，未完成pair按生产机制重做。

```bash
# 查看实时面板进度及已完成统计；不启动对战
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_recent_pool_crossplay_20261010/run.py status
# 请求后台runner优雅停止
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_recent_pool_crossplay_20261010/run.py stop
# 重新汇总已有结果；不启动对战
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_recent_pool_crossplay_20261010/run.py analyze
tail -f runtime/entity_attention_recent_pool_crossplay_20261010/driver.log
```

各组日志为`runtime/entity_attention_recent_pool_crossplay_20261010/<面板ID>.log`，当前owner/子进程及面板见同目录`live.json`。组合checkpoint在忽略的`runs/entity_attention_recent_pool_crossplay_20261010/compositions/`；正式配置与最终raw JSON分别位于本目录`configs/`、`panels/`，重型逐局恢复sidecar按仓库约定忽略。

每完成一组，脚本核验完整200pair、互换先后手、实际checkpoint/config身份、bootstrap区间与S1024/N1实际预算，再更新结果表和机器实验索引；全部完成后索引状态自动从incomplete改为valid。对局异常会明确停止队列并保留现场，不静默跳过。CI按200个整pair重采样10000次；和棋半分，循环裁和单独记录。矩阵为固定组合的探索性比较，未作多重比较校正，主要看直接互打与受控分支变化。
