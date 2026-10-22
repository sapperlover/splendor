# 第20轮模型：128个固定局面的搜索稳定性

用户授权：准备脚本并后台启动，不持续轮询。固定根诊断，不是棋力赛，不更换在线教师。使用前面BGA分析的第20轮原模型，SHA `fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd`，128维、2层、购买关系特征及多实体value读取，无双辅助头。

## 局面与规则

从混合预算自迭代第15–20轮各抽48个validation整局，每局只取一个有多个合法选项的主行动。完整checked重放至自然终局，抽中局面的所有状态及动作特征与原NPZ逐项核对。每轮16个基础样本、32个预筛候选；基础96个和预筛192个分别按实际主行动轨迹的前三分之一、中三分之一、后三分之一均衡分配。阶段表示相对对局进程，不是固定步数或分数阈值。

基础局面前、中、后各32个。预筛192个各跑4个独立seed的S1024N1，先按两两首选一致率从低到高、再按JS从高到低选32个，最终128根来自128个不同整局。保留全部预筛排名；若筛出的部分根仍完全一致，报告实际分歧根数。筛选组只保证高分歧，不保证包含特定行动类型；不把BGA的061局面混入随机样本。

旧对局使用显式冻结的10贵族目录重放，搜索用当前12贵族目录。两种目录对冻结的已入场3张贵族、卡牌及常量一致，每个根分别在两个目录下重建并核对DecisionRequest。未入场贵族不进入后续游戏，因此这些条件局面等效；本样本没有新增两张贵族入场的局面，不能推广为12贵族开局分布的全面评估。

搜索只接收当前行动者可见的DecisionRequest，保留其本人合法获知的盲保留身份，排除对手盲保留身份和真实未来牌序。整局历史仅供来源核验，不传入搜索。它是“行动者信息集”，并非删掉己方私有信息的纯旁观者视图。

## 方法与预算

本次及后续用户口径：**总预算=S×N**。S是每个确定化世界预算；内部`MCTSConfig.simulations`仍传总预算，不修改底层接口。

| 方法 | 每世界S | N | 总模拟数/根/seed |
| --- | ---: | ---: | ---: |
| S1024N1 | 1024 | 1 | 1024 |
| 随机节点S1024 | 1024 | 1个共享统计池 | 1024 |
| S256N4 | 256 | 4 | 1024 |
| S1024N4 | 1024 | 4 | 4096 |
| greedy | 0 | — | 0 |
| S128N1 | 128 | 1 | 128 |
| 随机节点S128 | 128 | 1个共享统计池 | 128 |
| 参考S16384N4 | 16384 | 4 | 65536 |
| 参考随机节点S65536 | 65536 | 1个共享统计池 | 65536 |

每种搜索均8个seed。候选使用42100000–42100007，确定化参考42200000–42200007，随机节点参考42300000–42300007，预筛42010000–42010003，抽样42000000。两种参考各自以8次访问策略的均值定义参考分布与首选，并保留自身跨seed稳定性。greedy固定网络不受seed影响，每根一次前向结果明确标记为重复使用，不伪称8次独立随机试验。

无根噪声、T=0、c_puct=1.5、FP32、TF32关闭，CPW、Forced Playouts、Policy Target Pruning均关闭。随机节点每次模拟重采隐藏世界并在真实随机事件抽样、共享信息历史树，N=1表示共享池，不表示只有一个隐藏世界。

正式8192次搜索，共141819904次模拟；另1024条greedy结果。预筛768次搜索，共786432次模拟。候选最多64个活跃搜索、高预算参考最多16个，GPU批容量500，单槽流水线；参考限制并发以控制CPU树内存，不能简单让所有根同时活跃。编译静态批桶16/32/64/128/256/500，候选桶16/32/48/64，GPU明确失败时停止，不静默换CPU。

## 指标定义与限制

- 跨seed Top-1一致率：8个seed的28对中首选相同的比例，同时输出众数占比、8次全同率。Policy差异为28对JS均值，单位nat，范围[0,ln2]。
- 与参考比较：8次候选首选命中参考均值分布首选的比例，以及候选分布到参考均值分布的平均JS。两种参考分别统计。参考与自身比较包含相同seed，只作自一致性描述，不能当独立准确率。
- 漏搜：参考首选在候选根获得0次访问的比例；另报≤1、≤4次访问。greedy没有搜索，这三项为null。
- Top-k，k=1/3/5：参考首选进入候选Top-k、候选首选进入参考Top-k、参考Top-k集合覆盖比例、候选Top-k覆盖的参考概率质量。零访问动作不因任意并列排序而算作覆盖。
- 重要决策：参考按平均访问量排序的首选与次选，其Q估分差达到0.5/1/2/5个百分点分别统计命中、JS、漏搜、Top-k等；与访问份额差分开。Q先在每个seed每个世界计算W/N，再等权平均，不能合并N4的W与N后相除。要求每个世界对相关动作均至少16次访问；否则Q或重要性未定义，报告有效根数。它是模型搜索的期望得分差，未校准，不是实际胜率或实测行动收益。
- 附带候选首选相对参考首选的有符号Q regret及有效seed覆盖率；负值保留，反映访问首选未必Q最高。未覆盖的动作不补零Q。
- 基础96、筛选32、全部128、基础各阶段、两个参考首选一致/不一致分别统计，并在这些组内再分重要性阈值。每根等权。128根合并受刻意筛选影响，不代表自然局面分布。无终局续局真值，因此所有“准确率”实际是参考吻合率。

## 执行与产物

首次准备、工程验证及后台启动：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u artifacts/entity_attention_r20_search_stability_128_20261021/run.py prepare
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/entity_attention_r20_search_stability_128_20261021/source/src SPLENDOR_PROBE_ROOT=/home/lc/下载/splendor OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/home/lc/下载/splendor/runtime/entity_attention_r20_search_stability_128_20261021/compile_cache TRITON_CACHE_DIR=/home/lc/下载/splendor/runtime/entity_attention_r20_search_stability_128_20261021/triton_cache python -u runtime/entity_attention_r20_search_stability_128_20261021/source/artifacts/entity_attention_r20_search_stability_128_20261021/run.py smoke
conda run --no-capture-output -n splendor python artifacts/entity_attention_r20_search_stability_128_20261021/launch.py
```

准备保存当前源码、native二进制及数据快照，后台进程从快照导入，不受后续源码或默认贵族数据修改影响。严格检查模型、脚本、来源局面及快照SHA；修改实验定义应另建实验。中断后原命令重启会跳过已写完的根/seed；启动器拒绝重复启动仍存在的PID。每次会话记录环境与命令；一次运行内进程锁避免重复写入。

现有产物：`plan.json`、`candidate_bank.json`、`source_games.jsonl`、两套冻结目录、`tests.xml`、`smoke.json`、`launch.json`、`execution.json`。其中后四项随相应步骤产生。最终128根由预筛完成后写入`root_bank.json`，正式搜索前冻结SHA。

进度和日志：`runtime/entity_attention_r20_search_stability_128_20261021/live.json`、`run.log`；逐根恢复结果在同目录`raw/`和`screen/`。完成后自动输出本目录的`summary.json`、`per_root_metrics.json`、`reference_roots.json`、`results.md`及可独立复算的`raw_searches.jsonl.gz`、`screen_searches.jsonl.gz`。原始行保留对齐的访问、策略、分世界Q和诊断，JSON根保留动作与候选顺序。

验证：7项相关测试通过，包含预算、零概率JS、两两一致率、N4等权Q、覆盖不足、漏搜/greedy口径、真实旧对局重放与当前目录条件等效、完整汇总及压缩原始数据导出。GPU工程验证是两个小预算根，不计正式统计；结果见`smoke.json`。正式实验状态以runtime进度和最终summary为准，启动不代表结果完成。
