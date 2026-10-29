# Policy辅助最优离线模型 → R31–R37历史／60k在线自迭代

2026-10-27进度核对：state已提交本地13轮。第12轮greedy为66.075%，S1024N1搜索已完整完成500 pair、得分60.20%；第13轮训练3000步及greedy1000 pair完成，greedy为63.75%。第14轮目录已建立，但尚无完成训练记录。此前973/1000局和只提交11轮是中断时点记录；[前12轮平台分析](plateau_review_20261025/results.md)保留当时统计，不扩展重算。

```bash
bash scripts/run_policy140221_r31_r37_p60k_iteration_20261025.sh
```

只检查输入、源码配置和checkpoint，不执行工作负载：

```bash
bash scripts/run_policy140221_r31_r37_p60k_iteration_20261025.sh --check
```

默认无限自迭代，同一命令恢复；本次只发布缓存释放实现并分析既有结果，没有恢复采集、训练或正式评测。`--rounds N`限制本分支绝对完成轮数；`--max-steps-this-session`只限制新增训练步，仍可能先采完整10000局，不能用作只读检查。原在线、离线、后续队列及其产物保持原身份。

## 起点和继承

起点固定为[仅policy辅助离线实验](../entity_attention_128x2_opponent_policy_r21_r37_20261024/README.md)按**完整留出主policy KL＋主终局value MSE最低**预先选中的step140221，不使用final192804或按对战表现重选：

- SHA `e4edeb20e8caa3c1abec0146d3757594dd6e0428bd833badd0cb3915c2299a9f`。
- 主KL0.2053704366、主MSE0.7359864915，合计0.9413569281；原selection、selected_candidate和11个完整留出节点已核对。
- attention128×2／4头／FFN512、market relations、global/players/market value读取，784778参数。继承完整权重、AdamW矩／计数、sampling RNG及Torch CPU/CUDA RNG；在线lr改为2e-5，累计步数从140221继续，局部轮号重新从1开始。
- 仅对手policy辅助权重0.15，短期value辅助0，不注册短期头。主policy／终局value权重仍1/1；原始目标及对手下一主行动的类型／买牌／保留牌标签协议保持。

## 历史、容量与训练

历史为原R31完整4000局＋R32–R37每轮10000局，**合计64000局**。零基NPZ `round_index=73–79`，配置一基范围74–80，逐轮局数使用完整映射检查。Train57511局／4594761位置／956337有效policy位置；Validation6489局／517380位置／107958有效policy位置。保留原始NPZ、标签、路径及整局split，不继承旧run已淘汰后的60k活动池；来源index／manifest／summary、教师谱系、牌池及每局size／mtime见[plan.json](plan.json)。

每轮仍新采10000局、容量60000、训练3000步、lr2e-5、两路独立B256、AdamW weight decay1e-4、全局clip1、FP32。首轮采集结束后，64000历史＋10000新局共74000，**沿用原retention seed及随机原始历史优先淘汰机制，移出14000历史整局**；保留50000历史＋10000新局，开始首轮训练。完整原历史清单保留，原NPZ不会删除。`initial_history_eviction=true`显式允许这一启动方式，旧配置默认仍拒绝首轮超容量。

首轮policy新局配额1/6，3000步累计128000／768000条policy抽样来自本分支新局；第二轮为1/3，依次增加至第六轮全新。原历史局先按seed随机退出，原历史全部退出后，在线新局按整局FIFO淘汰。主policy的历史正权重位置仍使用原固定25%子集；value从全部预算Train位置均匀抽样。policy辅助在同一value batch上按其原未来主行动policy权重监督，不另套主policy的历史位置子集，不增加额外抽样RNG。

局部第1轮标记谱系R38：采集`round_index=80`、checkpoint `round_index=81`；首轮完成后累计143221步。之后教师使用每轮3000步final，不由轮内留出或棋力重新选点。

## 采集、评测与留出

- 采集沿用当前attention较优执行：12 search worker、2000活跃局、B500、microbatch128、owner双槽、FP32/native-cudagraph、shared slots2／等待5ms；B桶`[16,32,64,128,192,256,384,500]`、C桶`[16,32,48,64]`。75% S128N1（总128），25% S256N4（每世界256×4，总1024）；高预算根噪声0.25、alpha0.3，低预算无根噪声／无主policy监督，前10完成回合温度1后0，FP/PTP关闭，12贵族。
- 固定对手是原raw R37 final，SHA `611f4f4ed18f397b03c79bb3a2beae51e7a77a747883680d9189eed3052a1045`。**每轮greedy1000pair**，CPU8 worker，双方零搜索；**每3个本地轮次搜索500pair**，双方S1024N1（每世界1024×1，总1024）。配对开局／种子和bootstrap沿用当前协议。
- 搜索采用较优配置A：12 search worker、1000活跃局、B500、microbatch128、owner双槽、B桶`[64,128,256,500]`、C桶`[32,48,64]`、shared slots2／等待5ms、FP32/native-cudagraph、fail-fast；无根噪声／确定性T0、FP/PTP关闭。采集与搜索A不混用活跃数和桶列表。
- 轮内原2048位置面板继续在`0,100,250,500,1000,2000,3000`步评估，轮内固定、跨轮按原规则重抽。每500步保存恢复点，并强制保存全部诊断点；每步保留主指标，policy辅助诊断随训练记录，编码器分任务梯度第100步整倍数记录（沿用累计步口径）。
- 每六轮固定使用此前六轮全部原始validation。新本地1–6轮（谱系R38–R43）用原R32–R37的6092局／485932 value／101425 policy位置；本地7–12轮使用新本地1–6轮原始validation，之后每六轮推进。独立于活动池及其淘汰，不进入训练或教师选点。

这部分使用旧R37在线配置的规则和当前较优执行候选；不宣称采集全局最优，不以小规模工程测试给出棋力结论。

## 当前执行实现与验证

使用本次独立冻结的**当前源码、native扩展和12贵族牌池**。在线driver新增按轮局数映射、显式启动淘汰，以及continuation模型／缓存标签／replay身份／联合更新的实际调用链。旧无辅助配置序列化保持原样、首轮超限仍默认报错；本分支配置严格冻结辅助权重和启动规则。

当前训练器内建流式会话累计、state-only value／辅助传输和批量标量回读；使用最新轻量记录：逐步主metrics、小型会话身份及batch SHA，不写大抽样`.bin`／`.steps.jsonl`或重复metrics。在线沿用16GiB按需compact LRU及同步draw；2026-10-25起，完整轮训练结束、final保存后清空LRU并删除采样引用，经GC及glibc malloc_trim将空闲页归还系统，再进入greedy／搜索及下一轮采集。固定留出面板仍常驻CPU，缓存不是进程RSS硬限制。新发布及冷加载代价见[缓存释放记录](cache_release_20261025/README.md)。`training_workers=4`是新建**离线**固定池的默认，此在线变动回放池不额外套用离线memfd路径。

[tests.xml](tests.xml)记录实际冻结实现下29项相关回归通过，覆盖不等局数／旧格式、初始淘汰、原辅助类型检查、两轮训练与精确恢复、初始权重／AdamW／RNG继承、policy辅助指标、六轮完整留出及恢复和正式配置。初始微型数据只有1个高预算位置，原25%整除后为空，已将测试样本补足4个正权重位置；另修正断言以排除会话事件行，原测试记录分别保留于initial_tests／fixture_logging_tests，不改正式抽样规则。

真实模型CPU工程验证见cpu_validation及verification：使用原选中模型的权重／AdamW／RNG，CPU执行副本只将learner设备元数据从cuda改为cpu；7个历史源轮各取1个真实Train对局，2次B16更新及3个S8N1原生根（共24模拟），推理不执行辅助头。正式配置和原checkpoint不改变，不运行GPU采集或正式对战。

- `configs/entity_attention_v1/random_budget_policy140221_r31_r37_p60k_20261025.json`：正式配置。
- 本目录：prepare、plan／source、完整初始留出、输入检查、回归和有限验证身份。
- 同名`runtime/`：旧源码及独立CPU工程验证保留；实际启动脚本已指向新`source_cache_release_20261025`，仅替换在线driver。
- 同名`runs/`：64k历史清单、12轮新采集／训练／留出／greedy，以及3／6／9轮完整搜索和第12轮中断搜索原始结果。当前未恢复正式运行；同一命令先恢复第12轮缺失搜索。
