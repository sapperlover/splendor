# 第20轮续跑：高预算S256N4＋12贵族池

状态：已后台启动，trainer PID **131091**，第21轮4000局采集已发出`collect_start`，生产manifest已确认总1024／4世界、低预算128／1世界和12贵族指纹；启动核验时在初始化／编译，未等待整轮完成。见[startup_verification.json](startup_verification.json)。

2026-10-22按用户要求，从原预算随机化分支的**第20轮raw模型**继续无限自迭代。模型为原128×2 attention，不是新的8万局离线模型、EMA或拼接模型。原累计训练步57423；继承权重、完整AdamW、Torch及抽样RNG、已完成20轮的状态和第15–20轮共24000局活动历史池。原分支不改配置，新分支从编号**第21轮**（`round-020`，数据round_index63）开始。

仅更改高预算搜索及新局规则：

| 项目 | 新分支 |
| --- | --- |
| 高预算多候选决策 | 概率25%，S256N4，每世界256次，总1024次；根噪声沿用原配置 |
| 低预算多候选决策 | 概率75%，S128N1，总128次；继续关闭根噪声、不提供policy监督 |
| 贵族池 | 新采集及评测使用12张，加入N11蓝绿黑333、N12白红44；双人每局仍随机出3张 |
| 每轮采集／历史池 | 4000局／24000局；旧原始20k局早已全部退出，当前均为在线混合预算局，继续每轮FIFO淘汰最早4000局 |
| policy／value | policy从整个活动池的高预算正权重步抽；value从高低预算全部步抽。新N4局首轮占活动池1/6，六轮后旧N1整局全部退出 |
| 训练 | 每轮1000步；lr2e-5、batch256，两头各256独立抽样后一次联合更新；AdamW、权重、clip1及8GiB缓存不变 |
| 采集执行 | 12 search worker、1000活跃局、GPU B500、2槽流水线；与第20轮实际执行配置核对一致 |
| 留出 | 沿用每轮固定2048样本，在步0／100／250／500／1000评价 |
| 对战 | 沿用对R43的每轮1000pair greedy、每3轮500pair S1024N1；搜索执行复用采集，但pipeline_slots=1；无根噪声 |
| FP/PTP、PSW、辅助头 | 全部保持关闭 |

历史24000局保持原10贵族标签、train/validation划分、size/mtime和路径，不重新生成。新旧数据暂时混合是继承历史池的结果，不能把这次分支变化单独归因于N4，因为贵族规则也同步改变。两个新增贵族以数值需求表示，不改变网络输入维数或权重结构。

原第21轮不完整采集已按用户授权删除：2591个完整NPZ及批次元数据，共2605文件，释放187736064分配字节。它不在第20轮活动池内、没有训练checkpoint，没有接入新分支；保留[删除记录及原manifest](discarded_round21.json)。原state SHA未变。

从仓库根目录启动或中断后恢复，默认无轮数上限：

```bash
bash scripts/run_r20_s256n4_iteration_20261022.sh
```

只检查继承输入，不采集或训练：

```bash
bash scripts/run_r20_s256n4_iteration_20261022.sh --check
```

配置见[random_budget_r20_s256n4_noble12_20261022.json](../../configs/entity_attention_v1/random_budget_r20_s256n4_noble12_20261022.json)，输入、种子、父状态／checkpoint／池身份见[plan.json](plan.json)。父状态冻结副本在`runs/entity_attention_r20_s256n4_noble12_iteration_20261022/inputs/parent_state.json`；继承的完成轮记录保留原始路径，新增轮与checkpoint写入新run。

执行源码、native扩展及12贵族数据已冻结于`runtime/entity_attention_r20_s256n4_noble12_iteration_20261022/source/`，身份见[source.json](source.json)；脚本使用这份源码和数据，后续工作区修改不会热切换算法或规则。输出在同名`runs/`，后台启动信息和日志在同名`runtime/`，实际执行命令及状态见[launch.json](launch.json)。如需改B500或pipeline_slots，仍可修改外部启动配置，下一新轮生效，已开始轮保留冻结设置。

实现补充：`SearchBudgetRandomization.low_determinizations`允许低预算单独指定世界数，默认省略时继承高预算世界数，保持旧配置身份。v3数组仍按高预算世界数分配，低预算只使用前`valid_world_count`行，其余补零；校验按实际世界数复算预算、behavior及policy，不把零行纳入平均。`random_budget_iteration.continuation_state`继承完整轮界状态，校验父checkpoint、绝对轮号及活动池身份；普通旧配置省略该字段时行为不变。

相关CPU回归37项通过，见[tests.xml](tests.xml)，其中包含真实混合世界采集／回放、正权重过滤、训练更新及续跑继承；GPU短采集的范围和结果见[gpu_smoke_verification.json](gpu_smoke_verification.json)，是工程验证，不构成棋力或吞吐结论。正式运行不会等待整轮完成才交付，也不自动轮询。
