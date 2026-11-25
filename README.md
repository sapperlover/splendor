# Splendor 双人 AI

2026-11-24已完成[四类冷产物T7归档](artifacts/repository_archive_20261124/README.md)：74个MC／λ普通模型、14份已完成主日志、30份旧在线回放快照及旧104k池93537个Train。包4.9205GiB，扣除恢复链接净释放6.6236GiB。最新λ在线输入／近期60k、全部固定留出及关键模型保留；旧104k续训与旧在线恢复需先恢复所需原件。较早“104k／初始64k全留本机”的表述仅为当时状态。

2026-11-24 已接入[在线归档搜索 λ 训练](docs/guides/value_outcome_zh-CN.md#在线混合预算自迭代)。用每局自身教师的原始 W/N 打标签，跨轮复用，按活动回放校验并恢复。可用 λ161924 与 FPU 的新分支启动器预检和续训。

2026-11-23新增[λ161924的历史最强FPU配方／无FPU对L12](artifacts/lambda161924_vs_l12_fpu2000_20261123/README.md)两个脚本，各2000pair、双方S128N1（各总128）。λ侧分别r0.3／c_puct1.0与None／1.5；L12均保持原搜索None／1.5，共用新种子、独立恢复。27项回归及双模型CPU根验证通过，正式评测未启动。

2026-11-23新增[固定λ161924对同池MC70842／原R37的两组搜索脚本](artifacts/lambda161924_search3000_20261123/README.md)：双方S128N1（各总128），每组3000pair／6000局，配置A、共同新种子、独立恢复。核对当前60k池和指定节点身份，16项回归与三模型CPU根检查通过；仅给命令，正式评测未启动。

2026-11-20已实现并准备[近期60k池标量MC／归档搜索λ从零对照](artifacts/value_outcome_60k_20261120/README.md)：当前执行标量MC与归档搜索λ两组，WDL两组搁置；原53980／6020整局split、4 worker共享只读缓存10GiB、固定12次value曝光；第二组由归档搜索根W/N一次生成λ=0.8标签与缺失掩码，无教师推理、MC预热或周期刷新，验证保持真实终局。仅完成工程验证、容量核算和prepare，未启动正式训练或棋力评测。

2026-11-01归档位置更新：已完成[普通模型／淘汰训练局／评测明细T7归档](artifacts/repository_archive_20261101/README.md)：234543个文件，包10.95GiB，扣除恢复链接后净释放13.74GiB。关键模型、初始64k／活动60k、104k共享池及固定留出保留；R37低学习率228000步恢复点与主日志保留。冷原始面板、旧普通节点和三份已结束逐步日志使用前需解包，较早“全留本机”仅是当时位置记录。

2026-10-31：[Kirameki八组各20局串行套件](docs/guides/kirameki_advisor_zh-CN.md)，壳脚本`scripts/run_kirameki_suite_20261031.sh`保留原六组，并在末尾追加独立世界S1024N1、S256N4，各20局，共160局。两组总预算均1024，按在线自迭代的原始访问份额等权平均选点，不使用Q补评；原随机节点S1024/b1单独保留。沿用第9轮、非排位宝神、Random／不限时、单调试连接、完整过程记录与恢复。当前六组进程继续运行，冻结计划不改写；结束后重跑原命令，备份升级计划并接续新增两组。116项回归及两组实际GPU离线验证通过，本次未另启网站对局。历史分组结果、连接和平局修复见专题。

2026-10-31：新增[Kirameki非排位自动对弈](docs/guides/kirameki_advisor_zh-CN.md)，入口`python -m splendor.kirameki_autoplay`，固定第9轮、CUDA随机节点S65536N1、batch32、compiled/native-cudagraph；精确执行支付／归还／贵族选择，核对网站棋谱，保存终局后回菜单，以Random先后手重开宝神对局。默认持续，`--games`限制局数；原`kirameki_live`仍只读。有限完整小预算对局及正式预算一手／重开验证完成，不作为棋力结论。

2026-10-31：[单树批量叶子与编译推理加速](artifacts/single_tree_batch_20261031/README.md)已接入网页推荐器；速度优先的batch64三个保存盘面65536次约2.82–3.20秒，同GPU串行编译约18.3–19.1秒。后续按搜索质量关注采用较保守的batch32（约3.14–3.56秒）。启动预热单列，条件支付／归还另计预算；批量可能改变选点，尚无正式棋力对照。[当前启动命令](docs/guides/kirameki_advisor_zh-CN.md)。

2026-10-30新增[Kirameki Lab 非排位只读推荐](docs/guides/kirameki_advisor_zh-CN.md)：连接已有浏览器，固定policy在线第9轮，S4096N16（总65536）；支持GPU多世界批量／CPU多进程搜索、购买支付与超限归还，并逐次保存安全盘面和模型输入。默认在原预算内给共同候选补足每世界32次，再按世界等权平均Q选点；完整逐世界统计用于复核。先搜1024／补到128的改动已按用户要求撤回，可显式试用每次决策65536次的 `native_chance` 随机节点共享树。仅给命令、用户手动执行；有限工程验证完成，未通过脚本操作网站，尚无新选点规则的棋力结论。

2026-10-29已准备[R37初值lr2e-5／104k池的固定140221步对原R37脚本](artifacts/r37_lr2e5_step140221_vs_r37_10k_20261029/README.md)：按用户更正固定约14万步最佳节点，先greedy10000pair，再双方S128N1（各总128）10000pair，沿用此前10k种子／配置A，严格串行可恢复。15项回归与双模型CPU根验证通过，正式评测未启动。

2026-10-27新增[R37权重起点的两组104k无辅助离线脚本](artifacts/entity_attention_r37_104k_offline_lr_20261027/README.md)，lr分别1e-4／2e-5；仅加载模型权重、新AdamW及原离线种子，原5%完整留出早停／选点和其他离线配置保持。各自独立，支持仅train或完整原评测流程；只给命令，未启动。

2026-10-27按[Git产物整理规则](artifacts/repository_git_cleanup_20261027/README.md)区分版本管理与原始数据保全：Git保存实现、协议、紧凑结论及数据身份；完整评测、诊断数组和大型局清单本地／外置保存。历史原件仍在原路径，未在此次整理中迁移至T7；新环境复核前按清单恢复所需原件。

2026-10-27[五个104k离线best对raw R37的大面板](artifacts/offline5_vs_r37_10k_20261027/README.md)十组全部完成：各模型greedy及双方S128N1（总128）各10000 pair，共20万局。Greedy为53.265%–55.390%，S128N1为56.290%–57.098%；辅助组greedy略高，搜索未明显拉开。原训练／原S1024评测与新版大面板分别见[近期辅助目标完成结果](docs/reports/auxiliary_offline_results_20261027_zh-CN.md)。

2026-10-27核对：[104k无辅助离线实验](artifacts/entity_attention_128x2_noaux_r21_r37_20261026/README.md)及前序policy＋得分均已完整结束；无辅助210332步早停、选中122694步，原R37 greedy1000 pair／S1024N1搜索500 pair为51.075%／57.50%。后续10000 pair的新greedy／S128N1为53.265%／56.70%。

2026-10-27核对：[104k池policy＋得分归属辅助实验](artifacts/entity_attention_128x2_policy_score_r21_r37_20261026/README.md)已完成训练和原两组对战，192804步早停、选中140221步；原greedy／S1024N1为53.475%／56.80%，新版10000 pair greedy／S128N1为55.080%／57.013%。

2026-10-26新增[到终局分数归属辅助](docs/guides/score_ownership_zh-CN.md)：双方各预测一级／二级／三级牌及贵族的未来新增分，共8输出；接通共享只读供数、训练／恢复／验证与梯度诊断。policy＋得分组合已完成正式训练和评测，见文首；独立仅得分辅助预设仍只完成准备及[有限工程验证](artifacts/score_ownership_20261026/README.md)。

2026-10-25已完成[四组离线普通模型／逐步日志和原80k局池归档](artifacts/repository_archive_20261025/README.md)：672个普通checkpoint、4份主日志、80000个NPZ，包8.84GiB，净释放11.06GiB。最优／完整留出／initial/final模型和当前在线输入保留；原80k离线复训或完整复核须先恢复本包。下方较早“原80k留本机”是当时记录，当前位置以本批清单为准。

- 2026-10-25[policy辅助最优离线140221步→60k在线自迭代命令](artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/README.md)：继续policy辅助0.15，原R31–R37历史64k，首采10k后随机移出14k历史至60k；lr2e-5／每轮3000步，采集active2000、搜索A，固定对raw R37。已提交13轮；第12轮greedy66.075%、完整搜索60.20%，第13轮greedy63.75%；此前平台快照见[分析](artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/plateau_review_20261025/results.md)。入口已接入训练结束释放CPU缓存的新发布，本次未续跑。

2026-10-27核对：[离线训练执行默认](docs/guides/offline_execution_zh-CN.md)的流式累计、4 worker共享只读供数及传输优化已用于完成后续训练；104k仅value／仅policy／双辅助及44k仅value均已完成原两组评测。115000步停训及仅给命令是此前时点，当前结果见[近期辅助目标完成结果](docs/reports/auxiliary_offline_results_20261027_zh-CN.md)。缓存仍按池大小配置。

- 2026-10-24[当前104k value辅助之后的三组串行消融](artifacts/entity_attention_auxiliary_queue_20261024/README.md)：104k仅policy → 104k双辅助 → 原44k仅value，全部独立从零／lr1e-4／相同早停与R37两组对战。等待前驱整条流程锁释放及成功完成后再启动；失败／中断不推进。90项回归与CPU真实数据／原生搜索验证通过，当前任务不修改。

- 2026-10-23[第21–37轮104000局attention128＋仅短期value辅助离线训练](artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023/README.md)：从零／新AdamW，lr1e-4，H2/H6/H14权重0.25，对手policy辅助关闭；12次value曝光上限350552步，每5%完整主KL+MSE三次严格上升早停，最低点先对raw R37 greedy1000pair、再配置A搜索500pair。使用最新轻量抽样记录；80项回归和真实GPU短验证通过；按最新要求已停止此前启动的队列，只保留手动命令。

2026-10-23新增[短期value与对手policy辅助目标](docs/guides/continuation_auxiliary_zh-CN.md)：2／6／14玩家回合KataGo式指数标签，以及对手下一主行动的类型／购买牌／保留牌软标签CE。提供独立离线配置、恢复及主指标选点；[有限工程验证](artifacts/continuation_auxiliary_20261023/README.md)不代表棋力收益，现行在线配置未启用。

2026-10-23训练产物调整：当前在线／近期离线默认停写policy、value两份抽样`.bin`和两份`.steps.jsonl`，每步指标统一保留主日志；R31在线分支改为500步恢复点＋七个固定诊断点。启动脚本已切换新冻结发布，历史文件保持；见[记录与checkpoint规则](docs/guides/training_artifacts_zh-CN.md)。

双人基础版规则引擎、Deep-Sets／实体注意力 Actor–Critic、PPO、冻结网络搜索与 AlphaZero 式训练，含 C++ 原生后端和本地人机网页。游戏行为以[标准规则](docs/splendor_standard_rules_zh-CN.md)为准。

从[文档导航](docs/index.md)查找命令和契约；[当前状态](docs/current_project_state_zh-CN.md)集中说明研究进展与限制；[实验索引](artifacts/experiment_index.json)定位实验身份和原始证据。历史实验的完成状态不等于确认棋力收益。

2026-10-21规则修正：默认贵族池统一为**12张**，加入蓝绿黑333、红白44，均3分；双人局仍随机抽3张。Python／C++、BGA推荐、新自对弈采集和评测共用此规范表，恢复身份冻结牌池指纹。旧10张池的历史产物保持原身份；旧模型输入形状不变。详情见[规则与兼容记录](artifacts/noble_pool_bga12_20261021/README.md)。

## 当前研究入口（近期更新至2026-10-23）

- [五类冷产物已迁移至T7](artifacts/repository_archive_20261023/README.md)：223个离线普通模型、1011个账本／逐步日志及76038个旧NPZ，压缩包5.8104GB，净释放13.9211GB。当前局池／恢复输入、原8万局和关键模型保留，旧实验原件按需从归档恢复。

- [在线固定留出已接入并完成六轮回补](artifacts/entity_attention_r31_10k_p60k_iteration_20261023/fixed_holdout_20261023/results.md)：固定R26–R31全部2412个validation对局，R32–R37共42节点；累计MSE 0.744220→0.737498，KL 0.208406→0.209699。以后每六轮自动改用此前六轮留出（R38–R43用R32–R37），段内固定并支持恢复补齐；新启动脚本已生效，本次未续训。

- 2026-10-23[raw R31、每轮10k新局／60k池／3000步自迭代](artifacts/entity_attention_r31_10k_p60k_iteration_20261023/README.md)已完成6轮至谱系R37，对原R43的greedy68.90%、S1024N1总1024搜索74.55%。采集12worker／active2000／B500／microbatch128／双槽、16GiB缓存保持；后续新搜索改用A（active1000、双槽、精简B/C桶），日常／在线greedy仍CPU8 worker／1000 pair，搜索默认500 pair；[五种场景的统一配置、种子与命令](docs/guides/evaluation_zh-CN.md)。新轮界读取源配置，旧面板不回写，本次不启动或重启任务。

- [新增购买结算特征方案已撤回](artifacts/entity_attention_first_round_20261022/README.md)：2026-10-23删除全部新增购买／贵族／终局派生分支及对应候选入口，恢复原583682参数关系attention128。短期进展辅助保持关闭；原准备、短验证及冗余分析保留为历史记录，未新增终局设计或启动训练。

- [第21–31轮attention128最低节点降lr续训](artifacts/entity_attention_128x2_r21_r31_lr2e5_20261022/README.md)：新增10000步三次上升早停，选中新增4000步／累计63453步，KL0.217213、MSE0.759487。对R31 greedy1000pair为50.150% [48.025%,52.300%]；相对父模型同开局差+0.625pp，区间跨0，未确认greedy棋力提升。训练不恢复；另对R43的搜索补测为72.20%，见[三组独立搜索](artifacts/eval_speed_20261023/README.md)，不等同于对R31的搜索结果。

- [第21–31轮44000局attention128从零离线训练](artifacts/entity_attention_128x2_offline_r21_r31_20261022/README.md)：用户主动停止，保留84000步，原148632步预算未完成。完整留出选中59453步，对R20／R31各greedy1000pair已完成：53.400% [51.250%,55.550%]／49.525% [47.375%,51.601%]，支持超R20，未确认超R31。原R20／R31搜索取消、训练不恢复；后续另对R43完成搜索70.70%，见[独立补测](artifacts/eval_speed_20261023/README.md)。

- [第21–30轮及Cross64离线冷产物归档](artifacts/repository_archive_20261022/README.md)：291个模型／43个账本已校验外置到T7的`archive-20261022.tar.zst`（2.45GB），另12个重复账本去重，净释放4.3250GB。**第21–31轮全部44000局、未完成轮已采对局及原80k池留本机**；各轮final与离线关键节点保留，在线续跑输入不变。
- [Cross64 MLP角色初始化标定＋8万局离线训练](artifacts/entity_set_mlp_cross64_role005_offline_80k_20261022/README.md)：角色std0.05、906434参数，保留乘积及动作旁路；用户交付后启动，现中断于已保存81168步，原270557步计划未完成。六个已评估节点中完整留出KL＋MSE最低为54112步；[对R20 greedy1000pair](artifacts/entity_set_mlp_cross64_54112_vs_r20_greedy_20261022/README.md)得43.225% [41.075%,45.325%]，弱于R20；未测搜索。
- [第20轮继承续跑：S256N4＋12贵族](artifacts/entity_attention_r20_s256n4_noble12_iteration_20261022/README.md)：继承raw R20、AdamW/RNG、24k活动池；25%高预算改为每世界256×4，总1024，75%低预算保持S128N1；其余在线设置不变，默认无限自迭代，旧第21轮残留已按授权删除。
- [Checkpoint外置归档](artifacts/repository_archive_20261021/README.md)：161个副本去重、1885个冷模型已校验打包到T7的`archive-20261021.tar.zst`（10.80GB），释放本机12.85GB；加此前缓存／96个恢复点清理共约16.28GB。当前训练、关键模型及8万局数据不动，冷模型使用前需解压对应批次。
- [Attention128同8万局离线对照](artifacts/entity_attention_128x2_offline_80k_20261021/README.md)最近提交135279步，核验时日志为KeyboardInterrupt退出；未自动恢复。预先选定的[94695步对R20 greedy1000pair](artifacts/entity_attention_128x2_94695_vs_r20_greedy_20261021/README.md)得54.400%，95%配对区间[52.299%,56.525%]；后续另对R43搜索500 pair得74.05%，见[独立补测](artifacts/eval_speed_20261023/README.md)。模型128×2／4头／FFN512、583682参数；原270557步训练计划未完成，本次未改配置或中断训练。
- [当前完整实体MLP的8万局离线训练](artifacts/entity_set_mlp_256x4_offline_80k_20261021/README.md)已停止并保留97000步；94695步完整留出KL/MSE为0.530577/0.782832，raw R20同留出为0.445110/0.755265。随后单独授权的[1000pair greedy](artifacts/entity_set_mlp_94695_vs_r20_greedy_20261021/README.md)已完成，MLP得37.075%，95%配对区间[35.0%,39.175%]；没有恢复训练或执行搜索对战。

- [实体汇总＋256宽、4残差块MLP](artifacts/entity_set_mlp_256x4_20261021/README.md)当前预设已加入64维／2头动作cross-attention，共906,434参数；读取25个未汇总实体，保留动作直连、非线性旁路和乘积的独立开关，value-only跳过新增分支。197项通过、1项CUDA跳过，CPU编译／原生短验证见[cross升级](artifacts/entity_set_mlp_256x4_20261021/action_cross_64x2_20261022/README.md)；无新正式训练／棋力结论。[S1024N1对Greedy同开局1000pair](artifacts/entity_attention_r20_s1024n1_vs_greedy_20261021/README.md)已完成1000 pair，得分78.05%；同开局S256N4为77.20%，结果见专题。

- [固定配对评测](docs/guides/evaluation_zh-CN.md)现行默认：greedy为CPU8 worker／1000 pair；日常及在线新搜索统一GPU配置A，12 search worker／active1000／B500／microbatch128／owner双槽、精简B/C桶，双方S1024N1总1024、无根噪声／温度0，从500 pair起步。历史单槽与200 pair原件保留；日常[greedy](configs/entity_attention_v1/evaluation_greedy_r43.json)／[搜索A](configs/entity_attention_v1/evaluation_search_a_r43.json)模板固定R37对R43示例。

## 环境与使用

从仓库根目录执行，使用 Conda `splendor`，设置 `PYTHONPATH=src`。依赖见 `pyproject.toml`，C++ 扩展构建见[原生后端指南](docs/guides/native_search_zh-CN.md)。

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q tests/test_rules_and_phases.py
conda run -n splendor env PYTHONPATH=src python -m splendor.evaluate \
  --player-a score --player-b development --seed-start 0 --seed-count 100
conda run -n splendor env PYTHONPATH=src python -m splendor.web_app
```

网页默认地址为 `http://127.0.0.1:8765`。支持固定策略、greedy、固定深度搜索及原生MCTS；模型和预算、评估条、悔棋及policy在线第9轮S1024N4命令见[评测与网页指南](docs/guides/evaluation_zh-CN.md)。网页模型使用CPU单线程，浏览器只接收当前玩家视图和合法动作。新启动的服务自动保存完整回放至 `runtime/web_replays/`，支持 `--replay-dir`；旧服务内存对局的无重启导出方法见同一指南。

BGA 双人基础版可用[只读招法推荐器](docs/guides/bga_advisor_zh-CN.md)直接连接已有 Chrome、自动识别牌桌，任意轮次接入并输出双方行动／翻牌后的模型胜率估计；支持主行动、支付组合和弃筹码推荐，默认原生MCTS S16384/N1，也可用 greedy，用户自行执行。首次接入时若已在弃筹码阶段且没有本手记录，需手动完成该阶段；也支持离线局面推荐。

训练入口见[AZ循环](docs/guides/alphazero_zh-CN.md)、[离线学习](docs/guides/entity_attention_offline_zh-CN.md)和[PPO协议](docs/specs/ppo_training_protocol_zh-CN.md)。checkpoint和训练NPZ不随Git分发；只读复核需要对应本地或已挂载归档；恢复训练前先按迁移清单撤回该run的冷文件，避免写入冻结归档。

## 仓库与归档

| 目录 | 内容与入口 |
| --- | --- |
| `src/splendor/`、`tests/`、`data/` | 实现、回归、卡牌与常量 |
| `docs/specs/`、`docs/guides/`、`docs/reports/` | 现行契约、操作说明、限定版本的专题结论；[导航](docs/index.md) |
| `docs/archive/`、`configs/archive/` | 被替代的设计与旧准备说明；[历史导航](docs/archive/README.md) |
| `configs/`、`scripts/` | [配置导航](configs/README.md)、[工具导航](scripts/README.md) |
| `artifacts/` | 纳管源码、冻结协议与紧凑结论；原始明细本地／外置保全，[产物约定](artifacts/README.md) |
| `runs/`、`runtime/` | 训练数据、必要模型、运行状态及本地保全；Git忽略 |
| `archive/` | [冷产物登记](archive/README.md)；本次[7.03GiB集中目录](archive/20261019/README.md)及[迁移核验](artifacts/repository_archive_20261019/README.md) |

2026-10-19已将7个离线run的334个中间checkpoint及27个账本／逐步指标文件集中到 `archive/20261019/`，原路径保留相对链接；24个关键模型和两份共72000局训练池留在原处。同盘移动尚未释放磁盘空间，外置时应移动该归档的真实目录内容。20261012旧归档已由用户移往其他位置，当前未挂载、外置位置未记录；原[迁移清单与结论索引](artifacts/repository_archive_20261012/README.md)继续保留。
