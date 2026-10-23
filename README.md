# Splendor 双人 AI

双人基础版规则引擎、Deep-Sets／实体注意力 Actor–Critic、PPO、冻结网络搜索与 AlphaZero 式训练，含 C++ 原生后端和本地人机网页。游戏行为以[标准规则](docs/splendor_standard_rules_zh-CN.md)为准。

从[文档导航](docs/index.md)查找命令和契约；[当前状态](docs/current_project_state_zh-CN.md)集中说明研究进展与限制；[实验索引](artifacts/experiment_index.json)定位实验身份和原始证据。历史实验的完成状态不等于确认棋力收益。

2026-10-21规则修正：默认贵族池统一为**12张**，加入蓝绿黑333、红白44，均3分；双人局仍随机抽3张。Python／C++、BGA推荐、新自对弈采集和评测共用此规范表，恢复身份冻结牌池指纹。旧10张池的历史产物保持原身份；旧模型输入形状不变。详情见[规则与兼容记录](artifacts/noble_pool_bga12_20261021/README.md)。

## 当前研究入口（近期更新至2026-10-23）

- 2026-10-23[raw R31、每轮10k新局／60k池／3000步自迭代命令](artifacts/entity_attention_r31_10k_p60k_iteration_20261023/README.md)：历史R22–R31共40k，首轮50k训练、第二轮填满；lr2e-5，完整继承AdamW/RNG，原混合预算及R43评测不变。最新采集12worker／2000活跃局／B500／microbatch128，回放缓存16GiB；已按要求清除首次未完成采集，**重置后未重启**。

- [新增购买结算特征方案已撤回](artifacts/entity_attention_first_round_20261022/README.md)：2026-10-23删除全部新增购买／贵族／终局派生分支及对应候选入口，恢复原583682参数关系attention128。短期进展辅助保持关闭；原准备、短验证及冗余分析保留为历史记录，未新增终局设计或启动训练。

- [第21–31轮attention128最低节点降lr续训](artifacts/entity_attention_128x2_r21_r31_lr2e5_20261022/README.md)：新增10000步三次上升早停，选中新增4000步／累计63453步，KL0.217213、MSE0.759487。对R31 greedy1000pair为50.150% [48.025%,52.300%]；相对父模型同开局差+0.625pp，区间跨0，未确认greedy棋力提升。训练不恢复、搜索不运行；原始结果与核验见专题。

- [第21–31轮44000局attention128从零离线训练](artifacts/entity_attention_128x2_offline_r21_r31_20261022/README.md)：用户主动停止，保留84000步，原148632步预算未完成。完整留出选中59453步，对R20／R31各greedy1000pair已完成：53.400% [51.250%,55.550%]／49.525% [47.375%,51.601%]，支持超R20，未确认超R31。搜索取消，训练不恢复；原始结果与核验见专题。

- [第21–30轮及Cross64离线冷产物归档](artifacts/repository_archive_20261022/README.md)：291个模型／43个账本已校验外置到T7的`archive-20261022.tar.zst`（2.45GB），另12个重复账本去重，净释放4.3250GB。**第21–31轮全部44000局、未完成轮已采对局及原80k池留本机**；各轮final与离线关键节点保留，在线续跑输入不变。
- [Cross64 MLP角色初始化标定＋8万局离线训练](artifacts/entity_set_mlp_cross64_role005_offline_80k_20261022/README.md)：角色std0.05、906434参数，保留乘积及动作旁路；用户交付后启动，现中断于已保存81168步，原270557步计划未完成。六个已评估节点中完整留出KL＋MSE最低为54112步；[对R20 greedy1000pair](artifacts/entity_set_mlp_cross64_54112_vs_r20_greedy_20261022/README.md)得43.225% [41.075%,45.325%]，弱于R20；未测搜索。
- [第20轮继承续跑：S256N4＋12贵族](artifacts/entity_attention_r20_s256n4_noble12_iteration_20261022/README.md)：继承raw R20、AdamW/RNG、24k活动池；25%高预算改为每世界256×4，总1024，75%低预算保持S128N1；其余在线设置不变，默认无限自迭代，旧第21轮残留已按授权删除。
- [Checkpoint外置归档](artifacts/repository_archive_20261021/README.md)：161个副本去重、1885个冷模型已校验打包到T7的`archive-20261021.tar.zst`（10.80GB），释放本机12.85GB；加此前缓存／96个恢复点清理共约16.28GB。当前训练、关键模型及8万局数据不动，冷模型使用前需解压对应批次。
- [Attention128同8万局离线对照](artifacts/entity_attention_128x2_offline_80k_20261021/README.md)最近提交135279步，核验时日志为KeyboardInterrupt退出；未自动恢复。预先选定的[94695步对R20 greedy1000pair](artifacts/entity_attention_128x2_94695_vs_r20_greedy_20261021/README.md)得54.400%，95%配对区间[52.299%,56.525%]；未测搜索。模型128×2／4头／FFN512、583682参数；原270557步训练计划未完成，本次未改配置或中断训练。
- [当前完整实体MLP的8万局离线训练](artifacts/entity_set_mlp_256x4_offline_80k_20261021/README.md)已停止并保留97000步；94695步完整留出KL/MSE为0.530577/0.782832，raw R20同留出为0.445110/0.755265。随后单独授权的[1000pair greedy](artifacts/entity_set_mlp_94695_vs_r20_greedy_20261021/README.md)已完成，MLP得37.075%，95%配对区间[35.0%,39.175%]；没有恢复训练或执行搜索对战。

- [实体汇总＋256宽、4残差块MLP](artifacts/entity_set_mlp_256x4_20261021/README.md)当前预设已加入64维／2头动作cross-attention，共906,434参数；读取25个未汇总实体，保留动作直连、非线性旁路和乘积的独立开关，value-only跳过新增分支。197项通过、1项CUDA跳过，CPU编译／原生短验证见[cross升级](artifacts/entity_set_mlp_256x4_20261021/action_cross_64x2_20261022/README.md)；无新正式训练／棋力结论。[S1024N1对Greedy同开局1000pair](artifacts/entity_attention_r20_s1024n1_vs_greedy_20261021/README.md)独立后台运行，用于比较已完成S256N4的77.20%得分。

- [固定配对搜索评测](docs/guides/evaluation_zh-CN.md)新正式面板从500pair起步；本次启动前持续迭代已完成20轮，搜索使用12 worker／1000活跃局／B500及owner单槽（采集仍双槽），保留全程S1024与评测裁决。旧200pair结果保留，未自动补跑。
- 2026-10-21的12 worker评测入口也支持搜索对greedy：greedy共享GPU owner批量前向根logit、零搜索。[第20轮S256N4对Greedy](artifacts/entity_attention_r20_s256n4_vs_greedy_20261021/README.md)已准备1000pair面板，256×4为总1024次模拟，使用当前12贵族规则；状态以该实验runtime及最终summary为准。
- [第17–19轮EMA回溯测试](artifacts/entity_attention_ema_17_19_20261020/README.md)已后台启动：各用原轮内留出统一预选250／500／1000步半衰期，再串行测第19轮EMA对同轮raw的greedy1000pair及搜索500pair。稀疏checkpoint EMA近似，原训练／采集配置不变，尚无完成棋力结论。
- [六轮固定留出回补](artifacts/entity_attention_random_budget_iteration_20261019/fixed_holdouts_20261020/results.md)：第6–11轮value MSE确认改善，第12–17轮最终变化区间跨0，但第12–16轮仍有小幅改善、第17轮末尾明显反复；固定旧teacher的policy KL持平／变差。只做历史标签前向评估，不代表当前棋力结论。
- [搜索预算随机化、Forced Playouts与Policy Target Pruning](docs/guides/search_training_zh-CN.md)已接入采集及在线训练，默认关闭；新数据可区分低预算value样本与高预算policy样本，保留原始/裁剪访问量供复核。仅做工程验证，尚无棋力收益结论。
- [新版128×2无辅助头从零训练](artifacts/entity_attention_128x2_relations_no_aux_r24_r43_20261019/results.md)已完成68041步，按完整留出KL＋MSE选中37423步；对原R43的greedy1000pair得分53.775%，S1024/N1搜索200pair为56.625% [51.750%,61.375%]。限定本训练seed、对手和面板，未独立隔离关系特征、value读取或训练日程的贡献。
- [原小模型辅助目标消融](artifacts/entity_attention_r24_r43_aux_ablation_20261019/results.md)已完成。无辅助／仅回合／旧双辅助的greedy为53.850%／53.150%／53.900%；旧双辅助减无辅助的配对差+0.050pp，区间跨0，未确认辅助头的额外收益。两组新模型搜索收益亦未确认。
- [R43历史池离线](artifacts/entity_attention_r43_history_pool_20261012/results.md)已完成，选中节点独立500pair为50.50% [47.35%,53.70%]，未确认超过原R43。[256×4双辅助分支](artifacts/entity_attention_256x4_aux_r24_r43_20261014/README.md)中断于约4.44万步，终点与搜索评测未完成；中期结果不能代表整个扩容方案无效。
- 较早的[P4九轮自迭代](docs/reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)、[截至R43的在线主线](artifacts/entity_attention_r9_online_lr2e5_1000_forever_20261011/results.md)、[M4诊断](docs/reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md)分别保留其版本和证据范围。搜索预算、value信号与动作排序诊断从[实验导航](artifacts/index.md)进入。

## 环境与使用

从仓库根目录执行，使用 Conda `splendor`，设置 `PYTHONPATH=src`。依赖见 `pyproject.toml`，C++ 扩展构建见[原生后端指南](docs/guides/native_search_zh-CN.md)。

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q tests/test_rules_and_phases.py
conda run -n splendor env PYTHONPATH=src python -m splendor.evaluate \
  --player-a score --player-b development --seed-start 0 --seed-count 100
conda run -n splendor env PYTHONPATH=src python -m splendor.web_app
```

网页默认地址为 `http://127.0.0.1:8765`。支持固定策略、greedy、固定深度搜索及原生MCTS；模型和预算、评估条、悔棋及第24轮模型命令见[评测与网页指南](docs/guides/evaluation_zh-CN.md)。网页模型使用CPU单线程，浏览器只接收当前玩家视图和合法动作。

BGA 双人基础版可用[只读招法推荐器](docs/guides/bga_advisor_zh-CN.md)直接连接已有 Chrome、自动识别牌桌，任意轮次接入并输出双方行动／翻牌后的模型胜率估计；支持主行动、支付组合和弃筹码推荐，默认原生MCTS S16384/N1，也可用 greedy，用户自行执行。首次接入时若已在弃筹码阶段且没有本手记录，需手动完成该阶段；也支持离线局面推荐。

训练入口见[AZ循环](docs/guides/alphazero_zh-CN.md)、[离线学习](docs/guides/entity_attention_offline_zh-CN.md)和[PPO协议](docs/specs/ppo_training_protocol_zh-CN.md)。checkpoint和训练NPZ不随Git分发；只读复核需要对应本地或已挂载归档；恢复训练前先按迁移清单撤回该run的冷文件，避免写入冻结归档。

## 仓库与归档

| 目录 | 内容与入口 |
| --- | --- |
| `src/splendor/`、`tests/`、`data/` | 实现、回归、卡牌与常量 |
| `docs/specs/`、`docs/guides/`、`docs/reports/` | 现行契约、操作说明、限定版本的专题结论；[导航](docs/index.md) |
| `docs/archive/`、`configs/archive/` | 被替代的设计与旧准备说明；[历史导航](docs/archive/README.md) |
| `configs/`、`scripts/` | [配置导航](configs/README.md)、[工具导航](scripts/README.md) |
| `artifacts/` | 长期实验交付物、必要原始统计；[产物约定](artifacts/README.md) |
| `runs/`、`runtime/` | 训练数据、必要模型、运行状态及本地保全；Git忽略 |
| `archive/` | [冷产物登记](archive/README.md)；本次[7.03GiB集中目录](archive/20261019/README.md)及[迁移核验](artifacts/repository_archive_20261019/README.md) |

2026-10-19已将7个离线run的334个中间checkpoint及27个账本／逐步指标文件集中到 `archive/20261019/`，原路径保留相对链接；24个关键模型和两份共72000局训练池留在原处。同盘移动尚未释放磁盘空间，外置时应移动该归档的真实目录内容。20261012旧归档已由用户移往其他位置，当前未挂载、外置位置未记录；原[迁移清单与结论索引](artifacts/repository_archive_20261012/README.md)继续保留。
