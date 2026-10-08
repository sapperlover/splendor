# Splendor 双人 AI

双人基础版规则引擎、Deep-Sets 与实体注意力 Actor–Critic、PPO、冻结网络搜索和 AlphaZero 式训练循环，含 C++ 原生后端与本地人机网页。玩法仅实现双人基础版；不含 3–4 人、扩展或 Duel。

- [当前状态](docs/current_project_state_zh-CN.md) 与 [文档导航](docs/index.md)。
- [标准规则](docs/splendor_standard_rules_zh-CN.md)；卡牌、贵族与常量位于 `data/`。
- [实验索引](artifacts/experiment_index.json)、[产物约定](artifacts/README.md)、[配置导航](configs/README.md)。
- [采集、learner 与固定评测时间统计](docs/reports/pipeline_profile_zh-CN.md)：有限诊断、瓶颈归因与复现命令。
- [端到端性能优化与完整验收](docs/reports/end_to_end_speed_optimization_zh-CN.md)：两组各1000局采集、各50 pairs评测、固定replay cache对照；实际加速与剩余瓶颈。
- [旧θ5起点10×1000配置与记录](docs/guides/next_az_training_zh-CN.md)：原准备说明保留；十轮现已完成，其归档用作新模型离线数据源。
- [实体注意力与离线训练](docs/guides/entity_attention_offline_zh-CN.md)：已完成12045步及六个PPO搜索/greedy面板，完整离线final搜索得分59.5%；重复DEV结果和版本范围见专题。
- [新模型采集执行优化](docs/reports/entity_attention_execution_speed_zh-CN.md)：固定地址CUDA Graph、独立双缓冲及供给控制的有限GPU对照。
- [新模型10×1000自迭代](docs/guides/entity_attention_self_iteration_zh-CN.md)：同时调度1000局，每轮对固定起点搜索/greedy各50 pairs；有限验证通过，用户已运行；现有run保留冻结配置。
- [从5000步接入连续自迭代](docs/guides/entity_attention_hybrid_zh-CN.md)：原run已完成9轮、9500步和有效棋力面板；[第九轮N2分支](docs/guides/entity_attention_hybrid_fork_zh-CN.md)支持继承优化器/RNG、100局/100步及旧数据逐步替换，S512/N2曾在18新轮/11300步后RAM OOM，退出清理已修复；2026-10-06本次检查至少已完成33新轮/12800步，最新棋力面板为新轮30。
- [从原第九轮恢复S1024/N1小步更新](docs/guides/entity_attention_hybrid_s1024_n1_zh-CN.md)：200局/100步、3000局窗口，用户完成50新轮/14500步后停止；[共同PPO补测](artifacts/entity_attention_strong_anchor_20261006/README.md)测得原第9轮63.75%、第50轮61.75%，未确认后续增益或退化。原RAM缓存修复保留；旧[最新单轮份额方案](docs/guides/entity_attention_fresh_share_zh-CN.md)保留为历史候选。
- [固定数据 policy/value 消融](artifacts/entity_attention_policy_value_ablation_20261006/README.md)：三个主分支各500步、greedy/MCTS/固定value共17面板2700局有效。最新175训练局上，联合500步搜索明显退化；保留其policy、换回完整起点value后恢复。未确认policy泛化增益，历史窗口与长期分离方案仍待验证；正式训练未恢复。

- [分头窗口与detach value续训](docs/guides/entity_attention_split_training_zh-CN.md)：第50轮/14500步fork，policy近4轮、value3000局且近4轮占50%，各batch256、独立clip1；200局/100步、S1024/N1。2026-10-07已完成20新轮/16500步，长期收益未确认。
- [冻结encoder、只练critic](artifacts/entity_attention_critic_only_replay_20261008/README.md)：从16550原样重放B2的50步value批次；完整15代3000局，整体留出MSE 0.802914→0.789075，正式16600为0.808967。固定表征下value能够学习，支持共享表征变化干扰拟合；最新B2的12局留出改善未确认，未测棋力。
- [16500／16600 encoder的critic重新拟合](artifacts/entity_attention_encoder_critic_refit_20261008/README.md)：冻结两版encoder，用相同新critic、空AdamW状态、相同15代数据与500个batch记录0/50/200/500步；500步留出MSE 0.792473／0.792415，差值区间跨0，未发现16600整体可拟合性明显下降。仍未测出绝对拟合上限或验证棋力。
- [原16600 critic适配与棋力检验准备](artifacts/entity_attention_critic_adaptation_strength_20261008/README.md)：固定policy/encoder、保留原critic及AdamW，额外0/50/200/500步的完整留出MSE 0.808967／0.789053／0.787500／0.786814。四个模型已完成；独立确认与前中期／后期critic替换的可恢复对战脚本已准备，按用户要求尚未启动搜索。
- [连续步数、学习率与policy窗口实验](docs/guides/entity_attention_iteration_suite_zh-CN.md)：16500步共同起点，按B→C→D顺序运行三组各seed1×16轮；2/4/8/12/16轮统一测冻结起点100pair。一条命令启动/恢复，保留16GiB内存上限；已完成B前两轮及首次评测51.25%（44.5%–58.0%），尚未确认净收益，C/D未运行。
- [只搜后期／只搜前中期](artifacts/entity_attention_stage_search_20261008/README.md)：冻结16500步同权重，以任一方公开分数≥10切换，两组各500pair对同权重greedy；独立连续/可恢复入口，active200＋CUDA Graph，不设RAM硬上限。两组500pair已完成：后期67.0%、前中期69.35%，差值区间跨0；新增全程搜索500pair为81.8% [79.3%,84.3%]，两项配对差值均大于0；本次耗时仅比前中期组多5.1%。
- [搜索目标噪声四组实验](artifacts/entity_attention_target_noise_20261007/README.md)：冻结第20新轮模型，200个留出局面各重复8次；世界抽样的目标JS约为根噪声的2倍，尚未证明其是训练平台期主因或新搜索算法更强。
- [随机节点搜索与同预算复测](artifacts/entity_attention_chance_search_20261007/README.md)：新增可选`native_chance`及评测入口；C组JS下降90.8%、D下降63.8%，耗时1.52倍、搜索更浅。随后[同模型100 pair棋力对照](artifacts/entity_attention_chance_strength_20261007/README.md)新搜索仅31.25%（95%区间24.5%–38.5%，200局有效），稳定性收益未转化为当前棋力；正式训练教师仍用旧native。
- [Chance progressive widening有限诊断](artifacts/entity_attention_chance_pw_20261007/README.md)：可选扩宽已接通，45项回归及六局smoke通过。新增[k0.25/α0.5的6400次稳定性复测](artifacts/entity_attention_chance_pw_noise_20261007/README.md)：D组深度2.560→3.401，但JS0.03072→0.04835，仍比旧确定化稳定。[同模型100 pair棋力](artifacts/entity_attention_chance_pw_strength_20261007/README.md)为37.0%（95%区间30.75%–43.25%），仍弱于旧确定化；[对同权重greedy的100 pair](artifacts/entity_attention_chance_pw_greedy_20261007/README.md)为145胜1和54负、得分72.75%（66.5%–78.5%），搜索收益明确。混合greedy/native批量评测已接通、35项相关回归通过。当次棋力对照没有训练，后续有限短训见下一项；正式教师未切换，大S后续已取消。

- [冻结教师采集与短训对照](artifacts/entity_attention_chance_training_probe_20261007/README.md)：同一16500步起点，旧确定化／完全随机节点／CPW各200局＋500步，detach value与独立clip1；16主面板1600局＋独立复测400局均完成，未确认新教师训练棋力收益；CPW100初筛优势复测未复现（49.75%对50%，差值区间跨0）。记录完整留出、梯度、真实抽样覆盖、value参数块漂移和采集成本；100/500步差异的分层诊断与系列整合见[专题报告](docs/reports/entity_attention_search_teachers_zh-CN.md)。
- [多搜索平均标签对照](artifacts/entity_attention_target_averaging_20261007/README.md)：同一200局native数据，四次完整S1024/N1取平均；保持抽样账本、value输入和优化配置，对独立八次均值的KL比单次标签低0.0189/0.0204/0.0176，修订的77局聚类区间均低于0。不同学生权重、相同旧搜索的直接对打60.5%（区间54.25%–66.75%），尚未确认超过冻结起点。标签预算S4096，另加原采集后本次总模拟量约5倍；正式入口未接入。后续平均标签、自迭代观察与value分析见[报告第九节](docs/reports/entity_attention_search_teachers_zh-CN.md#九如何改善当前自迭代先改变标签再判断剩余限制)。
- [200局短训剂量对照](artifacts/entity_attention_step_dose_20261007/README.md)：同一起点、同一batch账本下对照步数（25–500）、学习率（1e-4/2.5e-5）、value分支（detach/冻结critic）与标签（avg4/single）。棋力峰值在50步附近（avg4 56.5%、single 58.0%），100步回落（47%/45%）；lr2.5e-5@100为54.0%；冻结critic与detach无差异；拟合最优与棋力最优不一致。200局/100步/lr1e-4处于最差区间，优先降步数或学习率；自迭代启动仍未证明。

PPO seed1 已完成 5M learning decisions；其 DEV best 为 `local_004150000_lifetime_004662112.pt`，latest 为 `local_005000000_lifetime_005512112.pt`。AZ seed1 已完成 5 轮 × 500 局及 θ1–θ5 扩展对照。native CUDA pool 已实现并保存性能对照；窗口与整轮收益分别解读。具体结论和未验证范围见文档导航中的报告。

## 环境与验证

从仓库根目录执行，使用 Conda `splendor` 并设置 `PYTHONPATH=src`。依赖见 `pyproject.toml`；native 扩展构建见 [指南](docs/guides/native_search_zh-CN.md)。

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q tests/test_rules_and_phases.py
conda run -n splendor env PYTHONPATH=src python -m splendor.evaluate \
  --player-a score --player-b development --seed-start 0 --seed-count 100
```

不以旧报告的测试数量代表当前工作树。训练与搜索命令见 [AZ 指南](docs/guides/alphazero_zh-CN.md)、[PPO 协议](docs/specs/ppo_training_protocol_zh-CN.md)、[评测指南](docs/guides/evaluation_zh-CN.md)。checkpoint 和训练 NPZ 不随 Git 分发，需要另行备份或提供。

## 本地人机网页

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.web_app
```

浏览器打开 `http://127.0.0.1:8765`。支持 Score/Development、`play_checkpoints/` 中的 PPO greedy，以及启动时指定的固定深度搜索对手：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.web_app \
  --search-checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --search-depth 6 --search-top-k 8 --search-determinizations 1 \
  --search-seed 9200000 --search-leaf-value-clip 1.0 --search-fold-forced-phases
```

网页模型使用 CPU；浏览器只接收当前玩家视图和合法动作。规则仍在服务端引擎执行。

## 仓库分工

| 目录 | 用途 |
| --- | --- |
| `src/splendor/`、`tests/`、`data/` | 源码、回归、规范数据 |
| `docs/specs/`、`docs/guides/`、`docs/reports/` | 契约、操作、可复核结论 |
| `configs/`、`scripts/` | 稳定配置入口与运行/分析工具 |
| `artifacts/` | 长期实验交付物；相邻 `*.games/` 忽略 |
| `runs/` | 训练 NPZ、checkpoint、TensorBoard，忽略 |
| `runtime/` | 日志、运行状态、失败现场、生成配置及历史归档，忽略 |

旧工作包/计划原文已完整本地打包到 `runtime/document_archive/2026-10-01/`，不继续留在 docs。原实验与恢复目录保持原位，失效测量有哈希清单和替代依据。
