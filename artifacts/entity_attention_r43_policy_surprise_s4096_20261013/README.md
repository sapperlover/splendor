# R43：固定 4k S4096 的 Policy Surprise Weighting

用户指定：只使用现有4000局S4096/N1，policy与value使用同一固定池。从**原R43**（72898步）出发，各做1000步加权／不加权对照；原AdamW、Torch与采样RNG、lr2e-5、B256、统一clip1和loss系数1:1保留。没有新采集、重新搜索或改value结构。数据来自原S4096分支第44–47代，教师是R43–R46，不能称为统一由R43重新标注的数据。

2026-10-13已完成，结果入口：[results.md](results.md)、[summary.json](summary.json)。逐代来源、原整局Train/Validation划分、checkpoint哈希与种子见[配置](../../configs/entity_attention_v1/r43_policy_surprise_s4096_fixed4k_20261013.json)及`plan.json`。重型NPZ和checkpoint仍位于`runs/`。

## 本轮结论

- 加权确实生效：Train上surprise最高的10%局面获得22.73%的policy权重，均值1，最大8.61；有效样本量约为原policy局面数的76%。没有新增搜索成本。
- 普通留出KL：原R43为0.590099，不加权1000步为0.568885，加权为0.588634。加权相对对照差`+0.019749`，整局配对95% CI `[+0.018035,+0.021502]`，说明平均拟合明显较差。
- 同一固定surprise指标：留出加权KL从0.948990降至对照0.915460／加权0.898535。加权相对对照`−0.016925 [−0.020258,−0.013666]`。高surprise训练重点学进去了，但与普通平均拟合存在权衡。
- 原1:1联合留出loss（普通CE+MSE）为R43 2.348805／对照2.328098／加权2.345362；共同surprise CE+MSE为2.676659／2.643637／2.624227。直接比较两条分支日志中各自不同权重的训练loss没有意义。
- Value MSE为R43 0.721888／对照0.722396／加权0.719911。加权比对照略低，但相对原R43差值区间跨0，尚无净改善证据。
- Greedy对P4V4：R43 82.25%／对照80.875%／加权81.625%；加权−对照`+0.750pp [−3.875,+5.375]`，未确认棋力收益。没有测试带搜索棋力或长期自迭代，本轮不足以支持把0.5默认开启。

两个分支初始模型、AdamW和全部保存的RNG已与R43逐项精确核对；1000个batch、共256000次抽样完全一致，Validation无混入。原3610局Train共294310局面、390局Validation共31787局面。56项相关回归通过，另补验加权拟合诊断与原pool流程。每支1000步训练含节点诊断约39／68秒；加载诊断、首次缓存编码和greedy另计，不能把该差值视为稳定吞吐benchmark。

## 算法与接口

设归档搜索目标为q，加根噪声前的归档教师网络策略为p，`s=KL(q || p)`。对于每局原policy权重大于0的局面，`w=0.5+0.5*s/mean_game(s)`，保留该局policy总权重；forced move仍为0；全局面surprise为0的局保留原权重。概率先用float64重归一化，零prior取1e-100以计算有限KL。没有权重截断。一般形式的比例由`fraction`控制，默认为0.5。

训练依旧均匀抽样；只将policy CE替换为`sum(w*CE)/sum(w)`，value的样本、权重、标签和loss系数不变。共享encoder仍正常接收两个头的梯度，因此value预测和棋力可能受policy改动影响。归档NPZ不变，权重在训练时确定性计算并缓存。配置进入replay/checkpoint身份，恢复时不能切换加权目标。

参考[KataGo权重分配实现](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/cpp/program/play.cpp#L2092-L2170)、[KL计算](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/cpp/search/searchresults.cpp#L631-L690)和[0.5配置](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/cpp/configs/training/selfplay1.cfg)。这是本项目的policy-only适配：KataGo调整数据行权重，可能同时影响多头；其当前KL实现读取`getPolicyProbsMaybeNoised()`，本项目明确使用`root_network_prior`，避免直接把根噪声当作原网络策略。没有移植其廉价搜索、重分析或value surprise分支。

常规online-only联合训练可在hybrid配置加入：

```json
"policy_surprise_weighting": {"fraction": 0.5}
```

支持相同池或既有policy/value不同窗口；当前配置要求`online_only=true`、无split/detach训练、无policy averaging，避免把原单搜索KL与重分析目标混用。省略字段保留旧序列化和行为。常规训练的固定诊断仍默认不加权；若需同时报告固定surprise指标，在`fitting_diagnostics`中也加入同一字段。实验中对两条分支统一启用了该附加诊断。

## 重现

仓库根目录，使用conda环境splendor：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python -u artifacts/entity_attention_r43_policy_surprise_s4096_20261013/run.py
```

同一命令可恢复已提交checkpoint；不会重新采集。固定Train每代2048局面沿用已有引用，Validation为全部原留出整局，诊断节点0/100/250/500/1000。普通CE/KL/MSE与surprise CE/KL并列；对照与处理分支用同一套指标，不能直接比较两条分支各自不同定义的训练loss。

Greedy仅复用原R43对P4V4的200对开局DEV面板，检查policy端的初步变化；不运行完整搜索棋力赛，不用中间节点挑选教师。逐步抽样记录与终点RNG核对后，保存公共引用NPZ及紧凑学习曲线。执行环境、Git HEAD、dirty源码身份保存于`execution_sessions/`，完整源码快照和运行日志在`runtime/`。

## 下一阶段：value辅助头

本轮不增加辅助头，以便单独判断policy加权。现有归档已有`terminal_result.scores`、`terminal_result.completed_turns`和逐局面的`player_id`、`completed_player_turns`。无需重新采集即可生成：

- 分差：`scores[player_id] - scores[1-player_id]`，当前玩家视角。保留原终局胜负头；分差不能完全替代胜负，因为同分时还有购卡数判定。
- 距终局：`sum(terminal_result.completed_turns) - sample.completed_player_turns`，单位为剩余**玩家行动回合**，不是decision条数；归还宝石／选择贵族等子阶段不会额外增加回合。

建议下一阶段显式增加两个辅助预测头，先加载现有encoder及policy/value参数，用固定数据做离线适配，再接回自迭代。需定义标签缩放、辅助loss系数与新增参数的optimizer迁移，避免原始分差／回合数尺度使共享encoder梯度失衡。离线适配是建议的稳妥流程，并非数学上必须另采数据或先完全训练新模型；新头也可直接随在线训练，但会同时引入随机新头与新目标，难以归因。本次未实现或验证这部分。
