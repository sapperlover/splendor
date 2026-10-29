# Attention128：R21–R37从零训练，仅短期value辅助

2026-10-27完成状态核对：训练、完整留出选点及原两组对战均已完成；192804步早停，选中140221步。原raw R37 greedy1000 pair得分53.025%，双方S1024N1（总1024）500 pair为51.400%，见[原结果](results.md)。同一候选的新版10000 pair greedy／S128N1结果见[大面板](../offline5_vs_r37_10k_20261027/README.md)。下方按日期保留首次准备、停训和恢复过程，不再代表当前进度。

2026-10-25存储更新：本实验训练／选点及greedy、搜索两组对战均已完成，结果见[results.md](results.md)。按本次授权将192个普通checkpoint及`training_metrics.jsonl`外置至[归档20261025](../repository_archive_20261025/README.md)，原路径保留恢复链接；选中140221步、全部完整留出节点、initial/final以及配置／源码／结果保留。细查逐步曲线或普通模型须恢复对应文件，继续写入前须恢复为原位真实文件；下文运行进度及旧存储描述保留为历史记录。

2026-10-24，用户已自行恢复本实验；本任务首次核验时监督PID7398／训练PID7467正在运行、最近提交132000步。下方2026-10-23“停止／0步”仅为当时状态。本次不修改或中断当前流程，按用户要求将[三组辅助消融队列](../entity_attention_auxiliary_queue_20261024/README.md)接在其后：完整训练、选点、greedy及搜索均成功结束后才依次启动104k仅policy、104k双辅助、44k仅value。当前启动脚本、运行配置与冻结源码保持原身份。

2026-10-23当时指令改为**不要自动启动，只给命令**。此前启动的本次队列已SIGINT停止，监督／训练／Conda进程均退出；保留initial.pt与所有已有输入／日志，已提交训练步为0，未执行正式对战。未重启。手动运行下方命令将从保留的从零初始化点开始；原启动及本次更正见[manual_stop_20261023.json](manual_stop_20261023.json)。下方原授权启动描述及launch保留事件来源，当前状态以本段为准。

2026-10-23，用户授权写好后启动，完成训练后自动选点并依次对raw R37执行greedy和搜索评测；启动后不持续轮询。当前准备与启动身份分别见`verification.json`、`launch.json`，训练进度以新run原件为准。旧在线／离线run及其他用户进程不修改。

```bash
# 默认：训练 → 完整留出选点 → R37 greedy1000pair → R37搜索500pair
bash scripts/run_attention128_shortterm_value_r21_r37_20261023.sh

# 同一入口支持中断恢复；以下命令不启动工作负载。
bash scripts/run_attention128_shortterm_value_r21_r37_20261023.sh status
```

`prepare`只冻结／核验输入与源码；`train`训练并选点，不运行对战。当前仅交付手动命令，不自动启动。

## 冻结数据与训练

- R21–R31来自原S256N4／12贵族分支，每轮4000局，共44000局；R32–R37来自R31起点10k／60k分支的局部1–6轮，每轮10000局，共60000局。合计**104000局**，不受在线当前60k活动池淘汰影响，均读取原始采集档案。
- NPZ的零基`round_index=63–79`，离线配置的一基范围64–80。`expected_games_per_round`使用一基轮号字符串到局数的完整映射，逐轮检查，不能只以总数相同替代某轮完整性；旧整数配置及其冻结身份保持兼容。
- 保留每局原标签、原路径和整局split，不重划分／重标注。Train93537局／7478428位置／1559755有效policy位置；Validation10463局／835367位置／174056有效policy位置。教师谱系、原index／manifest／summary、牌池指纹和每局size／mtime均已核验，见[plan.json](plan.json)。
- 从零初始化attention128×2／4头／FFN512，保留原market relations和global/players/market value读取；增加H2/H6/H14短期value头，共**650757参数**。不继承R37权重或AdamW；主干初始化seed43100000，辅助seed36100000，抽样seed43110000，固定4096留出seed43120000。
- lr1e-4、AdamW weight decay1e-4、全局clip1、主policy／终局value权重1/1、FP32。policy仅高预算正权重位置和value全部预算位置各独立抽B256，一次联合更新；不沿用在线的历史25%policy子集或新局配额。
- 16GiB按需compact回放缓存；12次value抽样曝光上限**350552步**，对应policy约57.5355次曝光。缓存容量不是进程RSS硬限制。

短期value使用[最新continuation实现](../../docs/guides/continuation_auxiliary_zh-CN.md)：H2/H6/H14平均前视距离的几何EMA，终局吸收剩余尾部；只监督可精确还原的主行动，未搜索／缺回合影响的前缀明确屏蔽，记录有效覆盖率。`shortterm_value_weight=0.25`，`opponent_policy_weight=0`，模型不注册对手policy头、不生成其标签。旧分差／剩余回合、旧短期进展、PSW及已撤回的购买结算特征均关闭。编码器分任务梯度诊断沿用新配置默认第1步及每100步。

完整配置：[offline_shortterm_value_r21_r37_20261023.json](../../configs/entity_attention_v1/offline_shortterm_value_r21_r37_20261023.json)。除池路径／SHA、末轮和逐轮局数映射外，与此前仅短期value离线预设逐字段相同。

## 早停与预先选点

每5%在同一固定4096子集和**全部835367留出位置**评估主policy KL、主终局value MSE及辅助诊断。完整节点为17528、35056、52583、70111……350552步，具体20个节点见plan。

若完整留出`主KL+主MSE`连续三次严格大于上一百分比节点则停止；持平／下降重置，step0不参与比较，最早20%停止。辅助loss及其覆盖率不参与早停或选点。若未触发则到12次曝光上限。

在所有已到达的百分比节点中选取主KL+MSE最低的**整个checkpoint**，同值取较早步；选点及SHA固定后才开始对战。不按对战表现重选、不组合policy／value权重。

## 对战顺序与最新配置

对手固定为原在线分支局部第6轮的raw R37 final，累计86423步，SHA `611f4f4ed18f397b03c79bb3a2beae51e7a77a747883680d9189eed3052a1045`；不自动追踪后续模型。

1. **greedy1000pair／2000局**：CPU8 worker、双方零搜索、根policy确定性argmax。开局37000000–37000999，policy seed37010000，bootstrap seed37020000。
2. **双方S1024N1搜索500pair／1000局**：每世界1024次×1世界，总1024。采用最新配置A：CUDA pool、12 search worker、1000活跃局、B500、microbatch128、owner双槽、B桶`[64,128,256,500]`、C桶`[32,48,64]`、shared slots2、等待5ms、FP32/native-cudagraph、fail-fast。开局37100000–37100499，policy seed37110000，搜索seed37130000，bootstrap seed37120000。

两者均使用当前12贵族、同一开局交换先后手，最多400玩家回合，确定性policy-cycle裁决；搜索无根噪声／温度0、FP/PTP关闭。整pair bootstrap10000次。复用[最新日常greedy／搜索A预设](../../docs/guides/evaluation_zh-CN.md)，不复制采集active2000／旧桶，也不沿用旧owner单槽。开局库为已有DEV复核库，单训练seed；有限工程smoke不作为棋力结论。

## 最新记录形式与产物

采用[最新训练产物保留实现](../../docs/guides/training_artifacts_zh-CN.md)：每步指标写`training_metrics.jsonl`，会话只保留小型参考身份JSON和batch SHA；不写policy/value两份`.bin`、两份`.steps.jsonl`或重复`.metrics.jsonl`。配置、源代码、池／split／留出清单及模型、AdamW、全部RNG继续保留。恢复不依赖大账本；哈希不能反推出逐样本顺序。

离线恢复点沿用此前配置**每1000步**，额外强制保存全部5%节点；initial、final及被选中节点保留。在线分支后来改为500步的周期不自动套用到本离线配置。

- `runs/entity_attention_128x2_shortterm_value_r21_r37_20261023/`：原始池清单、dataset／split／固定留出、模型／优化器／RNG、逐步主指标、百分比完整留出、early_stopping、selection及progress。
- 同名`runtime/`：本次当前源码、native扩展和12贵族规则的冻结快照；后台`chain.log`、阶段日志、live身份及独立GPU smoke。旧实验快照不复用为新训练实现。
- 本目录：runner、smoke、plan／source身份、回归、GPU验证、verification、launch，以及后续自动更新的summary／results／两组评测配置和原始汇总。逐pair恢复sidecar按仓库规则忽略。

[verification.json](verification.json)记录80项冻结源码CPU回归全部通过，以及34局真实源数据、正式B256共6次CUDA更新／第2步恢复、greedy和编译搜索各1pair（合计4局）的工程验证通过；已核对新模型650757参数、无对手policy头和轻量记录。小规模结果不作为棋力结论。启动后只做一次启动核验，不等待训练结束或定期查询。
