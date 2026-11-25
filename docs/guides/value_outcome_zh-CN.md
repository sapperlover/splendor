# 归档搜索根价值与 λ-return 训练

2026-11-24存储更新：[MC／λ普通模型及已完成λ主日志已归档](../../artifacts/repository_archive_20261124/README.md)。74个5000步普通保存点和λ的逐步主日志使用前需恢复；λ161924、MC70842、全部完整验证节点、近期60k、标签及最新λ在线输入保持本机。MC未完成，主日志和172044步恢复点仍在原位。旧104k池的Train已外置，与本页近期60k没有交集。

## 在线混合预算自迭代

在 `splendor.random_budget_iteration` 的配置中设置 `outcome_lambda: 0.8`，启用标量 λ 训练。每一轮先用冻结教师完成采集，再为进入活动回放的 Train 整局从其原始 V3 搜索 W/N 计算标签和掩码。计算规则与下文离线实验相同。高预算访问分布继续监督 policy，低预算局面也参与 value 抽样。留出指标使用真实终局，下轮教师取本轮 final。

每个采集来源保存一份不可变标签。首轮从保留的历史局各自的原搜索记录生成标签，之后仅为新进池的局生成，历史局随回放淘汰退出抽样。带 FPU 的新局按采集时记录的 W/N 计算。另一个 checkpoint 也可以作为任意新分支起点，配置需提供原始 V3 数据及无辅助头的标量 `entity_attention_v1`。省略 `outcome_lambda` 或设为 `null` 使用终局 MC。

- `lambda_targets/<来源>/manifest.json` 记录教师 SHA、采集搜索配置、训练局索引和两个 `.npy` 的 SHA。原局档案是标签来源，旧 V3 省略的 `fpu_reduction` 按关闭解释。
- `rounds/round-NNN/lambda_targets.json` 固定活动 Train 池和标签文件，记录整体及各来源覆盖率。训练 checkpoint 绑定该清单摘要，恢复时核对并复用原标签。逐步 `lambda_train_mse` 是实际 value 训练目标误差，`value_loss` 仍记录真实终局误差。

### λ161924 与 FPU 的 60k 分支

启动器使用 [固定配置](../../configs/entity_attention_v1/lambda161924_fpu_p60k_iteration_20261124.json)，从 161924 步继承模型、AdamW 和 RNG。初始历史为近期 60k 局，每轮采 10k、回放容量 60k、训练 10,000 步，学习率 `2e-5`。`historical_policy_fraction=1.0` 让全部有效历史 policy 局面可抽样，首轮历史与新局配额仍按 50k 与 10k 的局数分配。在原轮内诊断节点之外，10,000 步也做轮末验证。采集使用 `r=0.3, c_puct=1.0`、75% S128N1（总 128）、25% S256N4（总 1024）。轮间双方评测使用相同搜索参数，每轮对原 R37 greedy 1000 pair，每三轮用配置 A 做双方 S1024N1（各总 1024）500 pair。

从仓库根目录执行。`--prepare` 冻结源码、native 扩展、牌池和配置，`--check` 只做预检。正式执行可先设绝对轮数 `--rounds 1`，同一命令恢复。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/run_lambda_online_iteration_20261124.py --prepare
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/run_lambda_online_iteration_20261124.py --check
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/run_lambda_online_iteration_20261124.py --rounds 1
```

当前执行发布位于 `runtime/lambda161924_fpu_p60k_iteration_20261124/source_10000_steps/`，正式轮次及标签写入同名 `runs/` 目录。原 `source/` 对应 25% 历史 policy 和 3,000 步的准备版。

正式配置预检通过。原发布的 native CPU 工程验证从 λ161924 起步，采集两轮共 8 局并更新 4 步，确认首轮中断恢复、次轮教师身份、回放淘汰后的标签复用以及旧 V3 归档读取。正式 GPU 采集与训练尚未启动。

## 离线标量对照

2026-11-20 的准备范围包括标量 MC 与归档搜索 λ 两组，WDL、WDL＋λ暂时搁置。离线 λ 随后完成 202404 步并选中 161924 步。固定数据、命令和原准备记录见[60k 专题](../../artifacts/value_outcome_60k_20261120/README.md)。

## 当前标量λ目标

第二个脚本`run_outcome_60k_scalar_lambda_20261120.sh`使用`outcome_lambda_source="archive_search"`：读取原Train NPZ里的搜索统计，一次性生成标签与掩码，再从随机初始化训练。无需模型教师、网络前向或重新搜索；不依赖第9轮checkpoint，也不预热或周期刷新标签。主policy沿用原历史搜索分布CE，主value沿用原标量头，λ=0.8。

每个已记录主行动根i，从原始`per_world_root_value_sums`与`per_world_root_visit_counts`还原：

```text
root_i = mean_over_valid_worlds(sum_over_actions(W_iw) / sum_over_actions(N_iw))
```

先在每个世界内求ΣW/ΣN，再对有效世界等权平均；不取最大动作Q，不用剪枝后的policy计数，不按世界访问量再次加权。复用旧短期value辅助的根值还原函数；保留归档原本的混合预算与根噪声，不声称重分析为统一预算或无噪声标签。

对玩家p当前状态i，j为轨迹中严格晚于i、同一玩家的下一次MAIN_ACTION：

```text
g_i = z_p                                # 无后续自己的主行动
g_i = (1-lambda)*root_j + lambda*g_j       # 所需统计完整时
```

支付、退币、选贵族子状态也可监督，但不增加λ时钟。只用同一玩家的信息视角，不将对手私有视角的Q简单取负。终局使用归档真实胜负z（胜1、和0、负−1）；自然终局之外的未结束片段不接受。每个玩家最后一次决策的尾部可直接用真实终局。

## 缺失统计与指标

无搜索的单候选主行动可能没有有效根值。λ目标依赖的后续根值缺失时，该位置及同一玩家更早依赖它的目标被mask；不会用0、裸V或真实终局补造一段λ监督。数组中无效位置只存放有限的终局占位值，训练权重为0。λ=0仅需下一根有效，λ=1完全退化为真实终局MC，不受缺根影响。

- λ损失使用原value权重×mask，并以有效权重归一化；整批无有效λ标签时，λ损失为0、policy梯度继续，`lambda_train_mse=null`。
- 主`value_loss`及`value_sq_sum/value_weight_sum`始终按原权重统计真实终局MSE，包括mask位置。完整6020局validation始终使用真实终局，不读取软标签或训练mask。
- `lambda_train_mse`为实际软标签误差；`outcome_train_weight_sum`、`outcome_train_coverage`记录有效监督量，`outcome_target_source="archive_search"`标识来源。
- 固定相同更新数和value抽样曝光；mask使λ有效监督量略少，不重抽、不改变抽样RNG。全池实际覆盖率由生成时的manifest报告，不能用小样本覆盖率代替。

## 生成、供数与恢复

`targets`阶段只做CPU逐局读取与标签构造，无学生、优化器或CUDA模型。生成只读取Train原始NPZ的必要数组，每次保留一局临时数据；不将W/N放入训练共享池。

`lambda_targets/archived_search/`保存唯一版本：

- `targets.npy`：float32标量，当前Train池约16.5MiB。
- `mask.npy`：bool，约4.1MiB。
- `manifest.json`：固定数据digest、game顺序、λ、世界合并／时钟／缺失规则、两个内容SHA、覆盖率、预算分布和生成耗时。

以身份文件、partial目录和原子重命名发布；只允许同身份未发布partial重建。已发布标签或mask损坏明确失败，不静默重算。恢复继续使用同一对只读mmap，不更新标签。

2026-11-20补齐终点收尾恢复：终点checkpoint已经保存时，不再把剩余0步传给训练循环，也不重新预热训练共享池或生成标签。终点全量验证已经提交时，在CPU核对checkpoint训练身份、步数、配置及验证SHA／数据digest／位置数，然后原子发布`final.pt`并写`complete`；没有验证结果时，仅恢复模型补做终点全量验证后收尾。训练权重、Adam和采样状态不再更新，随后原脚本正常自动选点。验证或checkpoint身份不符时仍明确失败。

训练复用4 worker、Linux memfd共享只读完整压缩池、流式会话累计、state-only value传输和批量标量回读。10GiB共享容量依现有逐局核算约8.80GiB含对齐上界配置；标签／mask由父进程另外映射，不复制给worker。缓存容量不是进程树RSS上限。

每组固定12次value抽样曝光／202404更新，lr1e-4、policy/value各B256；每5%完整留出，统一按policy KL＋真实终局MSE选点，不独立早停。checkpoint每5000步并额外保留5%验证点。只写主metrics、轻量引用SHA与恢复元数据，不写抽样bin或重复steps日志。

正式棋力评测仅在显式`evaluate`阶段：对固定raw R37，greedy1000pair，再双方S128N1（每世界128×1，总128）配置A500pair；同种子交换先后手、无根噪声、T0。工程smoke不作为棋力结论。

## 暂时搁置的WDL路径

`entity_attention_wdl_v1`保留胜／平／负三类logits、交叉熵和推理标量`p(win)-p(loss)`。WDL-MC与固定WDL教师的向量λ实现及脚本仍在，当前不列入执行计划、不自动接续。未完成实验不能据此判断WDL没有收益。

[Ataraxos v2](https://arxiv.org/abs/2511.07312v2)S3.4的结果标签取采集时网络θₜ对同一玩家后续状态的WDL预测与终局one-hot，λ=0.8；每批走子数据只训练1个epoch。本文当前的归档搜索标量λ是结合本项目搜索自对弈数据作的改造，并非原论文的直接复现。旧的“第9轮裸V教师”及更早的“MC预热／阶段刷新”准备方案均已替换，此前没有正式训练结果。
