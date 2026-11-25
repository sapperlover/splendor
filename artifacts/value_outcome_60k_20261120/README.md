# 近期60k池：标量MC vs 归档搜索根价值＋λ

2026-11-24存储更新：[本批T7归档](../repository_archive_20261124/README.md)包含MC的34个和λ的40个普通checkpoint，以及已完成λ的 `training_metrics.jsonl`。完整验证节点、λ161924／MC70842、initial/final和恢复点保留；近期60k、固定留出、标签及MC主日志不动。读取已外置模型／逐步曲线前需恢复；本批不改变训练状态和选点结论。

2026-11-20。当前执行计划仅两组：原标量MC基线、归档搜索根价值＋λ；WDL与WDL＋λ实现／脚本保留，暂时搁置。正式训练、全池标签生成和棋力评测均未启动。目标、掩码和恢复契约见[指南](../../docs/guides/value_outcome_zh-CN.md)。

## 固定数据与训练配置

固定policy在线分支本地第8–13轮，即`round-007..012`、归档`round_index=87..92`，60000局。Train53980局／4317943位置，Validation6020局／481516位置，沿用原整局split。原NPZ、原始搜索统计和终局结果不改写；清单身份见[pool_identity.json](pool_identity.json)。

两组均从随机初始化和新AdamW开始：attention128×2、4头、FFN512、market relations、global/players/market value读取；无辅助头。初始化seed52200000、抽样52210000、验证52220000；lr1e-4、policy/value各B256、weight decay1e-4、clip1、FP32。保留相同抽样顺序、12次value抽样曝光／202404更新、每5%完整留出和统一policy KL＋真实终局MSE选点，不独立早停。

4 worker共享只读完整压缩训练池、流式会话统计、state-only value传输和批量标量回读。共享缓存10GiB，[逐局容量核算](cache_capacity.json)含对齐上界约8.80GiB；父进程λ标签约16.5MiB、mask约4.1MiB，不改变worker共享池布局。

## 第二组的归档搜索λ

`scalar_lambda`预设改为`outcome_lambda_source="archive_search"`，不再依赖第9轮模型。读取Train中原始W/N统计，各世界先ΣW/ΣN、再世界等权平均；用严格晚于当前状态、同一玩家的下一次主行动根值递推λ=0.8，真实终局吸收尾部。保留归档原本的S128N1／S256N4混合预算与噪声，不重新搜索。

单候选等无搜索根值会使依赖它的λ标签无效；显式mask该λ损失，保留policy训练和真实终局MSE。无效位置不使用0、裸V或终局作为替代监督。mask使有效λ曝光略少，日志报告覆盖率，不额外重抽。实际全池覆盖率在首次生成后报告；此前90局抽样不是全池结果。

标签只生成一次、之后固定复用，无MC预热、网络教师或周期刷新。生成阶段仅CPU逐局读取，不创建学生或优化器，不分配训练共享池。学生仍从头训练。

## 命令

从仓库根目录分别运行；以下是交付命令，本次没有执行正式训练：

```bash
bash scripts/run_outcome_60k_scalar_mc_20261120.sh
bash scripts/run_outcome_60k_scalar_lambda_20261120.sh
```

第二个命令会先生成缺失的标签，再训练；相同命令恢复。也可把一次性标签生成单独执行：

```bash
bash scripts/run_outcome_60k_scalar_lambda_20261120.sh targets
```

其他阶段：

```bash
bash scripts/run_outcome_60k_scalar_lambda_20261120.sh prepare
bash scripts/run_outcome_60k_scalar_lambda_20261120.sh train --max-steps-this-session 1000
bash scripts/run_outcome_60k_scalar_lambda_20261120.sh select
bash scripts/run_outcome_60k_scalar_lambda_20261120.sh evaluate
```

`prepare`只冻结数据／配置／源码并生成计划，不生成全池标签或初始化模型。预算完成后默认自动选点；不自动启动棋力赛。两组应串行运行，公共workflow锁拒绝并发；不接入其他实验队列。

2026-11-20修复终点收尾崩溃窗口：若终点checkpoint已保存，重启不再进入零步训练；已有终点验证则核对身份后直接原子发布`final.pt`、写`complete`，缺验证则只补终点全量验证。两种情况均不重新预热训练池或生成λ标签，不改变终点权重／Adam／RNG；恢复成功后原脚本自动选点。覆盖MC与归档搜索λ两组，已提交产物身份不符仍拒绝收尾。

`evaluate`要求训练完成并已选点，依次运行对raw R37的greedy1000pair和双方S128N1搜索A500pair。搜索无根噪声、T0；配置A及共同新种子见[evaluation_templates.json](evaluation_templates.json)，对手SHA见[opponent.json](opponent.json)。

## 产物、冻结与验证

- `runs/outcome_60k_scalar_lambda_20261120/lambda_targets/archived_search/`：唯一版本的`targets.npy`、`mask.npy`和manifest，绑定数据身份、λ、根值还原／时钟／缺失规则及两个SHA，记录覆盖率、预算分布和生成耗时。只从Train构造。
- 标量组保留原`value_loss`真实终局MSE；λ实际软标签MSE另记`lambda_train_mse`，有效监督量记`outcome_train_weight_sum/outcome_train_coverage`。
- [summary.json](summary.json)只将标量两组计入当前完成状态，WDL两组标记deferred。
- [source.json](source.json)绑定`runtime/value_outcome_60k_20261120/source`的实际执行文件SHA；工作区修改不会自动覆盖冻结发布。本次只在确认所有实验仍处于prepare、无checkpoint／训练metrics／全池标签后替换未启动的准备版本。
- [verification.json](verification.json)记录必要回归、真实归档smoke和未启动状态；原始JUnit位于忽略的runtime。历史裸V教师smoke保留其当时身份，不代表当前标签来源。
- 已发布标签或mask损坏会明确失败；只有同身份的未发布partial可重建。恢复保留学生权重、AdamW、抽样与Torch RNG，继续使用同一份标签和mask。

原WDL与WDL＋λ配置／脚本保留，当前不推荐执行，也不自动跟在第二组后启动。WDL＋λ原定依赖WDL-MC的选定模型，该依赖尚未训练；当前没有WDL棋力结论。
