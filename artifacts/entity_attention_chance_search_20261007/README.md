# 随机节点／信息历史搜索：同预算 A/B/C/D 复测

2026-10-07。已实现可选的 `native_chance` 后端，并复用[上次实验](../entity_attention_target_noise_20261007/README.md)的同一16500步冻结模型、200个留出请求、种子和指标代码，完成四组各八次重复，共6400次S1024搜索、6,553,600次模拟。

**目标稳定性明显改善：C组JS下降90.8%，D组下降63.8%；D的最优动作一致率从67.1%升到85.2%。根噪声本身没有明显改善。相同模拟预算下耗时增加51.8%，平均逻辑深度大幅下降；未证明棋力或后续训练收益。**

## 搜索改动与控制口径

旧S1024/N1在每份搜索开始时确定一个世界，包括对手暗保留牌和未来牌堆顺序，1024次模拟在这棵固定世界树中运行。新模式每次simulation重新抽样对手当前暗保留牌；未来市场补牌／盲保留发生时，从该simulation剩余牌池均匀无放回抽样。随机结果按概率出现，并回传到同一个决策动作的N/W/Q，不能通过PUCT选择最好的随机结果。

决策统计按行动玩家自己的观察／行动历史共享，双方各有自己的历史；对手暗牌和未来顺序不进入另一玩家的节点键。公开动作与翻牌进入双方历史，自己的私有牌只进入自己的历史。这是均匀隐藏牌信念下的信息集PUCT近似，没有根据对手真实过往行为拟合后验，也不代表博弈均衡解。

仍标记N=1，在新模式下表示一个共享统计池，**不再表示只采一个固定世界**。S1024仍是总模拟数，没有将预算放大1024倍。模型权重、网络输入、c_puct1.5、alpha0.3/fraction0.25、forced folding、cap512、根噪声及行为温度入口均保持一致。新模式cap按simulation累计实际动作数，上次及本次均没有cap命中，因此没有影响本次比较。

| 组 | 旧模式控制 | 新模式控制 | 根噪声 |
| --- | --- | --- | --- |
| A | 固定一个世界 | 固定整份随机抽样序列的seed | 关闭 |
| B | 固定一个世界 | 固定整份随机抽样序列的seed | 每次变化 |
| C | 每次更换世界seed | 每次更换整份随机抽样序列的seed | 关闭 |
| D | 每次更换世界seed | 每次更换整份随机抽样序列的seed | 每次变化 |

新A/B仍在每次simulation抽样，固定的是完整抽样序列，不是停止环境随机。A是执行可重复性控制，C测无根噪声时的抽样敏感性，D最接近原自对弈监督target的生成条件。四组不能当作完整交叉方差分解。

## 主要指标

以下为200个局面等权平均，单元格均为“旧确定化 → 新随机节点”。JS/KL用自然对数；一致率指同一局面八次搜索的28个重复对之间的一致率，不是对正确答案的准确率。Top-3为交集大小除以`min(3,候选数)`。

| 组 | Pairwise JS | Top-1一致率 | Top-3重合率 | 明显Top-1翻转率 |
| --- | ---: | ---: | ---: | ---: |
| A | 0 → 0 | 100% → 100% | 100% → 100% | 0% → 0% |
| B | 0.02889 → 0.03047 | 87.86% → 86.50% | 80.15% → 76.97% | 1.86% → 2.41% |
| C | 0.05866 → 0.00538 | 73.21% → 92.54% | 86.01% → 94.07% | 11.66% → 0.89% |
| D | 0.08477 → 0.03072 | 67.05% → 85.20% | 73.05% → 77.36% | 13.66% → 2.68% |

明显翻转要求两次各自的Top-1比Top-2均领先至少0.1概率。D中发生过Top-1变化的局面从124/200降至56/200，说明改善并不只是平手候选换序。

按77个来源对局做10,000次配对cluster bootstrap，新减旧的C组JS差为−0.05328，95%区间[−0.06260,−0.04432]；D为−0.05405，区间[−0.06432,−0.04433]。D的Top-1一致率提高18.14个百分点，区间[13.38,22.75]个百分点。B的JS差+0.00158，区间[−0.00436,+0.00702]，没有改善证据；B的Top-3重合率反而下降。区间只描述当前固定模型、局面和八个种子的局面抽样不确定性，不覆盖其他训练seed和所有随机序列。

分层结果见[summary.json](summary.json)：D的新JS早／中／晚为0.02439/0.02919/0.03869，晚期仍较敏感；没有对手当前暗保留牌的121个局面，新D的JS为0.02498。未来牌堆随机性仍是需要处理的因素，不能只关注已有暗牌。

## KL分解：一致性改善与教师质量要区分

令冻结网络prior为p、八次target为π_i、经验均值为π̄：

`mean_i KL(π_i || p) = KL(π̄ || p) + mean_i KL(π_i || π̄)`

| 组 | 单次目标平均KL | 平均目标KL | 重复目标分歧项 | 分歧项占平均KL |
| --- | ---: | ---: | ---: | ---: |
| A | 0.38070 → 0.30988 | 0.38070 → 0.30988 | 0 → 0 | 0% → 0% |
| B | 0.44488 → 0.35467 | 0.38494 → 0.29471 | 0.05994 → 0.05996 | 13.5% → 16.9% |
| C | 0.37350 → 0.31195 | 0.25708 → 0.30236 | 0.11641 → 0.00958 | 31.2% → 3.1% |
| D | 0.44152 → 0.35592 | 0.26987 → 0.29575 | 0.17165 → 0.06017 | 38.9% → 16.9% |

所有6400次新旧搜索的网络prior及加噪后的used prior逐项完全相同，比较没有换网络或噪声向量。D的平均KL下降0.08560，主要来自重复目标分歧项减少0.11148；网络与平均目标的KL反而增加0.02588，95%区间[−0.01456,+0.06975]，没有缩小这一差距的证据。C的平均目标KL增幅区间为[+0.00566,+0.08639]。

所以这次证据支持“降低固定世界抽样造成的监督标签抖动”，不能把更低的单次KL解释为已经学得更好，也不能把16.9%当作所有训练loss的噪声比例。π̄并非正确策略答案，稳定或接近网络都不直接证明棋力。

## 成本与搜索深度

搜索墙钟总耗时为211.10秒 → 320.52秒，即1.518倍。不含编译、输入准备、smoke和后处理，是本次固定200请求native batch诊断的成本，不是整局采集吞吐结论。每个组的200个作业并发执行，不能把逐作业累计total_seconds当墙钟时间。

| 组 | 每次simulation平均逻辑深度 | 命中真实终局的simulation比例 |
| --- | ---: | ---: |
| A | 4.912 → 2.623 | 1.0928% → 0.0039% |
| B | 4.829 → 2.560 | 1.0930% → 0.0041% |
| C | 4.939 → 2.626 | 0.8088% → 0.0054% |
| D | 4.854 → 2.560 | 0.7890% → 0.0050% |

逻辑深度是`tree_depth_sum / simulations`，不计被折叠的forced动作，不等于完整游戏回合数。大量随机结果展开了更多分支，同预算下搜索更浅、终局精确收益更少，结果更依赖叶节点value。这是本次实现的明确代价；稳定性改善不保证会弥补深度损失。完整计数及复算口径见[search_profile.json](search_profile.json)。

当前适合保留新后端，下一项能判断是否用于训练的证据是冻结同一模型的新旧搜索配对棋力比较，分别考虑同S与同墙钟预算。此次没有追加正式棋力赛、模型更新或训练采集；不能据此宣布解决训练平台期。

## 实现、验证与运行范围

实现位于`src/splendor/native_search.cpp`、`native_policy.py`、`evaluation.py`及`evaluation_batch.py`。直接API为`NativeChancePUCTSearchPolicy`，固定评测策略JSON设置`"search_backend": "native_chance"`且N=1，支持串行及native_batched，可与旧native混合对打。API和字段含义见[搜索指南](../../docs/guides/search_zh-CN.md#采样随机节点与信息历史-puctnative_chance)。现有训练采集入口／冻结run配置仍使用旧native，本次实验通过独立适配器切换，不自动改变正式教师。

- 39项相关测试通过，3项既有CUDA测试在沙箱中跳过；其中7项新回归验证无放回抽样／牌币守恒、概率抽样与期望回传、暗牌重采样、私有信息节点边界及逐叶／batch一致性。命令见[tests.json](tests.json)。
- 新后端8局面×2次S64 GPU smoke通过。更新后的旧后端重跑200局面A0/B0，共400次S1024搜索，target、prior、used prior、计数和动作与旧原始结果逐项完全相同，见[baseline_equivalence.json](baseline_equivalence.json)。
- 通过实际固定评测factory/native_batched/checked规则调用链完成两局S8 GPU smoke，零异常／截断／不变量失败；见[evaluation_smoke.json](evaluation_smoke.json)和[配置](evaluation_smoke_config.json)。两局不用于棋力判断。
- 6400次正式诊断全部完成，A重复逐项一致、root预算守恒、控制变量独立、cap命中0，KL恒等式及所有指标复算通过。没有声称验证新后端的CUDA Graph、多worker长采集或lazy chance失败现场的独立重放工具；旧`reproduce_failure`的固定世界分支重放不适用于本模式。

本机构建采用隔离wheel构建，没有安装额外包进conda环境。命令为`conda run --no-capture-output -n splendor env PIP_CACHE_DIR=/tmp/splendor_chance_pip_cache python -m pip wheel --no-deps --wheel-dir runtime/entity_attention_chance_search_20261007/wheels .`，再将wheel内native扩展解压至workspace `src/splendor/`。旧扩展先备份至本次runtime。构建身份见[manifest.json](manifest.json)的build字段；重建方式仍以[原生指南](../../docs/guides/native_search_zh-CN.md)为准，不要求其他机器使用本机ABI文件名。

## 产物和复核

[compare.py](compare.py)导入旧[probe.py](../entity_attention_target_noise_20261007/probe.py)，复用其原始结果结构和指标，仅在本次进程中将策略类换为新后端，不编辑旧实验文件。[manifest.json](manifest.json)记录checkpoint SHA、种子、全部命令、源码快照、编译扩展和环境；[root_bank.json](root_bank.json)与旧实验字节相同，仍来自77个validation对局，早／中／晚67/67/66，最近四轮、每局最多3个主行动请求。

[raw_targets.npz](raw_targets.npz)、[per_root.json](per_root.json)、[summary.json](summary.json)、[comparison.json](comparison.json)保留全部target、访问数、动作、分层与配对区间；[verification.json](verification.json)保存正式搜索控制检查，[delivery_verification.json](delivery_verification.json)保存83个源码快照、C++/setup/data附加快照、wheel/旧扩展、跨后端prior和性能复算审计。临时分片、编译产物及源码快照在忽略的`runtime/entity_attention_chance_search_20261007/`，旧原始产物保持原位。

以下命令只读复核，不执行GPU搜索／训练：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/entity_attention_chance_search_20261007/compare.py verify
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/entity_attention_chance_search_20261007/audit.py
```

首次完整流程命令见manifest：`prepare → smoke → baseline_check → run → analyze → verify`。已完成目录禁止重新prepare/run覆盖输入；`analyze`会重写汇总，`audit.py --write`仅用于首次生成审计。本机runtime、checkpoint及原始训练NPZ不随Git分发，迁移复核时需另行备份。
