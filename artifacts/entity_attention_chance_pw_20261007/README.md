# Chance progressive widening：实现与有限诊断

2026-10-07。用户停止大S深度／棋力试验，转为尝试chance progressive widening（CPW）。本次实现可选CPW，完成45项相关回归、200固定局面诊断、GPU CUDA Graph四局及CPU串行两局checked smoke、两个60秒真实采集窗口。**S1024下，k=0.25、α=0.5使平均逻辑深度由2.623增至3.461（+31.9%），但旧确定化为4.912；真实8-worker等量模拟工作时间约为旧搜索的1.34倍。没有CPW棋力／训练收益结论。**

## 方法与边界

决策动作集合与PUCT不变，只对市场补牌、盲抽等物理chance事件扩宽。对每个决策边下的实际剩余牌池上下文，第N次访问的结果上限为`min(remaining_cards,max(1,ceil(k*N**alpha)))`。未满时从完整牌池抽取并收入新卡牌ID；已满时从已有不同ID均匀选取。Splendor剩余卡牌各自物理概率相等，因此后者是受限支持集上的归一化概率。重复抽到的ID不另占名额，不按value／访问数挑结果，所有回报共同更新动作N/W/Q。

只缓存卡牌ID，实际后继状态与双方历史仍从当前simulation重新执行；按实际剩余牌池区分上下文，防止对手重采样的暗保留牌又被抽出。对手已有暗牌仍每simulation重采样，自己已知牌保持确定；没有对root belief粒子扩宽。玩家统计键仍仅用自己可见的行动／观察历史，物理抽牌缓存不用于选择玩家动作。真实游戏引擎仍按真实牌堆推进。

CPW在有限预算下会忽略未收入的结果，可能降低覆盖率并增加标签的样本间方差。此轮未复测A/B/C/D目标噪声，不把原来不扩宽模式的JS收益继承为CPW结论。参考[MCMS论文§2.2](https://www.ijcai.org/papers13/Papers/IJCAI13-093.pdf)的chance扩宽定义；本项目按单次均匀卡牌事件实现，不复用完整状态。

入口为`NativeChancePUCTSearchPolicy(...,chance_widening=ChanceWideningConfig(k=0.25,alpha=0.5))`；配置类默认k=1、α=0.5。固定评测单个策略的`search_backend="native_chance"`旁设置`"chance_widening":{"k":0.25,"alpha":0.5}`，已接通串行和native_batched。省略保留不扩宽行为；MCTSConfig与旧self-play序列化保持原样。正式训练配置未切换；本次采集计时使用实验目录内的spawn适配器，manifest含CPW搜索版本／参数并重新计算fingerprint。

## 同200个局面的S1024结果

同一16500步checkpoint，复用此前200个留出DecisionRequest，seed16710000／16720000、N1、c1.5、fold forced、cap512、根噪声关闭、T0；GPU FP32 eager、TF32关闭、单CPU线程，active/batch200、C16/32/48/64、单buffer。每根完整使用1024次simulation，零transition cap。计时仅包含固定根搜索，不含模型加载。

| 搜索 | 平均逻辑深度 | 200根耗时 | 精确终局simulation比例 | 平均KL(target∥当前裸网) |
| --- | ---: | ---: | ---: | ---: |
| 旧native确定化 | 4.912 | 6.37s | 1.0928% | 0.3807 |
| native_chance，不扩宽 | 2.623 | 9.89s | 0.00391% | 0.3099 |
| CPW，k1/α0.5 | 2.893 | 10.72s | 0.00342% | 0.2999 |
| CPW，k0.25/α0.5 | 3.461 | 11.54s | 0.02930% | 0.3086 |

逻辑深度=`tree_depth_sum/simulations`，不计折叠forced动作。k0.25比不扩宽多耗时16.7%，仍只有旧平均深度的70.5%。终局命中增加但远低于旧搜索，因此仍高度依赖当前value。KL表示搜索目标与固定裸网的差异，不是正确性／棋力指标，也没有明确改善。

| 同局面分层 | 根数 | 旧确定化深度 | 不扩宽深度 | CPW k0.25深度 |
| --- | ---: | ---: | ---: | ---: |
| 无对手暗保留牌 | 121 | 4.695 | 2.975 | 4.059 |
| 有对手暗保留牌 | 79 | 5.244 | 2.085 | 2.546 |

剩余差距主要集中在暗牌局面。这提示root暗牌重采样及多观察者信息历史分叉仍可能限制深度；未单独隔离其因果，不应直接把暗牌固定成一个世界来消除差距。

先在相同分层抽取的12根做两次小探测：不扩宽2.399，k1为2.681，k0.25为3.270。随后两组都完整测200根，未按棋力筛参数。k0.25是更紧的诊断候选，不是已证明最优的训练教师。

## 真实采集时间

实际RTX4050 Laptop6GB；8个搜索worker、200并发、max batch200、共享槽2、owner单缓冲、固定地址CUDA Graph B16/32/64/128/200与C16/32/48/64；根噪声0.25，开局10完成回合T1、之后T0。两个独立进程各先完成编译／图预热，再5秒运行预热，测共同60秒窗口。模型／种子／执行配置相同，搜索算法不同；都是起局后的短窗口，不代表整轮状态分布。

| 短窗指标 | 旧S1024 | CPW k0.25 S1024 |
| --- | ---: | ---: |
| simulation/s | 39,483 | 29,369 |
| 实际平均GPU batch | 198.65 | 197.70 |
| worker CPU微秒/simulation | 36.87 | 87.84 |
| owner等候输入时间 | 8.05s | 14.36s |
| owner等待凑批时间 | 7.79s | 11.68s |
| 60秒内完成整局 | 0 | 0 |

等量模拟工作时间增加34.4%；按每局65个搜索决策估算，200局搜索工作约从337秒（5分37秒）增至453秒（7分33秒），**这不是完整200局实测**，未含启动、尾批、落盘及对局长度变化。固定请求单CPU耗时比与实际8-worker成本比分开解读。

更多状态推进、信息历史与牌池缓存使worker每simulation CPU成本增加约2.38倍，等待输入／凑批也增加，平均batch仍接近200。可观察到供给／同步负担增加，未做kernel或函数级因果归因。`forward_seconds`主要是异步启动计时，GPU完成等待另计，不能据不到1秒的forward字段断言GPU闲置。

本轮旧窗口完整进程191.0秒、CPW80.3秒，其中全档位编译／图预热分别123.6秒和13.0秒；第二次可能有缓存收益，缓存冷暖状态未隔离。不能以完整进程时间误判CPW更快，也不能把这两个启动数字直接预测正式轮次。原始计时和owner明细见[old_window.json](collection/old_window.json)、[pw_window.json](collection/pw_window.json)，配置和搜索fingerprint在相邻文件。

## 验证与复核

- 45 passed、3 skipped；三项为现有沙箱条件下的CUDA skip。新增测试覆盖支持集增长、归一化抽样、平均回传、隐藏重采样后牌守恒／无重复、批量与逐叶一致、配置拒绝错误后端。
- 重编译后，旧native与关闭CPW的native_chance各200根逐项复现冻结A/repeat0的target、prior、visit counts及选招，共400次搜索一致；旧结果未被新功能改变。
- 四局GPU S16 CUDA Graph和两局CPU S8串行checked smoke均自然终局、零异常／不变量失败／工程截断，只证明实际调用链；不以六局得分判断棋力。60秒采集窗口结束后工作进程退出。
- [manifest.json](manifest.json)保存模型／source／native／参数／命令与产物哈希；源码和编译扩展快照、构建日志、运行日志与计时中间数据在忽略的`runtime/entity_attention_chance_pw_20261007/`。wheel隔离构建，仅提取.so到源码目录，未更改conda已安装包；模型、runtime及评测`.games/`按现行规则不纳管。
- [analysis.json](analysis.json)由固定根原始JSON与采集短窗复算，[analyze.py](analyze.py)可只读验证；[verification.json](verification.json)保存对配置、逐局seed／预算及文件身份的复核。未运行正式训练、CPW棋力赛或A/B/C/D复测。

复核命令：

```bash
conda run -n splendor env PYTHONPATH=src python artifacts/entity_attention_chance_pw_20261007/analyze.py
conda run -n splendor env PYTHONPATH=src python artifacts/entity_attention_chance_pw_20261007/audit.py
conda run -n splendor env PYTHONPATH=src python -m pytest -q tests/test_native_chance_search.py tests/test_native_search.py tests/test_native_batch.py tests/test_evaluation_3a.py
```

首次采集计时命令（输出已存在，不需重复覆盖）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u artifacts/entity_attention_chance_pw_20261007/collect_probe.py \
  --search-backend native_chance --k 0.25 --alpha 0.5 \
  --config artifacts/entity_attention_chance_pw_20261007/collection/pw_config.json \
  --output artifacts/entity_attention_chance_pw_20261007/collection/pw_window.json \
  --scratch runtime/entity_attention_chance_pw_20261007/collection \
  --seconds 60 --warmup-seconds 5 --games-per-worker 25
```

用户取消的[大S初筛](../entity_attention_chance_depth_20261007/README.md)作为未完成实验保留；没有把其中的12根匹配视为完整深度／棋力验证。
