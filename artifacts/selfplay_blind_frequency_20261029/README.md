# 自对弈盲保留频率（2026-10-29）

统计当前policy在线分支13个完整采集批次：`runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-000`至`round-012`，每批10000局，共130000局。`round-013`未完整，不纳入。第1批使用起始离线模型，之后使用在线第1—12轮模型；不是13批都由同一个checkpoint生成。

逐局读取NPZ中的实际动作`metadata_json.trace.actions`，而非候选动作或搜索访问。只把`main_action`计入主行动分母，支付、退币、选贵族不计入；双人对局中双方动作均统计。全部游戏的状态、身份、终局记录及决策长度均检查；没有重新跑自对弈或使用GPU。

## 结果

| 指标 | 13万局合计 | 本地人机模型：在线第9轮自对弈1万局 | 最新完整批次：在线第12轮模型1万局 |
|---|---:|---:|---:|
| 盲保留／全部主行动 | **4.54%** | 4.50% | 4.68% |
| 盲保留／全部保留动作 | **20.56%** | 20.38% | 21.12% |
| 至少一次盲保留的对局 | **84.26%** | 83.71% | 84.60% |
| 每局双方合计盲保留次数 | **2.50** | 2.48 | 2.58 |
| 单个玩家至少盲保留一次 | **65.89%** | 65.57% | 66.55% |
| 对手持有未公开暗牌时的主行动占比 | **24.64%** | 24.92% | 25.44% |
| 盲留牌最终被购买的比例 | **58.68%** | 58.41% | 58.68% |

合计7169512个主行动，1581383次保留，其中325164次盲保留。盲留牌最终买了190801张，134363张留到终局，占41.32%。后者可能用于取黄金、保留选择权或未能兑现，不能仅凭未购买判定为浪费。

盲留目标：一级116192次（35.73%），二级136133次（41.87%），三级72839次（22.40%）。盲留并非主要集中在三级。

按行动者本局自己的主行动序号分组，盲留占主行动：第1—10步3.73%，第11—20步4.61%，第21步起5.50%。这是各阶段内部的动作占比，不是盲留动作的阶段构成。13批盲留动作占比介于4.25%—4.73%，没有数量级变化。

因此：按单步占比看，盲保留属于少数动作；按整局覆盖率和持有期间看，暗牌处理是经常遇到的情况。当前人机模型同样如此，非个别一盘异常现象。它在用户那盘盲留3次，高于其自对弈平均每方约1.24次。

## 追加：实盘集中盲留三级牌，及每次决策的重采样

上述平均频率不能直接说明用户实盘的“3次且全为三级”很普通。对完全相同的在线第9轮模型，进一步扫描10000局、20000个玩家局：单方盲留至少3次（任意级）占15.65%；单方盲留至少3张三级牌占0.895%；恰好盲留3次且全部三级占0.345%。实盘AI26个主行动中盲留3次，占11.54%，三次分别在第14、19、23回合且全为三级；加上人类一次，共4次三级盲留。它的次数偏多、三级集中程度更突出。统计见[tier3_distribution.json](tier3_distribution.json)，重新扫描的输入metadata流SHA与原统计完全一致。由于自对弈配置与人机配置不同，上述频率只是该参考人群的描述，不作为异常显著性检验。

当前人机使用独立世界的`NativePUCTSearchPolicy`。每次新决策调用`start_job`创建新的`NativeSearchJob`，根据本次`DecisionRequest`和固定base seed派生各世界种子，重新分配对手尚未公开的盲留牌和未来牌序；不保留最初盲留那一回合的假想牌。局面变化通常会改变请求hash，因此base seed同为0也会产生新的四世界。相同请求与相同配置／seed则可复现相同四世界，不能将这些抽样当成每回合相互独立的随机试验。

一次S1024N4搜索内部，每世界固定其假想隐藏牌和未来牌序，运行1024次模拟；不会每次simulation再重抽。下一次实际决策才重新建立世界。AI自己的盲留真实牌已经属于自身私人观察，固定保留；对手明保留和已揭示／购买的牌也按观察固定。当前实现没有根据完整历史走法维护对手暗牌的后验概率。

用户实盘的AI第20—23回合给人类分配的四世界暗牌依次为`088/079/081/089`、`078/089/075/085`、`079/087/085/085`、`074/087/081/075`。四组不同，均没有实际086。原始局面与重建结果见[局内知识诊断](../web_blind_reservation_20261029/advantage/knowledge.json)。代码依据为`native_policy.py`的`search/start_job`、`NativeSearchJob.__init__`，以及`native_search.cpp:1032`的`initialize_worlds`和`search_policy.py`的`_request_seed/_determinized_state`；当前人机不是每次simulation重采样的`NativeChancePUCTSearchPolicy`。

## 适用范围与验证

这是训练采集分布：带25%根噪声，预算随机化75%为S128N1（总128）、25%为S256N4（总1024），开局温度1、后续0。不能直接替代无噪声S1024N4（总4096）人机／评测分布。原始配置、checkpoint SHA、manifest SHA及逐局metadata流摘要均保存于[summary.json](summary.json)。

对手暗牌持有比例由动作账本跟踪：`ReserveDeck`增加暗留数量，`BuyReserved`按此前市场保留的卡ID区分明留／暗留并移除。每局均校验“暗留总数＝已买＋终局未买”。另固定seed20261029，每批抽10局共130局，把该账本与NPZ实际输入中对手未知保留槽交叉核对，并对实际选中候选类型核对：10331个动作、7142个主行动局面的暗牌数量全部一致，见[verification.json](verification.json)。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/selfplay_blind_frequency_20261029/analyze.py
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/selfplay_blind_frequency_20261029/verify.py
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/selfplay_blind_frequency_20261029/tier3_distribution.py
```

第一条命令约114秒、单进程CPU流式扫描，不加载训练特征池；运行中的`progress.json`为临时汇总，完整结果以`summary.json`为准。原始NPZ保持在`runs/`，未改动游戏、训练、搜索实现或正在运行的用户任务。
