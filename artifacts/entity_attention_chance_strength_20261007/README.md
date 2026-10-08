# 同一冻结模型：随机节点与确定化搜索的100 pair棋力对照

2026-10-07。完成100 pair／200局。**当前S1024随机节点搜索明显弱于旧S1024/N1确定化搜索：62胜、1和、137负，得分31.25%，按pair抽样的95%区间24.5%–38.5%。200局全部自然结束，零异常、非法动作、不变量失败、工程截断或周期裁定和局。**

这次比较搜索算法，双方网络权重完全相同，使用[上次目标噪声复测](../entity_attention_chance_search_20261007/README.md)的16500步checkpoint；没有重新训练“新模型”。目前不建议将这版随机节点后端直接替换正式训练教师。

## 结果

表中均从新随机节点搜索一方计分，胜1分、和0.5分、负0分。

| 新搜索位置 | 局数 | 胜 | 和 | 负 | 得分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 先手 | 100 | 25 | 1 | 74 | 25.50% |
| 后手 | 100 | 37 | 0 | 63 | 37.00% |
| 合计 | 200 | 62 | 1 | 137 | 31.25% |

配对bootstrap以同一游戏seed的两局平均得分为一个单位，100个单位、10,000次重采样，seed16790000。区间[24.5%,38.5%]整体低于50%；这是当前固定模型／配置／种子面板的清晰劣势，不代表所有随机节点算法、预算或训练后的模型都更弱。100个pair中，新搜索两局全输52对、两局全赢14对、各赢一局33对、1胜1和1对。

评测墙钟551.37秒，约9分11秒，不含输入准备和后处理；平均每局55.78个完成玩家回合、81.655次decision。200局并发共享一个模型推理owner，逐作业累加的`total_seconds`不是墙钟耗时，也不能据此算两种搜索各自速度。

## 与稳定性复测的关系

此前同一批200个固定留出请求上，新搜索C组JS下降90.8%、D组下降63.8%，说明target对环境抽样更稳定。这次直接对打却显示棋力明显下降，**不能再将“标签更稳定”当作这版教师质量更高的证据。**

本次全对局的搜索统计：

| 指标 | 新随机节点 | 旧确定化 |
| --- | ---: | ---: |
| 可搜索决策 | 6475 | 6463 |
| 总simulation | 6,630,400 | 6,618,112 |
| 平均逻辑深度 | 2.454 | 4.808 |
| simulation命中精确终局比例 | 2.823% | 6.994% |
| transition cap命中 | 0 | 0 |

每份可搜索决策都完整用了1024次simulation；两个总数略有不同来自对局动作序列中的可搜索决策数。逻辑深度是`tree_depth_sum / simulations`，不计折叠的forced动作。

新搜索明显更浅、使用精确终局收益更少，因此更依赖现有value。此前在**相同固定局面**上的同预算诊断也观测到深度从约4.85降至2.56，支持优先检查随机结果分支对预算的分散。但本次整局两方经过的状态分布不同，深度／终局命中率不能单独证明退化因果；尚未隔离value在新叶节点分布上的误差或其他搜索近似的影响。

下一步若继续探索随机节点，应先改善有限预算下的决策深度与估值质量，例如研究随机结果分支的控制，再复测棋力。这是待验证方向，不是本次已实现修复。当前正式训练配置／教师仍保留旧native确定化，未启动任何训练或追加预算矩阵。本测试比较相同S1024，尚未测试相同墙钟预算。

## 固定配置与身份

- 双方checkpoint：`runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1/rounds/round-019/checkpoints/final.pt`，SHA-256 `dbfbe066144a485d15ec633aa87c85ac27864f142183725b86562e4ba44ed2c4`。固定评测summary记录两方相同身份；同一GPU冻结模型实例供两方推理。
- A：`native_chance`；B：旧`native`。双方S1024/N1、c_puct1.5、fold forced phases、transition cap512、根噪声关闭、行为温度0；N=1在新模式表示一个共享统计池。
- 100个游戏seed16770000–16770099，每seed交换先后手。基础搜索seed16740000，两方分别与固定策略seed`16780000 + 2*pair_id + player_index`异或；同一pair两局保持同一策略seed，沿用现行固定评测协议。
- 单GPUFP32 eager，TF32关闭，PyTorch/OMP/MKL线程1；native_batched、active_games200、max_batch_size200、candidate buckets16/32/48/64、actor_projection=false、同步单buffer。实际GPU为RTX4050 Laptop 6GB，软件／环境见[environment.json](environment.json)。
- 真实游戏运行Python checked规则、由真实游戏牌堆决定抽牌；随机节点仅用于搜索模拟。max_completed_turns400，保留现行周期裁定，但本面板未使用周期裁定，200局均自然终局。

## 产物与复核

[evaluation_config.json](evaluation_config.json)为完整可执行配置，[manifest.json](manifest.json)冻结命令、seed、模型、源码与原生扩展身份；[evaluation.json](evaluation.json)保存200局结果、每pair记录、两方搜索统计、置信区间及执行时间；[analysis.json](analysis.json)从逐局JSONL复算，并检查预算、策略seed、交换先后手、源码快照及模型哈希。

本次没有修改训练／搜索源码，不重复运行此前已通过的39项回归。必要验证是实际200局checked对打和只读逐局复核。原始`evaluation.json.games/` sidecar按现行规则忽略，源码／编译扩展快照及运行日志在`runtime/entity_attention_chance_strength_20261007/`，未改忽略规则。模型、runtime和sidecar不随Git分发，迁移完整复核需备份；summary本身已保留全部200局结果。

首次运行命令：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u artifacts/entity_attention_chance_strength_20261007/run.py \
  --config artifacts/entity_attention_chance_strength_20261007/evaluation_config.json \
  --output artifacts/entity_attention_chance_strength_20261007/evaluation.json
```

已有summary时需`--resume`，完整pair会跳过；不要覆盖或换配置复用本目录。[run.py](run.py)设置与上次诊断相同的FP32及线程选项，再调用现有固定评测入口。

只读复核，不执行GPU搜索／训练：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/entity_attention_chance_strength_20261007/analyze.py
```

`analyze.py --write`仅用于首次写入分析结果；后续默认复算、检查已保存结果与输入输出哈希。
