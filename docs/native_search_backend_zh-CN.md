# 原生 PUCT 搜索后端

## 入口与边界

`src/splendor/native_search.cpp` 是 C++17/pybind11 扩展。它保存数值化卡牌、贵族、筹码、牌堆、市场、玩家资源、阶段及 pending context，执行候选枚举、支付与归还、抽牌、贵族结算、强制跳过和终局判定。`NativeGame` 可从完整真实状态启动、按候选索引运行完整对局；规则差分测试把每一步候选、状态及编码与 Python 引擎比较。搜索只从 `DecisionRequest.observation` 构造独立世界，不读取真实引擎的牌堆顺序或对手暗牌。

原生编码直接从数值状态生成 `splendor-observation-action-v1` 的 FP32 数组。卡牌静态目录在每个进程初始化一次；卡牌支付缺口、行动者视角、公开已购、pending 卡和候选动作等字段按局面生成。Python `features.py` 保持参照实现。C++ 搜索负责节点池、选择、展开、forced phase 折叠、回传、终局、transition cap 和各世界根访问统计。它沿用每世界最多一个未返回叶请求；Python `native_policy.py` 给请求分配唯一 ID，绑定冻结模型版本，负责网络推理、温度行为策略接口和结果归档适配。`search_inference.py` 的 encoded 路径把多个请求合批交给现有 ActorCritic FP32 模型。原生搜索推进期间释放 GIL；编码返回独立 NumPy 数组，推理完成前不会复用其内存。

算法位置：`SearchJob::select` 是 PUCT 与 tie-break；`SearchJob::finish` 是各世界按实际预算归一、等权 policy target、温度行为分布与选招；`NativeSearchJob._collect_result` 汇集归档统计。`MCTSConfig` 仍控制 S/N、噪声、温度以外的搜索参数；行为温度由调用方传入。训练损失、模型结构、replay 和 learner 没有改动。

native 自对弈使用 `NativeMatchStepper`：初始洗牌由现有 Python 引擎完成，随后 C++ `NativeGame` 是真实对局状态和动作推进的权威；每个真实决策边界才从原生快照构造行动者可见的 Python `DecisionRequest`，供搜索入口、trace 和归档使用。Python 后端保留原 `MatchStepper`。native 固定评测复用现有 Python 对局 runner，但其 MCTS 搜索世界、规则和树在 C++。跨局待推理输入仍以每个叶节点的 NumPy 字段对象穿过 Python 适配层，再由 Python collator 合并；这是继续优化 GPU 小批量和调度开销的位置。

## 构建、配置与兼容

在仓库根目录及项目 Conda 环境：

```bash
conda run -n splendor python -m pip install -e . --no-deps
conda run -n splendor env PYTHONPATH=src python -m pytest -q tests/test_native_search.py
```

构建隔离自动安装 pybind11，使用 C++17、`-O3 -DNDEBUG`，没有 fast-math。生成的 `.so` 被 Git 忽略。选择 native 而未构建时，`native_policy.py` 报出安装命令；不会回落到 Python。

- `configs/native_search/alphazero_loop_cpu.json`：独立生产 run，原生 CPU 采集和原生 CPU MCTS 评测，8 workers，FP32。
- `configs/native_search/alphazero_loop_cuda.json`：独立生产 run，单 GPU 模型 owner、8 条活跃对局的原生采集；MCTS 固定评测仍在 CPU 执行。GPU 采集已有限验证，但在本机 8 局对照慢于 CPU。
- `configs/native_search/alphazero_loop_smoke.json`：1 轮、4 局、4 学习步、每项 1 pair 的完整循环验证。
- `configs/native_search/selfplay_cpu_smoke.json`、`selfplay_cuda_smoke.json` 与 `fixed_evaluation_smoke.json`：独立小任务。

旧配置默认 Python 后端。native 的 manifest 把 `search`、`numeric`、`rng` 版本和 checkpoint 内容 SHA-256 纳入语义指纹，不能接到不兼容的半成品 run。归档仍为 `mcts-selfplay-v2`，旧 Dataset、collator 与跨轮 replay 可读取。`execution.backend` 区分搜索实现和模型设备：`native_cpu` 或 `native_cuda`；`engine_mode` 仍选择真实 Python 引擎的 checked/fast 模式。CUDA 不可用时报错。协作调度有可运行请求就立即发批，native 配置把 `max_batch_wait_ms` 设为 0；不存在等待凑满 batch 的队列。

隐藏世界洗牌使用原版 SplitMix64，语义种子按公开请求、对局、决策和世界派生。原生根噪声采用 C++ `mt19937_64` gamma，非零温度抽样采用 SplitMix64，故同一个整数种子不保证与 Python 搜索得到同一根噪声或执行动作。噪声关闭、使用同一完整状态和受控网络输出时，差分测试核对编码及访问计数。跨 CPU/GPU、不同 batch 形状的 FP32 输出可能略异，近平票可能改变选招；没有强制舍入。真实动作 trace 可以在 checked Python 引擎重放，失败现场还记录世界、模拟序号和分支动作；这不同于逐位重现神经搜索。

## 已运行验证与性能

- 原生规则完整轨迹差分：3 个固定随机种子，每步比较有序候选、完整可比状态和所有特征字段；另有 2 个种子核对 native 真实对局重建的请求、终局与 checked trace。覆盖购买支付、保留/盲留、超额归还、贵族及真实终局。
- 受控搜索差分：N=1/2、S=17 的非整除预算、相同采样世界和相同 evaluator 输出，逐叶编码、根访问、选招一致。
- 5.5M 初始 checkpoint 与已有 AlphaZero round-004 checkpoint 均可加载原生搜索。2 局 native CPU 与 2 局 native CUDA smoke 归档成功，checked replay 通过；拷贝出的一局缺失 run 只补采缺失 game_id。故意提交非有限 value 的失败分支可由 `reproduce_failure` 重建。
- 原生固定换座评测 1 pair 通过；微型 AlphaZero 循环完成采集、Python learner、原生 MCTS 评测、greedy 评测和恢复。完整测试结果见本次交付说明。

同一机器、同一初始 5.5M checkpoint、CPU FP32、S=1024/N=1 的 8 局测量，四档总计 32 局。Python 与 native 主对照均配置 8 workers、active/batch=4/4；只有 8 局时实际每个 worker 恰好 1 局，所以实际 batch 均为 1。native 2-worker 档是另一个调度设置。对局 RNG/浮点差异改变了实际局长，因此同时列出模拟和模型位置数。原始 config、checkpoint SHA、summary、批分布、资源统计与固定面板重复值见 [`artifacts/native_search/performance_2026-09-29.json`](../artifacts/native_search/performance_2026-09-29.json)。

| 路径 | 墙钟秒 | 局/分钟 | 有效 policy 样本/秒 | 模拟/秒 | 模拟数 | 网络位置数 | 平均 batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Python CPU，8 workers | 186.39 | 2.58 | 2.76 | 2824 | 526,336 | 475,276 | 1.00 |
| native CPU，8 workers | 97.62 | 4.92 | 5.50 | 5633 | 549,888 | 496,147 | 1.00 |
| native CPU，2 workers × 4 局 | 103.31 | 4.65 | 5.20 | 5322 | 549,888 | 496,148 | 3.51 |
| native CUDA，1 owner × 8 局 | 114.05 | 4.21 | 4.71 | 4821 | 549,888 | 496,147 | 6.76 |

主对照端到端墙钟快 **1.91 倍**，模拟数多 4.5%、网络位置数多 4.4%。这 32 局测量完成后接入了原生真实对局推进，并修正原生请求身份、边界检查和 RNG 版本标签；它们没有改变 S/N 或模型，原生真实对局的 2 局 smoke 和 checked replay 已通过，但未在最后改动后再做 8 局端到端复测，以遵守 32 局预算。性能 run 的 manifest 仍保留测量时的旧通用 RNG 标签，当前代码已把同一实际 RNG 实现显式标成 native 版本。两种合批档减少物理前向次数，但较少的 CPU 执行单元或 GPU 小批量开销使墙钟未超过 native CPU 8-worker 档。主对照推理前向累计 499.71 个 worker 秒，Python 基线为 614.33 个 worker 秒；搜索规则累计分别 2.34 和 173.23 个 worker 秒。native 编码在主对照累计 10.50 个 worker 秒，Python collate/传输累计 66.61 个 worker 秒；归档写入累计 0.21 秒，队列等待 0。这些都是各 worker 累计且可与别的阶段重叠，不能直接相加或除以墙钟求百分比。native CPU 8-worker 档单个 worker 的峰值 RSS 约 523 MiB（未测得全 run 同时峰值），CUDA 单 owner 约 843 MiB CPU RSS 和 10.8 MiB GPU allocation。当前剩余主要开销是模型前向、Python 批适配及 worker 调度；CUDA 平均 batch 6.76 尚未覆盖它的高吞吐区间。

最终 C++ 边界修复后重新运行固定开局/中盘/后段面板，预热 1 次、重复 3 次；每次均执行 1024 次模拟、1025 个逻辑网络位置，选招在三次重复和两后端间一致。首次面板也保留在性能 JSON 中，以展示机器运行间波动：

| 局面 | Python 中位秒 | native 中位秒 | 加速 |
| --- | ---: | ---: | ---: |
| 开局 | 1.097 | 0.489 | 2.24× |
| 中盘 | 1.068 | 0.469 | 2.28× |
| 后段 | 1.120 | 0.487 | 2.30× |

## 可复制命令

```bash
# 固定局面板
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_native_search.py \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --output runtime/native_search/fixed_panel_cpu_new.json --device cpu \
  --simulations 1024 --determinizations 1 --warmup 1 --repeats 3

# 小型自对弈、checked 重放与恢复
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast generate \
  --config configs/native_search/selfplay_cpu_smoke.json --output runtime/native_search/cpu_smoke
conda run -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast validate \
  --input runtime/native_search/cpu_smoke --replay
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast generate \
  --config configs/native_search/selfplay_cpu_smoke.json --output runtime/native_search/cpu_smoke --resume

# 固定换座评测与微型循环
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.evaluation \
  --config configs/native_search/fixed_evaluation_smoke.json \
  --output runtime/native_search/fixed_evaluation_smoke.json
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.alphazero_loop \
  --config configs/native_search/alphazero_loop_smoke.json \
  --output runtime/native_search/alphazero_loop_smoke
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.alphazero_loop \
  --config configs/native_search/alphazero_loop_smoke.json \
  --output runtime/native_search/alphazero_loop_smoke --resume

# 新生产 run，仅在决定启动时运行
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.alphazero_loop \
  --config configs/native_search/alphazero_loop_cpu.json --output runs/az_native_cpu_seed1
```

本次只执行了有限 smoke 和 32 局性能测量；未启动上述生产 run。
