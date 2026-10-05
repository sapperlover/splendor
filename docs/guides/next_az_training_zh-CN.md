# 旧 θ5 起点的 10×1000 AZ 训练准备

准备时记录（2026-10-04）：配置、记录链路、命令和有限回归已准备，当时正式实验及独立 CUDA smoke 未启动；以下准备期“尚未实测”表述仅适用于该时点。

后续状态：`runs/az_theta5_10x1000_seed1/loop_state.json` 已记录轮0–9完成。2026-10-04新模型实现任务复用这10000局归档和原split，数据元信息依据见 [离线学习专题](entity_attention_offline_zh-CN.md)；本次不重做旧run棋力验收，也不将准备期CPU微型测试当作native长训或棋力证据。

## 冻结的实验设置

| 项目 | 设置与口径 |
| --- | --- |
| run / 输出 | `az_theta5_10x1000_seed1` / `runs/az_theta5_10x1000_seed1`，独立于旧 run |
| 起点 | 旧 `az_loop_seed1/round-004/final.pt`，只继承权重；首轮新建 AdamW、replay、累计步数 |
| 模型 | 现有 Deep-Sets ActorCritic，302,626 参数；card/noble/state/actor/critic hidden=64/32/256/128/128 |
| 轮数 / 数据 | 一次冻结 10 轮，每轮完整 1000 局；双方合计完成回合与根决策数分别记录 |
| replay / split | 最近 6000 整局；确定性逐局 10% validation，实际比例看种子派生与归档；诊断样本不加入 train |
| 缓存 | 4 GiB 保留数组预算；不是 RSS 上限。此前验收为约2500局池、2 GiB，4 GiB/6000局尚未实测 |
| 教师搜索 | S1024/N1、c_puct1.5、折叠强制阶段、max_simulation_transitions512 |
| 探索 | 根噪声 fraction0.25/alpha0.3；双方合计回合 `<10` 温度1，之后0 |
| 采集 | native_cuda_pool/fast；外层1、search workers8、active500、batch500、micro64、slot1、wait5ms、shared width256 |
| 推理 | FP32、torch threads1、native接口、actor projection、reused transport、compiled16/32/48/64、fail_fast |
| learner | CUDA FP32、batch256/history128、AdamW lr1e-4/wd1e-4、policy/value权重各1、clip norm1 |
| 步数 | `ceil(4 × 当轮新 train 全部局面数 / 256)`；按 value/all positions 算，不按有效policy局面算；不设正式max_steps上限 |
| 实际曝光 | 首轮current256、平均约4次；有历史时current/history128/128，新数据平均约2次。随机抽样有重复/未覆盖，使用引用计数报告实际分布 |
| checkpoint | 每100步及轮末；optimizer、抽样RNG和累计步数跨轮延续，同轮resume恢复checkpoint；final是最后step文件的字节副本 |
| 当轮验证 | 所有当轮validation根局面、batch256；同一集合SHA分别评估父权重/轮末权重，含总体和早/中/后段 |
| 固定诊断 | 首轮validation最多2048引用，阶段轮流分配、空组让出配额、不足全取；初始/每100步/final在训练结束后评估；原NPZ即使replay淘汰也保留 |
| 每轮棋力 | 新MCTS vs父代50 pairs；新greedy vs固定旧θ5 50 pairs；各自固定DEV seeds，10000次paired bootstrap |
| 阶段锚点 | 人类第3/5轮=`round-002/004`，分别vs旧θ5、PPO各200 pairs；同一开发面板重复看趋势 |
| 结束验收 | 第10轮=`round-009`，两锚点各200 pairs，预留game/policy/bootstrap输入；调参期间不运行、不查看 |
| 评测执行 | MCTS双方S1024/N1、根噪声0、温度0；native_batched、唯一CUDA owner、active100/B64、compiled16/32/48/64、projection、wait0、外层1；greedy使用现有独立路径 |

起点 SHA：`e0cba7733de4977cd0a606d35d2f328f47f285a0be81a1bd3895633dd105f97b`。PPO latest锚点 SHA：`009d17d2e9bc0d879142d7ca3ff1777a6f8caa9dda9243c6a2d5eb3538b5d5b0`，路径 `runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt`。两者架构相同。完整身份、实际机器与配置摘要见 [准备清单](../../artifacts/az_theta5_10x1000_seed1/preparation_manifest.json)。

新起点只是旧θ5权重热启动，不能称旧run的第6轮；新round-004也不是固定锚点旧θ5。新run内跨轮继承AdamW，首轮重新创建；同轮恢复继承已提交的AdamW和RNG。没有跨run完整延续入口，不修改旧 `runs/az_loop_seed1`。恢复时不改变正式轮数/采集数/种子/设备/缓存等冻结配置。

## 种子与面板身份

完整10轮collection/split/sampling派生值、首局game/search输入和实际train/validation局数预测见 [seeds.json](../../artifacts/az_theta5_10x1000_seed1/seeds.json)。基种子collection=11300000、split=11310000、sampling=11320000；采集search base=11330000，固定诊断选择=11340000。split是整局确定性分配，1000局不是强制恰好100个validation。

| 面板 | game seeds（含端点） | policy seeds（含端点） | bootstrap seed | search base |
| --- | --- | --- | ---: | ---: |
| 每轮MCTS DEV 50pairs | 11400000–11400049 | 11410000–11410099 | 11420000 | 4306397296 |
| 每轮greedy DEV 50pairs | 11500000–11500049 | 11510000–11510099 | 11520000 | 无 |
| 第3/5轮vs旧θ5 DEV 200pairs | 11600000–11600199 | 11610000–11610399 | 11620000 | 8601564592 |
| 第3/5轮vsPPO DEV 200pairs | 11700000–11700199 | 11710000–11710399 | 11720000 | 17191599184 |
| 第10轮vs旧θ5 RESERVED 200pairs | 11800000–11800199 | 11810000–11810399 | 11820000 | 34371568368 |
| 第10轮vsPPO RESERVED 200pairs | 11900000–11900199 | 11910000–11910399 | 11920000 | 68731406736 |
| 可选500pairs扩展vs旧θ5 | 12000000–12000499 | 12010000–12010999 | 12020000 | 137450983472 |
| 可选500pairs扩展vsPPO | 12100000–12100499 | 12110000–12110999 | 12120000 | 274890036944 |

LoopSeeds用 `derive_seed(base, round, 0, "splendor-alphazero-loop-seed-v1:<domain>")`。逐局game和split继续用各自派生基种子、round/game/domain做SHA派生；搜索world为search base XOR派生的`search_world/player`；行为抽样还有`search_sampling/player/decision`。评测policy输入为`base+2×pair_id+player_id`，native搜索seed再与search base异或。同一pair两局交换座位、共用同一game seed及策略输入。

准备脚本枚举10000个新训练game输入、20000个search world输入，核对旧AZ、现有fast采集配置的同域输入与配置中的评测game区间；初始候选配置在两个开发锚点间经XOR后有64个搜索输入重合，已改用不同高位search域（2^32至2^38）并枚举实际XOR输入复核；最终配置除了第3/5轮同锚点的刻意复用，没有发现所审计范围内的重合。逐决策行为输入要等实际归档后才能完整枚举；审计范围写在JSON内。不同数字不能证明统计独立，重复DEV的CI更不能视为独立最终验收。两次开发锚点故意复用输入，第10轮与扩展面板另用未查看输入。扩样因查看结果而决定时仍需保留查看/选择记录。

## 记录实现和诊断口径

现有采集NPZ/trace、校验、owner统计、checkpoint、metrics、评测sidecar继续作为原件。新增 `diagnostics` 为opt-in配置，旧配置不自动加入固定集和资源线程；记录开关参与冻结身份。loss和采样概率/RNG推进保持原式；目标熵在CPU标签上计算，原scalar日志合成一次CPU读取，不增加逐步CUDA synchronize或cProfile。

`policy_loss` 是CE，`policy_entropy`一直是预测熵。新增target_entropy和policy_kl来自同一实际batch：`KL=(CE加权和-H(target)加权和)/policy_weight_sum`。单候选/零权重排除；日志保留加权和、分母及有效样本数。policy分母为0时诊断比值为null，训练仍沿用可微零policy loss。总体必须加总原始分子/分母，不能平均各batch的均值。value同理；零预测MSE由训练/验证标签平方和导出。常数基线只在该轮保留train池的标签上拟合，记录replay digest，不用validation拟合。

当轮父权重与新权重比较在该轮训练结束后离线做：父checkpoint冻结且未被修改，含义是“训练前权重对同一留出集合的指标”，不是再训练父模型。评估单独加载模型、不访问learner cache，用torch fork_rng恢复模型构造消耗的全局RNG，也不抽取训练RNG。早/中/后段统一为根决策时双方合计完成回合 `<10 / 10–49 / ≥50`，叶请求统计保持原owner口径，二者分开。

固定集来自首轮validation，无跨split；保存引用/原件SHA、实际分层组成。empty组写unavailable，整个首轮没有validation时也明确空集，不伪造指标。每100步曲线在训练后读取已存checkpoint补齐；final与最后step字节一致时只评估一次。每轮常数基线可能变化，固定模型的KL/value MSE才是跨轮可比较的主指标。固定集是开发诊断，不是独立棋力测试。

每轮追加二进制引用`[step,round,game,decision]`，每个invocation独立文件，checkpoint保存前fsync。分析以最终已提交step截断，重复step的引用必须一致，生成完整局面覆盖计数NPZ、未覆盖比例与含0次数的直方图；保留重复/未提交attempt。checkpoint仍保存原RNG，摘要不能替代计数。恢复会先保全旧metrics再按checkpoint对齐，分会话attempt日志保留恢复开销。

invocation记录源码文件SHA及真实副本、HEAD/dirty patch、native SHA、设备/版本/线程/性能环境、命令、恢复起点/结束状态、起止与错误。5秒线程采样/proc进程树和nvidia-smi，短阶段报告样本数；当前进程的torch allocator峰值为invocation范围，不重置。CPU利用率按存活进程差分，一核100%；退出子进程尾部未知。RSS含共享页重复、PSS按比例分摊、设备显存含桌面。资源采样线程失败会在会话末写resource_error，不能据此宣称资源监测完整。

阶段计时互斥：collect、replay_index、validation_prepare、learner_initialize、learner_train内的draw/collate/update/checkpoint/validation/remainder、checkpoint_commit、diagnostics、两种eval、archive_summary及invocation remainder。更新是主机区间，不是GPU kernel时间；worker/owner计时属于并行细分，不相加成墙钟。额外完整checked replay在诊断dataset中单列成本，失败直接阻断，成功记录全量game数。50/90/99/100%与last100/10/8/1在throttle前取点，尾区间嵌套；恢复前已越过的点不会伪造为本段0秒。

累计吞吐使用完整已提交工作/全部记录会话实际运行秒数，暂停另列；同轮无模型提交的重复步骤计作恢复开销。SIGKILL不能写结束时间时报告到最后资源样本的时间下界，完整吞吐标记无法精确计算。分析不把旧数据的缺字段补成0。详细原件/分母/频率见 [字段清单](next_az_record_fields_zh-CN.md)。

## 可直接使用的命令

从仓库根目录执行；下面训练、smoke和评测命令**本次均未运行**。终端使用`--no-capture-output`使conda实时转发日志；`--progress-format human`输出中文进度，原始JSON事件仍追加保存。默认每10秒进度，阶段起止/完成不节流。

```bash
cd /home/lc/下载/splendor
# 只读检查冻结配置、种子与现有模型，绝不启动实验
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python scripts/prepare_next_az.py

# 正式新run（独立目录，10轮一次冻结）
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m splendor.alphazero_loop \
  --config configs/az_theta5_10x1000/formal.json \
  --output runs/az_theta5_10x1000_seed1 \
  --progress-format human --progress-interval-seconds 10

# 同run恢复：使用同一配置，加--resume
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m splendor.alphazero_loop \
  --config configs/az_theta5_10x1000/formal.json \
  --output runs/az_theta5_10x1000_seed1 --resume \
  --progress-format human --progress-interval-seconds 10

# 独立native CUDA小验证：2轮×6局，S4，最多2训练步/轮，各1pair，固定诊断最多16
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m splendor.alphazero_loop \
  --config configs/az_theta5_10x1000/smoke.json \
  --output runtime/az_theta5_diagnostics_smoke/run \
  --progress-format human --progress-interval-seconds 1
# smoke恢复同上命令加--resume，不改正式配置缩小后冒充继续

# 独立终端观察；未启动时这些文件尚不存在
tail -f runs/az_theta5_10x1000_seed1/recording/events.jsonl
cat runs/az_theta5_10x1000_seed1/loop_state.json
# 只分析已存在日志，不加载模型、不启动对局；loop在每次会话结束时也自动执行
conda run -n splendor env PYTHONPATH=src python scripts/analyze_az_run.py \
  --input runs/az_theta5_10x1000_seed1 --output artifacts/az_theta5_10x1000_seed1
```

采集默认不要设置`SPLENDOR_SPEED_ROW_STATS=1`或`SPLENDOR_SPEED_CPROFILE=1`。准备清单和实际会话会记录环境，已有性能验收开启的昂贵逐叶统计不能直接套进正式训练。编译缓存没有被清理，冷/暖状态记录为unknown，由owner编译计数/首次图耗时进一步判断。

第3轮是人工观察点，不自动暂停。loop会继续第4轮；需要及时看锚点时，在终端确认`round-002`完成、下一轮尚未占用GPU的窗口人工Ctrl-C，原件支持恢复；若窗口错过，后续在GPU空闲时延后执行并记下延后原因。没有自动阶段调度器/暂停入口，不修改num_rounds。第5轮同理。

```bash
# GPU空闲后串行执行第3轮两锚点；第5轮将002改成004，第10轮改成009
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/run_recorded_evaluation.py --config configs/az_theta5_10x1000/round-002-vs-theta5.json \
  --output runs/az_theta5_10x1000_seed1/anchors/round-002-vs-theta5.json --progress-every 10
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/run_recorded_evaluation.py --config configs/az_theta5_10x1000/round-002-vs-ppo.json \
  --output runs/az_theta5_10x1000_seed1/anchors/round-002-vs-ppo.json --progress-every 10
# 外部评测恢复同一命令加--resume；不能换config在同一sidecar恢复
# 可选第10轮500pairs扩展（不是默认执行；theta5/ppo各有独立配置）
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/run_recorded_evaluation.py --config configs/az_theta5_10x1000/round-009-vs-theta5-extension500.json \
  --output runs/az_theta5_10x1000_seed1/anchors/round-009-vs-theta5-extension500.json --progress-every 10
```

以上轻量runner复用通用`FixedEvaluationConfig/run_fixed_evaluation`，加入外部单面板会话与5秒资源记录；也可直接用`python -m splendor.evaluation`读取相同配置，但直接入口没有这个额外资源/会话账本。sidecar由现有evaluation自动保存在相邻`*.games/`，含配置、完整双方checkpoint身份和每局记录。外部锚点评测成本保存在`anchors/recording/<panel>/wall_summary.json`，不进入主loop账本，保持串行且单列。`anchors/*.json`由分析脚本收集，按panel身份分别看趋势。

## 时间预算与解释

[既有性能验收](../reports/end_to_end_speed_optimization_zh-CN.md)是不同固定模型/数据的独立对照：PPO采集1000局1384.04秒；旧replay 8预热+128步及准备/验证总计11.41秒；θ5/θ4 50pairs批量评测207.21秒。只作规划依据，不是新AZ闭环实测。

| 阶段 | 每轮暂估 | 依据/不确定性 |
| --- | --- | --- |
| 1000局采集 | 20–40分钟 | 已测23.1分钟，换模型/长度/编译/温度会变；前几轮实测再修正 |
| 索引、验证准备、learner训练/保存 | 1–8分钟 | 旧500局新train约38k，对1000局估约76k→约1200步；缓存升温/6000局池/4GiB/保存数未验证，不线性用11.41秒冒充整轮 |
| 当轮/固定诊断＋额外checked replay | 1–10分钟 | 当轮约100 validation局，约12个100步节点×固定2048局面；新的离线成本尚未GPU实测，独立列账 |
| 每轮MCTS50pairs | 3–7分钟 | 已测3.45分钟；父代变化导致长度/难度变化 |
| greedy50pairs | 1–6分钟 | 执行路径独立，未套用MCTS加速比；需首轮测量 |
| 第3/5/10轮各两个200pairs锚点 | 每个节点约24–56分钟，另列 | 按已测50pairs约3–7分钟规划，active仍100；不同对手可能超出此区间 |

十轮主流程粗规划约4.3–11.8小时，另加三节点锚点约1.2–2.8小时；这是上述范围的预算推算，既非完整闭环测量也非时间保证。暂停、恢复重复工作、GPU争用和可选500pairs不在基础估计中。采集等待48.44%和评测非推理批次44.78%是现有画像，不代表可直接消除；本次不再优化执行路径。以后用本run阶段账本重新统计真实占比。

磁盘预算保留10000局NPZ、每100步约120个模型/optimizer checkpoint、引用二进制、TensorBoard及sidecar；固定集原件不能随replay淘汰删除。压缩大小和RSS/缓存抖动用实际记录更新，不能仅因4GiB预算断言全池常驻。

## 第3/5/10轮判断和后续4/8规格

在 [决策记录模板](../../artifacts/az_theta5_10x1000_seed1/decision_template.json) 记录查看时间、checkpoint SHA、已查看面板、依据、是否延后评测，以及“继续/调整候选/仍不确定”。每次查看追加到`decisions.jsonl`，含扩样理由。主run没有棋力门槛/自动回滚。单轮父代约50%、甚至暂降不能单独证明停训必要；先看固定锚点累计棋力、固定验证KL/value和同集前后差异。

若棋力和泛化没有改善，下一项只比较**固定数据强度4/8**：两次同数据/split/parent权重/AdamW起点/抽样seed/batch/current-history规则，独立输出，8从同一个parent开始，不能接4的final。先冻结已完成round-002的`replay_snapshot.json`和全归档SHA，令`P=当轮新train positions`，N4=ceil(4P/256)、N8=ceil(8P/256)，warmup0。跨轮来源/分母和实际曝光不能改。

现有`profile_speed_pipeline.py learner`可独立读取同一parent optimizer并执行指定步数，提供数据拟合/耗时诊断；它不是完整离线训练系统，只保存末端`diagnostic.pt`，不自动生成100步固定诊断曲线/覆盖计数/正式棋力面板。本次不扩建这个入口，不能将其输出称为记录完整的4/8正式实验。

```bash
# 前提：正式run至少完成round-002，原件不变。读取冻结snapshot只计算步数
conda run -n splendor env PYTHONPATH=src python -c \
  'import json,math; d=json.load(open("runs/az_theta5_10x1000_seed1/rounds/round-002/replay_snapshot.json")); p=sum(g["decision_count"] for g in d["games"] if g["round_index"]==2 and g["split"]=="train"); print("P",p,"N4",math.ceil(4*p/256),"N8",math.ceil(8*p/256))'
# 读取实际步数；该入口的config强度仍为4，实际步数由--steps控制
read -r az_steps4 az_steps8 < <(conda run -n splendor env PYTHONPATH=src python -c \
  'import json,math; d=json.load(open("runs/az_theta5_10x1000_seed1/rounds/round-002/replay_snapshot.json")); p=sum(g["decision_count"] for g in d["games"] if g["round_index"]==2 and g["split"]=="train"); print(math.ceil(4*p/256),math.ceil(8*p/256))')
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/profile_speed_pipeline.py --output runtime/fixed_intensity4/result.json \
  --scratch runtime/fixed_intensity4 --torch-threads 1 learner \
  --run runs/az_theta5_10x1000_seed1 --round 2 --cache-mib 4096 --device cuda --warmup-steps 0 --steps "$az_steps4"
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/profile_speed_pipeline.py --output runtime/fixed_intensity8/result.json \
  --scratch runtime/fixed_intensity8 --torch-threads 1 learner \
  --run runs/az_theta5_10x1000_seed1 --round 2 --cache-mib 4096 --device cuda --warmup-steps 0 --steps "$az_steps8"
```

两个入口会重建截至round-002的同一最近6000局池，后续run继续也不能替换这批原件；需核对生成digest与冻结snapshot一致。该profiling入口每步synchronize用于计时，不能把其耗时当正式无同步训练成本。验证/棋力有收益再考虑在独立run冻结8；训练误差降、固定验证不改善时不能仅因训练便宜选8。结构实验（实体attention/动作状态交互）只列前置：先固定数据拟合再独立自对弈，当前没有相应CLI。

## 验证与完成范围

相关回归49项通过：mask/分母、固定集身份和无train泄漏、诊断全局RNG恢复、原式更新逐参数相等、实际抽样序列相等、二进制重复step去重/撕裂记录、checkpoint后未提交一步的恢复、CPU微型完整闭环与无工作resume。旧Python配置的默认eval字段序列化也已核对恢复身份；此前loop绕过EvaluationSection.to_dict导致默认字段多出，本次修正为其现有规范序列化。

正式run、native CUDA独立smoke、4GiB/6000局长运行、阶段棋力面板和4/8比较均未执行。未据准备和测试宣称棋力提升、资源稳定或新的性能收益。运行前先核对准备清单和当前源码SHA；native小smoke是提供的下一步检查，不是已完成验收。
