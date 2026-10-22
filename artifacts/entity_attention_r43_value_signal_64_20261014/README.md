# R43 小模型：单次终局监督的可预测信号

用户授权的64局面诊断，冻结原R43的128维×2层小模型。每根16次独立续局，共1024个正式自然终局，另有一局同种子的重复控制。配置、输入及运行程序身份见 [plan.json](plan.json)，执行状态与环境见 `execution/`；完成结果见 `results.md` 和 `summary.json`。

2026-10-14开始、2026-10-15完成。正式执行约23.5分钟；1024个正式终局全部回放核对，每根一局另做checked不变量验证，正式重复控制完全一致。单次MSE为0.755510，分解为结果方差0.693555与修正后期望value误差0.061955；结果方差占自身MSE的91.8% [86.8%,96.6%]。局面之间的期望差异占总结果方差约29.4% [21.2%,36.6%]，模型与16次均值相关系数0.829。54/64根在16次中同时出现过赢和输。上述比例只对应本次采样和续局策略。

这支持“单局终局z监督高方差”，同时表明R43已学到不少期望胜负信号；仅凭单次终局留出MSE停在0.7多，不能证明value停止进步。修正后期望误差仍显著大于0，且中期点估计最大，因此也不能声称value已达到极限。该诊断没有改变标签后训练或验证棋力，不证明平均标签一定能增加搜索强度。

## 样本与续局口径

从原R24–R43固定池的整局留出中选64个不同对局，20轮各3局，另4个轮次随机各加1局。每局只取一个多候选主行动局面，按原轨迹主行动的15%／50%／85%位置取早／中／晚22／21／21根。选择seed37000000；这是一组轮次／阶段配额样本，不是训练池全部局面的频率加权估计。全部来源轨迹checked回放，候选顺序、状态与候选特征逐项对齐原NPZ。

每次续局从同一当前玩家可见请求重新采样对手暗保留牌和未来牌序；本方暗牌保持可见信息所确定的身份。使用现有 `determinize_request` 均匀采样器，生成后逐项验证根请求不变。采样完整世界在实际续局中保持固定；后续搜索仅接收各行动方可见请求，独立采样搜索世界，不能直接读取实际世界。

双方固定为R43、native S1024/N1、c_puct1.5、Dirichlet alpha0.3／fraction0.25、行为温度0、fold forced phases、搜索transition cap512。起始世界、搜索世界与根噪声采用独立种子域37100000／37200000／37300000并由既有 `derive_seed` 派生；每个续局、行动方及决策使用固定可复核的种子。R43 checkpoint SHA为 `b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14`。GPU compiled FP32、TF32关闭，最多128个续局同时活动，推理batch64。

温度差异：原主线采集的前10个已完成玩家回合采用温度1，之后为0；本诊断从一开始就固定全程温度0。搜索参数与原主线相同，不代表行为分布完全相同。早期22根处在第7–9回合，存在这一差异；中、晚期根已超过第10回合。完整冻结参数见plan.json，额外口径核对见protocol_difference.json；没有在运行途中切换策略或覆盖冻结计划。

价值与最终z都从根行动方视角计算；重建器把这个玩家映射为0。胜／和／负为+1／0／−1。400总玩家回合仅为工程上限，达到时明确失败，不能以截断代替终局或算作和棋。正式续局独立改变世界和搜索种子；额外重复控制固定两者，排除在统计之外。

## 统计含义

对第i个根，预测为v_i，16个终局为z_ij，均值为m_i，无偏样本方差为s_i²。所有根等权：

- 单次终局误差：mean((v_i−z_ij)²)。
- 结果方差：mean(s_i²)，估计同一可见局面下续局随机性造成的误差。
- 模型对期望结果的误差：mean((v_i−m_i)²−s_i²/16)。扣除16次均值自身的抽样方差。
- 局面间信号方差：Var(m_i, ddof=1)−mean(s_i²/16)。判断不同局面的期望结果是否存在可利用的差别。

前两类误差满足精确样本恒等式：单次MSE＝结果方差估计＋修正后的期望结果MSE。修正估计在有限样本中可以为负，比例可以超过1，不裁剪；这种情况应解释为估计不确定性。区间采用阶段内独立根bootstrap10000次，seed37400000。局面阶段、根的有限样本和16次Monte Carlo仍限制精度，不把区间当成全部潜在世界的精确范围。

这里估计的是统一R43续局策略和现有均匀世界采样器下的结果分布。旧训练标签来自多个历史教师；采样器也没有利用过去动作推断对手暗牌后验。因此不能直接声称测出了原训练池或真实游戏后验下的不可约MSE，不能仅因平均标签更易拟合就推断棋力提高。

图表见 [value_signal.png](value_signal.png)，左图为固定value与16次终局均值，误差棒为±1个Monte Carlo标准误；右图为MSE分解及按根bootstrap区间。首次自动绘图因splendor环境缺少Matplotlib而失败，统计与全部回放已完成，原日志保留在runtime的analysis.log；随后绘图依赖仅安装到忽略的runtime/plot_dependencies，使用同一个splendor Python渲染，没有修改conda环境。对应绘图程序为plot.py，临时依赖版本见plot_environment.json。

## 工程检查、运行与复核

[tests.xml](tests.xml)：6项有限CPU测试，覆盖方差／MSE分解、有限重复修正、确定性零方差、保留负修正、种子流独立及分层根bootstrap。`smoke.json` 保存两次相同种子的S16完整续局工程验证，不计入正式统计。

从仓库根目录运行，使用conda环境splendor：

```bash
# 首次冻结64根输入；已经冻结时只核验身份。
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_r43_value_signal_64_20261014/run.py prepare

# 正式续局；相同命令跳过已写出的终局记录，可中断恢复。
conda run --no-capture-output -n splendor env \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u artifacts/entity_attention_r43_value_signal_64_20261014/run.py run

# 无GPU、无新搜索：回放全部续局，核对终局并统计。
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 \
  python artifacts/entity_attention_r43_value_signal_64_20261014/run.py analyze
```

`root_bank.json` 保存safe请求、来源NPZ身份与历史z；`predictions.json` 保存固定模型输出；`raw/` 保存每次续局种子、全部动作候选索引、最终结果和状态digest。`analyze` 对1024条轨迹重建世界并回放到终局，其中每根第0次另做checked不变量验证；不再执行搜索。完整输入、源代码快照和执行日志在忽略的 `runtime/entity_attention_r43_value_signal_64_20261014/`，身份和环境在本目录 `execution/`。正式实验没有训练或修改模型。
