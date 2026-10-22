# BGA #927500015：第20轮模型逐步分析

2026-10-20准备，随后由用户启动。**正式搜索全部完成：1824次根搜索、15876096次模拟，用时约973秒，无编译失败或fallback。** [全部方式统计](statistics/results.md)汇总双方实际走法的平均概率、排名、一致率、Q差、低访问敏感性、种子稳定性、支付阶段及两两方法比较；[逐步报告](report.html)保留完整过程。原始捕获目录保持不变：[对局资料](../bga_927500015_20261020/README.md)。

模型为`entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt`，即当前混合预算分支第20轮的原始终点，不是旧主线R20、R43或EMA。SHA为`fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd`；128维、2层、4注意力头、FFN512，共583682参数，购买关系特征与多实体value读取启用，没有辅助头。

## 执行

EMA结束后，从仓库根目录执行；单进程串行跑完整矩阵，不启动训练或棋力对战：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u artifacts/bga_927500015_r20_analysis_20261020/analyze.py run
```

相同命令可断点续跑，按已保存决策跳过；最多重算尚未保存的一批16个局面。文件锁阻止同目录重复启动。冻结输入、模型和源码SHA不符时明确报错，不混用版本。`run`不主动等待EMA，也不中止现有进程；应由用户在EMA结束后启动。

后台执行与保存日志：

```bash
mkdir -p runtime/bga_927500015_r20_analysis_20261020
nohup conda run --no-capture-output -n splendor env \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u artifacts/bga_927500015_r20_analysis_20261020/analyze.py run \
  > runtime/bga_927500015_r20_analysis_20261020/run.log 2>&1 < /dev/null &
```

`prepare`只校验并冻结输入、算CPU Greedy；`smoke --device cpu`做有限工程检查；`report`从已保存数据重建报告。运行中的状态每批写`live.json`，每个种子完成后刷新报告；不需要持续轮询。

## 冻结矩阵与统计口径

| 组别 | 总S | 世界／随机机制 |
| --- | ---: | --- |
| Greedy | 0 | policy argmax，88个决策 |
| 原生确定化PUCT | 1024、16384 | N=1 |
| 原生确定化PUCT | 1024、16384 | N=4，分别每世界256、4096次模拟 |
| 随机节点搜索 | 1024、16384 | N=1共享信息集树，每次模拟重采未知牌、转移时采补牌 |
| CPW | 1024、16384 | 上述随机节点搜索加chance progressive widening，k=0.25、α=0.5 |

所有搜索组各3个固定种子：39700000、39701000、39702000。同一决策、同一种子、原生搜索的不同预算共享世界抽样种子，N4的第一个世界与N1相同。随机节点和CPW使用相同基础种子，但搜索树路径变化会改变随机调用，不能声称内部补牌完全配对。

复用现行搜索实现：c_puct=1.5，折叠强制阶段，单模拟最多512次状态转移，无根噪声、温度0，关闭Forced Playouts与Policy Target Pruning。S始终是**每次根搜索的总模拟数**，不乘N。主行动56次＋有选择支付20次，共76个可搜索根；8组×3种子×76=1824次根搜索，共15876096次模拟。另12个只有一个合法候选的决策保存Greedy和原行动，不执行搜索，也不计入一致率。自动贵族选择由引擎折叠，不能作为可选择根重建。

CUDA推理使用原生CUDA Graph、B16/B64、候选桶16/32/48/64、owner单槽，每批16个根，N4最多64个并行世界。TF32关闭、编译失败直接报错；CPU线程1。保存配置、环境、命令、代码身份与checkpoint SHA。小批设计控制搜索树内存，不代表最优吞吐。

逐步保存实际走法、完整候选、policy概率及排名、原始value、搜索访问数／份额、逐世界value_sum与Q、各种子首选、深度和截断次数。汇总首选取3个种子的平均访问份额argmax，平局沿用network prior再按候选顺序；这只是报告共识，不冒充一次标准MCTS的输出。原始每次MCTS首选保留。

Q为行动者视角的搜索回传均值，搜索内部value裁剪至[-1,1]；原始网络V仍无界。N4的Q先汇总4个世界的value_sum和visits，不能误用只含world0的`chance_diagnostics.root_value_sums`。跨种子Q使用有访问种子的等权均值，未访问标空；首选减实际的Q差只对两者均有访问的同种子求差后平均。输出Q覆盖数、最小访问数、种子范围；3个种子的范围不是置信区间。`stable_disagreement`仅标出3次均选同一替代动作、3个配对Q差均正、且实际动作每次至少16访问的情况，是进一步检查线索，不是客观失误认定。

汇总同时比较：双方主行动一致率、同配置跨种子的首选变化、同种子的预算／方法首选变化及访问分布JS差异。支付阶段单独列出，不和主行动混在分母里。原始V按行动者和先手两种视角保存，不能把相邻双方V直接相减当作某步损失。

## 信息边界和结论限制

唯一模型入口为导出的当时行动者`DecisionRequest`。全部88条输入特征逐字段与捕获NPZ一致，卡表指纹一致；86个非自动请求各验证两个独立未知世界均能原样重建。己方盲保留可见，对手盲牌不可见；不读取`reconstruction.json`，不向模型传真实后续牌序、最终结果或另一玩家私有信息。

单局的一致率不能测模型棋力，高预算也不是正确答案。Q差不是已验证的胜率差／真实收益损失；它同时受value偏差、未知世界、搜索方法与分配影响。提高N在固定总预算下会减少每世界深度，方法间结果需同时看种子敏感性。这里不进行独立续局来验证具体动作优劣。

## 产物与有限验证

- [plan.json](plan.json)：冻结模型、种子、矩阵、来源与运行参数。
- [input_validation.json](input_validation.json)：原始捕获校验、88条特征与安全世界重建检查。
- [smoke_cpu.json](smoke_cpu.json)：双方、支付、对手未知盲保留及后期局面，4种方法各5根、S16，检查N4总预算平分、逐世界Q汇总和根prior一致；仅工程验证。
- [results.md](results.md)：汇总与逐步中文表；[report.html](report.html)提供双方／阶段筛选和可见局面、前5候选展开。
- [decisions.csv](decisions.csv)：每步每组比较；[action_details.csv](action_details.csv)：完整候选排序、访问和Q覆盖。
- `search/<组名>/<种子>.json`：可恢复的逐根、逐种子、逐世界原始数据；`summary.json`、`aggregates.json`为派生汇总。
- `execution.json`、`live.json`在正式运行时生成；源码快照和临时日志在忽略的`runtime/bga_927500015_r20_analysis_20261020/`。

Greedy与实际主行动一致26/56，先手14/28、后手12/28。8个搜索组一致25–29/56，实际走法平均访问份额33.95%–43.89%。Q差换算相对各组共识首选平均约每步−1.95至−3.46个百分点得分率，受极低访问动作强烈影响；8组共同有至少16访问的39步子集为−0.47至−1.16个百分点。该子集排除了部分最大分歧，不能替代全局估计，也不能把Q差当作已确认的人类失误或棋力差。

全部派生统计由`summarize.py`生成，只读冻结模型、配置与原始搜索结果，不运行新搜索或训练，不修改原`analyze.py`。`statistics/manifest.json`记录31份来源的SHA和统计脚本身份；`statistics/validation.json`核对1824次逐世界访问／Q汇总，608条逐决策与原报告的推荐及配对Q差一致。`statistics_tests.xml`记录6项有限回归，含缺失Q的成对计算、保留正向变化及局面等权；原3项检查的`tests.xml`保留。

重建全部统计：

```bash
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 python \
  artifacts/bga_927500015_r20_analysis_20261020/summarize.py
```
