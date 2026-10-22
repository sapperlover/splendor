# 第20轮S256N4对同模型Greedy，1000 pair

用户授权本组正式对战并允许复用采集的12 worker配置。双方均使用前面BGA分析及128根稳定性实验的第20轮原模型：`runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt`，SHA `fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd`。

搜索方是native独立确定化：**每世界256次 × 4世界 = 总1024次模拟**，无根噪声、T=0、c_puct=1.5，CPW、Forced Playouts、Policy Target Pruning及预算随机化关闭。Greedy为原策略logit的argmax、并列取最小候选下标，零模拟。

当前规范数据为12贵族池，从中抽3张入场，牌池指纹 `084574b026e41ad6e5c64dfe36d74fb84a7604f6a0b0eb0ab59227c32a7ff17b`。与之前已完成的随机节点S128对Greedy（旧10贵族池）开局分布不同，不把两组分数直接当作同开局配对差。

## 执行与统计

- 1000 pair，共2000局；种子42500000–42500999，每个种子交换先后手。策略seed基数42510000，搜索seed42530000；bootstrap42520000、10000次按完整pair重采样，和棋半分。
- `native_cuda_pool`：12个搜索/规则worker、1000活跃局上限、GPU B500、local microbatch64、每worker共享slots2、owner流水线1槽、等待5ms。候选桶16/32/48/64，静态batch桶16/32/64/128/192/256/384/500，FP32、native-cudagraph、失败明确停止。
- 双方同checkpoint SHA，GPU owner只加载一个模型；worker不创建CUDA上下文。每个请求只计算当前行动方。Greedy现在通过root-only native作业接入相同批处理，只推理根，不创建搜索世界或展开树。
- 真实对局由checked runner推进，保留自然终局、确定性策略循环判和、异常、非法行动、工程截断和不变量失败分类；400完成回合上限不作为普通平局计分。
- 结束自动核验1000个完整pair、双方模型身份、12贵族指纹、每局总预算1024及N4、greedy零模拟/零世界/零搜索转移，并复算配对区间；输出胜和负、先后手、决策量、搜索与owner/worker执行统计。

## 代码变化与验证

`NativeGreedyPolicy`和C++可选`root_only`只用于裸网根前向，原native/native_chance默认仍为false。`evaluation_pool`允许native MCTS和greedy任意配对；纯greedy也可走相同执行器。已有`native_batched`的eager greedy路径保留。

相关回归共53项，50通过、3项条件CUDA跳过：原native/chance搜索、批传输、双模型pool，以及新增的原始logit argmax/并列、零搜索计数、greedy放任一侧和两侧时与串行checked终局逐局对照、恢复与错误分类。见`tests.xml`。扩展重建记录见`native_build.json`；旧二进制保存在本实验runtime的`before-*.so`，实际替换为原子操作，已有进程不热切换。

真实GPU工程验证见`smoke.py`和完成后生成的`smoke.json`：128个已有主行动根的批量greedy与原GPU eager逐一对比；一个完整256×4根核验预算；另6 pair小预算（每世界2次×4）走完整12 worker、单槽GPU owner及checked终局。均不计正式棋力结果。

## 命令、恢复与结果

准备和启动（仓库根目录）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u artifacts/entity_attention_r20_s256n4_vs_greedy_20261021/run.py prepare
conda run --no-capture-output -n splendor python artifacts/entity_attention_r20_s256n4_vs_greedy_20261021/launch.py
```

本组使用新目录，未覆盖先前实验。准备时冻结源码、native二进制和12贵族规范表；后台从runtime快照导入，后续源码或默认牌池改变不影响本组。中断后原启动命令可恢复，已完成pair跳过；启动器拒绝重复启动仍存在的PID。模型、配置、runner及源码快照SHA改变时明确拒绝恢复。

- 固定配置、版本身份：[config.json](config.json)、[plan.json](plan.json)、`catalog/`、`execution.json`。
- 后台日志：`runtime/entity_attention_r20_s256n4_vs_greedy_20261021/run.log`；状态同目录`live.json`。`launch.json`记录PID和命令。
- 正式逐局恢复记录：本目录`evaluation.json.games/`；每10 pair写`evaluation.json`，不能把未完成面板当最终结果。
- 完成后自动生成`summary.json`、`paired_outcomes.json`、`results.md`，并更新实验机器索引；启动不代表评测完成，也不证明改作教师后的训练收益。
