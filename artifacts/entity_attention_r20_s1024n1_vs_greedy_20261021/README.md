# 第20轮S1024N1对同模型Greedy，1000 pair

用户要求先启动本组，再继续实现MLP。使用第20轮原模型，SHA `fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd`。当前12贵族规则；开局42500000–42500999、策略seed42510000、搜索seed42530000，与已完成S256N4面板相同，1000个种子分别交换先后手，共2000局。

native单个独立世界、每世界1024次、总1024次模拟；同模型greedy为logit argmax、零模拟。无根噪声、T0、c_puct1.5，FP/PTP、CPW、预算随机化关闭。复用已通过CPU和真实GPU验证的混合评测入口：12 worker、1000活跃上限、B500、microbatch64、共享slots2、owner流水线1槽、wait5ms、native CUDA Graph，checked真实对局与固定裁决。仅改变N4为N1，不改生产搜索算法。

`prepare`核对两组开局、策略seed、执行配置及完整greedy配置相同；搜索配置仅世界数不同。源码、二进制及规范数据冻结于本组runtime，后续MLP开发不影响该进程。CPU单根工程验证见`smoke.json`，不计棋力统计。结束自动核验每局S1024/N1和greedy零搜索、输出得分及10000次整pair bootstrap区间，并生成`comparison_vs_s256n4.json`：同开局共同Greedy对手下的 **S256N4减S1024N1** 得分差与配对区间，不是两种搜索直接交手的胜率。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u artifacts/entity_attention_r20_s1024n1_vs_greedy_20261021/run.py prepare
conda run --no-capture-output -n splendor python artifacts/entity_attention_r20_s1024n1_vs_greedy_20261021/launch.py
```

日志与状态：`runtime/entity_attention_r20_s1024n1_vs_greedy_20261021/run.log`、`live.json`。最终产物：`evaluation.json`、`summary.json`、`paired_outcomes.json`、`results.md`、`comparison_vs_s256n4.json`。中断后原启动命令恢复完整pair；源码、模型、牌池或固定配置改变时拒绝混接。与旧R43/10贵族S1024结果分开解释。
