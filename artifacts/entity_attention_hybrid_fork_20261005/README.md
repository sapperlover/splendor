# 第九轮训练分支入口有限验证

交付入口为 `splendor.hybrid_train --fork-from <父hybrid run> --fork-round 9`，两组配置与完整启动/恢复命令见[专题指南](../../docs/guides/entity_attention_hybrid_fork_zh-CN.md)。从9500步继承模型、AdamW和RNG，不重新离线；每轮100局/100步，继承最近2000局并滚动替换。新轮20后在线池全N2，新轮61起不再抽原离线数据。这是交付时的状态；S512/N2后续至少完成33新轮／12800步，S1024/N2未发现正式运行产物；当前状态见专题指南。

[verification.json](verification.json) 记录最终源码身份、配置/父模型哈希、两次真实初始化命令、继承数据的split/位置数及测试结果。37项CPU回归全部通过；其中新增分支回归覆盖实际S4/N2采集/更新、已提交进度接续、优化器状态与逐参数/RNG一致的中断恢复、旧N1淘汰、父run不变、共同留出集保留、部分淘汰的父轮数据不重新导入。原混合训练、learner和replay回归也通过，未降低原resume的配置一致性要求。

两次最终源码CLI验证位于忽略的 `runtime/entity_attention_hybrid_fork_validation_20261005/final_s512_n2` 和 `final_s1024_n2`，均使用正式CUDA配置但以 `--online-rounds 0` 仅初始化：模型字节哈希与父9500步checkpoint完全一致，69个参数的AdamW状态均为9500步，继承2000局，离线/在线固定验证与共同theta_5000保持相同。未采集新局、未执行更新、未创建正式 `runs/entity_attention_hybrid_s*_n2_from_round9_seed1`。初始化过程不持有CUDA learner或推理owner；这些记录不能当成两组完整GPU训练或棋力收益验证。

CPU测试计数和用例保留在[cpu_tests.xml](cpu_tests.xml)。冗余终端输出已本地保全，位置和原始SHA见[整理清单](../repository_cleanup_20261009/manifest.json)；runtime保留源码快照与大份继承引用清单。父run数据未复制或修改，新run必须保留来源NPZ路径可读。

复核本地初始化产物并重新生成验证汇总，不训练、不采集：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_hybrid_fork_20261005/verify_initialization.py
```

该复核要求当时的源码与父run状态，后续代码变动或父run继续训练后应使用记录的源码快照/身份，不能以新的状态补填历史验证。此前有限S/N筛选的原始测量及源码身份保持不变，教师初筛不等于长期学习收益已得到证明。
