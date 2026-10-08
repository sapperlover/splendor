# OOM 后孤儿进程清理与 owner 死亡退出修复

事件发生于2026-10-05，收尾验证于2026-10-06。本次授权范围为清理已确认的孤儿进程、修复退出清理及有限回归；没有修改训练配置、缓存预算或采样算法，没有启动新训练。

内核日志 `oom_kernel.txt` 确认23:37:29发生global OOM，杀死主进程PID 435906。S512/N2分支第18个新轮次、11300步已完整保存，第19轮仅有采集manifest，尚无落盘对局。冻结配置仍为离线/在线各8 GiB数组LRU；预算不包含metadata、Python/PyTorch、编译进程及搜索worker，机器当时没有swap。

## 清理

逐一检查UID、本仓库cwd、splendor环境的multiprocessing.spawn命令、systemd收养关系和进程启动标识后，向16个已确认的孤儿进程发送SIGTERM。全部退出，无需SIGKILL。旧8个PID为360718–360725，本次新8个为469043–469050；清理前RSS合计8.4155 GiB，该数字不是完全独占物理内存的计量。逐进程记录见 `cleanup.json`。收尾的 `cleanup_verification.json` 显示本仓库残留孤儿worker/resource tracker为零，系统MemAvailable约22.1 GiB。

## 修复

`src/splendor/multiworker_gpu.py` 在worker入口通过独立daemon线程等待multiprocessing父进程sentinel。owner死亡时直接退出worker，避免Queue.put、replies.get和Python退出时队列feeder刷新阻塞。owner在worker启动之前死亡也会被已经就绪的sentinel检测到。操作系统回收worker映射及文件描述符；共享resource tracker在最后一个客户端退出后清除共享段。

正常结果发送、失败现场、owner的finally与join/terminate协议保持原调用链。owner被SIGKILL时无法承诺生成失败现场；完整提交的对局和模型保留，未完成局可通过原resume入口重新采集。此修复防止无主worker继续占用内存，不解决正常训练缓存逐步填充造成的容量压力。

## 验证

四个真实子进程SIGKILL回归分别覆盖worker启动阶段、Queue.put阻塞、等待推理结果和退出时等待feeder刷新，均确认worker退出及共享段unlink。其启动的owner使用独立进程组；测试失败时也只清理测试自己的组和共享段。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m pytest -q \
  tests/test_multiworker_gpu.py -k killed_owner --tb=short \
  --junitxml=artifacts/entity_attention_worker_cleanup_20261005/owner_death_tests.xml

conda run --no-capture-output -n splendor env PYTHONPATH=src python -m pytest -q \
  tests/test_multiworker_gpu.py tests/test_hybrid_training.py --tb=short \
  --junitxml=artifacts/entity_attention_worker_cleanup_20261005/regression_tests.xml
```

相关回归35项通过、2项因CUDA不可用跳过，包括正常采集的策略目标/访问数一致、owner/worker异常清理及分支训练的实际CPU调用链与恢复。未执行新的GPU长训练或完整棋力赛。`verification.json` 保存测试计数、源码身份、冻结配置一致性及11300步checkpoint哈希核对。

## 恢复

清理后可从第19个新轮次恢复，不重复离线学习和前18轮；此处仅交付命令，没有运行：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_fork_s512_n2.json \
  --output runs/entity_attention_hybrid_s512_n2_from_round9_seed1 \
  --resume --forever
```

本次没有增加缓存运行时覆盖入口。不能直接修改冻结配置冒充resume；若需要降低缓存预算，应另行实现执行覆盖并保留原算法配置。
