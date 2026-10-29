# R37低学习率104k续训最佳点对raw R37：Greedy

用户要求独立评测，固定启动时完整留出KL＋MSE最低的step140221，对原raw R37。CPU8 worker、greedy零搜索、1000 pair／2000局，同seed交换先后手，12贵族。原DEV种子37000000–37000999，策略seed37010000，整pair bootstrap10000次／seed37020000。

配置、命令、SHA及冻结执行源码身份见config.json、plan.json、source.json。原始结果与sidecar在同名runtime目录；本次不恢复训练。现已完成，得分**51.100%**，95%整pair区间**[49.000%, 53.175%]**；未确认超过R37。详见[results.md](results.md)与summary.json。
