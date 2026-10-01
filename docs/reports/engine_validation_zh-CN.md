# Python 规则引擎历史验收

这是 `splendor-base-two-player-v2-forced-skip` / replay schema 2 的 checked 引擎历史验收，不是当前提交全量测试，也不自动覆盖后来 C++ 规则实现。

[原始统计](../../artifacts/work_package_2d/stress-10000-rules-v2.json) 保存 10,000 局、seed 0–9999。passed=true，engineering_truncations=0、exceptions=0、illegal_action_failures=0、invariant_failures=0。checked replay 样本 132 个，mismatch=0；恢复样本 72 个，mismatch=0。

死锁审计促成本地 forced-skip 增补：MAIN_ACTION 无候选时引擎自动跳过，双方连续跳过则平局；它不是玩家可选 pass，不进入模型候选。现行权威规则与接口见 [标准规则](../splendor_standard_rules_zh-CN.md)、[引擎契约](../specs/engine_contract_zh-CN.md)。

原汇总没有记录精确测量 commit，不补填当前 HEAD。native 的维护入口为 `test_native_search.py` / `test_native_batch.py`；历史压力结果不能代替规则改动后的相关回归。
