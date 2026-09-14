# Splendor 基础版规则数据

- 中文规则与引擎规格：[`docs/splendor_standard_rules_zh-CN.md`](docs/splendor_standard_rules_zh-CN.md)
- 90 张发展卡：[`data/splendor_cards.csv`](data/splendor_cards.csv)
- 10 块贵族：[`data/splendor_nobles.csv`](data/splendor_nobles.csv)
- token、人数与终局常量：[`data/splendor_config.json`](data/splendor_config.json)
- 工作包 1 数据契约与阶段机：[`docs/work_package_1_contract_zh-CN.md`](docs/work_package_1_contract_zh-CN.md)
- 工作包 2A 非购买动作切片：[`docs/work_package_2a_zh-CN.md`](docs/work_package_2a_zh-CN.md)
- 工作包 2B 购买与支付切片：[`docs/work_package_2b_zh-CN.md`](docs/work_package_2b_zh-CN.md)
- 工作包 2C 贵族、完整回合与终局：[`docs/work_package_2c_zh-CN.md`](docs/work_package_2c_zh-CN.md)
- 工作包 2D 引擎集成验证报告：[`docs/work_package_2d_engine_validation_zh-CN.md`](docs/work_package_2d_engine_validation_zh-CN.md)
- 工作包 3A 基线策略与评测框架：[`docs/work_package_3a_zh-CN.md`](docs/work_package_3a_zh-CN.md)
- 工作包 3B Observation 与合法候选数值编码：[`docs/work_package_3b_zh-CN.md`](docs/work_package_3b_zh-CN.md)

当前代码已完成工作包 1、2A～2D、3A 和 3B。完整双人规则引擎已通过 10,000 局压力验收；Random、Score 和 Development 三种 baseline、成对换先手评测和纯 NumPy Observation/action feature 编码已可用。PyTorch 模型与训练仍未实现。

运行测试：`python -m pytest`

从仓库根目录运行基线评测：`PYTHONPATH=src python -m splendor.evaluate --player-a score --player-b development --seed-start 0 --seed-count 100`

数据范围仅限 2–4 人基础标准版，不含扩展、《璀璨宝石：对决》或漫威版。
