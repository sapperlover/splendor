# 贵族规则池统一为12张

2026-10-21完成，按用户明确要求将蓝绿黑333、红白44加入所有新开局使用的规范贵族池。原10张不删除、不改需求，两种新增需求各一张，均3分。

| 稳定ID | 牌名 | 白 | 蓝 | 绿 | 红 | 黑 | 分数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N11 | Rani Chennabhairadevi | 0 | 3 | 3 | 0 | 3 | 3 |
| N12 | Hino Tomiko | 4 | 0 | 0 | 4 | 0 | 3 |

双人局从完整12张洗牌抽3张、场外9张。需求只比较已购发展卡的永久加成；token和黄金不计入。规则依据及贵族来源见[规范规则](../../docs/splendor_standard_rules_zh-CN.md)。

## 生效路径

- `data/splendor_nobles.csv`补入N11/N12，`splendor_config.json`标明当前规则数据范围；Python开局读取完整目录，没有另一个自对弈贵族池。
- C++原生目录不再硬编码恰好10张，只保留90张发展卡的特征约束及贵族池足够开局抽取的检查。本机扩展已重建并原子替换，构建命令、源码与前后二进制SHA见`native_build.json`。这允许显式加载历史10张目录复现旧局，同时默认目录为12张。
- BGA选中／已领取的贵族按需求和分值映射到同一规范ID；不再临时造`BGA_N11/N12`、凑10张目录。规范池之外的需求明确失败。
- Python与批量/native两种自对弈manifest均冻结`catalog_fingerprint`并纳入semantic fingerprint。固定配对评测的manifest及summary也保存此指纹，恢复时拒绝不匹配的旧目录。
- 规则、引擎契约、特征契约、采集、原生搜索、评测、BGA指南及总览已同步；历史实验原件不改成新规则结果。

## 规则身份与旧产物

动作转移、forced-skip、replay字段都未改变，`ruleset_id`仍为`splendor-base-two-player-v2-forced-skip`，replay schema仍为2。完整规则身份必须同时比较规范数据指纹：

| 牌池 | catalog fingerprint |
| --- | --- |
| 修改前10张 | `2ccb91f7c2a8803360760ca8af13d3cc4b468605f462535046004448b6bbca08` |
| 当前12张 | `084574b026e41ad6e5c64dfe36d74fb84a7604f6a0b0eb0ab59227c32a7ff17b` |

旧规范数据完整保存在`previous_catalog/`，原SHA见`previous_identity.json`；本次12张目录快照在`current_catalog/`。旧保存局／trace可在对应代码及显式旧目录下复现；旧采集和配对评测不能直接在原目录用新规则续写，新任务使用新输出目录。既有NPZ格式可继续读取并标注为历史数据，不能宣称其来自12张池。评测的legacy恢复开关也不能绕过缺失的牌池身份。

贵族输入仍为3个槽、每张17维牌面属性，没有贵族ID embedding。现有第20轮模型严格加载成功、参数量仍为583682，checkpoint SHA未变；可以继承权重开始新规则采集，无需为这两张改网络形状。这里未验证新牌池下的棋力，也未启动训练。

历史scalar特征golden保留原JSON和digest，其回放测试改为显式读取本次保留的10张目录；新规则由当前12张目录的编码、状态和采集测试覆盖，避免用新贵族开局去重写旧golden。

## 验证

- 180项回归通过，3项普通测试进程的CUDA条件用例跳过。覆盖12张唯一需求、开局抽取、两种新贵族的资格／自动领取／3分结算、Python与C++差分、原生批次、随机节点、BGA映射、旧golden、采集manifest和评测恢复拒绝不同目录／缺失身份。原始JUnit为`tests.xml`和`pool_tests.xml`。
- 用原第20轮模型做2局CPU工程采集：native固定预算v2，以及纯随机节点＋预算随机化v3，各1局；两局开局都含N11和N12，均产生自然终局并通过Python checked完整重放。见`collection_validation.json`。
- 另做2局真实CUDA多worker工程采集：纯随机节点、随机预算、2个worker／2个活跃局／B4／双槽；两局均完成并通过checked重放。见`gpu_collection_validation.json`。
- 以上4局只验证调用链，不作正式棋力结论。未执行新规则下的大规模采集、训练或正式对战。

从仓库根目录复核相关回归：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_catalog_and_rng.py tests/test_rules_and_phases.py \
  tests/test_engine_2a.py tests/test_engine_2c.py tests/test_native_search.py \
  tests/test_native_chance_search.py tests/test_native_batch.py tests/test_bga_live.py \
  tests/test_bga_watch.py tests/test_features_golden_3b.py tests/test_noble_pool_identity.py \
  tests/test_selfplay.py tests/test_evaluation_3a.py tests/test_evaluation_pool.py
```

常规重建命令见[原生指南](../../docs/guides/native_search_zh-CN.md)。本次`build_native.py`使用已安装Torch提供的pybind11头文件，无需额外安装环境依赖；它会保留替换前的二进制，已有备份时拒绝覆盖，不要作为日常重复构建入口。验证脚本和完整命令身份见`validation.json`；NPZ、源码快照及临时产物位于`runtime/noble_pool_bga12_20261021/`。
