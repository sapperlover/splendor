# 256维×4层实体注意力：结构与有限验证

2026-10-14。按用户指定扩大现有实体注意力网络：hidden256、8 heads、4层状态编码、FFN1024、dropout0。预设见 [model_256x4.json](../../configs/entity_attention_v1/model_256x4.json)。复用 `entity_attention_v1` 工厂、输入特征和 policy/value 接口；checkpoint记录完整尺寸，不另建重复模型实现。

## 结构与容量

输入仍为148维全局特征、29维卡牌、17维贵族、82维候选动作。状态共有25个实体token：全局1、玩家2、市场12、双方保留牌6、贵族3、pending1。投影／类型／归属／层级embedding变为256维；同角色集合无槽位位置编码，unknown保留牌与padding区分。

状态encoder为4个Pre-LN残差self-attention block，每层8头、每头32维，FFN256→1024→256、GELU。policy将每个候选独立投影到256维，经过一层8头cross-attention读取共享状态，接LayerNorm→Linear256→GELU→Linear1输出logit。胜负value由全局token独立经过LayerNorm→Linear256→GELU→Linear1，无界输出。完整前向共享一次状态编码；value不依赖候选动作。

| 部分 | 参数 |
| --- | ---: |
| 输入投影与角色embedding | 50,432 |
| 4个状态self-attention／FFN block | 3,159,040 |
| 动作投影 | 21,248 |
| query／memory LayerNorm | 1,024 |
| policy cross-attention | 263,168 |
| policy输出头 | 66,561 |
| 胜负value头 | 66,561 |
| **主网络总计** | **3,628,034** |

历史128×2主网络532,738参数，扩大为6.810倍。新主网络FP32参数本体14,512,136 bytes，即13.83985 MiB；这不是训练总显存，还需梯度、AdamW状态、激活和运行缓存。启用已有分差／剩余回合两个辅助头时各增加66,561参数，合计3,761,156参数。辅助头升级保留同样的256×4尺寸；本次未改变辅助目标或搜索value定义。

`model_256x4.json`是模型配置片段，不是完整训练CLI配置：将其内容放入新训练配置的 `model_config`，或直接交给 `build_actor_critic`。省略尺寸的历史配置仍构建128×2；原R43权重没有被放大或覆盖，不可直接作为256×4续训状态。本次只实现与验证新容量，尚无正式训练或棋力收益。现有双网络组合类仍固定128×2，适配大模型组合属于后续工作。

## 验证

- [tests.xml](tests.xml)：22项CPU测试通过。覆盖128×2和256×4的前向、state-only／policy-only一致、候选置换等变、实体集合置换不变、padding与unknown处理；AZ／纯权重／冻结加载、未知架构／缺失权重拒绝；大模型辅助头升级保留输出、梯度到达全部参数及保存加载。
- [cpu_validation.json](cpu_validation.json)：取原R24归档Train对局的前4个真实局面，调用现行AZ learner更新1步，保存／恢复模型、AdamW与进度，继续1步后与不中断路径逐参数、逐optimizer状态完全一致。纯权重加载通过；开／中／后期三个固定根S8/N1，原生连续批量路径通过。总计约1.6秒。
- [cuda_validation.json](cuda_validation.json)：同样4样本更新1步、保存与恢复加载通过；CUDA eager与compiled的policy/value及state-only输出一致（atol2e-5、rtol2e-4），3个S8/N1原生批量根完整完成，实际调用compiled且无编译失败fallback。约30.3秒，包含编译／预热。该验证不测正式B256显存／吞吐，不进行整局棋力赛。

复核命令，从仓库根目录执行：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_entity_attention.py tests/test_value_auxiliary.py --junitxml=artifacts/entity_attention_256x4_20261014/tests.xml

conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/entity_attention_256x4_20261014/validate.py --device cpu

conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python artifacts/entity_attention_256x4_20261014/validate.py --device cuda
```

[validate.py](validate.py)只执行上述有限工程验证，随机初始化seed36400000；数据archive SHA、decision索引、preset／脚本SHA、source/native身份、设备、训练loss和编译计数保存在对应JSON。源码快照及冒烟checkpoint位于忽略的 `runtime/entity_attention_256x4_20261014/{cpu,cuda}/`，这些1步模型不用于棋力结论。测试临时模型／运行缓存不纳管，历史实验原件保留。
