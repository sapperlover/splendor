# 当前policy是否接近无噪声S1024

模型固定为主线原第30轮终点，global59898；不使用BCD诊断分支。评估完整第30轮留出集：来源124局，由第29轮教师S1024/N1自对弈采集，当前模型训练未使用这些局。使用所有有效policy决策、所有选择阶段，排除只有一个合法动作的局面。

每个公开DecisionRequest用**当前第30轮完整网络**重新做S1024/N1；`root_noise_fraction=0`且`add_root_noise=False`，温度0。分别使用两个独立世界seed33510000／33510001，保留独立结果，不称为N2搜索。单GPU FP32 eager native批处理，actor projection关闭，TF32关闭，每块最多200个根。已有自对弈的带噪声标签不作为这次无噪声目标。

主要指标为policy logits首选与访问数首选的严格Top-1一致率，以及`KL(访问数/1024 ∥ policy softmax)`，自然对数。另记录并列容忍一致率、policy首选的搜索访问份额、搜索首选的policy概率、访问分布熵、搜索重复Top-1一致率／JSD、两目标均值KL、搜索终局命中率和深度。按前中期／后期、主行动和搜索高margin组拆分，区间按整局bootstrap10000次，保留局内相关性和局面权重。

搜索重建仅消费公开请求，世界由搜索自身采样；不会把档案真实未来牌堆或对手隐藏牌传入搜索。原局轨迹回放用于恢复实际决策局面，候选顺序与编码特征对原归档逐项核对。模型SHA、NPZ SHA、请求SHA、搜索配置与种子均冻结。

首次准备输入：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_r30_search_alignment_20261011/run.py prepare
```

启动／中断后续跑：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u artifacts/entity_attention_r30_search_alignment_20261011/run.py run
```

每200个根原子保存访问数与policy log概率，恢复跳过完整分块。模型和主训练配置不修改；不训练、不新采对局、不做棋力对战。运行日志／源码快照在同名`runtime/`；必要可复算分块在本目录`raw/`，输入引用见`root_references.json`。

[结果](results.md)、[汇总](summary.json)、[逐局面数据](per_root.csv)。只重算报告使用`run.py analyze`，不重新搜索。

已完成，主要解读见[结论](conclusions.md)。[补充数据](supplement.json)给出仅主行动的前中期／后期拆分，以及“两次搜索相同、但均改写policy”的比例；复算用`python artifacts/entity_attention_r30_search_alignment_20261011/supplement.py`，只读既有搜索分块，不追加搜索。

本测量回答当前已采自对弈分布上网络与自身搜索还有多少差距。无根噪声仍保留N1世界抽样随机性；两个seed不是噪声总体估计。一致率不能直接证明决策正确或棋力；单个checkpoint也不能证明训练前后学会了多少。这里不是完全由第30轮自身生成的新on-policy局面。
