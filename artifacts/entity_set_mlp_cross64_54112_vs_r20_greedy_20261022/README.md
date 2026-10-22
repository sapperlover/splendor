# Cross64／角色std0.05 MLP 54112步对R20：1000pair greedy

2026-10-22用户授权独立评测。本面板在对战前固定候选为54112步：原8万局训练已中断，六个已完成百分比节点（截至81168步）内完整留出KL＋胜负MSE最低，分别为0.4729014151／0.7726106593，联合1.2455120743。选择整个checkpoint，不拼接value；没有把未完成的原270557步训练视为结束，也不根据棋力结果重新选点。六节点及SHA见[candidate_selection.json](candidate_selection.json)。

候选为906434参数的256×4残差MLP，包含64维／2头动作cross-attention、角色初始化std0.05、32维动作旁路及乘积。对手为原第20轮raw attention。双方确定性greedy，当前12贵族池；1000个开局seed43200000–43200999，每个交换先后手，共2000局。8个CPU worker，无搜索；整pair bootstrap10000次，seed43220000。

复用此前MLP／attention对R20面板的开局、评测配置及同版本评测代码；当前候选训练源码／native／规则快照见[execution_plan.json](execution_plan.json)中的source记录。模型SHA、规则指纹和选择口径冻结在[config.json](config.json)。本次没有恢复训练、改配置或写原训练的selection.json。

已完成1000pair／2000局，耗时117.98秒。新MLP **858胜、13和、1129负，得分43.225%**（平局折半），整pair bootstrap 95%区间 **[41.075%,45.325%]**。先手得分43.55%，后手42.90%；区间整体低于50%，支持本次候选greedy弱于raw R20。

异常、非法动作、工程截断及不变量失败均为0；有1次规则允许的强制跳过（seed43200705、候选先手，该局正常终局）及1局确定性策略循环判和。已重新核对双方checkpoint、配置、冻结源码与规则身份、1000组交换先后手开局及配对置信区间，原始结果见[result.json](result.json)，核验见[summary.json](summary.json)，复现命令及退出状态见[execution_plan.json](execution_plan.json)。

相同开局、规则及R20对手下，旧无cross MLP的94695步候选曾得37.075%，attention128的94695步候选得54.400%。这些是各自对共同对手的结果；新MLP仅训至54112步的选点与另外两者训练步数不同，不能据此独立归因于cross、角色初始化或网络结构。Greedy只评价policy行为，未测本候选value或搜索棋力。
