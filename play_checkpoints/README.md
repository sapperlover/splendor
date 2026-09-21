# 前端可用对手 checkpoint

将与当前规则版本、Observation feature schema 和模型配置兼容的
PPO 训练 checkpoint（`*.pt`）放在此目录。重新打开前端开局设置后，
文件会自动出现在“对手策略”列表。

前端使用 CPU 加载模型，并以 greedy/argmax 方式选择合法动作，
不会连接或修改正在训练的 learner。不兼容的 checkpoint 在选择开局时
会明确报错。

checkpoint 二进制文件默认不纳入 Git。
