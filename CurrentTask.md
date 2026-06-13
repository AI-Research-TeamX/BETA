/auto-research 完成 proposal.md 中的Parse 1，优先探究当前目录下 Qwen 文件夹('/root/storage/cuisijia.csj/vlmarch/BETA/Qwen')下的模型。将整个auto-research过程中的各种有价值的重要的结果存放在 results 文件夹下

请根据 phase 1中的主要findings，优化 proposal.md 中的 phase 2 训练规划，例如：训练固定使用 第 \alpha * MAX_L 层，使用mean pooling，或者其他需要修改的。

/auto-research 先完成 proposal.md 中的Parse 2中的 Full + Probe, 只使用一层（L/2），使用 mean pooling的实验。backbone和probe都要训练，使用GRPO算法。充分、高效利用8张卡资源。

/auto-research 完成 Full 不加 Probe的GRPO训练，为了和之前完成的Full + Probe实验结果进行对比，突出Probe的作用。只需要训练backbone，不利用Probe，使用GRPO算法。充分、高效利用8张卡资源。

现在的结果展示出SFT的效果最好，是不是因为训练和测试数据是同分布的，可能RL方法优势在泛化性上？在分布外（out-of- distribution）测试任务上效果更好（我的猜想）。 /auto-research 请完成OOD数据的生成（例如：更难的/形式差距大的/... 博弈数据），然后对不同的方法产生的checkpoints进行测试，并分析测试结果。充分利用8卡资源。

/auto-research 使用 ./verl 训练框架，重跑 GRPO 实验，实验名字为 grpo_verl，与之前transformers实现的grpo训练保持相同的训练数据/训练参数，保存并评估checkpoints，分析实验结果。grpo_verl训练需要写的代码和脚本放在 verl_scripts 中。充分利用8卡资源，遇到报错自主分析解决。

/auto-research 这个grpo_verl 是不是没有考虑probe，对 grpo_verl + probe 进行实验，保存并评估checkpoints，分析实验结果。验证probe的作用。充分利用8卡资源，遇到报错自主分析解决。

现在正在进行 /root/storage/cuisijia.csj/vlmarch/BETA/proposal.md 的实验，目前看结果不符合预期。请想一些其他可以发顶会（如：ICML, ICLR, NeurIPS）的research idea，充分深度调研，deep research，将所有的调研和idea记录在 /root/storage/cuisijia.csj/vlmarch/BETA/ideas 文件夹中。

/auto-research 测试 原始模型,SFT,GRPO(verl),GRPO(verl)+Probe 的checkpoints在GTBench, TextArena benchmark上的表现（不需要测试TextArena中所有的，挑选一些）。全面测试、分析结果，评测代码、脚本、结果分析完成在 ./benchmark_eval 文件夹下。遇到报错自主解决。
