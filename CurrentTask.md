/auto-research 完成 proposal.md 中的Parse 1，优先探究当前目录下 Qwen 文件夹('/root/storage/cuisijia.csj/vlmarch/BETA/Qwen')下的模型。将整个auto-research过程中的各种有价值的重要的结果存放在 results 文件夹下

请根据 phase 1中的主要findings，优化 proposal.md 中的 phase 2 训练规划，例如：训练固定使用 第 \alpha * MAX_L 层，使用mean pooling，或者其他需要修改的。

/auto-research 先完成 proposal.md 中的Parse 2中的 Full + Probe, 只使用一层（L/2），使用 mean pooling的实验。backbone和probe都要训练，使用GRPO算法。充分、高效利用8张卡资源。

/auto-research 完成 Full 不加 Probe的GRPO训练，为了和之前完成的Full + Probe实验结果进行对比，突出Probe的作用。只需要训练backbone，不利用Probe，使用GRPO算法。充分、高效利用8张卡资源。
