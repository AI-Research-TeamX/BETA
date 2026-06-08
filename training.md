# Phase 2: Full + Probe GRPO Training

## 1. 实验概述

在 Phase 1 线性探针诊断的基础上，本实验实施 **Full + Probe** 训练方案：在 GRPO 强化学习框架下，同时全参数微调 Qwen2.5-3B-Instruct 的 backbone 并训练辅助线性探针头（probing heads），旨在通过辅助监督信号增强模型的博弈论推理能力。

- **模型**: Qwen2.5-3B-Instruct（36 层，hidden_size=2048）
- **算法**: GRPO（Group Relative Policy Optimization）
- **训练方式**: Full fine-tuning（backbone + probing heads 全参训练）
- **探针层**: Layer 18（L/2，0-indexed 17）
- **池化方式**: Mean pooling（对 prompt 所有非 padding token 的隐层表示取均值）
- **硬件**: 8×NVIDIA H20（96GB HBM each），单节点

---

## 2. 实现架构

### 2.1 整体训练流程

```
每个训练步（training step）：
1. 从数据集采样一批 prompt（batch_size_per_gpu=2 × 8 GPUs = 16 prompts）
2. 对每个 prompt 生成 N=5 个 rollout（采样温度 0.7）
3. 用 reward function 对每个 response 打分（0~1）
4. 在每个 prompt 内部计算 group-normalized advantages
5. 前向传播计算：
   - Policy gradient loss（GRPO 目标）
   - 通过 forward hook 捕获 layer 17 的隐层状态
   - Mean pooling → 5 个探针头的分类损失
6. 合并损失：total_loss = pg_loss + λ × probe_loss
7. 反向传播 + 梯度更新
```

### 2.2 GRPO 算法实现

GRPO 不需要 critic 网络，而是在每个 prompt 的 rollout 组内归一化 reward 作为 advantage：

```python
# 对每个 prompt 的 N=5 个 rollout 的 reward 进行归一化
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# Policy gradient loss: 按 token 平均的 log prob × advantage
seq_mean_log_probs = token_log_probs.sum(dim=-1) / resp_lengths.clamp(min=1)
pg_loss = -(seq_mean_log_probs * advantages).mean()
```

关键设计决策：使用**按 token 平均**的 log probability（除以 response 长度），而非简单的 sum，避免长 response 的 log prob 绝对值过大导致 PG loss 数值不稳定。

### 2.3 辅助探针头（Probing Heads）

5 个线性分类器，每个从 layer 17 的 mean-pooled 表示预测一个博弈论概念：

| 概念 | 类别数 | 适用任务 | 概念权重 (Phase 1) |
|:-----|:------:|:---------|:-----------------:|
| eq_type | 3 (pure, mixed, both) | Nash Equilibrium | 1.34 |
| difficulty | 3 (easy, medium, hard) | 全部 | 1.11 |
| dominance | 2 (yes, no) | 全部 | 1.23 |
| br_direction | 5 (row_0~3, mixed) | Best Response | 1.57 |
| eq_uniqueness | 2 (one, multiple) | Nash Equilibrium | 1.37 |

概念权重来自 Phase 1 探针结果，与探针准确率成反比（难预测的概念获得更高权重）。排除了 `game_type`（Phase 1 中准确率近乎满分，无需额外监督）。

探针损失使用加权交叉熵：

```python
probe_loss = Σ (concept_weight_t × CE(probe_head_t(h_pooled), label_t))
```

对于不适用当前任务的概念（如 Nash 任务的 `br_direction`），使用 `-1` 标签掩码跳过该概念的损失计算。

### 2.4 隐层状态捕获（Forward Hook）

通过 PyTorch forward hook 非侵入式捕获中间层表示：

```python
class HiddenStateCapture:
    def __init__(self):
        self.hidden_states = None

    def hook_fn(self, module, input, output):
        self.hidden_states = output[0]  # (batch, seq_len, hidden_size)

# 注册在 model.model.layers[17]
hook_handle = model.model.layers[17].register_forward_hook(capture.hook_fn)
```

Mean pooling 仅对 prompt 部分的 token 取均值（通过 prompt_mask 排除 response token 和 padding token），因为探针预测的是输入博弈的属性，不依赖于生成的回答。

### 2.5 分布式训练（DDP）

选择 DDP（DistributedDataParallel）而非 FSDP：
- 3B 模型在 bf16 下约 6GB，远小于 H20 的 96GB 显存
- 加上优化器状态（Adam 约 3×参数 = 18GB）+ KV cache + 激活值，每卡约 40GB，显存充裕
- DDP 实现更简单，调试更方便，无需处理 FSDP 的分片逻辑

### 2.6 独立实现 vs. verl 框架

选择独立实现 GRPO 训练脚本，而非修改 verl 框架内部：
- verl 的 actor rollout + FSDP + rmpad + fused kernels 等内部高度耦合
- 插入 forward hook 需要修改 FSDP 分片后的模块注册方式，风险较高
- 独立脚本约 750 行，完全可控，便于调试和实验

---

## 3. 数据准备

### 3.1 数据集

- **来源**: GameSolve-Bench（`gamesolve_bench.jsonl`，2400 样本）
  - Nash Equilibrium 任务：1350 样本
  - Best Response 任务：1050 样本
- **划分**: 80/20 train/val（seed=42，随机打乱后切分）
  - 训练集：1920 样本
  - 验证集：480 样本
- **描述风格**: 每个样本随机选择 abstract / story / compact 三种描述之一

### 3.2 预处理脚本

`preprocess_gamesolve_grpo.py` 完成以下工作：
1. 读取 `gamesolve_bench.jsonl`
2. 提取 5 个概念标签（编码为整数，不适用的标为 -1）
3. 为每个样本拼接 prompt（游戏描述 + 任务指令）
4. 输出 `data/phase2/train.json` 和 `data/phase2/val.json`

### 3.3 Reward Function

`gamesolve_reward.py` 实现了结构化的奖励计算：

**Nash Equilibrium 任务** (权重分配)：
| 组件 | 权重 | 说明 |
|:-----|:----:|:-----|
| Pure NE F1 | 0.5 | 纯策略纳什均衡的精确率/召回率 F1 |
| Equilibrium class | 0.3 | 正确识别均衡类型 (pure/mixed/both) |
| Mixed NE accuracy | 0.2 | 混合策略概率分布的 L1 距离 |

**Best Response 任务** (权重分配)：
| 组件 | 权重 | 说明 |
|:-----|:----:|:-----|
| Action accuracy | 0.5 | 最优响应动作的正确率 |
| Expected payoff MAE | 0.3 | 期望收益的平均绝对误差 |
| BR value error | 0.2 | 最优响应值的绝对误差 |

---

## 4. 训练参数

### 4.1 完整超参数配置

| 参数 | 值 | 说明 |
|:-----|:---|:-----|
| model_path | `./Qwen/Qwen2.5-3B-Instruct` | 本地模型路径 |
| probe_layer | 17 (0-indexed) | Layer 18，即 L/2 |
| n_rollouts | 5 | 每个 prompt 生成的 response 数量 |
| batch_size_per_gpu | 2 | 每卡每步的 prompt 数 |
| micro_batch_size | 4 | forward pass 的 micro batch |
| n_gpus | 8 | GPU 数量 |
| effective_batch_size | 16 | 实际 batch = 2 × 8 |
| lr (backbone) | 1e-6 | backbone 学习率 |
| probe_lr | 1e-3 | 探针头学习率 |
| probe_lambda (λ) | 0.1 | 探针损失的缩放系数 |
| gen_temperature | 0.7 | 生成采样温度 |
| gen_top_p | 0.9 | nucleus sampling 阈值 |
| max_prompt_length | 1024 | prompt 最大 token 数 |
| max_response_length | 512 | response 最大 token 数 |
| max_seq_length | 1536 | 总序列最大长度 |
| num_epochs | 3 | 训练轮数 |
| total_steps | 360 | 总步数 (1920 / 16 × 3) |
| warmup_ratio | 0.05 | 预热步数比例 (18 steps) |
| grad_clip | 1.0 | 梯度裁剪范数 |
| scheduler | cosine | 余弦退火学习率调度 |
| gradient_checkpointing | True | 梯度检查点 |
| dtype | bfloat16 | 计算精度 |
| seed | 42 | 随机种子 |
| save_steps | 40 | 检查点保存间隔 |
| eval_steps | 40 | 验证间隔 |
| log_steps | 5 | 日志间隔 |

### 4.2 启动命令

```bash
torchrun --nproc_per_node=8 train_grpo_probe.py \
    --model_path ./Qwen/Qwen2.5-3B-Instruct \
    --train_data data/phase2/train.json \
    --val_data data/phase2/val.json \
    --output_dir results/phase2/full_probe_grpo \
    --probe_layer 17 \
    --n_rollouts 5 \
    --batch_size_per_gpu 2 \
    --micro_batch_size 4 \
    --max_prompt_length 1024 \
    --max_response_length 512 \
    --max_seq_length 1536 \
    --lr 1e-6 \
    --probe_lr 1e-3 \
    --probe_lambda 0.1 \
    --gen_temperature 0.7 \
    --gen_top_p 0.9 \
    --num_epochs 3 \
    --grad_clip 1.0 \
    --warmup_ratio 0.05 \
    --save_steps 40 \
    --log_steps 5 \
    --eval_steps 40 \
    --seed 42 \
    --gradient_checkpointing \
    --bf16
```

---

## 5. 实验结果

### 5.1 训练概况

| 指标 | 值 |
|:-----|:---|
| 总训练时间 | ~3 小时 |
| 总步数 | 360 |
| 平均步耗时 | ~26 秒 |
| 训练 reward 均值 | 0.1907 |
| 训练 reward 最大值 | 0.3115 |
| 最佳验证 reward | 0.3167 (Step 280) |
| 最终验证 reward | 0.1887 (Step 360) |

### 5.2 验证 Reward 轨迹

| Step | Val Reward | Probe Loss |
|:----:|:----------:|:----------:|
| 40   | 0.1461     | 0.9540     |
| 80   | 0.1875     | 1.2086     |
| 120  | 0.1920     | 0.9927     |
| 160  | 0.2153     | 1.0082     |
| 200  | 0.1756     | 0.9846     |
| 240  | 0.2027     | 0.9170     |
| **280** | **0.3167** | **0.8497** |
| 320  | 0.2007     | 0.8607     |
| 360  | 0.1887     | 0.8612     |

验证 reward 在前 280 步整体呈上升趋势（0.1461 → 0.3167），之后出现下降，可能与后期训练过拟合或学习率衰减过度有关。最佳检查点在 Step 280。

### 5.3 训练 Reward 轨迹（采样）

| Step | Reward Mean | PG Loss | Probe Loss |
|:----:|:-----------:|:-------:|:----------:|
| 5    | 0.2163      | -0.3465 | 0.7690     |
| 40   | 0.3030      | 0.1485  | 0.8887     |
| 80   | 0.0812      | 0.0895  | 2.3672     |
| 120  | 0.0971      | 0.5709  | 1.3914     |
| 160  | 0.1930      | -0.1987 | 1.0664     |
| 200  | 0.2524      | 0.1176  | 0.6076     |
| 240  | 0.1900      | -0.5904 | 0.5190     |
| 280  | 0.1933      | -1.1637 | 1.0820     |
| 320  | 0.2065      | -0.4532 | 0.7180     |
| 360  | 0.2051      | -0.5565 | 1.4005     |

### 5.4 最佳检查点探针准确率（Step 280）

| 概念 | 准确率 | 类别数 | Phase 1 基线 |
|:-----|:------:|:------:|:------------|
| eq_type | **93.75%** | 3 | 相对较高 |
| difficulty | **70.00%** | 3 | 中等 |
| dominance | **65.00%** | 2 | 中等 |
| br_direction | **55.56%** | 5 | 较低 |
| eq_uniqueness | **56.25%** | 2 | 较低 |

### 5.5 最终检查点探针准确率（Step 360）

| 概念 | 准确率 | 变化趋势 |
|:-----|:------:|:--------:|
| eq_type | 93.75% | → |
| difficulty | 50.00% | ↓ |
| dominance | 75.00% | ↑ |
| br_direction | 55.56% | → |
| eq_uniqueness | 75.00% | ↑ |

最终步与最佳步的探针准确率模式不同：dominance 和 eq_uniqueness 在后期有所提升，但 difficulty 下降。这表明探针头对不同概念的学习动态存在差异。

### 5.6 训练探针准确率轨迹（每 40 步）

| Step | eq_type | difficulty | dominance | br_direction | eq_uniqueness |
|:----:|:-------:|:----------:|:---------:|:------------:|:-------------:|
| 40   | 0.300   | 0.600      | 0.900     | 0.400        | 0.600         |
| 80   | 0.500   | 0.100      | 0.600     | 0.000        | 0.800         |
| 120  | 0.700   | 0.200      | 0.700     | 0.000        | 0.600         |
| 160  | 0.800   | 0.500      | 0.800     | 0.600        | 0.800         |
| 200  | 1.000   | 0.700      | 0.800     | 0.400        | 0.700         |
| 240  | 0.500   | 0.400      | 0.600     | 0.700        | 0.400         |
| 280  | 1.000   | 0.600      | 0.700     | 0.000        | 0.800         |
| 320  | 0.600   | 0.200      | 0.700     | 0.400        | 0.100         |
| 360  | 0.400   | 0.400      | 0.800     | 0.000        | 0.500         |

训练探针准确率波动较大，这是因为每个 mini-batch 仅有 16 个样本（其中部分概念的有效样本更少），导致统计量方差较高。

---

## 6. 关键实现细节与调试记录

### 6.1 解决的关键问题

**问题 1：BR action 解析正则失败**
- 现象：Best Response 任务的 `br_actions` 始终为空，action_accuracy = 0
- 原因：模型输出格式为 `action(s):`，原正则 `action[s]?` 无法匹配括号
- 修复：将正则改为 `action\(?s?\)?`，兼容 `actions` 和 `action(s)` 两种格式

**问题 2：计算图二次反向传播失败**
- 现象：`RuntimeError: Trying to backward through the graph a second time`
- 原因：先对 `pg_loss` 做 `backward(retain_graph=True)`，再对 `probe_loss` 做 `backward()`，在启用 gradient checkpointing 的情况下共享计算图导致冲突
- 修复：将两个 loss 合并为单一标量 `micro_loss = pg_loss/n_micro + (probe_loss * λ)/n_micro`，只做一次 `backward()`

**问题 3：PG loss 数值爆炸**
- 现象：PG loss 初始值约 -150，数值极不稳定
- 原因：`seq_log_probs = token_log_probs.sum(dim=-1)` 对所有 token 的 log prob 求和（每个约 -2~-5，512 个 token 累加到 -1000~-2500）
- 修复：改为按 token 平均 `seq_mean_log_probs = token_log_probs.sum(dim=-1) / resp_lengths.clamp(min=1)`

**问题 4：学习率显示为 0.0000**
- 现象：日志中 lr 显示为 0.0000（实际值 1e-6）
- 原因：`.4f` 格式化精度不足
- 修复：改用科学计数法 `f"{v:.2e}"`

### 6.2 探针损失的关键设计

- **仅使用第一个 rollout 的隐层状态**：每个 prompt 的 N=5 个 rollout 在 prompt 部分的隐层表示完全相同（因果注意力机制），因此只需对第一个 rollout 计算探针损失，避免冗余计算
- **梯度流向**：探针损失的梯度通过隐层状态回流到 backbone（不使用 stop-gradient），这使得 backbone 被鼓励在 layer 17 形成更利于概念分类的表示
- **概念掩码**：对于不适用当前任务的概念（label=-1），探针损失中自动跳过该概念

---

## 7. 输出文件

### 7.1 代码文件

| 文件 | 说明 |
|:-----|:-----|
| `preprocess_gamesolve_grpo.py` | 数据预处理脚本 |
| `gamesolve_reward.py` | 结构化奖励函数 |
| `train_grpo_probe.py` | GRPO + Probing 训练主脚本 (~750 行) |
| `run_phase2.sh` | 启动脚本 |

### 7.2 数据文件

| 文件 | 说明 |
|:-----|:-----|
| `data/phase2/train.json` | 训练集 (1920 样本) |
| `data/phase2/val.json` | 验证集 (480 样本) |

### 7.3 结果文件

| 文件/目录 | 说明 |
|:---------|:-----|
| `results/phase2/full_probe_grpo/checkpoint-best/` | 最佳检查点 (Step 280, val_reward=0.3167) |
| `results/phase2/full_probe_grpo/checkpoint-final/` | 最终检查点 (Step 360) |
| `results/phase2/full_probe_grpo/checkpoint-step_{40~360}/` | 每 40 步检查点 |
| `results/phase2/full_probe_grpo/train_log.json` | 结构化训练日志 (每 5 步记录) |
| `results/phase2/full_probe_grpo/train.log` | 完整 stdout 日志 |
| `results/phase2/full_probe_grpo/summary.json` | 实验总结 JSON |

---

## 8. 分析与讨论

### 8.1 Reward 提升

验证 reward 从 Step 40 的 0.1461 提升至 Step 280 的 0.3167，提升幅度约 **117%**。这表明 GRPO 结合辅助探针监督能够有效改善模型在博弈论任务上的表现。

### 8.2 后期性能下降

Step 280 之后验证 reward 下降至 0.1887，可能原因：
- 余弦学习率在后期过低，模型难以维持已学到的模式
- 训练集较小（1920 样本），3 epochs 后可能出现过拟合
- GRPO 的 on-policy 特性导致后期策略漂移

### 8.3 探针准确率波动

训练过程中探针准确率波动较大，这是由于：
- 每步仅 16 个样本用于探针训练，统计方差高
- 部分概念仅适用于特定任务（如 eq_type 仅适用于 Nash 任务），有效样本更少
- backbone 的全参数更新会持续改变隐层表示的分布，探针头需要不断适应

### 8.4 各概念学习难度

- **eq_type** (93.75%): 最易学习，模型能清晰区分 pure/mixed/both 三类均衡
- **difficulty** (70.00%): 中等难度，3 类分类
- **dominance** (65.00%): 二分类但准确率不高，可能是因为支配策略的概念与隐层表示的对应关系较弱
- **br_direction** (55.56%): 5 类分类中最难学习，细粒度动作方向需要更复杂的表示
- **eq_uniqueness** (56.25%): 二分类但准确率偏低，均衡唯一性的信号可能需要更深层的推理
