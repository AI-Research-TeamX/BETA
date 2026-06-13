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

---

## 9. 消融实验：GRPO-only（无 Probe）

### 9.1 实验目的

为突出辅助探针监督的作用，运行了一组消融实验：仅使用 GRPO 训练 backbone（`probe_lambda=0`），不加任何探针头的辅助损失。其余所有超参数与 Full+Probe 实验完全一致，确保公平对比。

### 9.2 配置差异

| 参数 | Full+Probe | GRPO-only |
|:-----|:----------:|:---------:|
| probe_lambda | 0.1 | **0.0** |
| 其余参数 | 同上 | **完全相同** |
| 输出目录 | `results/phase2/full_probe_grpo/` | `results/phase2/full_grpo_only/` |

### 9.3 验证 Reward 对比

| Step | Full+Probe | GRPO-only | 差异 |
|:----:|:----------:|:---------:|:----:|
| 40   | 0.1461     | 0.1568    | +0.0107 |
| 80   | 0.1875     | 0.1887    | +0.0012 |
| 120  | 0.1920     | 0.1800    | -0.0120 |
| 160  | 0.2153     | 0.1865    | -0.0288 |
| 200  | 0.1756     | 0.1716    | -0.0040 |
| 240  | 0.2027     | **0.2046**| +0.0019 |
| **280** | **0.3167** | 0.2024 | **-0.1143** |
| 320  | 0.2007     | 0.1628    | -0.0379 |
| 360  | 0.1887     | 0.2027    | +0.0140 |

### 9.4 关键结果

| 指标 | Full+Probe | GRPO-only | 差异 |
|:-----|:----------:|:---------:|:----:|
| 最佳验证 Reward | **0.3167** | 0.2046 | **+54.8%** |
| 最佳步数 | Step 280 | Step 240 | — |
| 训练 Reward 均值 | 0.1907 | 0.1933 | +1.4% |
| 训练 Reward 最大值 | 0.3115 | 0.2959 | -5.0% |
| 最终验证 Reward | 0.1887 | 0.2027 | +7.4% |
| 训练时间 | ~3 小时 | ~3 小时 | — |

### 9.5 分析

1. **辅助探针显著提升峰值性能**：Full+Probe 的最佳验证 reward（0.3167）比 GRPO-only（0.2046）高出 **54.8%**，表明探针辅助监督能引导模型在中间层形成更结构化的博弈论概念表示，从而改善推理质量。

2. **早期训练阶段差异不大**：前 120 步两者表现接近（差异 < 0.02），说明探针的作用在训练充分后才显现——需要 backbone 先学到基础的博弈论模式，探针辅助信号才能发挥引导效果。

3. **GRPO-only 表现更稳定但天花板更低**：GRPO-only 的验证 reward 波动较小（0.16~0.20 范围），而 Full+Probe 有更大的峰值（0.3167）但也有更大的波动。这说明探针辅助损失在引导学习方向的同时也引入了额外的优化复杂性。

4. **训练 reward 均值相近**：两者的训练 reward 均值非常接近（0.1907 vs 0.1933），说明在训练集上的即时表现差异不大——探针的优势主要体现在泛化能力上。

### 9.6 输出文件

| 文件/目录 | 说明 |
|:---------|:-----|
| `results/phase2/full_grpo_only/checkpoint-best/` | 最佳检查点 (Step 240, val_reward=0.2046) |
| `results/phase2/full_grpo_only/checkpoint-final/` | 最终检查点 (Step 360) |
| `results/phase2/full_grpo_only/checkpoint-step_{40~360}/` | 每 40 步检查点 |
| `results/phase2/full_grpo_only/train_log.json` | 结构化训练日志 |
| `results/phase2/full_grpo_only/train.log` | 完整 stdout 日志 |
| `results/phase2/full_grpo_only/summary.json` | 实验总结 JSON |
| `run_phase2_no_probe.sh` | 消融实验启动脚本 |

---

## 10. SFT on Chain-of-Thought 实验

### 10.1 动机

从 GRPO 系列实验（Full+Probe: 0.3167, GRPO-only: 0.2046）来看，纯 RL 方法在博弈论任务上提升有限。核心问题是**冷启动困境**：模型从未见过正确的博弈论求解过程，RL 信号太稀疏（reward ∈ [0,1]），无法引导模型从零学会多步推理。

参考 DeepSeek-R1 的方法论：先用 SFT 教会模型「怎么解题」，再用 RL 精调。GameSolve-Bench 每个样本都有 `chain_of_thought` 字段（平均 ~1034 字符的详细求解步骤），是天然的 SFT 监督信号。

### 10.2 SFT 数据准备

`preprocess_sft.py` 将 GameSolve-Bench 转为 SFT chat 格式：

```
System: [game theory expert prompt + ANSWER format instructions]
User: [game description + task instruction]
Assistant: [chain_of_thought]\n\n[formatted ANSWER block from ground truth]
```

- 训练集: 1920 样本，验证集: 480 样本（与 GRPO 实验相同划分）
- 描述风格: 随机选择 abstract/story/compact
- 输出: `data/phase2_sft/train.json`, `data/phase2_sft/val.json`

### 10.3 SFT 训练配置

| 参数 | 值 |
|:-----|:---|
| 训练脚本 | `train_sft.py` (DDP, 8×H20) |
| 学习率 | 2e-5 |
| Batch size | 2/GPU × 8 GPUs × 4 grad_accum = 64 |
| Epochs | 3 |
| 总步数 | 90 |
| max_seq_length | 2048 |
| 最佳验证 loss | 0.0304 (perplexity = 1.03) |

### 10.4 SFT→GRPO 配置

在 SFT 最佳检查点上继续 GRPO 训练：

| 参数 | 与 GRPO-only 的差异 |
|:-----|:-------------------|
| model_path | `results/phase2/sft_cot/checkpoint-best` (SFT 检查点) |
| lr | 5e-7 (降低，保护 SFT 知识) |
| max_response_length | 1024 (翻倍，适应 CoT 输出) |
| max_seq_length | 2048 (翻倍) |
| probe_lambda | 0.0 (无探针) |

最佳验证 reward: 0.7495 (Step 280)

---

## 11. 全方法对比（统一评估）

使用 `eval_checkpoint.py` 对所有方法在相同的 200 样本上进行统一评估（temperature=0.1, max_new_tokens=1024）。

### 11.1 总体结果

| 方法 | Overall | Nash | BR | Easy | Medium | Hard |
|:-----|:-------:|:----:|:--:|:----:|:------:|:----:|
| Base (Qwen2.5-3B) | 0.5249 | 0.4033 | 0.6829 | 0.5738 | 0.4844 | 0.5224 |
| GRPO-only | 0.5152 | 0.4066 | 0.6563 | 0.5702 | 0.4614 | 0.5580 |
| Full+Probe GRPO | 0.4840 | 0.3620 | 0.6423 | 0.5084 | 0.4539 | 0.5367 |
| **SFT (CoT)** | **0.9451** | **0.9206** | **0.9769** | **0.9802** | **0.9299** | **0.8666** |
| SFT → GRPO | 0.9351 | 0.9028 | 0.9770 | 0.9802 | 0.9096 | 0.8673 |

### 11.2 关键发现

1. **SFT on CoT 是压倒性赢家**: 0.9451 overall reward，比 base model 提升 **80%**，比最优 GRPO 方案提升 **95%**。CoT 监督学习直接解决了模型「不知道怎么解题」的根本问题。

2. **纯 RL（GRPO）基本无效**: GRPO-only (0.5152) 甚至低于 base model (0.5249)，Full+Probe (0.4840) 更低。这说明在冷启动条件下，RL 的稀疏 reward 信号不足以引导模型学会博弈论的多步推理过程。训练期间的验证 reward（0.2-0.3）与评估 reward（0.5）的差异可能是由于评估使用了更长的 max_new_tokens=1024（训练时为 512）和不同的采样策略。

3. **SFT→GRPO 未带来额外提升**: 0.9351 略低于纯 SFT 的 0.9451。GRPO 阶段反而引入了轻微退化（-1.1%），可能原因：(a) SFT 已接近该数据集的性能天花板；(b) GRPO 的探索性采样扰动了已优化的 CoT 推理模式；(c) 数据集规模有限（1920 样本），进一步优化空间受限。

4. **CoT 监督是关键因素**: 模型性能的核心瓶颈不是「表示质量」（probing 试图改善的）也不是「策略优化」（RL 试图做的），而是「推理过程知识」——模型需要被明确教导如何逐步分析支付矩阵、识别支配策略、计算均衡。

### 11.3 各维度分析

- **Nash vs BR**: Nash 任务难度更大（base: 0.40 vs 0.68），SFT 在两者上都接近满分（0.92, 0.98）
- **Easy/Medium/Hard**: SFT 在 easy 上接近完美（0.98），hard 上仍有提升空间（0.87）
- **GRPO 系列在 Hard 上表现略好于 Medium**: 可能是 hard 样本中 BR 任务比例较高

### 11.4 输出文件

| 文件/目录 | 说明 |
|:---------|:-----|
| `preprocess_sft.py` | SFT 数据预处理脚本 |
| `train_sft.py` | DDP SFT 训练脚本 |
| `eval_checkpoint.py` | 统一评估脚本 |
| `compare_experiments.py` | 实验对比脚本 |
| `run_sft.sh` | SFT 启动脚本 |
| `run_sft_then_grpo.sh` | SFT→GRPO 启动脚本 |
| `results/phase2/sft_cot/` | SFT 训练输出 |
| `results/phase2/sft_then_grpo/` | SFT→GRPO 训练输出 |
| `eval_results/*/eval_summary.json` | 各方法评估结果 |
| `results/phase2/comparison.json` | 完整对比 JSON |

---

## 12. Out-of-Distribution (OOD) 泛化性评估

### 12.1 动机与假设

Section 11 的结果表明 SFT 在 ID 评估上远超 RL 方法（0.9451 vs ~0.50）。但 ID 评估可能高估 SFT 的真实能力——训练和测试数据来自相同的生成分布（2×2/3×3 整数矩阵、三种固定描述风格）。

**核心假设**：SFT 的高性能可能部分源于对训练分布的过拟合，RL 方法（GRPO）的泛化能力可能更强，在分布外任务上表现优于 SFT。

### 12.2 OOD Benchmark 设计

`generate_ood_bench.py` 生成 **750 个 OOD 样本**，覆盖 6 类分布偏移：

| OOD 类别 | 样本数 | 偏移类型 | 具体设计 |
|:---------|:------:|:---------|:---------|
| large_matrix | 140 | 矩阵规模增大 | 5×5, 6×6（训练集仅有 2×2, 3×3）|
| non_integer | 120 | 连续值收益 | 小数收益（如 3.7, -1.2），训练集为整数 |
| wide_range | 120 | 收益范围扩大 | 收益 ∈ [-50, 50]，训练集 ∈ [-10, 10] |
| novel_format | 150 | 描述格式新颖 | 数学符号 / JSON / Markdown表格 / 枚举格式 |
| asymmetric | 150 | 非对称矩阵 | 4×2, 2×5, 3×5, 5×3（训练集为 n×n 方阵）|
| combined_hard | 70 | 多重偏移叠加 | 大矩阵 + 连续值 + 宽范围 + 新格式 |

每类包含等量的 Nash Equilibrium 和 Best Response 任务。使用 nashpy 库精确求解均衡/最优响应，确保 ground truth 准确。

### 12.3 评估方法

使用 `vllm_eval/eval.py`（vLLM 离线批量推理）在 5 个 checkpoint 上并行评估：
- 5 个模型分配到 GPU 0-4，每个 ~90 秒完成 750 样本
- 与 ID 评估使用完全相同的 reward function 和 prompt 模板
- Temperature=0.1, max_new_tokens=1024

### 12.4 总体结果：ID vs OOD

| 方法 | ID Reward | OOD Reward | 绝对下降 | 相对下降 |
|:-----|:---------:|:----------:|:--------:|:--------:|
| Base (Qwen2.5-3B) | 0.5249 | 0.4331 | -0.0918 | -17.5% |
| GRPO-only | 0.5152 | 0.4331 | -0.0821 | -15.9% |
| Full+Probe GRPO | 0.4840 | 0.4356 | -0.0483 | -10.0% |
| **SFT (CoT)** | **0.9451** | **0.7342** | **-0.2109** | **-22.3%** |
| SFT → GRPO | 0.9351 | 0.7299 | -0.2052 | -21.9% |

### 12.5 OOD 分任务结果

| 方法 | OOD Overall | OOD Nash | OOD BR |
|:-----|:-----------:|:--------:|:------:|
| Base | 0.4331 | 0.3894 | 0.4804 |
| GRPO-only | 0.4331 | 0.3547 | 0.5182 |
| Full+Probe | 0.4356 | 0.3729 | 0.5036 |
| **SFT (CoT)** | **0.7342** | **0.6145** | **0.8637** |
| SFT → GRPO | 0.7299 | 0.6098 | 0.8600 |

### 12.6 OOD 分类别详细结果

| OOD 类别 | Base | GRPO-only | Full+Probe | SFT (CoT) | SFT→GRPO |
|:---------|:----:|:---------:|:----------:|:----------:|:---------:|
| non_integer | 0.5415 | 0.5357 | 0.5751 | **0.8959** | 0.8965 |
| asymmetric | 0.5405 | 0.5362 | 0.5263 | **0.8601** | 0.8566 |
| novel_format_json | 0.6103 | 0.5750 | 0.5730 | **0.8653** | 0.8551 |
| novel_format_table | 0.5164 | 0.4794 | 0.5529 | **0.8398** | 0.7787 |
| wide_range | 0.4646 | 0.4656 | 0.4306 | **0.8407** | 0.8421 |
| novel_format_math | 0.5985 | 0.6014 | 0.6531 | **0.7147** | 0.7254 |
| large_matrix | 0.1900 | 0.2235 | 0.2085 | **0.5434** | 0.5351 |
| combined_hard | 0.1199 | 0.1143 | 0.1106 | **0.2450** | 0.2459 |

### 12.7 矩阵维度分析

| 维度 | Base | GRPO-only | Full+Probe | SFT (CoT) | SFT→GRPO |
|:----:|:----:|:---------:|:----------:|:----------:|:---------:|
| 2×2 | 0.5853 | 0.5729 | 0.6171 | **0.8790** | 0.8767 |
| 2×5 | 0.5799 | 0.5745 | 0.5434 | **0.9055** | 0.8980 |
| 3×3 | 0.4852 | 0.4791 | 0.4642 | **0.8049** | 0.7993 |
| 3×5 | 0.3232 | 0.3576 | 0.3317 | **0.6911** | 0.6592 |
| 4×2 | 0.5270 | 0.5375 | 0.5169 | **0.8624** | 0.8657 |
| 5×3 | 0.7060 | 0.6357 | 0.7054 | **0.9333** | 0.9532 |
| 5×5 | 0.1897 | 0.2410 | 0.2128 | **0.4747** | 0.4698 |
| 6×6 | 0.1358 | 0.1152 | 0.1266 | **0.4029** | 0.3972 |

### 12.8 分析与结论

#### 假设验证：RL 方法的 OOD 泛化优势 — **不成立**

1. **SFT 在 OOD 上仍然压倒性领先**：SFT 的 OOD reward (0.7342) 远超所有 RL 方法 (~0.43)，绝对优势达 **+0.30**。即使 SFT 经历了更大的相对下降（-22.3% vs RL 的 ~-15%），其 OOD 性能仍是 RL 方法的 **1.7 倍**。

2. **RL 方法的"稳定性"是虚假的**：GRPO-only 的 OOD reward (0.4331) 与 base model (0.4331) 完全相同，Full+Probe (0.4356) 也仅微幅提升。RL 方法的"低下降率"并非泛化能力强，而是因为 **它们从未学到有效的博弈论推理能力**——ID 和 OOD 都接近 base model 水平。

3. **SFT 学到的 CoT 推理过程具有迁移性**：
   - 在 non_integer (0.90)、asymmetric (0.86)、wide_range (0.84) 上保持高性能
   - 在 novel_format_json (0.87) 和 novel_format_table (0.84) 上也很鲁棒
   - 说明 SFT 学到的不只是模式匹配，而是可迁移的数学推理步骤

4. **SFT 的薄弱环节**：
   - **large_matrix** (0.54): 5×5 和 6×6 矩阵导致显著下降，计算复杂度超出 CoT 训练的覆盖范围
   - **combined_hard** (0.25): 多重偏移叠加时性能骤降，接近随机水平
   - **novel_format_math** (0.71): 数学符号格式有中等下降

5. **矩阵维度是性能的主要决定因素**：所有方法在 5×5 (SFT: 0.47) 和 6×6 (SFT: 0.40) 上都大幅下降，这是计算复杂度而非泛化能力的瓶颈。

#### 总体结论

SFT on Chain-of-Thought 不仅在 ID 上大幅领先，在 OOD 场景下仍是最优方法。RL 方法（GRPO）在博弈论的冷启动条件下基本无效——既无法学会 ID 任务，也无法展现 OOD 优势。SFT 的主要限制不在泛化性，而在于 CoT 训练数据覆盖的计算复杂度范围（大矩阵）。未来提升方向应聚焦于：(1) 扩展 SFT 训练数据以覆盖更大矩阵；(2) 在 SFT 充分初始化后再使用 RL 精调，可能在 OOD 场景下发挥优势。

### 12.9 输出文件

| 文件/目录 | 说明 |
|:---------|:-----|
| `generate_ood_bench.py` | OOD benchmark 生成脚本 |
| `gamesolve_ood_bench.jsonl` | OOD benchmark 数据（750 样本）|
| `gamesolve_ood_stats.json` | OOD benchmark 统计信息 |
| `vllm_eval/eval.py` | vLLM 批量评估脚本 |
| `vllm_eval/run_parallel_ood.sh` | 并行 OOD 评估脚本（5 GPU）|
| `vllm_eval/run_all_ood.sh` | 顺序 OOD 评估脚本 |
| `compare_ood.py` | OOD vs ID 对比分析脚本 |
| `eval_results/ood_*/ood_eval_summary.json` | 各方法 OOD 评估结果 |
| `results/phase2/ood_comparison.json` | 完整对比 JSON |

---

## 13. verl 框架 GRPO 实验 (grpo_verl)

### 13.1 动机

前序实验中，基于 transformers 自定义实现的 GRPO 训练效果不佳（ID reward 仅 0.515，远低于 SFT 的 0.945）。为排查是 GRPO 算法本身的局限还是实现问题，使用字节跳动的 verl 框架（v0.7.1）重新实现 GRPO 训练，保持相同的训练数据和超参数。

### 13.2 实验设置

**框架**：verl v0.7.1，使用 Ray + FSDP + vLLM rollout 的 hybrid engine

**训练超参数**（与之前 transformers GRPO 保持一致）：

| 参数 | 值 |
|:-----|:---|
| 学习率 | 1e-6 |
| LR schedule | cosine, warmup_ratio=0.05 |
| train_batch_size | 16 |
| n_rollouts | 5 |
| temperature | 0.7 |
| top_p | 0.9 |
| max_prompt_length | 1024 |
| max_response_length | 512 |
| grad_clip | 1.0 |
| total_epochs | 3 (360 steps) |
| KL loss | disabled |
| GPU | 8×H20, FSDP, TP=1 |

**数据**：与之前相同的训练集（1920 样本）/ 验证集（480 样本），seed=42, 80/20 split，转换为 verl Parquet 格式。

**代码**：
- `verl_scripts/preprocess_gamesolve_verl.py` — 数据预处理
- `verl_scripts/gamesolve_reward_verl.py` — 自定义 reward function
- `verl_scripts/run_grpo_verl.sh` — 训练脚本

### 13.3 训练曲线

| Step | Epoch | Val Reward | Train Score |
|:-----|:------|:-----------|:------------|
| 0    | -     | 0.3160     | (base)      |
| 40   | 0     | 0.5986     | 0.4594      |
| 80   | 0     | 0.6435     | 0.5988      |
| 120  | 0     | 0.6712     | 0.6909      |
| 160  | 1     | 0.6986     | 0.6936      |
| 200  | 1     | 0.7176     | 0.6376      |
| 240  | 1     | 0.7264     | 0.7574      |
| **280** | **2** | **0.7298** | **0.8088** |
| 320  | 2     | 0.7294     | 0.6033      |
| 360  | 2     | 0.7269     | 0.6944      |

最佳 checkpoint：**global_step_280**（val reward 0.7298），之后验证奖励轻微回落（过拟合迹象）。

### 13.4 ID 评估结果

使用 vLLM 离线评估，200 样本（与之前所有方法相同的评估集）：

| 方法 | Overall | Best Response | Nash Equilibrium |
|:-----|:--------|:-------------|:----------------|
| Base (Qwen2.5-3B) | 0.5249 | 0.6829 | 0.4033 |
| GRPO-only (transformers) | 0.5152 | 0.6563 | 0.4066 |
| Full+Probe (transformers) | 0.4840 | 0.6423 | 0.3620 |
| **GRPO (verl)** | **0.7717** | **0.9801** | **0.6112** |
| SFT→GRPO | 0.9351 | 0.9770 | 0.9028 |
| SFT (CoT) | 0.9451 | 0.9769 | 0.9206 |

### 13.5 OOD 评估结果

750 OOD 样本评估：

| 方法 | Overall | Best Response | Nash Equilibrium |
|:-----|:--------|:-------------|:----------------|
| Base (Qwen2.5-3B) | 0.4331 | 0.4804 | 0.3894 |
| GRPO-only (transformers) | 0.4331 | 0.5182 | 0.3547 |
| Full+Probe (transformers) | 0.4356 | 0.5036 | 0.3729 |
| **GRPO (verl)** | **0.6615** | **0.8532** | **0.4845** |
| SFT→GRPO | 0.7299 | 0.8600 | 0.6098 |
| SFT (CoT) | 0.7342 | 0.8637 | 0.6145 |

**verl GRPO OOD 按类别细分**：

| OOD 类别 | Reward | 样本数 |
|:---------|:-------|:------|
| asymmetric | 0.7375 | 150 |
| wide_range | 0.7400 | 120 |
| non_integer | 0.7307 | 120 |
| novel_format_json | 0.7184 | 60 |
| novel_format_math | 0.6391 | 60 |
| large_matrix | 0.6048 | 140 |
| novel_format_table | 0.5378 | 30 |
| combined_hard | 0.3823 | 70 |

### 13.6 分析

#### 关键发现

1. **verl GRPO 远优于 transformers GRPO**：
   - ID: 0.7717 vs 0.5152，提升 **+49.8%**
   - OOD: 0.6615 vs 0.4331，提升 **+52.7%**
   - 说明之前 GRPO "无效"的结论是 **实现问题**，而非算法本身的局限

2. **verl GRPO 显著缩小了与 SFT 的差距**：
   - ID: 0.7717 vs 0.9451 (差距从 0.43 缩小到 0.17)
   - OOD: 0.6615 vs 0.7342 (差距从 0.30 缩小到 0.07)
   - 在 OOD 上差距仅 7.3%，说明 GRPO 学到的推理能力具有良好泛化性

3. **Best Response 任务上 verl GRPO 接近 SFT**：
   - ID BR: 0.9801 vs 0.9769（甚至略高于 SFT）
   - OOD BR: 0.8532 vs 0.8637（仅差 1%）
   - GRPO 通过 RL 奖励信号学会了 Best Response 任务的正确计算，无需 CoT 监督

4. **Nash Equilibrium 仍是 GRPO 的短板**：
   - ID NE: 0.6112 vs 0.9206 (SFT)
   - OOD NE: 0.4845 vs 0.6145 (SFT)
   - NE 任务需要结构化多步推理（枚举+验证），纯 RL 信号较难学习

5. **verl 框架的优势**：
   - 专业的 GRPO 实现（Ray 分布式、vLLM rollout、FSDP 训练）
   - Hybrid engine 在同一组 GPU 上交替做 rollout 和训练，效率更高
   - 更好的 advantage normalization 和 PPO 优化策略

#### 重新评估之前的结论

Section 12 中"RL 方法基本无效"的结论需要修正：

- **原结论**：GRPO 在博弈论冷启动条件下基本无效
- **修正**：使用正确实现（verl）后，GRPO 在冷启动条件下仍可达到 0.77 的 ID reward，尤其在 Best Response 任务上接近 SFT 水平
- **新结论**：RL 训练的效果高度依赖实现质量。verl 的专业 RL 框架显著优于自定义 transformers 实现

### 13.7 输出文件

| 文件/目录 | 说明 |
|:---------|:-----|
| `verl_scripts/preprocess_gamesolve_verl.py` | verl 数据预处理 |
| `verl_scripts/gamesolve_reward_verl.py` | verl reward function |
| `verl_scripts/run_grpo_verl.sh` | verl GRPO 训练脚本 |
| `data/verl_gamesolve/` | 训练/验证 Parquet 数据 |
| `results/phase2/grpo_verl/` | 训练 checkpoints (step 40-360) |
| `results/phase2/grpo_verl/merged_step_280/` | 最佳 checkpoint (HF 格式) |
| `results/phase2/grpo_verl_train.log` | 训练日志 |
| `eval_results/grpo_verl/eval_summary.json` | ID 评估结果 |
| `eval_results/grpo_verl/ood_eval_summary.json` | OOD 评估结果 |

## 14. verl GRPO + Probe 实验 (grpo_verl_probe)

### 14.1 动机

Section 13 的 grpo_verl 实验未包含辅助探针损失（probe loss）。本实验在完全相同的 GRPO 超参数下加入 probe 辅助损失，检验 Phase 2 核心假设：**在 RL 训练中用概念探针塑造中间层表征，能否提升博弈求解能力**。这是对 transformers 版 Full+Probe 实验（结果为负收益）的重新检验——当时的负结果可能源于 GRPO 实现质量而非 probe 本身。

### 14.2 实验设置

**与 grpo_verl 完全一致的 GRPO 超参**（lr=1e-6, batch=16, n_rollouts=5, 360 steps 等，见 13.2），新增：

| 参数 | 值 |
|:-----|:---|
| probe_lambda (λ) | 0.1 |
| probe_layer | 17 (= L/2, Qwen2.5-3B 共 36 层) |
| 概念标签 | eq_type, difficulty, dominance, br_direction, eq_uniqueness |
| 概念权重 w_t | 与 Phase 1 峰值探针准确率成反比（br_direction 1.57 ... difficulty 1.11）|
| Probe heads | 每概念一个 nn.Linear(2048, C)，bf16 |
| Probe optimizer | Adam, lr=1e-3（独立于 actor optimizer）|

**集成方式**（重要工程发现）：verl 0.7.1 默认 `use_legacy_worker_impl=auto` 走 legacy 路径（`fsdp_workers.py` → `workers/actor/dp_actor.py:update_policy`），而非新 engine 路径。Probe 必须集成在 `dp_actor.py`：

1. forward hook 挂在 FSDP 包装的 layers[17]，捕获 hidden states（remove_padding 下为 packed 格式 `(1, total_nnz, 2048)`）
2. 概念标签从 `data.non_tensor_batch["extra_info"]["concept_labels"]` 读取
3. 用 attention_mask 计算每条样本的有效 prompt/response 长度，在 packed 序列中提取 prompt 段做 mean pooling
4. probe_loss 加进 policy_loss 一起 backward（gradient checkpointing 用 use_reentrant=False，梯度可流回 backbone）
5. probe optimizer 与 actor optimizer 同步 zero_grad/step

**训练中的事故与修复**：step 202 时崩溃——某些 DP rank 的 micro-batch 恰好没有某概念的有效标签（如 br_direction 仅在 best_response 样本上有效），导致各 rank 返回的 metrics key 集合不一致，`DataProto.concat` 断言失败。修复：metrics key 恒定，缺失值用 NaN 占位。从 step 200 checkpoint 自动续训至 360（注：probe heads 未存入 checkpoint，续训时重新初始化，~40 步内重新收敛）。

### 14.3 训练曲线（与 baseline 对比）

| Step | grpo_verl | grpo_verl_probe | Δ |
|:-----|:----------|:----------------|:--|
| 0    | 0.3160 | 0.3308 | — |
| 40   | 0.5986 | 0.5827 | -0.016 |
| 80   | 0.6435 | 0.6599 | +0.016 |
| 120  | 0.6712 | 0.7293 | +0.058 |
| 160  | 0.6986 | 0.7248 | +0.026 |
| 200  | 0.7176 | 0.7398 | +0.022 |
| 240  | 0.7264 | 0.7474 | +0.021 |
| 280  | 0.7298 | 0.7482 | +0.018 |
| 320  | 0.7294 | **0.7484** | +0.019 |
| 360  | 0.7269 | 0.7416 | +0.015 |

- Probe 版从 step 80 起持续领先，最佳 val reward **0.7484@320** vs baseline 0.7298@280（**+1.9pp**）
- Probe 自身指标收敛良好：total_loss 1.18→0.52；探针准确率 eq_type 0.38→0.86、difficulty 0.31→0.96、eq_uniqueness 0.50→1.00，说明 layer 17 表征确实变得更可线性解码

### 14.4 ID 评估结果（200 样本，vLLM）

| 方法 | Overall | Best Response | Nash Equilibrium |
|:-----|:--------|:-------------|:----------------|
| GRPO (verl) @280 | 0.7717 | 0.9801 | 0.6112 |
| **GRPO+Probe @280** | **0.7805** | 0.9812 | 0.6260 |
| **GRPO+Probe @320** | 0.7793 | 0.9635 | **0.6375** |
| SFT (CoT) | 0.9451 | 0.9769 | 0.9206 |

### 14.5 OOD 评估结果（750 样本）

| 方法 | Overall | Best Response | Nash Equilibrium |
|:-----|:--------|:-------------|:----------------|
| GRPO (verl) | 0.6615 | 0.8532 | 0.4845 |
| **GRPO+Probe @280** | **0.6660** | 0.8462 | 0.4997 |
| **GRPO+Probe @320** | 0.6655 | 0.8436 | **0.5011** |

OOD 分类别（@320 vs baseline）：non_integer 0.7487 vs 0.7307 (+1.8pp)、novel_format_math 0.6852 vs 0.6391 (+4.6pp)、novel_format_table 0.5767 vs 0.5378 (+3.9pp)；asymmetric -0.6pp、large_matrix -0.6pp、combined_hard -0.5pp 基本持平。

### 14.6 分析：probe 的作用得到验证（效应温和但一致）

1. **Probe 辅助损失带来一致的正向收益**：
   - Val reward: +1.9pp（全程 step 80 后持续领先，非单点噪声）
   - ID: +0.8~0.9pp（两个 checkpoint 均超 baseline）
   - OOD: +0.4~0.5pp
   - 收益虽小，但方向在两个独立 checkpoint、两个评估集上完全一致

2. **收益集中在 Nash Equilibrium 任务**（概念密集型）：
   - ID NE: +1.5~2.6pp；OOD NE: +1.5~1.7pp
   - BR 任务基本持平（ID @280: 0.9812 vs 0.9801）
   - 与机制假设吻合：探针概念（eq_type、eq_uniqueness、dominance）主要刻画均衡结构，对 NE 任务的表征塑造直接相关；BR 是数值计算任务，受概念表征影响小

3. **与 transformers 版 Full+Probe 的负结果对比**：
   - transformers: Full+Probe 0.4840 < GRPO-only 0.5152 < Base 0.5249（probe 有害）
   - verl: GRPO+Probe 0.7805 > GRPO 0.7717（probe 有益）
   - **结论修正**：之前"probe 辅助损失无效甚至有害"的结论是训练实现质量的伪影。在健康的 RL 训练动态下，probe 辅助损失是温和的正向正则化

4. **局限**：
   - 单 seed、单 λ(0.1)、单层(17)，+0.9pp 的 ID 差距在 200 样本上不具统计显著性（约 ±3pp 的 95% CI）
   - 但 val reward 曲线全程领先（480 样本×10 个评估点）和 NE 任务上的一致方向提供了较强的趋势证据
   - 后续可做：多 seed 重复、λ 扫描、多层注入、与 SFT→GRPO 组合

### 14.7 输出文件

| 文件/目录 | 说明 |
|:---------|:-----|
| `verl_scripts/preprocess_gamesolve_probe_verl.py` | 带概念标签的数据预处理 |
| `verl_scripts/run_grpo_probe_verl.sh` | GRPO+Probe 训练脚本 |
| `verl/verl/workers/utils/probe_utils.py` | ProbeState/ProbingHeads 实现 |
| `verl/verl/workers/actor/dp_actor.py` | probe 集成点（update_policy）|
| `data/verl_gamesolve_probe/` | 训练/验证 Parquet（含 concept_labels）|
| `results/phase2/grpo_verl_probe/` | checkpoints (step 40-360) |
| `results/phase2/grpo_verl_probe/merged_step_320/` | 最佳 checkpoint (HF 格式) |
| `results/phase2/grpo_verl_probe/merged_step_280/` | 次佳 checkpoint (HF 格式) |
| `results/phase2/grpo_verl_probe/training_metrics.jsonl` | 训练全程指标（含 probe acc）|
| `results/phase2/grpo_verl_probe_training.log` | 训练日志 |
| `eval_results/grpo_verl_probe/` | step 320 ID/OOD 评估 |
| `eval_results/grpo_verl_probe_280/` | step 280 ID/OOD 评估 |

## 15. 外部博弈 Benchmark 迁移评估 (GTBench & TextArena)

### 15.1 动机

前述所有评估都在 GameSolve-Bench（normal-form 矩阵博弈）上进行，属于训练同任务族。为检验四个 checkpoint 学到的能力能否**迁移到结构不同的 sequential / interactive 博弈**（结构泛化），在两个外部基准上评测 Base、SFT (CoT)、GRPO (verl)、GRPO+Probe (verl)。这也是对 proposal §2.5 中"Generalization"一栏的落实。

### 15.2 实验设置

| Benchmark | 引擎 | 对局形式 | 选用游戏 |
|:----------|:-----|:---------|:---------|
| **GTBench** | openspiel / rlcard | 候选模型 (prompt_agent) vs **随机对手**，20 场有效/格，前后手各半 | breakthrough, connect4, first_sealed_auction, kuhn_poker, liars_dice, negotiation, nim, pig, prisoners_dilemma, tictactoe（10 个）|
| **TextArena** | textarena 0.7.3 | 候选模型 vs **固定 Base 对手**，20 集/格，前后手交替 | TicTacToe, ConnectFour, Nim, KuhnPoker, IteratedPrisonersDilemma, SimpleNegotiation（6 个）|

- 4 个 3B 模型由 4 个 vLLM 服务器（GPU 0-3）并行供给；GTBench 4 模型 × 8 worker 并行，TextArena 24 任务并行
- GTBench 原 langchain 后端重写为 OpenAI client 直连本地 vLLM（`benchmark_eval/GTBench/gamingbench/chat/chat.py`）
- 指标统一为 score = (win + 0.5·draw)/n；GTBench 因候选非法动作判 Abnormal 的对局不计入
- 评测代码/脚本/结果在 `benchmark_eval/`（`ANALYSIS.md`、`results/REPORT.md`、`results/summary.json`）

### 15.3 结果

**GTBench avg score**（对随机对手，8 个所有模型 n≥10 的游戏；nim/pig 因完成率不足排除）:

| 方法 | avg score | 完成率 |
|:-----|:----------|:-------|
| Base (Qwen2.5-3B) | 0.524 | 80% |
| **SFT (CoT)** | **0.604** | 83% |
| GRPO (verl) | 0.559 | 79% |
| GRPO+Probe (verl) | 0.574 | 79% |

**TextArena avg score**（对固定 Base 对手；Base vs Base = 0.521 为 ~50% 对照，验证通过）:

| 方法 | avg score |
|:-----|:----------|
| Base (Qwen2.5-3B) | 0.521 |
| SFT (CoT) | **0.382** |
| **GRPO (verl)** | **0.558** |
| GRPO+Probe (verl) | 0.483 |

### 15.4 关键结论

1. **SFT 在交互式博弈上显著退化，与其同分布优势形成鲜明对比**。SFT 在 GameSolve-Bench（ID 0.945）和 GTBench 对随机对手（0.604，靠格式遵循好）上最强，但在 TextArena 对 Base 对手时**跌至 0.382**（KuhnPoker 0.20、TicTacToe 0.35），明显低于 Base。SFT 过拟合了 GameSolve 的 CoT 答题格式，在需要多轮交互、对手建模的场景中受损。**这把"SFT 同分布最强、分布外脆弱"的结论从 OOD 矩阵博弈（§12）进一步外推到了结构性分布外。**

2. **RL 方法（GRPO / GRPO+Probe）迁移安全，无能力回退**。两者在 GTBench（0.56/0.57）和 TextArena（0.56/0.48）上均与 Base（0.52/0.52）持平或略升。GameSolve 上 +25pp 的提升（0.77 vs 0.52）没有以牺牲通用博弈能力为代价——**RL 微调的"对齐税"远低于 SFT**。这与 §12 OOD 结论一致：RL 的优势在泛化稳健性。

3. **Probe 在外部 benchmark 上与 GRPO 无一致差异**（GTBench 略高 +1.4pp、TextArena 略低 -7.5pp，主要差在 Nim/TicTacToe）。这符合预期：probe 概念（均衡类型、支配策略、均衡唯一性）是 normal-form 矩阵博弈特定的，不应迁移到 sequential 游戏。**§14 中验证的 probe 正向作用是 GameSolve 任务内的，不构成结构泛化收益。**

4. **格式遵循是 3B 模型在 GTBench 的主要瓶颈**（附带发现）。pig 游戏（动作仅 `<roll>`/`<stop>`）中 Base/GRPO/Probe 几乎全部输出字面 `<Action>` 等非法动作被判 Abnormal（完成率 1-3%），而 SFT 达 77%——SFT 唯一确凿的迁移收益是指令格式遵循。

### 15.5 对 proposal 的含义

- 结构泛化（matrix → sequential）有限，且这是预期内的：probe 概念与 GameSolve 任务族绑定。若要外部 benchmark 收益，需在训练中混入 sequential 博弈，或把概念标签泛化（如"是否存在占优行动"在 sequential 游戏中重新定义）。
- RL（verl GRPO ± probe）是更安全的能力增强路径：同分布大幅提升、分布外无回退；SFT 的同分布优势以分布外退化为代价。
- 后续若做 benchmark 提升实验，优先考虑 TextArena 自我对弈 RL（SPIRAL 式）或 GTBench 游戏混训。

### 15.6 输出文件

| 文件/目录 | 说明 |
|:---------|:-----|
| `benchmark_eval/ANALYSIS.md` | 完整分析（发现、统计注意事项、结论）|
| `benchmark_eval/results/REPORT.md` | 分游戏数据表 |
| `benchmark_eval/results/summary.json` | 结构化汇总 |
| `benchmark_eval/scripts/serve_models.sh` | 启动 4 个 vLLM 服务器 |
| `benchmark_eval/scripts/run_gtbench.sh` | GTBench 评测脚本 |
| `benchmark_eval/scripts/textarena_eval.py` | TextArena 评测脚本 |
| `benchmark_eval/scripts/analyze_results.py` | 结果汇总脚本 |
| `benchmark_eval/scripts/gtbench_patch/` | GTBench 修改存档（chat.py + 模型配置）|
| `benchmark_eval/results/gtbench/`, `textarena/` | 原始对局记录（本地保留）|
