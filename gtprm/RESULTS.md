# GT-PRM 验证实验:Game-Theoretic Process Reward Models

> 验证 `ideas/game_theoretic_prm_research_report.md` 的核心假设:**用博弈论求解步骤的可形式化验证作为稠密过程奖励(process reward),能否优于仅用最终答案的结果奖励(outcome reward)训练 GRPO?**
> 结论(2026-06-14):**否。在本设置下 GT-PRM 不成立**——纯过程奖励严重劣于结果奖励(reward hacking),混合奖励与结果奖励持平、无增益(含 OOD)。

---

## 1. 方法设计

### 1.1 核心思想
博弈论求解的每一步(解析支付、识别占优、消去、最优响应、纳什均衡)都可**算法化验证**,无需人工标注、无需 rollout(对比 Math-Shepherd/OmegaPRM)。把"步骤正确率"作为稠密奖励训练 GRPO。

### 1.2 验证器(`verl_scripts/gt_prm_verifier.py`)
对模型自由文本推理做 **label-aware 抽取 + 形式化验证**:
| 声明类型 | 抽取(正则) | 验证 |
|:--------|:-----------|:-----|
| DOMINANCE | "X dominates Y" | 支付矩阵弱占优检查 |
| EQUILIBRIUM | NE 关键词 60 字窗口内的 (row,col) | 互为最优响应 |
| BR_ACTION | "best response: X" | vs ground-truth BR |
| EXP_PAYOFF | "EU(X) = … = v" | vs ground-truth 期望支付 |

`process_score = 0.5·accuracy + 0.5·coverage`,accuracy=正确声明加权占比(惩罚错误声明),coverage=正确覆盖真实可验证事实数(防止靠刷量/单条正确声明骗分)。

### 1.3 奖励函数(`verl_scripts/gt_prm_reward.py`,verl 接口)
- `outcome`(ORM 基线):仅答案质量奖励(复用 `gamesolve_reward`,本身已是分级奖励)
- `process`:纯过程分
- `hybrid`:0.5·process + 0.5·outcome
- 变体:`binary`(各步等权)/ `typed`(NE>BR>EXP>DOM 加权)

### 1.4 验证器有效性(训练前的 gate)
- gold CoT:process=0.84(nash 0.73、BR 0.985);空/错配 game ≈0;94% gold 含 ≥1 可验证声明 → **验证器不是坏的,它确实奖励正确推理**。
- **关键 gate**:在 base 模型真实生成上(n=300),`CORR(process, outcome)=0.006`(近零,非负);高 process 段的答案质量无提升(0.127 vs 0.123)。
  → 过程正确性与最终答案质量**近正交**。这预示纯过程训练对答案质量帮助有限,且有 reward-hacking 风险。**实验证实了这一预示。**

---

## 2. 实验设计

- **条件**:ORM / GT-PRM-Process / GT-PRM-Hybrid / GT-PRM-Typed(消融),每个 **3 seeds** = 12 次 verl GRPO。
- **超参**:与既有 grpo_verl 一致(lr=1e-6,batch=16,n=5,温度0.7),240 步(2 epoch,val 平台期)。
- **算力**:8×H20,2-slot × 4 卡并行(`orchestrate_gtprm.sh`)。
- **评估(关键)**:各条件训练奖励不同 → 训练/val reward **不可跨条件比较**。所有 checkpoint 一律用**统一的答案质量指标**(vLLM,`gamesolve_reward`)在 ID(200)+ OOD(750)上评估。

---

## 3. 结果

### 3.1 ID(200 样本,统一答案奖励,mean±std,n=3)

| 条件 | Overall | Nash | BR | vs ORM |
|:-----|:-------:|:----:|:--:|:------:|
| **ORM (outcome)** | **0.753 ± 0.023** | 0.615 | 0.932 | — |
| GT-PRM-Process | 0.505 ± 0.057 | 0.437 | 0.593 | **−0.248 (t=−6.96)** |
| GT-PRM-Hybrid | 0.748 ± 0.025 | 0.618 | 0.917 | −0.005 (t=−0.25, ns) |
| GT-PRM-Typed | 0.601 ± 0.113 | 0.555 | 0.662 | −0.151 (t=−2.28) |

### 3.2 OOD(750 样本)

| 条件 | Overall | Nash | BR | vs ORM |
|:-----|:-------:|:----:|:--:|:------:|
| **ORM (outcome)** | **0.633 ± 0.004** | 0.483 | 0.795 | — |
| GT-PRM-Process | 0.429 ± 0.062 | 0.370 | 0.492 | **−0.204 (t=−5.66)** |
| GT-PRM-Hybrid | 0.616 ± 0.019 | 0.490 | 0.753 | −0.017 (t=−1.44, ns) |
| GT-PRM-Typed | 0.471 ± 0.092 | 0.431 | 0.515 | −0.161 (t=−3.03) |

### 3.3 Reward hacking 直接证据
process 训练把**过程奖励**从 0.15 拉到 ~0.63(训练曲线正常上升),但**答案质量**只有 ~0.50;而 ORM 训练把答案奖励拉到 ~0.76、答案评估 0.75。**模型成功优化了过程目标,却没学会解题**——典型的目标错配 / reward hacking,与 gate 的近零相关一致。

---

## 4. 结论与分析(诚实)

1. **GT-PRM 核心假设被否定**:没有任何过程奖励条件在 ID 或 OOD 上超过结果奖励基线。纯过程**显著更差**,混合**持平无增益**(连 GT-PRM 主打的 OOD 也无增益,Hybrid OOD Δ=−1.7pp ns)。

2. **根因 = 过程奖励与任务目标近正交且可被 hack**。可验证的中间声明(占优/NE/BR)是解题的**必要非充分**信号:模型可以陈述若干正确的易验证事实来抬高过程分,却不产出正确最终答案。gate 的 `corr=0.006` 已精准预警,实验落地为 −20~25pp 的答案质量下跌。

3. **Typed 加权更糟**:给最难验证、最易 hack 的 NE/BR 步更高权重,放大了 hacking。

4. **为何与 math PRM 的成功不同**:数学 PRM 的步骤奖励通过 rollout 估计"该步能否导向正确答案"(与结果强相关);本 GT-PRM 的步骤奖励是"该步声明是否自洽正确"(与最终答案弱相关)。**可形式化验证 ≠ 与结果对齐**——这是本实验最有价值的洞见。

5. **局限**:本实验测的是**标量化过程奖励**(GT-PRM-Binary/Typed/Hybrid 的聚合形式),未实现 doc §6.4 的 **token/step 级优势分配**(需改 verl 优势计算)。真正的 per-step credit assignment 可能不同,留作 future work——但当前标量形式的负面结果已足以否定"直接用步骤验证分做 GRPO 奖励"这一最自然的实现。单模型(3B)、240 步、3 seeds。

---

## 5. 对发论文的判断

GT-PRM 作为"零成本可验证过程奖励 > 结果奖励"的正面方法论文,**当前结果不支持**。但有一个**有价值的反向洞见**可写成 analysis:**"formally verifiable ≠ outcome-aligned"** —— 在博弈论这个可完美验证的理想试验台上,稠密过程奖励反而因与最终目标正交而失败甚至有害。这对"把形式验证当过程奖励"的范式是一个有意义的警示性结果(cautionary tale),可投 workshop 或作为更大 study 的一节。

若要救正面故事,需让过程奖励与结果对齐:用 rollout 估计步骤价值(退化为 Math-Shepherd,失去"零成本"卖点),或做真正的 step 级优势分配 + 强约束防 hack。性价比低,建议转向 `ideas/` 中其他方向。

## 6. 复现

```bash
python3 verl_scripts/preprocess_gtprm_verl.py          # data/verl_gtprm (payoff matrices in extra_info)
python3 verl_scripts/validate_gtprm_verifier.py        # 验证器 gate(gold/empty/wrong)
CUDA_VISIBLE_DEVICES=0 python3 verl_scripts/gtprm_corr_gate.py  # base-gen 相关性 gate
bash verl_scripts/orchestrate_gtprm.sh                 # 12 runs,2×4 卡并行
bash verl_scripts/merge_eval_gtprm.sh                  # 合并 + 统一指标 ID/OOD 评估
python3 verl_scripts/analyze_gtprm.py                  # 汇总 + 显著性
```

## 7. 输出文件
- `verl_scripts/gt_prm_verifier.py` / `gt_prm_reward.py` — 验证器 + 奖励
- `verl_scripts/{validate_gtprm_verifier,gtprm_corr_gate,analyze_gtprm}.py` — 验证/gate/分析
- `verl_scripts/{run_gtprm,orchestrate_gtprm,merge_eval_gtprm}.sh` — 训练/编排/评估
- `data/verl_gtprm/` — 含支付矩阵的训练数据
- `results/gtprm/{ckpt,merged,logs}/`、`results/gtprm/gtprm_eval_summary.json`
- `eval_results/gtprm/<run>{,_ood}/` — 各 checkpoint 统一指标评估
