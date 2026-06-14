# Nash-GRPO 验证实验:Game-Theoretic Group Relative Policy Optimization

> 验证 `ideas/game_theoretic_rlhf_research_report.md` 的 Idea D:**把 GRPO 的"组均值基线"换成"偏好锦标赛的 Nash 均衡基线",能否改善策略优化(尤其在偏好不可传递 intransitive 的情形)?**
> 结论(2026-06-14):**在本设置(GameSolve 规则奖励)下 Nash-GRPO 不优于 GRPO**——ID 与 GRPO 统计持平,OOD 略差(β=4 显著 −2.1pp)。其理论卖点(处理 intransitive 偏好)在标量规则奖励下不出现(偏好天然可传递)。

---

## 1. 方法设计

### 标准 GRPO
对一组 K 个回答,优势 A(y_i) = (r_i − mean(r)) / std(r) —— **组均值基线**。

### Nash-GRPO(本实现)
把组内比较视为**偏好锦标赛博弈**:
1. 由标量奖励构造成对偏好 P[i,j] = σ(β·(r_i − r_j)),反对称收益 M = P − 0.5。
2. 求该对称零和博弈的 **Nash 均衡(maximin)混合策略 p\***(Hedge/乘性权重自博弈,时间平均收敛)。
3. 优势 A(y_i) = p\*[i] − 1/K(对均匀分布的偏离),再 std 归一化以与 GRPO 同尺度。

实现:`verl/verl/trainer/ppo/core_algos.py` 注册 `nash_grpo`(温度 `NASH_BETA`)与 `rank_grpo`(机制对照:用组内归一化 rank 作基线)。通过 `algorithm.adv_estimator=nash_grpo` 选择。

### 离线数学验证(已通过)
- 传递性奖励 [0.1,0.4,0.7,1.0]:GRPO 给分级优势 [−1.34..1.34];**Nash(β=4)给 winner-take-all**:最优 +1.73、其余≈−0.58。
- intransitive RPS(A>B>C>A):Nash 混合 = 均匀 [.33,.33,.33](Condorcet 一致,不武断选边)。
- 有 Condorcet 赢家:Nash 质量集中在赢家(0.991)。
→ 实现正确;关键行为差异 = **更激进的"赢家通吃"信用分配**。

---

## 2. 实验设计

- **条件**:GRPO(均值基线)/ Nash-GRPO(β=4,主)/ rank(机制对照)/ Nash β=1(软)/ Nash β=16(硬),每个 **3 seeds** = 15 次 verl GRPO。
- **关键**:所有条件**共享同一规则奖励与数据**,只换优势估计器 → 训练 val reward **可跨条件比较**(与 GT-PRM 不同),再加外部 ID/OOD 评估。
- 超参与 grpo_verl 一致;**rollout n=8**(给锦标赛更多结构,各条件一致),240 步,2×4 卡并行。
- 工程修复:n 5→8 使归一化 mini-batch=32,改 `ppo_micro_batch_size_per_gpu`=8。

---

## 3. 结果

### 3.1 训练峰值 val reward(480 验证,跨条件可比,mean±std,n=3)

| 条件 | peak val |
|:-----|:---------|
| **GRPO (mean)** | **0.7476 ± 0.0061** |
| Nash-GRPO (β=1) | 0.7474 ± 0.0066 |
| Nash-GRPO (β=16) | 0.7473 ± 0.0078 |
| rank baseline | 0.7452 ± 0.0067 |
| Nash-GRPO (β=4) | 0.7349 ± 0.0131 |

### 3.2 外部评估 overall reward（mean±std，n=3）

| 条件 | ID (200) | vs GRPO | OOD (750) | vs GRPO |
|:-----|:--------:|:-------:|:---------:|:-------:|
| **GRPO (mean)** | 0.788 ± 0.018 | — | **0.663 ± 0.005** | — |
| Nash-GRPO (β=1) | 0.788 ± 0.011 | +0.0002 (t=0.02) | 0.656 ± 0.006 | −0.008 (t=−1.68) |
| Nash-GRPO (β=4) | 0.782 ± 0.030 | −0.006 (t=−0.32) | 0.643 ± 0.008 | **−0.021 (t=−3.88)** |
| Nash-GRPO (β=16) | 0.790 ± 0.012 | +0.002 (t=0.14) | 0.649 ± 0.013 | −0.014 (t=−1.72) |
| rank baseline | 0.770 ± 0.009 | −0.018 (t=−1.56) | 0.654 ± 0.003 | −0.010 (t=−2.92) |

### 3.3 任务细分(ID / OOD,Nash | BR）
GRPO ID: nash 0.662 / br 0.952;OOD: nash 0.512 / br 0.827。Nash-GRPO 各变体在 Nash 子任务上略低(如 β=4 OOD nash 0.477 vs 0.512),BR 持平。

---

## 4. 结论与分析(诚实)

1. **Nash-GRPO 不优于 GRPO**。ID 上所有变体与 GRPO 统计持平(|t|<0.4,差异在 ±2-3pp 噪声内);OOD 上 GRPO 最佳,所有 Nash 变体略差,**主条件 β=4 显著更差(−2.1pp,t=−3.88)**。rank 基线也略差,说明并非 Nash 特有问题。

2. **根因:在标量规则奖励下,偏好天然可传递**。Nash 锦标赛此时退化为对均值基线的单调重加权;β 越大越"赢家通吃"(β=4/16),**丢弃了 GRPO 分级优势中有助于 OOD 泛化的信息**,故 OOD 略降。软化(β=1)≈ GRPO 但无增益。

3. **理论卖点(处理 intransitive 偏好)在本任务不出现**。离线演示证实:cyclic A>B>C>A 时 Nash 正确摊成均匀、有 Condorcet 赢家时集中——Nash 机制本身正确,但 GameSolve 的标量真值奖励**不产生偏好环**,因此 Nash 相对均值基线无用武之地。

4. **这指向 Nash-GRPO 的真正适用场景**:需要**成对偏好且可能不可传递**的设置(RLHF/preference,judge/PairRM 打分,或多目标冲突奖励),而非单一可验证标量奖励的推理任务。本仓库的 rule-reward 推理任务恰好是 Nash-GRPO **最不可能见效**的场景。

5. **局限**:单模型(3B)、240 步、3 seeds、标量奖励派生的偏好。未在真实偏好/RLHF 数据(UltraFeedback/judge)上测试——那才是 doc 设计的主战场,但不在本项目的 rule-reward 基础设施射程内。

---

## 5. 对发论文的判断

在本项目的 rule-reward 推理基础设施上,Nash-GRPO **不构成正面结果**——这是预期内的(机制与任务不匹配)。若要正面验证,必须切换到**成对偏好 RLHF 设置**(judge/PairRM + UltraFeedback/HH),并刻意构造/挖掘 intransitive 偏好作为主战场;那是一个独立的、更大的工程,与本仓库的博弈论推理主线关系较弱。性价比低,建议不在此方向继续。

可复用的正向产出:`nash_grpo`/`rank_grpo` 两个 verl 优势估计器(已验证正确)、2×4 卡并行编排、以及"同奖励→val 可跨条件比较"这一干净对照设计。

## 6. 复现
```bash
python3 -c "import sys;sys.path.insert(0,'verl');from verl.trainer.ppo.core_algos import _nash_minimax_mixture"  # 估计器
bash verl_scripts/orchestrate_nashgrpo.sh     # 15 runs, 2×4 卡
bash verl_scripts/merge_eval_nashgrpo.sh      # 合并 + ID/OOD 统一评估
python3 verl_scripts/analyze_nashgrpo.py      # 汇总 + 显著性 + intransitive 演示
```

## 7. 输出文件
- `verl/verl/trainer/ppo/core_algos.py` — `nash_grpo` / `rank_grpo` 估计器 + `_nash_minimax_mixture`
- `verl_scripts/{run,orchestrate,merge_eval,analyze}_nashgrpo.sh|py`
- `results/nashgrpo/{ckpt,merged,logs}/`、`eval_results/nashgrpo/<run>{,_ood}/`
