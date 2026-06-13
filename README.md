# Probing & Enhancing Game-Theoretic Reasoning in LLMs

> 用线性探针**诊断**、再用辅助探针损失 / RL **增强** LLM 的博弈论推理能力。
> 围绕自建基准 **GameSolve-Bench**（2,400 道矩阵博弈题）展开的两阶段研究。

本仓库包含完整的研究管线:基准构建 → Phase 1 表征探针诊断 → Phase 2 多种训练增强(SFT / GRPO / GRPO+Probe)→ 同分布/分布外/外部 benchmark 评估 → 严格的多 seed + 关键层消融。**逐次实验的完整中文记录见 [`training.md`](training.md)(共 16 节)**;研究设计见 [`proposal.md`](proposal.md)。

---

## TL;DR — 核心结论(含正面与负面)

| # | 结论 | 证据 |
|:-:|:-----|:-----|
| 1 | **CoT 监督是性能关键**,SFT 压倒性最强(同分布) | ID reward 0.945 vs base 0.525(+80%);§11 |
| 2 | **冷启动纯 GRPO(transformers 实现)无效**,但 **verl 框架的 GRPO 显著有效** | transformers GRPO 0.515 ≈ base;verl GRPO **0.772**(+47%);§9/§13 |
| 3 | **RL 泛化更稳,SFT 分布外脆弱** | OOD:verl GRPO 0.662 vs SFT 0.734(差距由 ID 的 0.17 缩到 0.07);交互式博弈(TextArena)SFT 退化到 0.38 < base 0.52;§12/§15 |
| 4 | ⚠️ **probe 辅助损失的"增益"是 seed 噪声;关键层假设被推翻** | 多 seed:probe@17 0.713 ≤ no-probe 0.723(不显著);层消融单调:早层≈无 probe > 中层 > 晚层 > 末层;§16 |

> **对研究方向的诚实判断**:proposal 的 Phase 2 核心假设(在 Phase 1 定位的关键层注入概念探针损失能提升博弈求解)在严格检验下**不成立**。probe 损失是一个随注入深度递增的"干扰项",而非概念局部化的"增益"。建议保留资产(基准、探针诊断、verl 管线、RL-vs-SFT 泛化结论),转向 [`ideas/`](ideas/) 中的其他方向。

---

## 研究框架(两阶段)

```
Phase 1 — 诊断 (Diagnose)                Phase 2 — 增强 (Enhance)
─────────────────────────                ─────────────────────────
逐层抽取隐藏表征                          联合损失 = 生成损失 + λ·Σ 探针损失
   ↓ 线性探针(逻辑回归)                     ↓ backbone + 探针头联合训练
定位 6 个博弈论概念的可解码层             多种训练范式对比 (SFT / GRPO / GRPO+Probe)
   ↓                                         ↓
critical layers L*                       同分布 / OOD / 外部 benchmark 评估
```

**6 个探针概念**:`eq_type`(均衡类型)、`game_type`(博弈类型)、`difficulty`(难度)、`dominance`(是否存在占优策略)、`br_direction`(最优响应方向)、`eq_uniqueness`(均衡唯一性)。

---

## 目录结构

```
BETA/
├── README.md                    # ← 本文件(项目总览)
├── proposal.md                  # 研究提案(两阶段方法论、评估协议)
├── training.md                  # ★ 逐次实验完整记录(16 节,中文)
├── research_log.jsonl           # 结构化研究日志(每行一个事件)
├── CurrentTask.md               # 历次 /auto-research 任务指令
├── docs/
│   └── gamesolve_bench.md        # GameSolve-Bench 基准详细文档(原 README)
│
├── ── 基准数据 ────────────────────────────────────────────
├── gamesolve_gen.py             # 基准生成器(采样博弈、nashpy 求解、生成 NL 描述)
├── gamesolve_bench.jsonl        # 主基准 2,400 样本(1,350 Nash + 1,050 BR)
├── gamesolve_stats.json         # 基准统计
├── generate_ood_bench.py        # OOD 基准生成器(更难/更大/异分布形式)
├── gamesolve_ood_bench.jsonl    # OOD 基准 750 样本(8 个 OOD 类别)
├── gamesolve_ood_stats.json
│
├── ── Phase 1:探针诊断 ────────────────────────────────────
├── extract_representations.py        # 单模型逐层表征抽取(forward hook)
├── extract_representations_multi.py  # 多模型/多卡表征抽取
├── train_probes.py / train_probes_parallel.py  # 逻辑回归探针训练
├── analyze_probes.py            # 探针准确率热图、critical layer 分析
├── analyze_pooling.py           # pooling 方法对比(last/mean/sum/weighted)
├── phase-1-results.md           # Phase 1 结论摘要
│
├── ── Phase 2:训练(transformers 实现)────────────────────
├── preprocess_sft.py            # SFT 数据预处理(CoT 监督)
├── train_sft.py                 # DDP SFT 训练
├── preprocess_gamesolve_grpo.py # GRPO 数据预处理
├── train_grpo_probe.py          # GRPO + 辅助探针损失训练(transformers)
├── gamesolve_reward.py          # 规则奖励函数(Nash/BR 评分)
├── run_sft.sh / run_phase2.sh / run_phase2_no_probe.sh / run_sft_then_grpo.sh / run_all_experiments.sh
│
├── ── Phase 2:训练(verl 框架,主力)──────────────────────
├── verl/                        # verl v0.7.1(ByteDance RL 框架,已改:见下)
├── verl_scripts/
│   ├── preprocess_gamesolve_verl.py        # → verl Parquet 格式
│   ├── preprocess_gamesolve_probe_verl.py  # + concept_labels(probe 用)
│   ├── gamesolve_reward_verl.py            # verl 接口奖励函数
│   ├── run_grpo_verl.sh                    # verl GRPO 训练
│   ├── run_grpo_probe_verl.sh              # verl GRPO + Probe 训练
│   ├── run_probe_ablation.sh               # 参数化(SEED/PROBE_LAYER/GPU)训练
│   ├── orchestrate_ablation.sh             # 2-slot(4 卡/run)并行编排器
│   └── analyze_ablation.py                 # 多 seed + 层消融分析
│
├── ── 评估 ────────────────────────────────────────────────
├── eval_gamesolve.py            # 主评估(vLLM/OpenAI 接口,解析+评分)
├── eval_all.sh                  # 多模型批量评估
├── eval_checkpoint.py           # checkpoint 统一评估(§11 用)
├── eval_ood.py / compare_ood.py # OOD 评估与对比
├── compare_experiments.py       # 全方法对比
├── vllm_eval/
│   ├── eval.py                  # vLLM 离线批量评估(ID+OOD,高吞吐)
│   └── run_all_ood.sh / run_parallel_ood.sh
├── benchmark_eval/              # 外部 benchmark 迁移评估(见其 README)
│   ├── GTBench/ TextArena/      # 两个外部博弈 benchmark 克隆(不入库)
│   ├── scripts/                 # serve_models / run_gtbench / textarena_eval / analyze
│   ├── results/  ANALYSIS.md  README.md
│
├── ── 产物 ────────────────────────────────────────────────
├── results/
│   ├── representations/<model>/ # 逐层隐藏表征
│   ├── probing/<model>/         # 探针准确率结果
│   ├── analysis/<model>/        # pooling 对比、热图(pooling_comparison.json)
│   └── phase2/                  # 各训练 checkpoint + 日志
│       ├── sft_cot/ sft_then_grpo/ full_grpo_only/ full_probe_grpo/
│       ├── grpo_verl/ grpo_verl_probe/        # verl 训练(含 merged_step_* HF 格式)
│       ├── ablation_summary.json             # §16 层消融汇总
│       ├── ablation_trajectories/*.jsonl     # §16 15 次 run 的 val 轨迹
│       └── comparison.json / ood_comparison.json
├── eval_results/                # 各方法 ID/OOD 评估 JSON
├── ideas/                       # 6 个后续顶会方向调研报告
└── Qwen/Qwen2.5-3B-Instruct     # 主实验模型(本地)
```

> **对 verl 的改动**(支持 probe):`verl/verl/workers/actor/dp_actor.py`(在 legacy 训练路径 `update_policy` 中集成探针)、`verl/verl/workers/utils/probe_utils.py`(`ProbeState`/`ProbingHeads` 实现)。注意 verl 0.7.1 默认走 legacy worker 路径(`dp_actor.py`),而非新 engine 路径。

---

## 研究流程与实验(对应 `training.md` 各节)

| 阶段 | 实验 | training.md | 一句话结论 |
|:-----|:-----|:-----------:|:-----------|
| 基准 | GameSolve-Bench 构建 | — | 2,400 题,nashpy 真值,3 种 NL 变体 |
| Phase 1 | 逐层线性探针 + pooling 对比 | (phase-1-results.md) | 概念多在**中间层**最可解码;mean/weighted 优于 last-token |
| Phase 2 | GRPO+Probe(transformers) | §1–8 | 冷启动效果差(0.484) |
| Phase 2 | GRPO-only 消融(transformers) | §9 | 0.515 ≈ base,纯 RL 冷启动无效 |
| Phase 2 | SFT on CoT | §10 | **0.945**,压倒性最强 |
| 评估 | 全方法统一对比(ID) | §11 | SFT > SFT→GRPO ≫ base ≈ GRPO > Probe |
| 评估 | OOD 泛化(750 样本) | §12 | RL 泛化更稳,SFT 同分布优势在 OOD 收窄 |
| Phase 2 | **verl GRPO**(重做 RL) | §13 | **0.772**,框架实现质量是关键 |
| Phase 2 | verl GRPO+Probe | §14 | 单 run +1.9pp(**后被证伪为噪声**) |
| 评估 | 外部 benchmark(GTBench/TextArena) | §15 | RL 迁移安全;SFT 交互式博弈退化到 0.38 |
| **验证** | **关键层消融 + 多 seed(15 run)** | §16 | **probe 增益=噪声;关键层假设被推翻** |

---

## 核心结果

### 同分布(ID,200 样本,reward∈[0,1])

| 方法 | Overall | Nash | Best Response |
|:-----|:-------:|:----:|:-------------:|
| Base (Qwen2.5-3B) | 0.525 | 0.403 | 0.683 |
| GRPO-only (transformers) | 0.515 | 0.407 | 0.656 |
| Full+Probe (transformers) | 0.484 | 0.362 | 0.642 |
| **GRPO (verl)** | 0.772 | 0.611 | 0.980 |
| SFT → GRPO | 0.935 | 0.903 | 0.977 |
| **SFT (CoT)** | **0.945** | **0.921** | **0.977** |

### 分布外(OOD,750 样本)

| 方法 | Overall | Nash | BR |
|:-----|:-------:|:----:|:--:|
| Base | 0.433 | 0.389 | 0.480 |
| GRPO (verl) | 0.662 | 0.485 | 0.853 |
| SFT → GRPO | 0.730 | 0.610 | 0.860 |
| SFT (CoT) | 0.734 | 0.615 | 0.864 |

ID 上 SFT 领先 verl GRPO 0.17,OOD 上仅领先 0.07 → RL 泛化更稳。

### 外部 benchmark 迁移(§15)

| | GTBench(vs 随机,8 游戏均分) | TextArena(vs Base 对手) |
|:--|:--:|:--:|
| Base | 0.52 | 0.52(对照) |
| SFT (CoT) | **0.60** | **0.38**(交互式博弈大幅退化) |
| GRPO (verl) | 0.56 | **0.56** |
| GRPO+Probe | 0.57 | 0.48 |

### 关键层消融 + 多 seed(§16,峰值 val reward,3 seeds)

| 条件 | 注入层 | peak val (mean ± std) | vs no-probe |
|:-----|:------:|:---------------------:|:-----------:|
| probe@6 | 早 | 0.7280 ± 0.0084 | +0.5pp |
| **no_probe** | — | 0.7227 ± 0.0059 | — |
| probe@17 | 中(关键) | 0.7133 ± 0.0118 | −0.9pp(t=−1.23,不显著) |
| probe@30 | 晚 | 0.7025 ± 0.0043 | −2.0pp |
| probe@35 | 末 | 0.6853 ± 0.0073 | −3.7pp |

**单调趋势**:注入层越晚,损害越大;"关键层"(17)并非最优。→ probe 损失是与深度相关的干扰项,Phase 1 的概念定位**未**转化为 Phase 2 收益。

---

## 复现指南

环境:8× H20 (96GB),单节点。模型:`Qwen/Qwen2.5-3B-Instruct`(本地)。

```bash
# ── 基准 ──────────────────────────────────────────
python gamesolve_gen.py            # 生成 gamesolve_bench.jsonl (seed=42)
python generate_ood_bench.py       # 生成 OOD 基准

# ── Phase 1:探针诊断 ─────────────────────────────
python extract_representations_multi.py    # 逐层表征
python train_probes_parallel.py            # 训练探针
python analyze_pooling.py                  # pooling 对比 + critical layer

# ── Phase 2:训练 ────────────────────────────────
bash run_sft.sh                            # SFT on CoT(最强方法)
bash verl_scripts/run_grpo_verl.sh         # verl GRPO(最强 RL)

# ── 评估 ─────────────────────────────────────────
python vllm_eval/eval.py --model_path <ckpt> --bench_path gamesolve_bench.jsonl --max_samples 200 --output_dir eval_results/<name>
bash benchmark_eval/scripts/serve_models.sh && bash benchmark_eval/scripts/run_gtbench.sh qwen3b-base 20   # 外部 benchmark

# ── 关键层消融 + 多 seed(§16)────────────────────
bash verl_scripts/orchestrate_ablation.sh  # 15 run,2×4卡并行,~3.7h
python verl_scripts/analyze_ablation.py    # 显著性 + 层消融分析
```

> 评估优先用 `vllm_eval/`(vLLM,非 transformers);训练充分利用 8 卡。verl 默认 legacy worker 路径,probe 集成在 `dp_actor.py`。

---

## 后续方向([`ideas/`](ideas/))

当前 probing-guided enhancement 不达预期后,调研了 6 个顶会方向(详见 `ideas/README.md`),推荐度排序:

1. **Game-Theoretic Process Reward Models (GT-PRM)** — 用可验证的博弈论求解步骤(消去占优→BR→NE)作零成本稠密过程奖励
2. **Representation Engineering for Reasoning** — 推理时的表征干预
3. **Curriculum Strategic Reasoning** — 难度课程
4. Structured Reasoning Distillation / 5. Nash-GRPO / 6. LLM Self-Play + Emergent Communication

复用资产:GameSolve-Bench 基准、Phase 1 探针诊断、verl GRPO 管线、RL-vs-SFT 泛化结论、2×4卡并行编排器。

---

## 依赖

Python 3.12 · `numpy` `nashpy`(博弈求解) · `torch` `transformers` `peft` · `vllm`(服务/评估) · `verl` v0.7.1(`pip install -e ./verl`,RL 训练) · `sklearn`(探针) · 外部 benchmark:`open_spiel` `rlcard`(GTBench)、`textarena`(`pip install -e benchmark_eval/TextArena`)
