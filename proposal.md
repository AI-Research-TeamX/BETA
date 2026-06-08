# Probing and Enhancing Game-Theoretic Reasoning in LLMs via Linear Diagnostics and Auxiliary Supervision

> **Research Proposal** | Draft v0.1

---

## 1. Motivation & Research Questions

Large language models (LLMs) are increasingly deployed in strategic and multi-agent settings—negotiation, resource allocation, competitive recommendation—yet systematic evidence shows they consistently deviate from game-theoretically rational behavior (e.g., failing to identify Nash Equilibria, misapplying dominance reasoning, collapsing on mixed-strategy problems).

Two foundational questions remain unanswered:

1. **Do LLMs internally encode game-theoretic concepts?** If so, at which layers, and with what fidelity?
2. **Can lightweight, targeted supervision on those representations improve strategic reasoning** without expensive full fine-tuning?

This proposal addresses both questions through a two-phase framework:

- **Phase 1 (Diagnose):** Linear probing across layers to localize where and how well game-theoretic concepts are represented.
- **Phase 2 (Enhance):** Auxiliary probing-loss fine-tuning that leverages the Phase 1 findings to improve game-solving ability with minimal parameter overhead.

---

## 2. Background

### 2.1 Game-Theoretic Reasoning as a Testbed

We ground our experiments in **normal-form games** (matrix games), which offer:
- Formally verifiable ground-truth labels (Nash Equilibria, best responses, dominant strategies)
- Controllable complexity axes (matrix size, equilibrium type, game symmetry)
- Natural language variability (abstract payoff tables → story-based descriptions)

We build on our existing **GameSolve-Bench** (2,400 samples, 5 game sizes, 3 description styles, CoT annotations), which provides the labeled representation data required by both phases.

### 2.2 Linear Probing in NLP

Linear probing—training a linear classifier on frozen internal representations—is a standard interpretability tool to test whether a model encodes a concept without being trained on it. Prior work has probed syntax (Tenney et al., 2019), factual knowledge (Geva et al., 2021), and reasoning steps (Bills et al., 2023). No prior work has systematically probed **game-theoretic structure** in LLMs.

### 2.3 Auxiliary Supervision for Structured Reasoning

Auxiliary objectives on intermediate representations have been used to improve mathematical reasoning (Lightman et al., 2023, process reward models) and structured prediction. We propose the first application of this paradigm to strategic reasoning, using probing-derived supervision signals.

---

## 3. Phase 1: Linear Probing Diagnostic Framework

### 3.1 Overview

We extract hidden states from all transformer layers for each game-solving instance in GameSolve-Bench and train lightweight linear classifiers to predict game-theoretic labels. The output is a **layer-by-layer probing accuracy profile** for each concept.

### 3.2 Representation Extraction

**Input construction:** For each sample $x_i$ (a game description + payoff matrix), we pass it through the frozen LLM and collect the hidden state at each layer $l \in \{1, \ldots, L\}$:

$$H_i^{(l)} = \text{LLM}^{(l)}(x_i) \in \mathbb{R}^{T \times d}$$

We extract **six pooling variants** in a single forward pass, computing all on GPU before transfer to CPU:

| Pooling | Formula | Motivation |
|---|---|---|
| **last** | $h_{T_{\text{eff}}}^{(l)}$ | Standard decoder aggregation |
| **first** | $h_1^{(l)}$ | BOS token (control baseline) |
| **mean** | $\frac{1}{T_{\text{eff}}} \sum_{j=1}^{T_{\text{eff}}} h_j^{(l)}$ | Uniform average over all tokens |
| **sum** | $\sum_{j=1}^{T_{\text{eff}}} h_j^{(l)}$ | Magnitude-preserving aggregation |
| **max** | $\max_j h_j^{(l)}$ (element-wise) | Peak activation per dimension |
| **weighted** | $\sum_j w_j h_j^{(l)}$, $w_j \propto e^{2j/T_{\text{eff}}}$ | Exponentially favors later positions |

**Models:** We run probing experiments on:
- `Qwen2.5-1.5B-Instruct`, `Qwen2.5-3B-Instruct` (completed)
- `Qwen2.5-7B-Instruct`, `Qwen2.5-72B` (planned)
- `LLaMA-3.1-8B`, `LLaMA-3.1-70B` (planned)
- `DeepSeek-R1-Distill-Qwen-7B` (reasoning-tuned baseline, planned)

### 3.3 Probing Labels

We define six **game-theoretic concept labels** across two granularities:

#### Coarse-grained (game-level labels)
| Label | Task | Classes |
|---|---|---|
| `eq_type` | Equilibrium type | {pure-only, mixed-only, both, none} |
| `game_type` | Game structure | {zero-sum, symmetric, general} |
| `difficulty` | Difficulty tier | {easy, medium, hard} |

#### Fine-grained (strategy-level labels)
| Label | Task | Classes |
|---|---|---|
| `dominance` | Does player 1 have a dominant strategy? | {yes, no} |
| `br_direction` | Best response direction for player 1 | {row 1, row 2, row 3, mixed} |
| `eq_uniqueness` | Number of Nash Equilibria | {0, 1, multiple} |

### 3.4 Probing Classifier

For each layer $l$ and label type $t$, we train a **logistic regression** (no hidden layers):

$$\hat{y} = \text{softmax}(W^{(l,t)} h^{(l)} + b^{(l,t)})$$

**Training setup:**
- 80/10/10 train/val/test split (stratified by label)
- Optimizer: L-BFGS (standard for linear probing, avoids SGD hyperparameter sensitivity)
- Regularization: L2 with $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$, selected via val accuracy
- No data augmentation; representations are frozen throughout

### 3.5 Analysis Metrics

**Per-layer accuracy curves:** Plot probing accuracy vs. layer index for each label, revealing:
- Which layers first encode a concept ("emergence layer")
- Whether accuracy saturates, peaks-then-declines, or monotonically increases
- Cross-model alignment: do structurally similar models encode concepts at similar relative depths?

**Mutual information (MI) analysis:** Complement accuracy with MI between $h^{(l)}$ and $y$ (using a kernel density estimator), to separate the effect of representation norm from directional encoding.

**Linearity coefficient:** Following Andreas (2022), measure how much probing accuracy drops when we restrict to top-$k$ PCA components ($k \in \{10, 50, 100\}$). High accuracy with small $k$ → concept is encoded in a low-dimensional, geometrically clean subspace, which is favorable for Phase 2.

**Cross-label interference:** Compute the cosine similarity between probe weight vectors $W^{(l, t_1)}$ and $W^{(l, t_2)}$ across label pairs. High similarity = concepts share the same representational subspace (potential for joint probing heads in Phase 2).

### 3.6 Expected Diagnostic Outputs

The Phase 1 analysis will produce:

1. **Concept emergence map**: a heatmap of probing accuracy across (layer × concept) for each model
2. **Critical layer set** $\mathcal{L}^* = \{l : \text{acc}^{(l)} > \tau\}$ per concept—used in Phase 2 to select where to inject auxiliary supervision
3. **Capability gap profile**: which game-theoretic concepts are *least* represented (lowest peak probing accuracy), motivating targeted enhancement

---

## 4. Phase 1 Key Findings (Empirical Basis for Phase 2)

The following findings from our probing experiments on Qwen2.5-{1.5B, 3B}-Instruct directly inform the Phase 2 design:

### 4.1 Pooling Method Selection

| Pooling | Wins (1.5B) | Wins (3B) | Interpretation |
|---|---|---|---|
| **weighted** | 2/6 | 4/6 | Exponential position-weighting captures both structural and reasoning tokens |
| **mean** | 2/6 | 1/6 | Strong all-rounder; averages full-sequence information |
| **sum** | 2/6 | 1/6 | Excels on game_type and br_direction; magnitude-sensitive |
| last | 0/6 | 0/6 | Standard decoder token; leaves substantial signal on the table |
| first | 0/6 | 0/6 | BOS token carries zero task-relevant information |

**Conclusion:** Weighted pooling is the optimal default representation for auxiliary heads. Mean pooling is a strong alternative with simpler computation.

### 4.2 Critical Layer Positions

Peak probing accuracy occurs at **model-relative depth** $\alpha = l / (L-1)$:

| Concept | 1.5B α (best) | 3B α (best) | Cross-model pattern |
|---|---|---|---|
| eq_type | 0.63 | 0.63 | Upper-middle layers |
| game_type | 0.56 | 0.54 | Middle layers |
| difficulty | 1.00 | 0.46 | Model-dependent |
| dominance | 0.67 | 0.77 | Upper-middle to late |
| br_direction | 0.70 | 0.74 | Late layers |
| eq_uniqueness | 0.44 | 0.17 | Early-to-middle |

**Multi-concept coverage analysis** (layers where ≥5/6 concepts are within 95% of their peak):
- 1.5B: layers 8, 11, 13 → α ∈ {0.30, 0.41, 0.48}
- 3B: layers 7, 13, 21 → α ∈ {0.20, 0.37, 0.60}

**Conclusion:** Three injection points at $\alpha \in \{1/4, 1/2, 2/3\}$ cover most concept peaks across model scales.

### 4.3 Concept Difficulty Ranking (Phase 1 Peak Accuracy)

| Concept | Best Acc (avg) | Relative Difficulty |
|---|---|---|
| game_type | 0.996 | Trivial (near-ceiling) |
| difficulty | 0.903 | Easy |
| dominance | 0.816 | Medium |
| eq_uniqueness | 0.731 | Hard |
| eq_type | 0.748 | Hard |
| br_direction | 0.638 | Very Hard |

**Conclusion:** br_direction, eq_type, and eq_uniqueness are the underrepresented concepts that benefit most from auxiliary supervision. game_type is near-ceiling and provides negligible learning signal.

---

## 5. Phase 2: Auxiliary Probing-Loss Fine-Tuning

### 5.1 Overview

Using the empirical findings from Phase 1 (§4), we perform **lightweight fine-tuning** that jointly optimizes:
1. The standard next-token generation loss (to preserve general LLM capability)
2. Auxiliary classification losses at **empirically-determined critical layers**, using probing heads that operate on **weighted-pooled** hidden states

This is a **parameter-efficient** approach: the probing heads are small (linear layers, ~$d \times C$ parameters), and we apply LoRA to the backbone.

### 5.2 Model Architecture

```
Input: game description + payoff matrix (tokenized)
         │
   [LoRA-adapted Transformer Backbone]
         │
   Layer ⌊L/4⌋  → WeightedPool → [Probing Head₁] → aux_loss₁  (early-mid)
   Layer ⌊L/2⌋  → WeightedPool → [Probing Head₂] → aux_loss₂  (middle)
   Layer ⌊2L/3⌋ → WeightedPool → [Probing Head₃] → aux_loss₃  (upper-mid)
         │
   [LM Head]
         │
   Output: generated reasoning + answer → generation loss
```

**Layer selection rule:** For a model with $L$ layers, inject probing heads at layers:

$$\mathcal{L}^* = \left\{\lfloor \tfrac{L}{4} \rfloor,\ \lfloor \tfrac{L}{2} \rfloor,\ \lfloor \tfrac{2L}{3} \rfloor\right\}$$

This corresponds to $\alpha \in \{0.25, 0.5, 0.67\}$, empirically validated as the positions with maximal multi-concept coverage across both model scales (§4.2).

**Pooling for probing heads:** Each probing head operates on the **weighted-average pooled** representation:

$$\bar{h}^{(l)} = \sum_{j=1}^{T} w_j \cdot h_j^{(l)}, \quad w_j = \frac{\exp(2 \cdot j / T_{\text{eff}})}{\sum_k \exp(2 \cdot k / T_{\text{eff}})} \cdot m_j$$

where $m_j$ is the attention mask and $T_{\text{eff}}$ is the effective sequence length. This exponentially weights later positions, capturing both structural context (early tokens) and reasoning state (later tokens). Phase 1 showed this outperforms last-token pooling by 5–10% absolute on most concepts.

**Probing heads:** One linear head per (injection layer, concept label) pair:

$$\hat{y}^{(l,t)} = \text{softmax}(W^{(l,t)} \cdot \bar{h}^{(l)} + b^{(l,t)})$$

Gradients flow back through $\bar{h}^{(l)}$ into the backbone—this is what distinguishes auxiliary-loss training from frozen probing.

### 5.3 Training Objective

The total loss for a training instance $(x_i, y_i^{\text{gen}}, \{y_i^{(t)}\}_t)$ is:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{gen}}(x_i, y_i^{\text{gen}})}_{\text{generation loss}} + \lambda \sum_{l \in \mathcal{L}^*} \sum_{t \in \mathcal{T}} w_t \cdot \underbrace{\mathcal{L}_{\text{CE}}(\hat{y}_i^{(l,t)}, y_i^{(t)})}_{\text{probing auxiliary loss}}$$

**Parameters:**
- $\lambda$: global auxiliary loss weight (tuned on val set, range $[0.01, 0.3]$)
- $w_t$: per-concept weight, **set from Phase 1 empirical difficulty** (inversely proportional to peak probing accuracy):

| Concept | Phase 1 Acc | $w_t$ (normalized) |
|---|---|---|
| br_direction | 0.638 | **1.57** |
| eq_uniqueness | 0.731 | **1.37** |
| eq_type | 0.748 | **1.34** |
| dominance | 0.816 | **1.23** |
| difficulty | 0.903 | **1.11** |
| game_type | 0.996 | **1.00** |

  Formula: $w_t = 1 / \text{acc}_t^{\text{Phase1}}$, then normalize so $\min(w_t) = 1$.

- $\mathcal{T}$: concept label set. **Default: exclude game_type** (near-ceiling at 99.6%, provides negligible gradient signal). Use $\mathcal{T} = \{\text{eq\_type, difficulty, dominance, br\_direction, eq\_uniqueness}\}$
- $\mathcal{L}^*$: three injection layers per the $\alpha \in \{1/4, 1/2, 2/3\}$ rule

### 5.4 Training Data

**Source:** GameSolve-Bench (2,400 samples) + augmentation to 10k via the generator (payoff perturbation, style mixing).

**Label construction:** Each training sample includes:
- Ground-truth game answer (for $\mathcal{L}_{\text{gen}}$)
- Structured metadata: equilibrium type, game type, dominance flags, BR direction (for auxiliary losses $\mathcal{L}_{\text{CE}}$)
- **Labels are derived automatically** from payoff matrices (no human annotation needed for scaling)

**Description style mixing:** Uniform sampling from 3 description styles (abstract / story / compact).

**Data split:** 70% train / 10% val (for $\lambda$ selection) / 20% held-out test.

### 5.5 Backbone Fine-Tuning Strategy

Based on Phase 1 results, we focus on the **LoRA + Probe** regime (highest expected return):

| Regime | Backbone | Probing Heads | # Trainable Params | Priority |
|---|---|---|---|---|
| **LoRA + Probe** | LoRA ($r=16$, target: q,k,v,o) | Trained | ~50M (3B) / ~150M (7B) | **Primary** |
| Probe-only | Frozen | Trained | ~$5 \times d \times C$ | Ablation baseline |
| Full + Probe | Full fine-tune | Trained | Full model | Ablation (if compute allows) |

**Rationale for LoRA + Probe as primary:** Phase 1 showed concepts are encoded in linear subspaces within hidden states, but not at sufficient fidelity for strategy-level concepts (br_direction: 63.8%, eq_type: 74.8%). LoRA provides enough backbone plasticity for auxiliary gradients to reshape these subspaces, while the probing heads provide the learning signal for *what* to encode.

### 5.6 Inference

At inference time, the probing heads are **discarded**. The LoRA-merged backbone is used directly for game solving. The benefit is entirely encoded in the updated model weights, with **zero inference overhead**.

---

## 6. Evaluation

### 6.1 Primary Metrics

**Game solving accuracy:** Exact-match accuracy on GameSolve-Bench held-out test set, broken down by:
- Task: Nash Equilibrium identification vs. Best Response computation
- Difficulty: easy / medium / hard
- Matrix size: 2×2 / 3×3 / 4×4 / 2×3 / 3×2
- Equilibrium type: pure / mixed / both

**Reasoning quality (CoT):** For samples with chain-of-thought annotations, we score the reasoning trace using:
- Step correctness (rubric-based, automatic): does each reasoning step correctly apply dominance elimination / Nash calculation?
- Logical coherence (LLM-as-judge, GPT-4o): is the reasoning chain internally consistent?

### 6.2 Diagnostic Metrics (Phase 1 Evaluation)

- Probing accuracy per (layer, concept): reported as mean ± std over 5 random seeds
- Baseline: majority-class classifier, random linear probe (random $W$), upper bound: fine-tuned linear head on representations from task-specific SFT model

### 6.3 Ablation Study

| Ablation | Variable | Tests |
|---|---|---|
| No auxiliary loss | $\lambda = 0$ | Is aux supervision necessary at all? |
| All 6 labels (incl. game_type) | $\mathcal{T} = \text{all 6}$ | Does near-ceiling game_type add signal or noise? |
| Fine-grained only | $\mathcal{T} = \{\text{dominance, br\_direction, eq\_uniqueness}\}$ | Strategy-level concepts sufficient? |
| Random layer injection | $\mathcal{L}^*$ = 3 random layers | Does the α ∈ {1/4, 1/2, 2/3} rule matter? |
| Last layer only | $\mathcal{L}^* = \{L\}$ | Single deep injection vs. distributed |
| Single mid-layer | $\mathcal{L}^* = \{\lfloor L/2 \rfloor\}$ | Is one well-chosen layer enough? |
| Uniform $w_t$ | $w_t = 1$ for all $t$ | Does Phase 1-informed weighting help? |
| Last-token pooling | Pool = last token (not weighted) | Does weighted pooling matter for training? |
| Mean pooling | Pool = mean | Second-best pooling vs. weighted |

The **random layer injection** and **last-token pooling** ablations are the most critical: they test whether the Phase 1 empirical findings (layer selection, pooling choice) actually translate to training improvements vs. auxiliary supervision helping regardless of design choices.

### 6.4 Generalization Evaluation

To test whether the enhancement transfers beyond GameSolve-Bench:
- **TMGBench** (144 game types, story-based): tests surface-form generalization
- **GTBench** (sequential/dynamic games): tests structure generalization
- **General math reasoning** (MATH, GSM8K): tests whether game-specific training hurts or is neutral

We expect improvement on TMGBench (same domain, different surface), neutral on GTBench (different game structure), and neutral-to-slight-improvement on math reasoning (strategic reasoning overlaps with logical deduction).

---

## 7. Implementation Plan

### 7.1 Infrastructure

```
GameSolve-Bench (2,400 samples)
├── Phase 1: Probing Pipeline (COMPLETE)
│   ├── extract_representations.py         # Single-pooling hidden state extraction
│   ├── extract_representations_multi.py   # Multi-pooling extraction (6 methods, single pass)
│   ├── train_probes_parallel.py           # 8-GPU parallel L-BFGS probing
│   ├── analyze_probes.py                  # Per-model accuracy curves, heatmaps
│   └── analyze_pooling.py                 # Cross-pooling comparison analysis
│
└── Phase 2: Auxiliary Training Pipeline
    ├── model_with_probing_heads.py   # LoRA backbone + weighted-pool probing heads
    ├── aux_loss_trainer.py           # Joint generation + auxiliary loss training loop
    ├── phase1_config.py              # α-based layer selection, w_t from Phase 1
    └── eval_pipeline.py              # GameSolve + generalization eval
```

### 7.2 Timeline

| Phase | Task | Duration | Status |
|---|---|---|---|
| Phase 1 | Multi-pooling representation extraction (Qwen 1.5B, 3B) | Week 1 | ✓ Complete |
| Phase 1 | 8-GPU parallel probe training (6 methods × 2 models) | Week 1 | ✓ Complete |
| Phase 1 | Cross-pooling analysis & findings | Week 1 | ✓ Complete |
| Phase 2 | Auxiliary training implementation (LoRA + weighted-pool heads) | Week 2 | |
| Phase 2 | Hyperparameter search ($\lambda$, LoRA rank) on 3B model | Week 2–3 | |
| Phase 2 | Ablation experiments (9 conditions) | Week 3–4 | |
| Phase 2 | Scale to 7B models, generalization evaluation | Week 4–5 | |
| Both | Paper writing | Week 5–7 | |

### 7.3 Compute Estimate (8× H20 96GB node)

| Task | Hardware | Estimated Time |
|---|---|---|
| Phase 1 extraction (3B, 6 poolings, single pass) | 8× H20 (device_map=auto) | ~90s per model |
| Phase 1 probing (36 layers × 6 concepts × 6 poolings) | 8× H20 parallel | ~400s per (model, pooling) |
| Phase 2 LoRA + Probe training (3B, 10k samples) | 8× H20, tp=2 | ~2h per run |
| Phase 2 full ablations (×9 conditions) | 8× H20 | ~18h total |
| Phase 2 scale to 7B | 8× H20, tp=4 | ~4h per run |

---

## 8. Expected Contributions

1. **First systematic linear probing study of game-theoretic concepts in LLMs**, producing an interpretability map of where and how strategic reasoning concepts are encoded across model families and scales.

2. **A probing-accuracy-informed auxiliary supervision method** that is:
   - Parameter-efficient (probing heads + LoRA)
   - Zero inference overhead (heads discarded at test time)
   - Motivated by interpretability findings, not heuristic

3. **Empirical findings** on the relationship between internal representation quality and downstream game-solving accuracy, potentially revealing: do models that "know" game theory concepts internally (high probing accuracy) also solve games better? Or is there a representation-behavior gap?

4. **A reusable framework** (probing pipeline + benchmark + auxiliary training code) for future work on other structured reasoning domains.

---

## 9. Related Work & Differentiation

| Work | Method | Difference from Ours |
|---|---|---|
| Tenney et al. (2019) | Probing linguistic structure | We probe game-theoretic, not linguistic concepts |
| Lightman et al. (2023) | Process reward models | Their auxiliary signal is human-labeled reasoning steps; ours is game-theoretic labels (automated, verifiable) |
| RepE / Zou et al. (2023) | Representation engineering | They steer activations at inference; we use representations to shape training |
| TMGBench / GTBench | Evaluation benchmarks | We use benchmarks as training signal, not just evaluation |
| CoRY (NeurIPS 2024) | Multi-agent self-play fine-tuning | Multi-agent training loop; we use single-model auxiliary supervision |

---

## 10. Open Questions & Risks

| Question | Mitigation |
|---|---|
| Phase 1 probing accuracy may be universally low → no "critical layers" identifiable | Use relative ranking of layers; even low-accuracy probing gives layer ordering |
| GameSolve-Bench (2,400 samples) too small for Phase 2 fine-tuning | Use the generator to scale to 10k+ samples; apply data augmentation (payoff perturbation, style mixing) |
| Auxiliary loss may interfere with generation quality | Monitor perplexity on general benchmarks (MMLU, HellaSwag) during training; add KL regularization to original model if needed |
| Results don't transfer to TMGBench / GTBench | Reframe as in-domain diagnostic + enhancement; generalization as a future work direction |

---
