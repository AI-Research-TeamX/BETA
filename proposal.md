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

$$h_i^{(l)} = \text{LLM}^{(l)}(x_i) \in \mathbb{R}^d$$

We use the hidden state at the **last token position** of the input (before generation begins), as this aggregates the full context. We additionally experiment with **mean pooling** over all input tokens.

**Models:** We run probing experiments on:
- `Qwen2.5-7B`, `Qwen2.5-72B`
- `LLaMA-3.1-8B`, `LLaMA-3.1-70B`
- `DeepSeek-R1-Distill-Qwen-7B` (reasoning-tuned baseline)

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

## 4. Phase 2: Auxiliary Probing-Loss Fine-Tuning

### 4.1 Overview

Using the diagnostic outputs from Phase 1, we perform **lightweight fine-tuning** that jointly optimizes:
1. The standard next-token generation loss (to preserve general LLM capability)
2. Auxiliary classification losses at the **critical layers** identified in Phase 1, using lightweight probing heads

This is a **parameter-efficient** approach: the probing heads are small (linear layers, ~$d \times C$ parameters), and we optionally apply LoRA to the backbone.

### 4.2 Model Architecture

```
Input: game description + payoff matrix (tokenized)
         │
   [Frozen or LoRA-adapted Transformer Backbone]
         │
   Layer 1 hidden states → [Probing Head 1] → concept loss₁
   Layer 2 hidden states → [Probing Head 2] → concept loss₂
         ⋮
   Layer L* hidden states → [Probing Head L*] → concept loss_{L*}
         │
   [LM Head]
         │
   Output: generated reasoning + answer → generation loss
```

**Probing heads:** One linear head per (critical layer, concept label) pair selected from Phase 1. Each head is:

$$\hat{y}^{(l,t)} = \text{softmax}(W^{(l,t)} \cdot \text{sg}(h^{(l)}) + b^{(l,t)})$$

where $\text{sg}(\cdot)$ denotes stop-gradient **on the forward pass to the head** only; gradients do **flow back** through $h^{(l)}$ into the backbone (this is what makes it "auxiliary loss" rather than frozen probing).

### 4.3 Training Objective

The total loss for a training instance $(x_i, y_i^{\text{gen}}, \{y_i^{(t)}\}_t)$ is:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{gen}}(x_i, y_i^{\text{gen}})}_{\text{generation loss}} + \lambda \sum_{l \in \mathcal{L}^*} \sum_{t \in \mathcal{T}} w_t \cdot \underbrace{\mathcal{L}_{\text{CE}}(\hat{y}_i^{(l,t)}, y_i^{(t)})}_{\text{probing auxiliary loss}}$$

**Parameters:**
- $\lambda$: global auxiliary loss weight (tuned on val set, typical range $[0.01, 0.5]$)
- $w_t$: per-concept weight, set inversely proportional to Phase 1 peak probing accuracy (up-weight harder-to-learn concepts)
- $\mathcal{T}$: set of concept labels used as supervision (we ablate using all 6 vs. only the 3 coarse-grained)
- $\mathcal{L}^*$: critical layers from Phase 1 (top-3 layers per concept by probing accuracy)

### 4.4 Training Data

**Source:** GameSolve-Bench (2,400 samples) + optional augmentation via our generator.

**Label construction:** Each training sample already includes:
- Ground-truth game answer (for $\mathcal{L}_{\text{gen}}$)
- Structured metadata: equilibrium type, game type, dominance flags, BR direction (for auxiliary losses $\mathcal{L}_{\text{CE}}$)

**Description style mixing:** We sample uniformly from the 3 description styles (abstract / story / compact) to improve generalization across surface forms.

**Data split:** 70% train / 10% val (for $\lambda$, $w_t$ selection) / 20% held-out test.

### 4.5 Backbone Fine-Tuning Strategy

We evaluate three backbone regimes:

| Regime | Backbone | Probing Heads | # Trainable Params |
|---|---|---|---|
| **Probe-only** | Frozen | Trained | ~$6 \times d \times C$ |
| **LoRA + Probe** | LoRA ($r=16$) | Trained | ~150M (7B model) |
| **Full + Probe** | Full fine-tune | Trained | ~7B |

We expect LoRA + Probe to dominate: enough backbone plasticity for the auxiliary signal to reshape representations, without the instability of full fine-tuning on a small dataset.

### 4.6 Inference

At inference time, the probing heads are **discarded**. The enhanced backbone is used directly for game solving. The benefit is entirely encoded in the updated LLM weights, with zero inference overhead.

---

## 5. Evaluation

### 5.1 Primary Metrics

**Game solving accuracy:** Exact-match accuracy on GameSolve-Bench held-out test set, broken down by:
- Task: Nash Equilibrium identification vs. Best Response computation
- Difficulty: easy / medium / hard
- Matrix size: 2×2 / 3×3 / 4×4 / 2×3 / 3×2
- Equilibrium type: pure / mixed / both

**Reasoning quality (CoT):** For samples with chain-of-thought annotations, we score the reasoning trace using:
- Step correctness (rubric-based, automatic): does each reasoning step correctly apply dominance elimination / Nash calculation?
- Logical coherence (LLM-as-judge, GPT-4o): is the reasoning chain internally consistent?

### 5.2 Diagnostic Metrics (Phase 1 Evaluation)

- Probing accuracy per (layer, concept): reported as mean ± std over 5 random seeds
- Baseline: majority-class classifier, random linear probe (random $W$), upper bound: fine-tuned linear head on representations from task-specific SFT model

### 5.3 Ablation Study

| Ablation | Variable |
|---|---|
| No auxiliary loss | $\lambda = 0$ |
| Coarse labels only | $\mathcal{T} = \{\text{eq\_type, game\_type, difficulty}\}$ |
| Fine-grained labels only | $\mathcal{T} = \{\text{dominance, br\_direction, eq\_uniqueness}\}$ |
| Random layer injection | $\mathcal{L}^*$ = random layers (not from Phase 1) |
| Last layer only | $\mathcal{L}^* = \{L\}$ |
| Uniform $w_t$ | $w_t = 1$ for all $t$ |

The **random layer injection** ablation is critical: it tests whether the Phase 1 layer selection actually matters, vs. auxiliary supervision helping regardless of where it's injected.

### 5.4 Generalization Evaluation

To test whether the enhancement transfers beyond GameSolve-Bench:
- **TMGBench** (144 game types, story-based): tests surface-form generalization
- **GTBench** (sequential/dynamic games): tests structure generalization
- **General math reasoning** (MATH, GSM8K): tests whether game-specific training hurts or is neutral

We expect improvement on TMGBench (same domain, different surface), neutral on GTBench (different game structure), and neutral-to-slight-improvement on math reasoning (strategic reasoning overlaps with logical deduction).

---

## 6. Implementation Plan

### 6.1 Infrastructure

```
GameSolve-Bench (2,400 samples)
├── Phase 1: Probing Pipeline
│   ├── representation_extractor.py   # Hook-based hidden state extraction
│   ├── probe_trainer.py              # L-BFGS logistic regression per (layer, concept)
│   ├── probe_analysis.py             # Accuracy curves, MI, linearity coefficient
│   └── critical_layer_selector.py   # Select L* per concept
│
└── Phase 2: Auxiliary Training Pipeline
    ├── probing_head.py               # Linear heads with configurable stop-grad
    ├── aux_loss_trainer.py           # Joint loss, LoRA + probe training loop
    ├── weight_scheduler.py           # Adaptive w_t from Phase 1 accuracy
    └── eval_pipeline.py              # GameSolve + TMGBench + GTBench eval
```

### 6.2 Timeline

| Phase | Task | Duration |
|---|---|---|
| Phase 1 | Representation extraction (all models × all layers) | Week 1–2 |
| Phase 1 | Probing classifier training + analysis | Week 2–3 |
| Phase 1 | Write-up of diagnostic findings | Week 3–4 |
| Phase 2 | Auxiliary training implementation | Week 4–5 |
| Phase 2 | Hyperparameter search ($\lambda$, $w_t$, LoRA rank) | Week 5–6 |
| Phase 2 | Ablation experiments | Week 6–7 |
| Phase 2 | Generalization evaluation | Week 7–8 |
| Both | Paper writing | Week 8–10 |

### 6.3 Compute Estimate

| Task | Hardware | Estimated Time |
|---|---|---|
| Phase 1 extraction (7B models, 2400 samples) | 1× A100 | ~4h per model |
| Phase 1 extraction (70B models) | 4× A100 | ~12h per model |
| Phase 1 probing training (all layers × 6 concepts) | CPU cluster | ~2h total |
| Phase 2 LoRA fine-tuning (7B) | 2× A100 | ~6h per run |
| Phase 2 full ablations (×8 conditions) | 2× A100 | ~50h total |

---

## 7. Expected Contributions

1. **First systematic linear probing study of game-theoretic concepts in LLMs**, producing an interpretability map of where and how strategic reasoning concepts are encoded across model families and scales.

2. **A probing-accuracy-informed auxiliary supervision method** that is:
   - Parameter-efficient (probing heads + LoRA)
   - Zero inference overhead (heads discarded at test time)
   - Motivated by interpretability findings, not heuristic

3. **Empirical findings** on the relationship between internal representation quality and downstream game-solving accuracy, potentially revealing: do models that "know" game theory concepts internally (high probing accuracy) also solve games better? Or is there a representation-behavior gap?

4. **A reusable framework** (probing pipeline + benchmark + auxiliary training code) for future work on other structured reasoning domains.

---

## 8. Related Work & Differentiation

| Work | Method | Difference from Ours |
|---|---|---|
| Tenney et al. (2019) | Probing linguistic structure | We probe game-theoretic, not linguistic concepts |
| Lightman et al. (2023) | Process reward models | Their auxiliary signal is human-labeled reasoning steps; ours is game-theoretic labels (automated, verifiable) |
| RepE / Zou et al. (2023) | Representation engineering | They steer activations at inference; we use representations to shape training |
| TMGBench / GTBench | Evaluation benchmarks | We use benchmarks as training signal, not just evaluation |
| CoRY (NeurIPS 2024) | Multi-agent self-play fine-tuning | Multi-agent training loop; we use single-model auxiliary supervision |

---

## 9. Open Questions & Risks

| Question | Mitigation |
|---|---|
| Phase 1 probing accuracy may be universally low → no "critical layers" identifiable | Use relative ranking of layers; even low-accuracy probing gives layer ordering |
| GameSolve-Bench (2,400 samples) too small for Phase 2 fine-tuning | Use the generator to scale to 10k+ samples; apply data augmentation (payoff perturbation, style mixing) |
| Auxiliary loss may interfere with generation quality | Monitor perplexity on general benchmarks (MMLU, HellaSwag) during training; add KL regularization to original model if needed |
| Results don't transfer to TMGBench / GTBench | Reframe as in-domain diagnostic + enhancement; generalization as a future work direction |

---
