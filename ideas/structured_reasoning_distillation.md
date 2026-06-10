# StructReason: Distilling Reasoning Skeletons via Probing-Guided Structural Transfer and Game-Theoretic Verification

> Deep Research Report | June 2025

---

## 1. Executive Summary

**Core Idea:** Instead of distilling a large reasoning model's token-level CoT, we distill its *reasoning skeleton* — the load-bearing structural operations identified through probing classifiers. We combine structural distillation (SFT) with game-theoretic step-level verification (GRPO) to train small models that reason concisely and correctly.

**Key Insight:** Recent work shows (1) RL does NOT create new reasoning patterns, only distillation does [arXiv 2504.13837], (2) reasoning verbosity is an optimization artifact not deeper thinking [arXiv 2504.05185], (3) cognitive behaviors (verification, backtracking) matter more than answer correctness for self-improvement [arXiv 2503.01307]. This motivates identifying and transferring only the *structurally critical* reasoning steps.

**Target Venue:** ICML 2026 or NeurIPS 2025

---

## 2. Literature Review

### 2.1 Reasoning Distillation (Core Papers)

| Paper | Venue | Key Contribution |
|---|---|---|
| **DeepSeek-R1** (Guo et al.) | arXiv 2501.12948 | Pure RL elicits reasoning; distilled models via SFT on CoT |
| **LIMO** (Ye et al.) | arXiv 2502.03387 | 1% training data sufficient for sophisticated reasoning via "cognitive templates" |
| **LIMR** (GAIR-NLP) | arXiv 2502.11886 | 1,389 strategically selected samples > 8,523 random for RL |
| **Light-R1** (Qihoo 360) | arXiv 2503.10460 | Curriculum SFT+DPO+RL; Light-R1-14B surpasses many 32B |
| **TinyR1-32B** | arXiv 2503.04872 | Branch-Merge distillation; +5.5 Math, +4.4 Coding over DeepSeek-R1-Distill |
| **ThinkPO** | arXiv 2502.13173 | DPO with short/long CoT as rejected/chosen; +3.8% on MATH500 |

### 2.2 Process Reward Models (PRMs)

| Paper | Venue | Key Contribution |
|---|---|---|
| **Let's Verify Step by Step** (Lightman et al., OpenAI) | ICLR 2024 | Process supervision >> outcome supervision; PRM800K |
| **PRIME** (Qwen Team) | arXiv 2502.01456 | Online PRM updates via implicit process rewards |
| **OmegaPRM** (Google DeepMind) | arXiv 2406.06592 | MCTS for efficient process supervision collection |
| **rStar-Math** (Microsoft) | arXiv 2501.04519 | MCTS + PRM; Qwen-7B from 58.8% to 90.0% on MATH |
| **MRT (Meta RL Fine-Tuning)** | arXiv 2503.07572 | Dense "progress" reward; 2-3× gain over outcome RL |

### 2.3 Critical Findings on RL vs. Distillation

| Paper | Venue | Key Finding |
|---|---|---|
| **"Does RL Really Incentivize Reasoning?"** | arXiv 2504.13837 | RL does NOT create new patterns; only reshuffles existing capabilities |
| **"SFT Memorizes, RL Generalizes"** | arXiv 2501.17161 | RL generalizes OOD; SFT memorizes. Two-stage optimal |
| **Dr. GRPO** (SAIL-SG) | arXiv 2503.20783 | GRPO has length-inflation bias; debiased version achieves 43.3% AIME |
| **Logic-RL** | arXiv 2502.14768 | 5K logic puzzles with rule-based RL → strong generalization |
| **"Cognitive Behaviors of Effective STaRs"** | arXiv 2503.01307 | Reasoning *behaviors* (verify, backtrack) matter more than correctness |

### 2.4 Structured/Implicit Reasoning

| Paper | Venue | Key Contribution |
|---|---|---|
| **Implicit CoT via KD** (Deng et al.) | NeurIPS 2023 | Reasoning in hidden states, not tokens |
| **From Explicit to Implicit CoT** | arXiv 2405.14838 | Gradual step removal → internalized reasoning |
| **Latent Reasoning (Recurrent Depth)** | arXiv 2502.05171 | Iterate recurrent block; 3.5B→50B equivalent reasoning |
| **Dualformer** (Meta) | arXiv 2410.09918 | Randomized trace training; 97.6% optimal with 45.5% fewer steps |
| **L1: Length-Controlled Reasoning** (CMU) | arXiv 2503.04697 | LCPO for short reasoning; 1.5B L1 > GPT-4o at equal lengths |
| **Concise Reasoning via RL** | arXiv 2504.05185 | Verbosity from loss minimization, not deeper thought |

### 2.5 Neurosymbolic and Structure-Aware

| Paper | Venue | Key Contribution |
|---|---|---|
| **Neurosymbolic Representations** (Dhanraj et al.) | arXiv 2502.01657 | Encode hidden states → solve in symbolic space → decode; 15.4× improvement |
| **"Underthinking of o1-Like LLMs"** | arXiv 2501.18585 | Premature thought switching; TIP penalty helps |
| **ZMath** | arXiv 2504.11919 | Adaptive difficulty grading; 2K samples beats full datasets |

---

## 3. Gap Analysis

### Gap 1: Token-Level vs. Structure-Level Distillation
**Current state:** ALL existing distillation (R1-Distill, LIMO, ThinkPO, Light-R1) operates at token level — student mimics teacher CoT via SFT/DPO.

**The gap:** No work identifies and transfers the *structural skeleton* — the dependency graph of load-bearing operations vs. filler/repetition/dead-end exploration. Papers show verbosity is an artifact [2504.05185] and models "underthink" via premature switching [2501.18585]. Much of CoT is structurally unnecessary.

### Gap 2: Probing-Based Step Importance
**Current state:** PRMs score steps by correctness but treat it as black-box. [2503.01307] identifies behavioral patterns.

**The gap:** No work uses *probing on internal representations* to identify which steps produce meaningful state changes toward the solution. If we can probe when the model genuinely "advances" vs. performs cosmetic reasoning, we can selectively distill.

### Gap 3: Game-Theoretic Verification as Reward
**Current state:** RL rewards are binary or PRM-based. [Logic-RL] shows formal verification enables strong RL with 5K examples.

**The gap:** Game theory offers formally verifiable intermediate steps (dominance elimination → BR computation → NE identification). Each step checkable. No work uses this as structured reward for distillation.

### Gap 4: The Missing Bridge Between Distillation and RL
**Current state:** Either pure distillation OR pure RL. [2504.13837] shows RL can't create new patterns; [2501.17161] shows RL generalizes.

**The gap:** No principled combination: structural distillation to introduce patterns + RL with structured rewards to generalize them.

---

## 4. Proposed Method

### Phase 1: Probing-Based Reasoning Structure Extraction

1. Run teacher model (DeepSeek-R1-Distill-Qwen-32B or QwQ-32B) on game-theoretic tasks
2. Collect hidden states at each reasoning step boundary
3. Train probing classifiers to detect:
   - **State-advancing steps**: representation meaningfully changes toward solution
   - **Verification steps**: performing validity checks
   - **Filler steps**: cosmetic restatements, dead-end exploration
4. Extract the **reasoning skeleton**: ordered subset of load-bearing steps + dependency structure

### Phase 2: Structural Distillation (SFT)

1. Convert skeletons to structured training data:
   ```
   <problem> → <skeleton: step1[advance] → step2[verify] → step3[advance]> → <answer>
   ```
2. SFT student (Qwen2.5-7B or 3B) on skeleton-annotated traces
3. Curriculum: easy games → hard games (following Light-R1 methodology)

### Phase 3: Game-Theoretic GRPO

1. Define composite reward:
   - **Correctness**: Is the final Nash/BR answer correct?
   - **Step verification**: Is each intermediate step valid? (automated game-theoretic check)
   - **Structural adherence**: Does reasoning follow skeleton patterns?
   - **Conciseness bonus**: Fewer tokens for same correctness
2. Train with GRPO (verl) + Dr. GRPO debiasing
3. Reward is fully automated (no human annotation)

---

## 5. Why This Is Novel and Publishable

### Independent Novelty Claims:

1. **First probing-based reasoning step importance scoring for distillation** — existing PRMs score externally; we probe internal state transitions
2. **First structured (graph-based) reasoning distillation** — all prior work transfers token sequences
3. **First game-theoretic verification for reasoning distillation** — richer verification than binary math correctness
4. **Bridges distillation vs. RL divide** — structural distillation introduces, RL generalizes
5. **Addresses verbosity/underthinking** by training on concise load-bearing skeletons

### Venue Fit:
- **Timeliness:** Reasoning distillation is THE hottest 2025 topic
- **Principled methodology:** Bridges interpretability (probing) + training (GRPO)
- **Strong baselines:** DeepSeek-R1-Distill, LIMO, ThinkPO, Light-R1
- **Game theory angle:** Differentiates from flood of math-only papers

---

## 6. Experimental Design

### Models
- **Teacher:** DeepSeek-R1-Distill-Qwen-32B (fits on 4×H20 in FP16)
- **Students:** Qwen2.5-7B-Base, Qwen2.5-3B-Base
- **Baselines:** (1) Standard CoT SFT, (2) GRPO-only, (3) ThinkPO, (4) LIMO-style minimal SFT

### Datasets
- Game-theoretic tasks: 10K normal-form games (auto-generated + verified)
- Sequential games: 5K backward induction tasks
- Standard: MATH, GSM8K, AIME2024 (transfer evaluation)

### Key Experiments

| Experiment | Tests |
|---|---|
| **Probing validation** | Can probes predict load-bearing steps? (ablation check) |
| **Skeleton vs. Full-CoT SFT** | Same model, same steps — which distillation is better? |
| **GRPO reward comparison** | Binary vs. PRM vs. game-theoretic step verification |
| **End-to-end pipeline** | Skeleton distill + GRPO vs. all baselines |
| **Transfer to math** | Does game reasoning training help MATH/AIME? |

### Metrics
- Accuracy (pass@1, pass@8)
- Token efficiency (correct answers / tokens generated)
- OOD generalization (train on 2-player, test on 3-player)
- Probing quality (can student's internal states be probed for solution components?)
- Structural faithfulness (does student follow skeleton patterns?)

---

## 7. Compute Budget (8×H20)

| Phase | Duration | Notes |
|---|---|---|
| Teacher inference (generate reasoning traces, 10K) | ~50 hours | 32B model on 4 GPUs, parallelized |
| Probing experiments | 1 week | Lightweight, 1-2 GPUs |
| Skeleton SFT training | 1 week | 8×H20 |
| GRPO training | 2-3 weeks | 8×H20, verl framework |
| Evaluation + ablations | 1-2 weeks | 4-8 GPUs via vLLM |
| **Total** | **8-10 weeks** | |

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Probing can't reliably identify load-bearing steps | Medium | Multiple probing methods; validate via causal ablation. Partial identification still publishable |
| Skeleton distillation underperforms full-CoT | Medium | Interesting negative result. Hybrid: skeleton + key expansions |
| Game tasks don't transfer to math | Medium-High | Include math in curriculum. If weak, focus on game domain (underserved) |
| GRPO instability at 7B | Low | Well-established [Dr. GRPO, Open-Reasoner-Zero]. Start from SFT checkpoint |
| Concurrent work scoops idea | Medium | Probing + game-theoretic verification combination is unique. Move fast |

---

## 9. Key Framing Insights

1. **"RL does not create new reasoning; distillation does"** → justifies distillation as primary
2. **"Reasoning behaviors > answer correctness"** → justifies distilling structure over content
3. **"SFT memorizes, RL generalizes"** → justifies two-phase approach
4. **"Verbosity is an artifact, not depth"** → justifies skeleton compression
5. **"Formal verification enables strong RL with minimal data"** → justifies game-theoretic domain
6. **"Strategic sample selection > scale"** → justifies probing-based step selection

---

## 10. Relationship to Your Current Work

This idea directly extends your existing infrastructure:
- **Phase 1 probing** → used here to identify load-bearing reasoning steps
- **GameSolve-Bench** → provides the formally verifiable tasks
- **verl GRPO** → the RL training framework
- **Your finding that SFT > GRPO** → motivates the two-phase approach (distill first, RL second)
- **Your OOD evaluation** → transfer testing methodology

The key pivot from your current proposal: instead of auxiliary probing loss during training (which isn't working), use probing as a *diagnostic tool* to identify what to distill, then distill + RL.
