# Research Ideas — Top Conference (ICML / ICLR / NeurIPS)

> Generated: June 2025 | Updated: June 2026 (added Ideas 7–9; annotated experimental outcomes)
> Context: probing + auxiliary-loss proposal not meeting expectations. **Ideas 7–9
> are grounded in this project's *experimental outcomes*** — including the
> refutations of Ideas 1 & 2 — rather than in prior literature alone.

> **Experimental status of the original six (as of June 2026):**
> - **Idea 1 (GT-PRM): REFUTED** as a *training* method — formally-verifiable ≠
>   outcome-aligned; scalar process reward is hackable & ~orthogonal to answer
>   quality (`gtprm/RESULTS.md`). **→ salvaged by new Idea 8** (verifier as a
>   *test-time selector* instead of a reward).
> - **Idea 2 (RepE/PGAS / probing): REFUTED** as a training signal — probing
>   aux-loss is a depth-dependent distractor; +1.9pp was seed noise (training.md §16).
>   The probing *stack* is reused as instrumentation in Ideas 7 & 9.
> - **Idea 5 (Nash-GRPO): IN PROGRESS** — multi-seed GRPO-vs-Nash runs executing now.
> - Ideas 3, 4, 6: not yet attempted.

---

## Quick Ranking (Publishability × Feasibility × Novelty)

| # | Idea | Novelty | Feasibility | Risk | Status | Venue | Time |
|---|---|---|---|---|---|---|---|
| **8** | **Verifier-as-Selector (test-time)** | ★★★★☆ | ★★★★★ | Low-Med | **NEW — top pick** | NeurIPS 2026 | 1.5-2 wk |
| **7** | **Generalization Anatomy of Strategic Reasoning** | ★★★★★ | ★★★★★ | Low | **NEW** | ICLR 2027 / COLM 2026 | ~2 wk |
| **9** | **Strategic-Depth (level-k) Dynamics** | ★★★★☆ | ★★★★★ | Low-Med | **NEW** | COLM 2026 | ~1.5 wk |
| 5 | **Nash-GRPO (Game-Theoretic RLHF)** | ★★★★☆ | ★★★★☆ | Medium | running | NeurIPS 2026 | 6-8 wk |
| 3 | **Curriculum Strategic Reasoning** | ★★★★☆ | ★★★★★ | Medium | open | ICLR 2027 | 4 d + writing |
| 6 | **LLM Self-Play + Emergent Communication** | ★★★★★ | ★★★☆☆ | High | open | ICML 2026 | 10-12 wk |
| 4 | **Structured Reasoning Distillation** | ★★★★★ | ★★★☆☆ | Medium | open | ICML 2026 | 8-10 wk |
| 1 | Game-Theoretic Process Reward Models | ★★★★★ | ★★★★★ | — | **refuted** (→8) | — | — |
| 2 | Representation Engineering for Reasoning | ★★★★★ | ★★★★★ | — | **refuted** | — | — |

> **Recommendation:** the three new ideas are the highest-EV directions left,
> because they (a) build on findings that are *already true* in our logs, (b) need
> little-to-no new training, and (c) convert sunk cost (refuted GT-PRM, probing
> stack) into assets. **Idea 8 is the single best next bet** — cheapest, and it
> flips a refutation into a positive result. See "New Ideas (June 2026)" below.

---

## Idea 1: Game-Theoretic Process Reward Models (GT-PRM)

**File:** [game_theoretic_prm_research_report.md](game_theoretic_prm_research_report.md)

**One-liner:** Use formally verifiable game-theoretic solution steps (dominance elimination → BR → NE) as zero-cost dense process rewards for GRPO training.

**Why #1 Recommendation:**
- PRMs are the hottest topic (2024-2025); first application to game theory
- Zero annotation cost (game solutions are algorithmically verifiable)
- Directly extends your verl GRPO + GameSolve-Bench infrastructure
- Clean story: PRM for math (expensive, human-labeled) vs GT-PRM (free, algorithmic)
- Low risk: even partial results are publishable

**Key Differentiation from Current Work:**
- Current: auxiliary loss on frozen representations (not working)
- GT-PRM: dense step-level reward during RL training (directly shapes behavior)

---

## Idea 2: Representation Engineering for Reasoning (PGAS)

**File:** [representation_engineering_reasoning.md](representation_engineering_reasoning.md)

**One-liner:** Use probing-derived contrastive directions to steer LLM activations toward correct game-theoretic reasoning at inference time — zero training cost.

**Why Strong:**
- Bridges RepE (safety-focused) with reasoning (unexplored)
- Your Phase 1 probing results become BOTH diagnostic tool AND intervention
- Training-free: just forward passes + vector arithmetic
- Compelling narrative: "probing finds the problem, steering fixes it"
- Very fast to execute (no training needed)

**Key Differentiation from Current Work:**
- Current: auxiliary training loss (expensive, not working)
- PGAS: inference-time intervention (free, immediate)

---

## Idea 3: StrategyGym — Curriculum-Driven Compositional Generalization

**File:** [curriculum_strategic_reasoning.md](curriculum_strategic_reasoning.md)

**One-liner:** Decompose strategic reasoning into atomic skills, design principled curricula, and show curriculum GRPO dramatically outperforms flat training for OOD generalization.

**Why Strong:**
- First curriculum framework for game-theoretic reasoning
- Directly addresses your observed SFT>>GRPO gap (curriculum may close it)
- Very fast execution (4 days compute)
- Builds on ALL existing infrastructure

**Key Differentiation from Current Work:**
- Current: train on all tasks simultaneously
- Curriculum: structured skill progression → better generalization

---

## Idea 4: StructReason — Reasoning Skeleton Distillation

**File:** [structured_reasoning_distillation.md](structured_reasoning_distillation.md)

**One-liner:** Distill large reasoning model's structural skeleton (not token sequence) identified via probing, then generalize via game-theoretic GRPO.

**Why Strong:**
- Addresses fundamental limitation: token-level distillation transfers noise
- Supported by multiple 2025 findings (RL doesn't create; verbosity is artifact)
- Combines your probing expertise with distillation (unique angle)
- Game theory as ideal testbed (formally verifiable steps)

**Risks:**
- Requires large teacher model inference (32B on 4 GPUs)
- Longer timeline (8-10 weeks)

---

## Idea 5: Nash-GRPO — Game-Theoretic Foundation for GRPO

**File:** [game_theoretic_rlhf_research_report.md](game_theoretic_rlhf_research_report.md)

**One-liner:** Formalize GRPO as a tournament game; replace mean-baseline advantages with Nash equilibrium advantages from pairwise preference matrices, handling intransitive preferences correctly.

**Why Strong:**
- Connects two hottest topics: GRPO (DeepSeek-R1) + Nash alignment (ICML 2024 oral)
- Theory + algorithm + experiments
- Minimal code change (solve KxK LP per group in GRPO)
- Applicable beyond game theory (general RLHF improvement)

**Risks:**
- More theoretical; needs strong math/game theory story
- Must demonstrate clear empirical gains over standard GRPO

---

## Idea 6: LLM Self-Play with Emergent Communication

**File:** [llm_selfplay_emergent_communication_research.md](llm_selfplay_emergent_communication_research.md)

**One-liner:** Train LLMs through communicative self-play (cheap talk before actions) in strategic games; analyze emergent signaling protocols against game-theoretic predictions (Crawford-Sobel, correlated equilibrium).

**Why Strong:**
- Very novel: merges emergent communication with LLM game-theoretic training
- Theoretically grounded (Aumann's correlated equilibrium, Crawford-Sobel)
- Recent evidence (April 2026 paper) that LLMs compute Nash internally but suppress it
- Beautiful interdisciplinary story

**Risks:**
- Most complex to implement (multi-agent training)
- Communication may degenerate to trivial announcements
- Longer timeline

---

## New Ideas (June 2026) — grounded in experimental outcomes

These three were generated *after* the GT-PRM and probing refutations, and after
the SFT-vs-RL generalization findings. Each is deliberately built so that **every
outcome is publishable** and **little or no new training is required** (they ride
on existing checkpoints + the verl/vLLM eval stack + the Nash-GRPO runs in flight).

### Idea 8: Verifier-as-Selector — "Sound verifiers make good selectors, not good rewards" ⭐ top pick

**File:** [verifier_as_selector_testtime.md](verifier_as_selector_testtime.md)

**One-liner:** Our GT verifier is *sound* (gold process 0.84) but failed as a
dense RL reward (hackable, orthogonal to outcome). Resolution: reward-hacking is a
phenomenon of *optimization pressure*; **selection applies no such pressure**, so
a sound process verifier that is useless for *shaping* is excellent for
*test-time selection* (best-of-N, step-level beam search, self-correction).

**Why top pick:** cheapest (training-free), and it *flips the refuted GT-PRM into a
positive result*. The controlled A/B — same verifier hurts as reward, helps as
selector — is a clean, reusable insight for the whole PRM literature.

### Idea 7: Generalization Anatomy of Strategic Reasoning

**File:** [generalization_anatomy_strategic.md](generalization_anatomy_strategic.md)

**One-liner:** Game theory is the only reasoning domain whose OOD axes are
*formally* definable (size / concept / payoff / information / surface / composition).
Use it to make "SFT memorizes, RL generalizes" rigorous and *mechanistic*, and ship
a diagnostic that predicts per-example OOD collapse from internal geometry alone.

**Why strong:** the finding is already in our logs (SFT 38% vs GRPO 56% on
TextArena); this turns it into a benchmark + mechanism + predictive diagnostic.
Reuses the (otherwise-refuted) probing stack as instrumentation. Near-zero new training.

### Idea 9: Strategic-Depth (level-k) Dynamics of Post-Training

**File:** [strategic_depth_levelk_dynamics.md](strategic_depth_levelk_dynamics.md)

**One-liner:** Use behavioral game theory's level-k / cognitive-hierarchy
instruments to measure *strategic depth* as a new evaluation axis: does RL raise
an LLM's k while SFT freezes/lowers it, and does depth (not accuracy) predict
transfer to interactive multi-agent games?

**Why strong:** reframes the SFT-below-base interactive collapse as a *depth*
phenomenon; inference-only; provides a theory-aligned metric that Nash-GRPO (#5) is
most likely to move — a free strengthener for the running experiment.

**Two-paper arc:** Ideas 7 + 9 together form a "science of strategic post-training"
story (generalization anatomy + depth anatomy); Idea 8 is the standalone method win.

---

## Recommended Strategy (revised June 2026, post-refutations)

> The original strategy below recommended Ideas 1 & 2 — both since **refuted** as
> training methods. Revised guidance reflects what survived contact with experiments.

### If you want ONE paper fast (and high-confidence):
→ **Idea 8 (Verifier-as-Selector)** — training-free, ~1.5-2 wk, converts the
refuted GT-PRM into a positive result. Lowest risk on the board.

### If you want the strongest single paper:
→ **Idea 7 (Generalization Anatomy)** — releasable benchmark + mechanism +
predictive diagnostic, built on a finding that's already true in our logs.

### If you want to maximize output (2 papers, shared infra):
→ **Idea 7 + Idea 9** — a "science of strategic post-training" arc (generalization
anatomy + strategic-depth anatomy), both inference-only on existing checkpoints,
can run in parallel. Add **Idea 8** as a near-free third (different stack).

### If you want the most ambitious / highest-ceiling project:
→ **Idea 6 (Self-Play + Emergent Communication)** or **Idea 4 (StructReason)** —
untested, longer timelines, higher novelty ceiling.

### Already committed:
→ **Idea 5 (Nash-GRPO)** is running; Ideas 9 (depth) and 7 (generalization) each
provide a *free extra metric* to strengthen its eventual paper.

<details><summary>Original (June 2025) strategy — superseded</summary>

- ONE paper fast: Idea 2 (PGAS) or Idea 3 (Curriculum)
- Strongest single: Idea 1 (GT-PRM)
- Max output: Idea 1 + Idea 2
- Most ambitious: Idea 4 (StructReason)

</details>

---

## Synergies Between Ideas

```
Idea 8 (Verifier-Select) ←→ Idea 1 (GT-PRM): same verifier, reward→selector (refutation→positive)
Idea 8 (Verifier-Select) ←→ Idea 7 (Gen-Anatomy): test-time selection as an OOD-recovery lever
Idea 7 (Gen-Anatomy) ←→ Idea 9 (Depth): generalization anatomy + strategic-depth anatomy = 1 arc
Idea 7 (Gen-Anatomy) ←→ Idea 2 (probing): probing stack reused as mechanism instrumentation
Idea 9 (Depth) ←→ Idea 5 (Nash-GRPO): depth is the metric Nash-GRPO should most move
Idea 8/9 ←→ Idea 5 (Nash-GRPO): selection + depth metric layer on the running policy
Idea 3 (Curriculum) ←→ Idea 4 (StructReason): curriculum for skeleton distillation
Idea 6 (Self-Play) ←→ Idea 5 (Nash-GRPO): multi-agent Nash optimization
```

All ideas share: GameSolve-Bench, verl GRPO, 8×H20, probing stack, **the sound GT
verifier** (Idea 8), and the existing base/SFT/GRPO/probe/Nash checkpoints.

**Meta-pattern:** the new ideas (7–9) deliberately *recycle refuted-method assets*
— GT-PRM's verifier becomes a selector (8), the probing stack becomes mechanism
instrumentation (7), and the SFT-vs-RL gap becomes the object of study (7, 9).
The refutations were not dead ends; they re-pointed the assets at better questions.
