# StrategyGym: Curriculum-Driven Training for Compositional Generalization in LLM Strategic Reasoning

> Deep Research Report | June 2025

---

## 1. Executive Summary

**Core Idea:** Decompose strategic reasoning into atomic skills (dominance detection, payoff calculation, best-response computation, NE identification), design principled curricula (skill-first, size-first, interleaved, self-paced), and demonstrate that curriculum-based GRPO training yields dramatically better generalization to unseen game types/sizes/formats than flat training or SFT.

**Why Publishable:** First curriculum learning framework for game-theoretic reasoning; first systematic transfer evaluation across a hierarchy of game types; builds directly on existing GameSolve-Bench + verl GRPO infrastructure.

**Target Venue:** ICLR 2026 or NeurIPS 2025

---

## 2. Literature Review

### 2.1 LLM Game-Theoretic Reasoning Benchmarks (2023-2025)

| Paper | Venue | Key Contribution |
|---|---|---|
| **GTBench** (Jinyu Zhang et al.) | NeurIPS 2024 D&B | 10 game environments; LLMs near-random on multi-step planning games |
| **TMGBench** (Haoran Sun et al.) | Preprint 2024 | 144 game types from Gamut; story-based NL descriptions |
| **Can LLMs Reason About Game Theory?** (Yiwei Chen et al.) | Preprint 2024 | Systematic failures in backward induction, iterated elimination |
| **ALYMPICS** (Shaoguang Mao et al., Microsoft) | Preprint 2024 | Auction/bargaining environments; GPT-4 approximates NE in simple settings |
| **GameBench** (Costarelli et al.) | Preprint 2024 | Board/card games; planning depth is bottleneck |
| **Evaluate LLMs on Game-Theoretic Reasoning** (Liang Xu et al.) | AAAI 2025 Workshop | Multi-agent interaction; LLMs are exploitable |

### 2.2 RL/Training for Strategic Reasoning

| Paper | Venue | Key Contribution |
|---|---|---|
| **CoRY** | NeurIPS 2024 Workshop | Multi-agent self-play RL for game-playing |
| **GRPO** (DeepSeek-AI) | DeepSeek-R1 report 2025 | Group Relative Policy Optimization for reasoning |
| **SPIN** (Zixiang Chen et al.) | ICML 2024 | Self-play fine-tuning converging to Nash |
| **Strategic Reasoning via Self-Improvement** | Various 2024 | RL from game outcomes; minimal cross-game transfer |

### 2.3 Transfer and Generalization in Reasoning

| Paper | Venue | Key Finding |
|---|---|---|
| **On the Generalization of Reasoning Abilities in LLMs** | 2024 preprints | Math RL partially transfers to science, NOT to strategic tasks |
| **CurricularLM** | 2024 | Easy-to-hard ordering for math improves LLM reasoning |
| **Skill-Mix** | 2024 | Tests compositional skill assembly |
| **Self-Evolved Curriculum Learning** | 2024 | Automated difficulty scheduling |
| **Beyond A*: Search Dynamics Bootstrapping** (Lehnert et al., Meta) | NeurIPS 2024 | Training on search traces improves planning generalization |
| **SFT Memorizes, RL Generalizes** | arXiv 2501.17161 | RL generalizes OOD; SFT memorizes |

### 2.4 Compositional Generalization

| Paper | Venue | Relevance |
|---|---|---|
| **SCAN** (Lake & Baroni) | ICML 2018 | Compositional generalization benchmark tradition |
| **COGS** (Kim & Linzen) | EMNLP 2020 | Structural compositional generalization |
| Relevance: Can an LLM that learns 2x2 dominance compose it to handle 5x5 games? |

---

## 3. Gap Analysis

### Gap 1: No Systematic Study of Transfer Across Game Types
- GTBench tests 10 games but never trains on one and tests on another
- TMGBench tests 144 types but only evaluates zero-shot
- **Nobody has asked: If you train on simple matrix games, does it generalize to larger matrices? Different structures? Sequential games?**

### Gap 2: No Curriculum Design for Game-Theoretic Reasoning
- Curriculum learning is well-studied for math but nobody has:
  - Defined a difficulty hierarchy across game types (not just matrix size)
  - Tested 2x2 → 3x3 → 4x4 → 5x5 ordering effects
  - Tested whether pure-strategy games first helps learn mixed-strategy
  - Tested whether dominance-solvable games should precede general NE computation

### Gap 3: Limited Understanding of What Generalizes
- Your results show: SFT dramatically outperforms GRPO on ID (94.5% vs 77.2%) but the OOD gap is smaller (73.4% vs 66.1%)
- Nobody has systematically analyzed WHY some training approaches generalize better

### Gap 4: No Compositional Analysis of Strategic Reasoning
- Nobody has decomposed strategic reasoning into atomic skills and tested compositional assembly
- Can "identify dominant strategy" + "compute expected payoff" compose to solve mixed-strategy NE?

### Gap 5: Disconnect Between Benchmarks and Training
- GTBench/TMGBench are evaluation-only; nobody uses them for curriculum-based training

---

## 4. Proposed Method

### 4.1 Skill Decomposition (6 Atomic Levels)

| Level | Skill | Example Task | Complexity |
|---|---|---|---|
| L1 | Payoff reading | "What is P1's payoff at (Row2, Col1)?" | Trivial |
| L2 | Dominance detection | "Does P1 have a strictly dominated strategy?" | Easy |
| L3 | Best response (pure) | "If P2 plays Col2, what is P1's BR?" | Easy |
| L4 | Best response (mixed) | "Given P2's mixed strategy (0.3, 0.7), P1's BR?" | Medium |
| L5 | Pure NE identification | "Find all pure-strategy Nash Equilibria" | Hard |
| L6 | Mixed NE computation | "Find the mixed-strategy NE" | Very Hard |

Generate 5,000 samples per level (30,000 total) across 2x2, 3x3, 4x4 matrices.

### 4.2 Curriculum Strategies (5 Variants)

| Curriculum | Description | Hypothesis |
|---|---|---|
| **Flat** | Uniform random (baseline) | No structure |
| **Skill-first** | L1→L2→...→L6 sequentially | Builds atomic skills before composition |
| **Size-first** | 2x2(all) → 3x3(all) → 4x4(all) | Complexity scaling |
| **Interleaved** | L1-2x2, L1-3x3, L2-2x2, ... | Broad-first exposure |
| **Self-paced** | Select lowest-reward batches | Adaptive difficulty |

### 4.3 Training Recipe

- Model: Qwen2.5-3B-Instruct (primary), 7B (scale-up)
- Algorithm: GRPO via verl framework, 8×H20
- Per curriculum: 500 GRPO steps, checkpoint every 50
- Reward: binary correctness (game solutions are formally verifiable)

### 4.4 Multi-Axis OOD Evaluation

| OOD Axis | Training Range | Test Range |
|---|---|---|
| **Size** | 2x2, 3x3, 4x4 | 5x5, 6x6, 8x8 |
| **Asymmetry** | Square matrices | 2x5, 5x2, 3x6 |
| **Game class** | General-sum | Zero-sum, Cooperative |
| **Format** | Abstract notation | JSON, LaTeX, story |
| **Equilibrium type** | Pure + simple mixed | Multiple NE, no pure NE |
| **Sequential** | Normal-form | Extensive-form (2-stage) |
| **Cross-benchmark** | GameSolve-Bench | TMGBench (144 types), GTBench |

### 4.5 Probing Analysis (Mechanistic Explanation)

For best-generalizing vs worst-generalizing models:
- Extract representations at critical layers (α = 0.25, 0.5, 0.67)
- Train probes on atomic skills at each layer
- Measure: Does curriculum produce cleaner, more modular skill representations?
- Hypothesis: Good generalization ↔ composable skill subspaces

---

## 5. Why This Is Publishable at Top Venues

### Novelty Claims (each independently novel):

1. **First curriculum learning framework for game-theoretic reasoning in LLMs**
2. **First systematic transfer evaluation across a hierarchy of game types**
3. **Compositional reasoning skills decomposition for strategic reasoning**
4. **Mechanistic explanation via probing** — existing Phase 1 infrastructure enables unique analysis
5. **Practical training recipe** that improves any LLM's strategic reasoning

### Venue Fit:
- **ICML/NeurIPS:** Core ML (curriculum + generalization) + experiments + interpretability
- **ICLR:** Understanding and improving reasoning, representation analysis
- **Timely:** Field moving from "LLMs fail at game theory" (2024) to "how to fix it" (2025)

---

## 6. Experimental Plan & Compute

| Experiment | Estimated Time (8×H20) |
|---|---|
| Data generation (30K samples, all levels) | ~2 hours |
| Training (5 curricula × Qwen-3B, 500 steps each) | ~50 hours |
| OOD evaluation (5 models × 7 axes) | ~10 hours |
| Probing analysis (5 models × 3 layers) | ~5 hours |
| 7B scaling (best 2 curricula) | ~16 hours |
| Ablations | ~20 hours |
| **Total** | **~100 hours (~4 days)** |

---

## 7. Expected Results

1. **Curriculum ordering matters significantly:** Skill-first or interleaved curriculum outperforms flat training by 10-20% on OOD
2. **Size generalization is hardest:** 2x2-4x4 → 6x6+ remains challenging, revealing fundamental scaling limitation
3. **Skill composition is partial:** Models learn L1-L4 well but composing for L5-L6 OOD is challenging
4. **Probing reveals modularity:** Well-generalizing models have more separable skill representations
5. **GRPO + curriculum closes the SFT gap:** Your results show SFT >> GRPO; with curriculum, GRPO may close this gap

### Paper Narrative:
"Strategic reasoning is not monolithic but a composition of atomic skills. Curriculum-based RL achieves SOTA generalization across game types and sizes, with mechanistic probing evidence that curriculum training produces modular internal representations."

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Curriculum doesn't help over flat | Medium | Include self-paced (adapts); study WHY for insight paper |
| Size generalization fundamentally limited | High | Focus on format/type transfer; 8x8 as "ceiling" analysis |
| verl GRPO instability at curriculum switches | Medium | Warm-start from previous stage; KL penalty |
| SFT+curriculum dominates GRPO+curriculum | Medium | Reframe as "best recipe" comparison |
| "Just matrix games" criticism | Medium | Include sequential game extension; connect to negotiation |

---

## 9. Your Unique Advantages

1. **GameSolve-Bench already built** — data generation ready, just extend to skill levels
2. **GRPO pipeline working** (verl, 8×H20) — 77.2% ID demonstrated
3. **Probing infrastructure** — immediate mechanistic analysis
4. **OOD evaluation pipeline** — 750-sample OOD with multiple axes
5. **Preliminary results showing the gap** — Base 43% → GRPO 66% → SFT 73% OOD. "How to close with better RL?" is the question

---

## 10. Quick Validation Experiment

**Run first:** Train GRPO on only 2x2 games, evaluate on 3x3+. If non-trivial transfer exists → curriculum is viable. If not → analysis of why is still publishable.
