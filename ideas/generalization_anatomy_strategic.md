# Idea 7 — The Generalization Anatomy of Strategic Reasoning

> **One-liner:** Game theory is the only reasoning domain where the *axes* of
> out-of-distribution generalization are formally definable. Use it to turn the
> vague "SFT memorizes, RL generalizes" folklore into a rigorous, mechanistic,
> *predictive* science — and ship a diagnostic that forecasts OOD collapse from
> a model's internal geometry before you ever run the OOD eval.

**Status:** New (June 2026). Grounded in this project's strongest robust finding.
**Type:** Analysis / "science of LLMs" paper (not a new-method paper).
**Venue fit:** ICLR 2027 / COLM 2026. Analysis papers with a crisp, falsifiable
claim and a released benchmark do very well at these.
**Risk:** Low. We already have most of the data and the infrastructure. The
finding is *already true* in our logs; the work is making it rigorous and
explaining the mechanism.

---

## 1. Why this idea, now

Two method directions in this project have been **refuted** by clean experiments:

- **Probing-guided auxiliary loss** (Phase 2): depth-dependent distractor, the
  apparent +1.9pp was seed noise (training.md §16, 15 runs).
- **GT-PRM** (process reward): formally-verifiable ≠ outcome-aligned; process
  reward is hackable and near-orthogonal to answer quality (gate corr 0.006).

But across *every* one of those experiments, one signal kept reappearing and was
never the thing we were testing — so it was never confounded by our manipulation:

> **SFT wins in-distribution but collapses out-of-distribution and on
> interactive games; RL transfers safely.**
>
> - In-distribution: SFT 0.83 overall >> verl-GRPO 0.77 (`research_log`).
> - OOD GameSolve: SFT 0.71 vs GRPO 0.66 — gap *shrinks* (SFT −17% ID but only −7% OOD vs GRPO).
> - **TextArena interactive: SFT 38% vs GRPO 56% vs Base 52%** — SFT *regresses
>   below base*, GRPO does not (`exp_benchmark_eval`).
> - Pig game: SFT 77% format-completion vs base/grpo 1–3% — SFT overfits a CoT
>   answer *format* that does not survive a new interaction protocol.

This is the textbook "SFT memorizes, RL generalizes" pattern (Chu et al., ICML
2025). The community-level result is established. **What nobody has** is a domain
where the OOD axes are *formal* rather than vibes. "Harder math," "different
visual texture," "new hop count" are all informal. **Game theory gives you a
lattice of provably-distinct generalization axes**, because a game is fully
specified by `(players, action sets, payoff matrix, information structure,
solution concept)`. You can move along exactly one axis at a time and *prove*
the move is OOD. That is the contribution prior work cannot make.

## 2. The thesis

> Strategic reasoning is the natural testbed for *anatomizing* generalization
> because its OOD directions are formally orthogonal. Using it, we (a) build a
> **typed generalization benchmark**, (b) measure SFT vs RL vs base along each
> typed axis, (c) **mechanistically explain** the SFT-collapse via
> representation geometry and solution-step coverage, and (d) deliver a
> **cheap internal diagnostic that predicts per-example OOD collapse** without
> running the OOD task.

## 3. The formal generalization lattice (the benchmark)

We already have `gamesolve_bench.jsonl`, `gamesolve_ood_bench.jsonl`, and the
generators `gamesolve_gen.py` / `generate_ood_bench.py`. We re-cut them into
**typed OOD axes**, each a single-variable move from the training distribution:

| Axis | Train dist | OOD move | Why it isolates a capability |
|---|---|---|---|
| **Size** | 2×2, 3×3 | 4×4, 5×5, n-player | Same solution *concept*, larger search — tests procedure vs lookup |
| **Concept** | NE, best-response | iterated dominance, mixed NE, Pareto, correlated eq | Tests whether the *algorithm* generalizes or only the seen concept |
| **Payoff structure** | random / coordination | zero-sum, anti-coordination, Prisoner's-Dilemma-class | Tests reliance on payoff-surface statistics |
| **Information** | simultaneous, complete info | sequential (subgame perfection), incomplete info | The TextArena gap lives here |
| **Surface form** | matrix as table | matrix as prose / transposed / relabeled actions | Tests format memorization directly (pig-game finding) |
| **Compositional** | atomic concepts seen separately | "do dominance elimination *then* NE on the residual" | Chu/KG-reward-model style compositional OOD |

Each cell is formally verifiable (we have the sound solver from GT-PRM). This
**typed lattice is itself a releasable artifact** — "GameGen-OOD: a controlled
generalization benchmark for strategic reasoning."

## 4. Experiments

All checkpoints already exist (base, SFT, verl-GRPO, GRPO+probe) plus the
Nash-GRPO runs landing now. The verl + vLLM eval pipeline already scores them.

**E1 — Generalization heatmap.** Method × axis × {accuracy, format-completion}.
Hypotheses, pre-registered:
- H1: SFT ≈ best on *surface-form-identical* ID, worst on *Surface form* and
  *Information* OOD (it memorized format).
- H2: RL's relative advantage is *monotone in OOD distance along the Concept and
  Compositional axes*, ~flat on Size.
- H3: SFT's collapse is *format-driven* — when scored on extracted answer only
  (ignoring format), part of the gap closes; the residual is the *reasoning* gap.
  (This cleanly separates "memorized format" from "memorized solutions.")

**E2 — Mechanism: representation geometry.** We have the full probing stack
(`extract_representations_multi.py`, `train_probes.py`). For each method, per
layer:
- *Effective dimensionality / participation ratio* of hidden states on OOD games.
  Prediction (from "RL concentrates into hubs", RL-Squeezes-SFT-Expands 2509.21128):
  RL representations stay lower-rank and more axis-aligned to the game-concept
  probe directions; SFT representations *rotate off* the concept axes under OOD.
- *Concept-probe transfer*: train a linear probe for "iterated dominance applies
  here" on ID, test on OOD. SFT's probe accuracy should fall faster — its
  features are distributional, not procedural.

**E3 — Mechanism: solution-step coverage.** Using the sound GT verifier, measure
the *fraction of required solution steps the trace actually executes* on OOD.
Hypothesis: SFT emits the *shape* of a derivation (right length, right tokens)
while skipping the steps that the new game size actually requires; RL executes
fewer-but-load-bearing steps (ties to the "RL = hubs" mechanism). This is a
*behavioral* confirmation of the geometric story.

**E4 — The diagnostic (the deliverable that makes it a paper, not a report).**
Train a tiny logistic model on *ID-only internal features* (participation ratio +
concept-probe margin + step-coverage) to **predict whether a given (model, game)
will be solved OOD**, without ever running OOD. If AUC is high and it transfers
across methods, you have a *predictive theory of generalization* — reviewers love
"we can forecast the failure before it happens." Practical pitch: a cheap
pre-deployment OOD-risk score for strategic agents.

## 5. Why it's publishable even if a hypothesis is wrong

Every outcome is a result:
- If H2 holds → mechanistic confirmation + benchmark + diagnostic. Strong, accept-shaped.
- If SFT's collapse is *purely* format (E1-H3 residual ≈ 0) → that is a sharp,
  surprising, citable claim ("SFT's OOD failure in reasoning is format
  memorization, not reasoning loss") that *reframes* the Chu narrative.
- If the diagnostic transfers across methods → standalone contribution.

No single hypothesis is load-bearing. This is the hallmark of a low-risk analysis paper.

## 6. Cost & timeline

- No new training required for the core (checkpoints exist). ~2 days to re-cut the
  typed benchmark, ~3 days eval sweep on 8×H20, ~4 days mechanism (probes/geometry),
  ~3 days diagnostic + writing. **~2 weeks** to a full draft.
- Optional strengthener: add 1–2 more base model families (Qwen sizes already in
  `Qwen/`) to show the effect is not model-specific.

## 7. Relationship to existing ideas / assets

- Consumes refuted-work assets productively: probing stack → E2; GT verifier → E3.
- Orthogonal to Nash-GRPO (#5, running): Nash-GRPO produces *another method* whose
  generalization this benchmark then *characterizes*. They compose.
- Subsumes the OOD work in `compare_ood.py` / `eval_ood.py` and the
  GTBench/TextArena harness as the "Information axis."

## 8. Open risks

- Need the typed axes to be genuinely single-variable (careful generator design).
- The mechanism (E2/E3) must *agree* with the behavior (E1) or the story
  fragments — budget time to reconcile; a clean agreement is the paper's spine.

## References (grounding)
- Chu et al., "SFT Memorizes, RL Generalizes," ICML 2025 — https://proceedings.mlr.press/v267/chu25c.html
- "RL Squeezes, SFT Expands" — https://arxiv.org/pdf/2509.21128 (mechanism: RL concentrates functionality into hub steps)
- "Rethinking Generalization in Reasoning SFT" (2026) — https://arxiv.org/abs/2604.06628v1 (generalization is *conditional*, not absolute)
- "Knowledge Graphs are Implicit Reward Models" — https://arxiv.org/pdf/2601.15160 (axiomatic primitives → compositional OOD)
- "Scaling Reasoning Hop Exposes Weaknesses" — https://arxiv.org/pdf/2601.21214 (informal OOD axis prior art we improve on)
