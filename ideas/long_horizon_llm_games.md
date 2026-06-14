# Idea 10 — Long-Horizon LLM Games: Research Problems & Ideas

> **Motivation:** GameSolve-Bench is *single-shot* (compute NE/BR once). Yet the
> project's most robust finding is that the SFT↔RL gap *widens on interactive,
> sequential games* (TextArena: SFT 38% < Base 52% < GRPO 56%). That gap lives in
> the **long-horizon** regime — where payoffs are delayed, planning depth matters,
> and equilibria multiply. Game tasks are the ideal long-horizon testbed because,
> unlike open-ended agent tasks, the *optimal long-horizon policy is formally
> computable* (subgame-perfect equilibrium, folk-theorem regions), so we can
> *grade* long-horizon reasoning instead of guessing.

**Status:** New (June 2026). Extends Ideas 7 (generalization) & 9 (depth) into time.
**Type:** Benchmark + analysis, with an optional method sub-project.
**Venue fit:** NeurIPS 2026 (benchmark track) / ICLR 2027.
**Risk:** Low–medium. Reuses `gamesolve_gen.py`, the verl GRPO pipeline, and the
sound GT solver (extended to extensive-form / repeated games).

---

## 1. The five research problems (why each is worth studying)

| # | Problem | Why it's hard & open for LLMs | Formal ground truth |
|---|---|---|---|
| **P1** | **Temporal credit assignment** | A sacrifice now pays off 10 turns later (reputation, traps). Outcome-only GRPO gets one scalar at episode end — does the gradient reach the load-bearing early move, or does the model stay myopic? | Per-turn regret vs SPE path; distance to subgame-perfect action |
| **P2** | **Planning / backward-induction depth** | Centipede, finite bargaining, finitely-repeated PD all require backward induction. Humans (and LLMs?) systematically deviate. How deep does the model induct? | SPE solvable by backward induction; depth = #correct induction steps |
| **P3** | **Equilibrium selection under repetition** | Folk theorem: repetition admits a *continuum* of equilibria (defect-always … full cooperation). Which one does RL self-play select, and can we steer it? | Folk-theorem feasible/individually-rational region is computable |
| **P4** | **Opponent adaptation / non-stationarity** | Over a long match the opponent shifts strategy. Does the LLM do *online* opponent modeling and best-respond, or replay a fixed policy? | Best-response-to-empirical-history is computable each turn |
| **P5** | **Memory as a strategic bottleneck** | Long histories overflow context; the model must *summarize* game history. Poor summarization → degraded play. A strategic-reasoning failure mode unique to long horizon. | Compare full-history vs summarized-history play vs optimal |

These are not five papers — they are a *coherent axis* (time/horizon) that the
current benchmark entirely omits. P1–P3 are the strongest; P4–P5 are
strengtheners / ablations.

## 2. The core artifact: LongGame-Bench

Extend `gamesolve_gen.py` from normal-form to **extensive-form + repeated games**,
each with a *computable long-horizon optimum*:

- **Finitely-repeated matrix games** (PD, coordination, Blotto) with known SPE and
  known folk-theorem region.
- **Sequential bargaining** (alternating-offer Rubinstein, finite horizon) — SPE in
  closed form.
- **Centipede / trust games** — backward induction predicts early stop; cooperation
  deviates measurably.
- **Repeated games vs scripted opponents** (tit-for-tat, grim-trigger, random,
  adaptive) — exact best response computable per turn (for P4).

Each instance is *parameterized by horizon H*, so we sweep H and watch where
reasoning breaks — the "horizon-scaling curve" is the headline figure.

## 3. Concrete ideas per problem

### Idea 10a — "Does outcome-RL learn non-myopic play?" (P1+P2, the flagship)
Train base / SFT / GRPO / Nash-GRPO on LongGame-Bench; evaluate **SPE-distance vs
horizon H**. Pre-registered hypotheses:
- H1: SFT imitates *stage-game* (myopic) responses → SPE-distance grows fast with H.
- H2: outcome-GRPO learns non-myopic play up to some *horizon cliff* H\*, beyond
  which credit assignment fails (gradient too diluted). Locate H\* per method.
- H3: the cliff aligns with where TextArena interactive performance dropped —
  *explaining* the original SFT-collapse finding as a horizon/credit-assignment limit.

This is publishable as pure analysis: first measurement of the **strategic horizon
cliff** of post-trained LLMs, with formal ground truth.

### Idea 10b — Temporal game-structure shaping (P1, method sub-project)
GT-PRM (single-shot *process* reward) was refuted. Long-horizon failure is a
*different* problem — **temporal**, not process-correctness. Test
game-theoretic *return shaping*: use per-turn regret-to-SPE or
potential-based shaping (Ng et al. — provably policy-invariant, so it *cannot*
introduce the reward-hacking that killed GT-PRM) as a baseline/credit signal for
long-horizon GRPO. Clean contrast: "scalar process reward hurts (Idea 8 / GT-PRM),
but *potential-based temporal* shaping helps long-horizon credit assignment."
Even a null result is a clean comparative claim.

### Idea 10c — Emergent cooperation & equilibrium steering (P3)
Self-play GRPO on finitely-repeated PD: does it select cooperation or defection?
Then *steer* selection via initialization / opponent curriculum / discount-like
horizon shaping, and map which equilibrium in the folk-theorem region is reachable.
Ties to Self-Play (#6) but with the sharper, gradable question of *which* equilibrium.

### Idea 10d — Planning-depth & opponent-adaptation as evaluation axes (P2+P4)
Companion to Idea 9 (level-k depth): add a **temporal depth** axis (backward-induction
depth) and an **adaptation** axis (best-response-to-history accuracy vs scripted
opponents). Inference-only on existing checkpoints. Gives a 2-D map: *strategic
depth (level-k) × planning horizon (induction depth)*.

## 4. Why this is strong & low-risk
- Fills the exact gap behind the project's headline finding (interactive collapse).
- Formal ground truth at every horizon → every result is gradable, not vibes.
- Reuses all infrastructure; the only build is the extensive-form generator + solver.
- Composes with Ideas 7 (horizon as another OOD axis), 8 (verifier-select per turn),
  9 (depth), 5 (Nash-GRPO as a method to place on the horizon-scaling curve).

## 5. Risks
- Extensive-form generator + SPE solver is real engineering (budget ~1 week).
- Context length limits maximum H — but that *is* P5, so turn the constraint into a
  studied variable rather than a nuisance.
- Self-play (10c) is the highest-variance sub-project; keep 10a as the safe spine.

## References (grounding)
- Folk theorem / repeated games (Fudenberg & Maskin) — equilibrium multiplicity under repetition.
- Backward induction & subgame perfection (Selten) — formal long-horizon optimum.
- Ng, Harada, Russell, "Policy invariance under reward shaping" (1999) — potential-based shaping cannot be hacked (basis for 10b).
- "LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory" — https://www.arxiv.org/pdf/2502.20432v3
- "MINDGAMES: Live Arena for Social/Strategic Reasoning" — https://arxiv.org/abs/2605.29512 (repeated/interactive multi-agent eval)
- See also: [[strategic_depth_levelk_dynamics]] (Idea 9), [[generalization_anatomy_strategic]] (Idea 7), [[verifier_as_selector_testtime]] (Idea 8).
