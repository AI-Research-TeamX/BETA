# Idea 9 — Does Post-Training Climb the Cognitive Hierarchy?

> **One-liner:** Behavioral game theory measures *strategic depth* (level-k /
> cognitive-hierarchy reasoning) on a calibrated scale. Use it as a measuring
> instrument to ask, for the first time with a *quantitative depth axis*: does RL
> raise an LLM's level-k, does SFT freeze or lower it, and does depth — not
> accuracy — predict transfer to interactive multi-agent games?

**Status:** New (June 2026). Analysis companion to Ideas 7 & 8.
**Type:** Analysis paper with a borrowed-but-underused measurement instrument.
**Venue fit:** COLM 2026 / ICLR 2027 / a behavioral-ML venue. Strong if framed as
"a new *axis* for evaluating post-training, not another leaderboard."
**Risk:** Low-medium. Instruments exist (level-k estimation from behavior); the
empirical question is open and our checkpoints answer it.

---

## 1. The gap

Everyone evaluates strategic reasoning with **accuracy on solved games**. But two
models can have identical accuracy on a 2×2 NE task while differing wildly in
*how many steps of recursive belief* ("I think that you think that I think…") they
actually run. Behavioral economics has measured exactly this for decades:

- **Level-k / cognitive hierarchy**: level-0 = non-strategic, level-1 = best-respond
  to level-0, level-k = best-respond to level-(k−1). Canonical elicitation games
  (p-beauty / Keynesian beauty contest, 11–20 money-request game) *estimate a
  model's k from its behavior alone*, without assuming equilibrium.
- LLMs are now being placed on this scale (K-Level Reasoning 2402.01521; "LLM
  Strategic Reasoning: Agentic Study through Behavioral Game Theory" 2502.20432;
  MINDGAMES 2605.29512). Reasoning models (o1, R1) sit higher.

**Nobody has used level-k as a *dependent variable to characterize post-training.*
** That is the move: treat k as the thing that changes, and ask which training
algorithm moves it.

## 2. Why this connects to *our* findings

Our most robust result is the **SFT-collapse / RL-transfer** asymmetry on
interactive games (TextArena: SFT 38% < Base 52% < GRPO 56%). The natural
mechanistic hypothesis, in game-theoretic language:

> SFT raises *accuracy on the seen, single-shot solution concept* but does **not**
> raise — possibly lowers — *strategic depth*, because it imitates final answers,
> not the recursive belief computation. RL, optimizing for *outcomes against a
> responsive environment*, has incentive to raise k. Therefore SFT collapses
> exactly where depth matters (multi-step, opponent-responsive games) while RL
> holds up.

This reframes the TextArena gap as a **depth** phenomenon, not a generic
"robustness" one — a sharper, more mechanistic claim, and *testable*.

## 3. Hypotheses (pre-registered)

- **H1 (depth ordering):** estimated k: RL ≥ Base > SFT on elicitation games,
  *even where SFT's single-shot accuracy is highest*. (Accuracy ⊥ depth.)
- **H2 (depth predicts transfer):** across checkpoints, estimated k correlates
  with TextArena/GTBench interactive win-rate *better than* in-distribution
  GameSolve accuracy does. I.e., depth is the right predictor of agentic transfer.
- **H3 (RL raises k over training):** k increases monotonically along the
  verl-GRPO / Nash-GRPO training trajectory (we have intermediate checkpoints:
  `global_step_*`). SFT's k is flat from step 0.
- **H4 (Nash-GRPO bonus):** if Nash-GRPO (#5) does anything, it should *raise k*
  more than mean-baseline GRPO, because its advantage signal is explicitly
  game-theoretic. A natural, almost-free add-on measurement for the running #5 runs.

## 4. Method

1. **Instrument battery.** Implement standard depth-elicitation games as
   text prompts: p-beauty (guess ⅔ of the mean), 11–20 game, and a small suite of
   2-player matrix games with known level-k predictions. Estimate k per checkpoint
   via the standard cognitive-hierarchy likelihood fit over repeated plays
   (vs. fixed level-0 / vs. self-play populations).
2. **Score every checkpoint we already have**: base, SFT, verl-GRPO (+ its step
   trajectory), GRPO+probe, Nash-GRPO seeds. Pure inference on 8×H20 via vLLM.
3. **Correlate** estimated k with (a) GameSolve ID accuracy, (b) OOD accuracy,
   (c) TextArena/GTBench win-rate. Test H2 (does k beat accuracy as a predictor?).
4. **Trajectory analysis** for H3/H4 using intermediate `global_step_*` ckpts.
5. **(Optional, deeper) internal depth.** Probe whether higher-k behavior is
   accompanied by deeper recursive belief representations (reuse probing stack
   from Idea 7) — links behavior to mechanism.

## 5. Why it stands alone as a paper

- **New evaluation axis**, not a leaderboard entry: "measure post-training by the
  strategic *depth* it induces, not just task accuracy." Reusable by the whole
  field — any post-training method can be placed on this axis.
- **Explains a real, surprising phenomenon** (SFT-below-base on interactive games)
  with a principled, decades-old instrument.
- **Cheap and inference-only**; rides entirely on existing checkpoints and the
  Nash-GRPO runs already in flight.

## 6. Predicted results & stories

- H1+H2 hold → "Strategic depth, not accuracy, is the right yardstick for
  post-training; SFT buys accuracy at the cost of depth." Clean, citable.
- H1 fails (SFT raises k too) → still interesting: depunbundles depth from
  generalization, refutes the easy story — a finding either way.
- H4 holds → a *third*, independent confirmation that Nash-GRPO does something,
  strengthening #5's paper for free.

## 7. Cost & timeline

Instrument implementation ~3 days, eval sweep ~2 days, analysis + optional probing
~4 days. **~1.5 weeks.** No training.

## 8. Relationship to existing ideas / assets

- **Companion to Idea 7**: Idea 7 anatomizes *generalization*; this anatomizes
  *strategic depth*. Together they're a strong two-paper "science of strategic
  post-training" story, or one larger paper.
- **Feeds Nash-GRPO (#5)**: provides an extra, theory-aligned metric (depth) that
  Nash-GRPO is most likely to move — measure it on the running seeds.
- Distinct from **Self-Play (#6)**: that *trains* via interaction; this *measures*
  depth of already-trained models.

## 9. Open risks

- Level-k estimation is noisy on single rollouts — need enough repeated plays /
  population settings for stable k fits; budget sampling.
- LLMs may verbalize high-k reasoning while *acting* low-k (or vice versa) — score
  on *actions*, and optionally contrast verbalized vs enacted depth (itself an
  interesting sub-finding).
- Prompt sensitivity of elicitation games; report across several framings.

## References (grounding)
- "K-Level Reasoning: Higher-Order Beliefs in LLMs for Strategic Reasoning" — https://arxiv.org/pdf/2402.01521
- "LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory" — https://www.arxiv.org/pdf/2502.20432v3
- "MINDGAMES: A Live Arena for Social and Strategic Reasoning in Multi-Agent LLMs" — https://arxiv.org/abs/2605.29512
- "LSTM ... exhibits representations of level-k thinking" — https://arxiv.org/pdf/2311.17211 (level-k is representable / probeable)
- "Take Caution in Using LLMs as Human Surrogates" — https://arxiv.org/pdf/2410.19599 (caveats on behavioral elicitation)
