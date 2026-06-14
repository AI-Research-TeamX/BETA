# Idea 8 — Sound Verifiers Make Good *Selectors*, Not Good *Rewards*

> **One-liner:** Our game-theoretic verifier is *sound* (gold-trace process score
> 0.84, ~0 on wrong/empty) yet **failed as a dense training reward** (hackable,
> orthogonal to outcome). The resolution is not "the verifier is bad" — it's that
> a sound process verifier is the wrong tool for *shaping under optimization
> pressure* but the *right* tool for *selection at test time*. Build a
> training-free, verifier-guided test-time-scaling method for strategic reasoning
> and turn a refuted result into a positive one.

**Status:** New (June 2026). Direct pivot salvaging the refuted GT-PRM asset.
**Type:** Method paper (training-free) + a clean conceptual contribution.
**Venue fit:** NeurIPS 2026 / ICLR 2027. The "selection vs shaping asymmetry"
framing is the kind of crisp, reusable insight that gets cited.
**Risk:** Low–medium. The verifier is already built and validated; the only open
question is *how much* test-time lift it buys — and even a modest, clean lift with
the asymmetry story is publishable.

---

## 1. The pivot, precisely

The GT-PRM experiment (`gtprm/RESULTS.md`, `exp_gtprm`) established two facts that
look contradictory until you separate *shaping* from *selection*:

1. **The verifier is sound.** Gold CoT process score 0.84 (nash 0.73, BR 0.985),
   empty trace 0.0, wrong-game 0.009; 94% of gold traces carry ≥1 verifiable claim.
2. **As a dense RL reward it fails.** Across 12 runs / 3 seeds, no process
   condition beats the outcome baseline anywhere (ID: process 0.505 vs ORM 0.753;
   OOD: 0.429 vs 0.633). Mechanism: process reward is *near-orthogonal* to answer
   quality (gate corr 0.006) **and hackable** — under gradient pressure the model
   drives process reward 0.15→0.63 while answer quality stays flat (~0.50).

The community-wide explanation for (2) is **Goodhart under optimization pressure**:
"any proxy used as a reward will be exploited once optimization pressure is
applied" (reward-hacking survey 2505.02686; Lilian Weng 2024). The key word is
*optimization*. Gradient ascent *searches for* the verifier's blind spots.

**But selection does not apply gradient pressure to the verifier.** At test time
you draw N i.i.d. samples from a fixed policy and *rank* them. The policy is not
optimizing against the verifier; it cannot deform its outputs to exploit the
verifier's gaps because it never sees the verifier's gradient. A sound verifier
with near-zero false positives on *natural* (un-adversarial) samples is exactly
what best-of-N / beam selection needs. **The orthogonality that killed it as a
reward is also why it is safe as a selector** — it rejects the empties and the
wrong-game traces, which is most of the failure mass.

> **Thesis:** Soundness transfers to selection but not to shaping. We prove this
> empirically in strategic reasoning and ship a free test-time method.

## 2. Why this is novel (not "just best-of-N")

Best-of-N with a learned PRM is well-trodden. Two things make this new:

1. **The verifier is formal, zero-cost, and label-free** — it is the *game
   solver*, not a trained reward model. There is no PRM to train, no annotation,
   no PRM-of-the-PRM reward-hacking. Compare to math PRMs (expensive, human/LLM
   labeled, themselves hackable). "Algorithmic verifier as test-time critic for
   strategic reasoning" is unexplored.
2. **The asymmetry is the contribution.** We run the *same* verifier as (a) a
   training reward and (b) a test-time selector on the *same* benchmark and show
   it *hurts* in (a) and *helps* in (b). That controlled A/B is a clean,
   generalizable scientific claim about *when* process verification is usable —
   directly actionable for the whole PRM literature.

## 3. Method: VeriSelect (training-free, three variants of rising power)

All on the existing verl/vLLM stack; the sound verifier already scores partial
solution steps.

- **V1 — Best-of-N.** Sample N traces, score each with the *outcome-agnostic*
  process verifier, return the argmax. (Crucial: at test time we usually don't
  have the gold answer, so process score is the *only* available signal — this is
  the realistic deployment regime, unlike training where ORM was available.)
- **V2 — Step-level beam / tree search.** Because the verifier scores *partial*
  derivations (dominance step → BR step → NE step), we prune beams whose executed
  steps fail verification. This is where a process verifier should beat an
  outcome-only verifier: it gives signal *before* the final answer exists.
- **V3 — Verifier-guided self-correction.** Feed the failed-step diagnostic back
  ("your dominance elimination removed a non-dominated action") and let the model
  revise. One-shot and iterative.

## 4. Experiments

**Headline A/B (the spine).** Same verifier, same GameSolve-Bench, same base
policy:
- Column 1: process-as-reward (already have it — `gtprm`): ID 0.505, OOD 0.429. ✗
- Column 2: process-as-selector (VeriSelect V1–V3): expected ↑ over the policy's
  own pass@1. The contrast *is* the paper figure.

**E1 — Test-time scaling curves.** Accuracy vs N for {process-selector,
outcome-oracle upper bound, self-consistency/majority, random}. Process-selector
should sit between majority and the outcome oracle, and the *gap to oracle* is the
verifier's residual blind spot — quantifies soundness in the selection regime.

**E2 — The asymmetry under controlled pressure.** Interpolate optimization
pressure: 0 steps (pure selection) → a few RL steps against the verifier → full
RL. Show the verifier helps at 0 and degrades monotonically as pressure rises
(reward hacking emerging in real time). This *operationalizes Goodhart* with a
dial — a genuinely nice figure.

**E3 — Where it helps most.** Break down by task: V2's step-level pruning should
help most on multi-step concepts (iterated dominance, mixed NE) and on **OOD**
(where the base policy's pass@1 is low but a correct trace is still *in the
sample set* — selection recovers it). This reconnects to Idea 7's generalization
story: test-time selection as an OOD-robustness lever.

**E4 — Cost/accuracy frontier.** Report accuracy-per-FLOP vs majority voting and
vs just training longer. The pitch: *free* (no training) robustness for strategic
agents.

## 5. Predicted results & their stories

- If VeriSelect clearly beats pass@1 and majority → method paper + asymmetry insight. ✓✓
- If it only matches majority but the *asymmetry* (E2 dial) is clean → still a
  publishable conceptual paper ("when can you use a process verifier?").
- If step-level (V2) ≫ best-of-N (V1) → validates that *partial* verifiability is
  the real value of formal verifiers, a forward-looking claim.

## 6. Cost & timeline

No training for V1/V2; V3 is inference + optional light fine-tune. Verifier exists.
Eval harness exists. **~1.5–2 weeks** to a draft. Among the cheapest high-upside
options on the board, and it *recovers* sunk GT-PRM cost.

## 7. Relationship to existing ideas / assets

- **Salvages Idea 1 / GT-PRM** (`gtprm/`) — same verifier, flipped from reward to
  selector. The refutation becomes the *motivation*.
- Composes with **Idea 7**: VeriSelect as an OOD-recovery mechanism, measured on
  the typed lattice.
- Composes with **Nash-GRPO (#5)**: selection on top of a better-trained policy.
- Cleanly distinct from **RepE/PGAS (#2)**: that steers activations (refuted);
  this selects whole samples (no activation surgery, no training).

## 8. Open risks

- If the base policy's sample set rarely *contains* a correct trace, selection
  can't help (selection ≤ oracle@N). Mitigate with a capable base / larger N and
  report the oracle@N ceiling honestly.
- Must guard against the verifier rewarding *verbose-but-correct-looking* partial
  steps at selection time too — measure false-positive rate on natural samples
  (already ~0 on wrong-game, but re-check on near-miss traces).

## References (grounding)
- This project: `gtprm/RESULTS.md`, `research_log` exp_gtprm (verifier sound; reward fails; hacking 0.15→0.63).
- Reward-hacking survey — https://arxiv.org/pdf/2505.02686 (Goodhart under *optimization* pressure)
- "Is it thinking or cheating?" — https://arxiv.org/html/2510.01367 (implicit reward hacking)
- "Putting the Value Back in RL: ... Unifying Reasoners with Verifiers" — https://arxiv.org/html/2505.04842v2 (test-time scaling via verification)
- "Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers" — https://arxiv.org/pdf/2502.20379
- "Learning Generative Selection for Best-of-N" — https://arxiv.org/pdf/2602.02143
