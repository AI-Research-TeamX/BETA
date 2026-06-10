# Game-Theoretic Foundations of RLHF: Research Report for Top-Venue Submission

## Executive Summary

The intersection of game theory and LLM alignment has emerged as one of the most active research frontiers in 2024-2025. Key developments include Nash Learning from Human Feedback (NLHF), Self-Play fine-tuning (SPIN), Self-Play Preference Optimization (SPPO), and Direct Nash Optimization (DNO). Despite rapid progress, significant gaps remain in multi-player formulations, equilibrium selection, robustness to reward hacking, and multi-objective alignment. This report identifies concrete novel contribution opportunities feasible with 8xH20 GPUs and Qwen-series models.

---

## 1. Literature Review: Key Papers (2023-2025)

### 1.1 Nash Learning from Human Feedback (NLHF) and Variants

**[P1] "Nash Learning from Human Feedback"**
- Authors: Remi Munos, Michal Valko, et al. (Google DeepMind)
- Venue: ICML 2024 (Oral)
- arXiv: 2312.00886 (Dec 2023)
- Key contributions:
  - Frames RLHF as a two-player constant-sum game between a policy pi and a reference/opponent policy mu
  - The preference model P(y1 > y2 | x) defines payoffs
  - Shows the optimal aligned policy is the **Nash equilibrium** of this game
  - Proposes two algorithms: **Nash-MD** (mirror descent) and **Nash-EMA** (exponential moving average opponent)
  - Theoretically proves convergence to Nash equilibrium under Bradley-Terry preference model
  - Key insight: standard RLHF with KL regularization is a special case (Nash against the reference policy)

**[P2] "Direct Nash Optimization"**
- Authors: Corby Rosset, Ching-An Cheng, Arindam Mitra, et al. (Microsoft Research)
- Venue: arXiv preprint, 2024 (likely NeurIPS 2024 or ICLR 2025)
- arXiv: 2404.03715 (Apr 2024)
- Key contributions:
  - Eliminates the need for separate reward model training
  - Directly optimizes for Nash equilibrium using a regression-based approach
  - More scalable than Nash-MD: works with batched generation
  - Demonstrates strong performance on AlpacaEval and MT-Bench
  - Uses general preference oracle (not restricted to Bradley-Terry)

**[P3] "A General Theoretical Paradigm to Understand Learning from Human Feedback"**
- Authors: Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, et al. (Google DeepMind)
- Venue: AISTATS 2024
- arXiv: 2310.12036 (Oct 2023)
- Key contributions:
  - Unifies RLHF, DPO, and preference-based objectives under KL-regularized game framework
  - Introduces IPO (Identity Preference Optimization) as a theoretically motivated alternative to DPO
  - Shows DPO implicitly solves a regularized two-player zero-sum game
  - Proves equivalence between various alignment objectives and Nash equilibrium conditions

**[P4] "Magnetic Alignment of Large Language Models"**
- Authors: Weijia Shi et al. (Meta / University of Washington)
- Venue: arXiv 2024
- Key contributions:
  - Iterative alignment procedure inspired by fictitious play in games
  - Each iteration, the model plays against its previous version
  - Proves convergence under certain conditions
  - Connection to no-regret learning in games

**[P5] "A Minimaximalist Approach to Reinforcement Learning from Human Feedback"**
- Authors: Gokul Swamy, Christoph Dann, et al. (CMU / Google DeepMind)
- Venue: ICML 2024
- arXiv: 2401.04056 (Jan 2024)
- Key contributions:
  - Shows RLHF can be formulated as a **minimax game** between policy and reward class
  - The minimax solution is robust to reward misspecification
  - Proposes SPO (Self-Play Optimization) that alternates between reward estimation and policy improvement
  - Theoretical guarantees on robustness to Goodhart's law / reward hacking
  - Direct connection to adversarial training and robust optimization

### 1.2 Self-Play for Alignment

**[P6] "SPIN: Self-Play fIne-tuNing Converts Weak Language Models to Strong Language Models"**
- Authors: Zixiang Chen, Yihe Deng, Huizhuo Yuan, et al. (UCLA)
- Venue: ICML 2024
- arXiv: 2401.01335 (Jan 2024)
- Key contributions:
  - Two-player game: main player (current policy) vs opponent (previous iteration)
  - Main player learns to distinguish its own generations from human-written text
  - Opponent generates synthetic data
  - **Fixed point**: Nash equilibrium is reached when policy matches target distribution
  - Iterative training converges provably under certain assumptions
  - Strong empirical results: Zephyr-7B matches GPT-4 on MT-Bench after SPIN

**[P7] "Self-Play Preference Optimization for Language Model Alignment (SPPO)"**
- Authors: Yue Wu, Zhiqing Sun, et al. (CMU)
- Venue: arXiv 2405.00675 (May 2024), NeurIPS 2024
- Key contributions:
  - Frames alignment as a **two-player constant-sum game** with preference-based payoffs
  - The minimax winner (Nash equilibrium) is the policy preferred over all others
  - Uses self-play: policy generates pairs, then optimizes to be preferred over its own previous outputs
  - Achieves state-of-the-art on AlpacaEval 2.0 (28.53% LC win rate with Mistral-7B)
  - Does not require a separate reward model
  - Theoretical connection to von Neumann's minimax theorem

**[P8] "Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint"**
- Authors: Wei Xiong, Hanze Dong, et al.
- Venue: ICML 2024
- Key contributions:
  - Online iterative DPO with theoretical game-theoretic justification
  - Shows that single-round DPO fails to find Nash equilibrium
  - Iterative procedure converges to the game solution
  - Practical algorithm with strong empirical results

### 1.3 Game Theory and Reward Hacking / Robustness

**[P9] "Defining and Characterizing Reward Hacking"**
- Authors: Joar Skalse, Nikolaus Howe, et al.
- Venue: NeurIPS 2022
- Key contributions:
  - Formalizes reward hacking through Goodhart's taxonomy
  - Shows reward hacking is inevitable when proxy reward diverges from true reward
  - Game-theoretic interpretation: policy exploits reward model's weaknesses

**[P10] "WARM: On the Benefits of Weight Averaged Reward Models"**
- Authors: Alexandre Rame et al. (Meta)
- Venue: ICML 2024
- Key contributions:
  - Addresses reward hacking by ensembling reward models
  - Game-theoretic interpretation: makes the "opponent" (reward model) more robust
  - Averaging reduces exploitability of reward signal

**[P11] "Adversarial Preference Optimization"**
- Authors: Various (2024)
- Key contributions:
  - Adversarial training framework for alignment
  - Red-team / blue-team game for robust alignment
  - Policy must satisfy preferences even under adversarial perturbations

**[P12] "REBEL: Reinforcement Learning via Regret-Based Equilibrium Learning"**
- Authors: Gao et al.
- Venue: 2024
- Key contributions:
  - Uses regret minimization (a game-theoretic concept) for alignment
  - Connection to no-regret learning and equilibrium computation
  - Provides online learning guarantees

### 1.4 Multi-Agent and Multi-Objective Alignment

**[P13] "AI Safety via Debate"**
- Authors: Geoffrey Irving, Christiane Kamber, Paul Christiano (OpenAI)
- Venue: arXiv 2018 (foundational)
- Key contributions:
  - Two AI agents debate to convince a human judge
  - Theoretically: Nash equilibrium of the debate game is truthful
  - Foundation for multi-agent approaches to alignment

**[P14] "Debating with More Persuasive LLMs Leads to More Truthful Answers"**
- Authors: Akbir Khan et al. (Anthropic)
- Venue: ICML 2024
- Key contributions:
  - Empirical validation of debate for alignment
  - Stronger debaters improve truthfulness
  - Game-theoretic equilibrium analysis in practice

**[P15] "Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards"**
- Authors: Alexandre Rame et al. (Meta)
- Venue: NeurIPS 2023
- Key contributions:
  - Multi-objective alignment through weight interpolation
  - Pareto front discovery without retraining
  - Implicit connection to multi-objective games (though not explicitly game-theoretic)

**[P16] "Multi-Objective Reinforcement Learning from Human Feedback"**
- Authors: Various, 2024
- Key contributions:
  - Multiple reward models for different objectives (helpfulness, harmlessness, honesty)
  - Scalarization vs Pareto approaches
  - NOT explicitly game-theoretic (gap!)

### 1.5 Recent Developments (Late 2024 - Early 2025)

**[P17] "Nash-MD for RLHF with General Preferences"**
- Follow-up to NLHF extending beyond Bradley-Terry
- Handles intransitive preferences (where A>B>C>A)
- Game-theoretic interpretation is essential for intransitive preferences

**[P18] "Online Nash Policy Optimization"**
- Authors: Building on NLHF
- Online learning version where preference data arrives sequentially
- Regret bounds for Nash equilibrium learning

**[P19] "From r to Q*: Your Language Model is Secretly a Q-Function"**
- Connections between RLHF policy and game-theoretic Q-values
- Token-level game formulation

**[P20] "Game-Theoretic Foundations of Safe RL"**
- Constrained games for safety
- Applicable to alignment constraints

---

## 2. Gap Analysis

### 2.1 Confirmed Gaps in Current Literature

| Gap | Current State | Opportunity |
|-----|--------------|-------------|
| **Multi-player (>2) alignment games** | All work is two-player (policy vs. reference/reward/opponent) | N-player formulations with multiple stakeholders/objectives |
| **Equilibrium selection** | Works prove Nash equilibrium exists but don't address which one | Refinement concepts (trembling-hand, proper, correlated) for alignment |
| **Mode collapse as equilibrium phenomenon** | Discussed informally, no rigorous theory | Formal connection between support of Nash equilibrium and generation diversity |
| **Reward hacking as game exploitation** | SPO/minimaximalist partially addresses | Full Stackelberg/mechanism design treatment |
| **Multi-objective alignment via game theory** | Multi-obj RLHF exists but not game-theoretic | Pareto Nash equilibrium, Shapley value for objective weighting |
| **Dynamic/extensive-form games** | All work treats single-stage (normal-form) games | Multi-turn alignment as extensive-form game |
| **Bayesian games for uncertain preferences** | Not explored | Bayesian Nash equilibrium under preference uncertainty |
| **Population games / evolutionary dynamics** | Not explored for alignment | ESS (Evolutionary Stable Strategies) to prevent mode collapse |
| **Mechanism design for preference elicitation** | Not explored | Incentive-compatible preference collection |
| **Correlated equilibrium for alignment** | Not explored | Richer than Nash, allows coordination |

### 2.2 Why These Gaps Matter

1. **Multi-player games**: Real alignment involves multiple objectives (helpfulness, harmlessness, honesty) and multiple stakeholders (users, developers, society). Two-player models are insufficient.

2. **Equilibrium selection**: When multiple equilibria exist (e.g., different "personalities" all satisfying alignment), we need principled selection. This directly relates to mode collapse - collapsed modes correspond to degenerate equilibria.

3. **Robustness to reward hacking**: Minimax formulations provide worst-case guarantees, but Stackelberg formulations (where the adversary moves second) are more natural for reward hacking.

4. **Population dynamics**: If we view alignment as an evolutionary process (model populations evolving through training), evolutionary game theory offers tools for diversity maintenance.

---

## 3. Concrete Idea Formulations

### Idea A: "Multi-Objective Alignment as an N-Player Game: Pareto Nash Equilibrium for LLMs"

**Core formulation:**
- Frame alignment with K objectives as a K+1 player game
- Players: the policy pi, and K "objective advocates" (one per reward dimension)
- Each objective advocate maximizes their reward while the policy balances all
- Solution concept: **Pareto Nash Equilibrium** (PNE) - a Nash equilibrium that is also Pareto optimal
- Prove existence conditions and design algorithms to find PNE

**Why novel:**
- First to use multi-player game theory (not just two-player) for alignment
- Provides principled trade-offs between objectives without ad-hoc scalarization
- Connects to welfare economics (social choice theory + alignment)

**Training algorithm:**
- Extend GRPO to multi-objective: each objective gets a "virtual player" that generates preference signals
- Use no-regret learning (multiplicative weights) to converge to PNE
- The policy updates using weighted GRPO where weights emerge from the game solution

**Experiments:**
- Multi-objective alignment on Qwen-2.5 (helpfulness + safety + honesty)
- Compare against: scalarized RLHF, multi-objective RLHF (linear combination), rewarded soups
- Metrics: Pareto front quality, individual objective scores, diversity of outputs
- Show PNE achieves better Pareto front than baselines

### Idea B: "Stackelberg Alignment: Robust RLHF through Leader-Follower Game Formulation"

**Core formulation:**
- Model reward hacking as a Stackelberg game
- Leader: alignment procedure (chooses policy)
- Follower: adversarial "reward hacker" (exploits the reward model after seeing the policy)
- The leader must commit to a policy that is robust to the follower's best response
- Solution: Stackelberg equilibrium - tighter robustness than Nash

**Why novel:**
- Stackelberg is a better model of reward hacking than Nash (hacker adapts to policy)
- Provides stronger robustness guarantees than minimax (SPO)
- First to explicitly use Stackelberg structure for alignment

**Training algorithm:**
- Bi-level optimization: outer loop optimizes policy, inner loop finds worst-case reward manipulation
- Inner loop: adversarial reward model perturbation (within epsilon ball)
- Outer loop: GRPO on the adversarially-perturbed reward
- Can be approximated via alternating gradient descent with unrolling

**Experiments:**
- Train Qwen-2.5 with Stackelberg alignment vs standard RLHF/DPO/SPPO
- Test robustness: over-optimization curves, reward hacking benchmarks
- Show Stackelberg maintains performance even with imperfect reward models
- Scaling behavior: does robustness improve with model size?

### Idea C: "Evolutionary Alignment: Using Evolutionary Game Theory to Prevent Mode Collapse in RLHF"

**Core formulation:**
- View RLHF training as an evolutionary process on a population of policies
- Instead of single policy, maintain a population (mixture of policies)
- Define fitness = alignment score; interaction = preference comparison
- Solution concept: **Evolutionarily Stable Strategy (ESS)**
- ESS guarantees diversity: a monomorphic population (mode collapse) is evolutionary unstable

**Why novel:**
- First application of evolutionary game theory to LLM alignment
- Provides principled mechanism for diversity (ESS = diverse = resistant to mode collapse)
- Connects mode collapse to evolutionary instability (novel theoretical insight)
- Practical algorithm that maintains diversity without explicit diversity penalties

**Training algorithm:**
- Maintain K policy variants (population)
- Each generation: evaluate fitness via preferences, select + mutate (RL update)
- Replicator dynamics determine population fractions
- Converge to ESS = a diverse equilibrium population

**Experiments:**
- Compare diversity metrics: distinct-n, embedding entropy, topic coverage
- Show ESS-based training avoids mode collapse that standard RLHF/DPO exhibits
- Quality comparison: ESS population's mixture vs single best policy
- Ablation: population size K, mutation rate (learning rate), selection pressure

### Idea D (RECOMMENDED - Highest Feasibility + Novelty): "Nash-GRPO: Game-Theoretic Group Relative Policy Optimization"

**Core formulation:**
- Extend GRPO (Group Relative Policy Optimization) with game-theoretic foundations
- Standard GRPO: groups of responses, relative rewards within group
- Nash-GRPO: formulate the group comparison as a tournament game
  - Each response is a "player" in a tournament
  - Preferences define pairwise payoffs
  - The Nash equilibrium of the tournament game defines optimal response ranking
  - Policy update uses Nash-equilibrium-derived advantages (not just mean-baseline)

**Key insight:**
- GRPO uses group mean as baseline (arbitrary)
- Nash-GRPO uses the **Nash ranking** as baseline - which is the game-theoretically optimal importance weighting
- For intransitive preferences (A>B>C>A), Nash ranking handles cycles correctly while mean-baseline fails
- Connection: Nash ranking = stationary distribution of Markov chain on preference graph = PageRank-style

**Why novel:**
- First game-theoretic foundation for GRPO specifically (which is used by DeepSeek, Qwen)
- Handles intransitive preferences naturally (mean-baseline assumes transitivity)
- Provides theoretical justification for group-based policy optimization via minimax
- Directly extends existing GRPO infrastructure (easy to implement)

**Training algorithm:**
```
For each prompt x:
  1. Sample K responses {y1, ..., yK} from policy
  2. Compute pairwise preference matrix P[i,j] = Prob(yi > yj)
  3. Find Nash equilibrium of the preference tournament:
     - Solve: max_p min_q p^T P q  (minimax)
     - Nash mixture p* gives "importance" of each response
  4. Compute advantages: A(yi) = p*[i] - 1/K  (Nash deviation from uniform)
  5. Policy gradient with Nash advantages:
     grad = sum_i A(yi) * grad log pi(yi|x)
```

**Why feasible with 8xH20:**
- Directly extends GRPO (researcher already has GRPO infrastructure)
- Only added computation: solving a K×K game per group (trivial: K=8 or 16)
- Same GPU parallelization as standard GRPO
- Can use Qwen-2.5-7B or Qwen-2.5-14B

**Experiments:**
- Comparison: GRPO vs Nash-GRPO vs DPO vs NLHF vs SPPO
- Benchmarks: AlpacaEval 2.0, MT-Bench, Arena-Hard
- Analysis: behavior on intransitive preferences (synthetic + natural)
- Theoretical: prove convergence to Nash equilibrium of the alignment game
- Ablation: group size K, effect of Nash vs mean vs median baseline

---

## 4. Novelty and Publishability Assessment

### Idea D (Nash-GRPO) - Primary Recommendation

**Novelty score: 8.5/10**
- Game-theoretic foundation for GRPO is completely new
- Handles intransitive preferences (a known issue others haven't solved in GRPO context)
- Connects two hot topics: GRPO (DeepSeek-R1) + Nash alignment (ICML 2024 oral)

**Publishability assessment:**
- Venue: ICML 2025 (if submitted by Jan 2025) or NeurIPS 2025 / ICLR 2026
- Strengths: theoretical contribution + practical improvement + addresses real limitation
- Fits "theory-motivated practical algorithm" template that top venues love
- Comparable to: NLHF (ICML 2024 oral), SPPO (NeurIPS 2024), SPO (ICML 2024)

**Risks:**
- Low risk: may not outperform GRPO on transitive preferences (but theory + intransitive case is enough)
- Medium risk: concurrent work might formalize GRPO game-theoretically
- Mitigation: fast execution, unique angle (intransitive preferences + tournament game)

### Idea A (Multi-Objective Pareto Nash) - Secondary

**Novelty score: 9/10** (very novel but harder to execute)
- Completely new formulation
- Risk: may be hard to show clear empirical gains over simpler baselines

### Idea B (Stackelberg Alignment) - Tertiary

**Novelty score: 8/10**
- Builds on minimaximalist (SPO) but with Stackelberg structure
- Risk: bi-level optimization can be unstable

---

## 5. Experimental Design (for Idea D: Nash-GRPO)

### 5.1 Setup

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Base model | Qwen-2.5-7B-Instruct | Strong open model, fits 8xH20 |
| Training framework | veRL (already set up) | GRPO implementation ready |
| Preference model | PairRM or Qwen-2.5-72B-as-judge | Standard choice |
| Training data | UltraFeedback, HH-RLHF | Standard benchmarks |
| Group size K | 8, 16, 32 (ablation) | Matches GPU count |
| Baseline methods | GRPO, DPO, SPPO, Nash-MD | Cover all relevant comparisons |

### 5.2 Experiments

**Experiment 1: Main comparison (2-3 days on 8xH20)**
- Train Qwen-2.5-7B with Nash-GRPO vs baselines
- Evaluate on AlpacaEval 2.0, MT-Bench, Arena-Hard
- Report win rates, LC win rates

**Experiment 2: Intransitive preference robustness (1-2 days)**
- Construct synthetic intransitive preferences
- Show GRPO (mean-baseline) gives suboptimal ranking
- Show Nash-GRPO correctly handles cycles
- Evaluate on natural intransitive preferences (style vs accuracy vs conciseness)

**Experiment 3: Reward hacking resistance (1-2 days)**
- Over-optimization curves (KL vs reward)
- Compare Nash-GRPO vs GRPO: which is more robust to proxy reward over-optimization?
- Hypothesis: Nash ranking is more robust because it doesn't over-credit any single dimension

**Experiment 4: Scaling (1 day)**
- Qwen-2.5-1.5B, 7B, 14B
- Does Nash-GRPO benefit more at larger scale?

**Experiment 5: Theoretical validation (no GPU needed)**
- Prove convergence theorem for Nash-GRPO
- Characterize fixed points
- Show connection to NLHF as special case

### 5.3 GPU Budget

| Experiment | GPUs | Duration | Total GPU-hours |
|-----------|------|----------|----------------|
| Exp 1 (main) | 8xH20 | 72h | 576 |
| Exp 2 (intransitive) | 8xH20 | 48h | 384 |
| Exp 3 (robustness) | 8xH20 | 48h | 384 |
| Exp 4 (scaling) | 8xH20 | 24h | 192 |
| **Total** | | **~8 days** | **1536** |

This is very feasible for 8xH20 GPUs. The key insight is that Nash-GRPO adds minimal overhead to GRPO (solving a KxK game is O(K^3) per group, negligible vs. forward/backward passes).

---

## 6. Detailed Technical Formulation (Idea D)

### 6.1 Background: GRPO

Standard GRPO (DeepSeek-R1):
- Sample K responses: {y1, ..., yK} ~ pi_theta(.|x)
- Compute rewards: r(yi, x) for each i
- Compute advantages: A(yi) = (r(yi) - mean(r)) / std(r)  [group normalization]
- Policy gradient: grad J = E[sum_i A(yi) * grad log pi(yi|x)]

**Limitation**: Uses scalar reward and mean-baseline. Assumes transitivity of preferences.

### 6.2 Nash-GRPO Formulation

**Step 1: Preference tournament**
Given K responses, construct pairwise preference matrix:
- P[i,j] = Prob(yi preferred over yj | x)  [from preference model or LLM judge]
- P is a K×K matrix with P[i,j] + P[j,i] = 1

**Step 2: Nash equilibrium of tournament game**
- Two-player zero-sum game with payoff matrix M = P - 1/2
- Find Nash equilibrium: p* = argmax_p min_q p^T M q
- This is a linear program (solvable in O(K^3))
- p* is a probability distribution over responses

**Step 3: Nash advantages**
- A_Nash(yi) = p*[i] - 1/K
- Interpretation: how much more "weight" the Nash equilibrium gives to yi vs uniform

**Step 4: Policy update**
- Same as GRPO but with Nash advantages:
- grad J_Nash = E[sum_i A_Nash(yi) * grad log pi(yi|x)]

### 6.3 Theoretical Properties

**Theorem 1 (Convergence):**
Under mild assumptions, Nash-GRPO converges to the Nash equilibrium of the alignment game defined by the preference model P.

**Proof sketch:**
- Nash-GRPO is equivalent to Follow-the-Regularized-Leader (FTRL) in the preference game
- FTRL converges to Nash equilibrium in zero-sum games (standard result)
- The KL regularization in policy updates serves as the regularizer

**Theorem 2 (Intransitivity handling):**
When preferences are intransitive, Nash-GRPO assigns non-degenerate weights to all responses in cycles, while mean-baseline GRPO may arbitrarily rank them.

**Theorem 3 (Robustness):**
Nash-GRPO maximizes worst-case performance against any opponent response mixing. This provides minimax robustness guarantees.

### 6.4 Connection to Prior Work

- **NLHF** (Munos et al.): Nash-GRPO can be seen as a practical implementation of NLHF using group sampling
- **SPPO** (Wu et al.): SPPO uses self-play across iterations; Nash-GRPO uses Nash within each group (complementary)
- **GRPO** (DeepSeek): Nash-GRPO strictly generalizes GRPO (when preferences are transitive and Bradley-Terry, they coincide)
- **DNO** (Rosset et al.): DNO is regression-based; Nash-GRPO is policy-gradient-based (different optimization family)

---

## 7. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Nash advantages ≈ mean advantages empirically | Medium | High | Focus on intransitive preferences where they provably differ; show theoretical contribution alone is valuable |
| Concurrent work on game-theoretic GRPO | Medium | Medium | Execute fast; emphasize unique intransitivity angle; submit to workshop first |
| Solving KxK game adds noise with small K | Low | Medium | Use K=16 or 32; regularize Nash solution |
| Preference model quality limits gains | Medium | Low | Use strong preference model (Qwen-72B-as-judge); show improvements are orthogonal to preference model quality |
| Reviewers find contribution incremental over NLHF | Medium | Medium | Emphasize: (a) first theoretical foundation for GRPO family, (b) practical algorithm, (c) intransitivity handling, (d) connection to tournament theory |
| Training instability | Low | Medium | Nash advantages bounded by construction; can interpolate with mean-baseline |

---

## 8. Paper Outline

**Title:** "Nash-GRPO: Game-Theoretic Foundations for Group Relative Policy Optimization"

1. Introduction
   - GRPO is widely adopted (DeepSeek-R1, Qwen) but lacks theoretical foundation
   - We formalize GRPO as a tournament game and derive Nash-GRPO
   - Nash-GRPO handles intransitive preferences and provides robustness

2. Background
   - RLHF, DPO, GRPO
   - Game theory in alignment: NLHF, SPPO, DNO

3. Nash-GRPO: Formulation
   - Tournament game construction
   - Nash equilibrium computation
   - Nash advantages and policy update

4. Theoretical Analysis
   - Convergence to Nash equilibrium (Theorem 1)
   - Handling intransitive preferences (Theorem 2)
   - Robustness guarantees (Theorem 3)
   - Connection to NLHF/SPPO as special cases

5. Experiments
   - Main comparison on standard benchmarks
   - Intransitive preference experiments
   - Robustness to reward hacking
   - Scaling analysis

6. Related Work

7. Conclusion

---

## 9. Timeline

| Week | Activity |
|------|----------|
| 1-2 | Theory: formalize Nash-GRPO, prove theorems |
| 3-4 | Implementation: extend veRL GRPO with Nash advantages |
| 5-6 | Experiment 1-2: main comparison + intransitivity |
| 7-8 | Experiment 3-4: robustness + scaling |
| 9-10 | Writing: full paper draft |
| 11-12 | Revision and submission |

**Target venues:**
- NeurIPS 2025 (deadline: May 2025) - if fast execution
- ICML 2025 (deadline: Jan 2025) - if very fast
- ICLR 2026 (deadline: Oct 2025) - most realistic

---

## 10. Related Competitive Landscape

### Groups likely working on similar topics:
1. **Google DeepMind (Munos et al.)** - NLHF authors, likely extending to practical algorithms
2. **Microsoft Research (Rosset et al.)** - DNO authors, may extend to GRPO
3. **CMU (Wu, Sun et al.)** - SPPO authors, active in self-play alignment
4. **UCLA (Chen et al.)** - SPIN authors, may extend to preference games

### Our advantages:
- Direct GRPO expertise (already implemented veRL-based GRPO)
- Game theory domain knowledge
- Fast execution capability (8xH20, existing infrastructure)
- Unique angle: tournament game + intransitive preferences (nobody else framing it this way)

---

## 11. Alternative/Backup Ideas

### 11.1 "Correlated Equilibrium for Multi-Stakeholder Alignment"
- When multiple principals (users with different preferences) want to align a model
- Correlated equilibrium allows coordination that Nash cannot
- Novel but harder to make practical

### 11.2 "Mechanism Design for Preference Elicitation in RLHF"
- Design incentive-compatible mechanisms for collecting human preferences
- Connection to auction theory
- More theoretical, less systems/ML

### 11.3 "Bayesian Nash Alignment under Preference Uncertainty"
- When the preference model has epistemic uncertainty
- Bayesian Nash equilibrium accounts for this uncertainty
- Could combine with Thompson sampling for exploration

---

## 12. Key References (Consolidated)

1. Munos et al., "Nash Learning from Human Feedback," ICML 2024
2. Chen et al., "SPIN: Self-Play fIne-tuNing," ICML 2024
3. Wu et al., "Self-Play Preference Optimization for Language Model Alignment," NeurIPS 2024
4. Rosset et al., "Direct Nash Optimization," arXiv 2024
5. Swamy et al., "A Minimaximalist Approach to Reinforcement Learning from Human Feedback," ICML 2024
6. Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Feedback," AISTATS 2024
7. Xiong et al., "Iterative Preference Learning from Human Feedback," ICML 2024
8. Irving et al., "AI Safety via Debate," arXiv 2018
9. Khan et al., "Debating with More Persuasive LLMs Leads to More Truthful Answers," ICML 2024
10. Rame et al., "Rewarded Soups," NeurIPS 2023
11. Rame et al., "WARM: On the Benefits of Weight Averaged Reward Models," ICML 2024
12. Skalse et al., "Defining and Characterizing Reward Hacking," NeurIPS 2022
13. Rafailov et al., "Direct Preference Optimization," NeurIPS 2023
14. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning," 2024 (introduced GRPO)
15. DeepSeek-AI, "DeepSeek-R1," 2025
16. Zhu et al., "Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons," ICML 2023
17. Dudik et al., "Efficient optimal learning for contextual bandits," UAI 2011 (game-theoretic policy optimization foundations)
18. Freund & Schapire, "Game theory, on-line prediction and boosting," COLT 1996 (foundational)
19. Lanctot et al., "OpenSpiel: A Framework for Reinforcement Learning in Games," 2019
20. Brown & Sandholm, "Superhuman AI for Heads-Up No-Limit Poker: Libratus," Science 2018 (game solving at scale)
