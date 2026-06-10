# Research Report: Learning Game-Theoretic Strategies via LLM Self-Play with Emergent Communication

**Date:** 2026-06-10  
**Target Venues:** ICML 2026 / ICLR 2026 / NeurIPS 2026  
**Hardware:** 8x H20 GPUs | Models: Qwen2.5-3B/7B, LLaMA-3-8B

---

## Executive Summary

This report surveys the rapidly evolving landscape of LLM self-play, multi-agent game-theoretic training, and emergent communication (2023-2026). A clear gap exists: while recent work trains LLMs via self-play for single-objective improvement (reasoning, alignment) or evaluates them in games without training, **no existing work trains LLMs through multi-agent self-play to discover and internalize game-theoretic equilibrium concepts while simultaneously developing emergent communication protocols**. This gap represents a publishable contribution at a top venue if framed correctly.

---

## 1. Literature Review

### 1.1 LLM Self-Play for Alignment and Reasoning (Non-Game-Theoretic)

| Paper | Authors | Venue/Year | Key Contribution |
|-------|---------|------------|------------------|
| **SPIN: Self-Play Fine-Tuning** | Chen, Deng, Yuan, Ji (2401.01335) | ICML 2024 | LLM plays against copies of itself; generator vs. discriminator game for alignment without new human data |
| **SPPO: Self-Play Preference Optimization** | Wu, Sun, Yuan, Ji, Yang, Gu (2405.00675) | NeurIPS 2024 | Two-player constant-sum game for Nash equilibrium of preference; models alignment as minimax |
| **SPAG: Self-Playing Adversarial Language Game** | Cheng, Hu, Xu, Zhang (2404.10642) | ACL 2024 | Two-player adversarial taboo game; RL on game outcomes improves reasoning benchmarks broadly |
| **DeepSeek-R1** | DeepSeek-AI (2501.12948) | 2025 | Pure RL (GRPO) on reasoning without human labels; emergent CoT through self-play-like reward |
| **Search Self-Play (SSP)** | Lu, Wen, Cheng (2510.18821) | 2025 | Proposer-solver co-evolution for search agents; task difficulty curriculum via self-play |
| **SAGE: Multi-Agent Self-Evolution** | Peng, Zhu, Wei (2603.15255) | 2026 | Four-agent closed-loop (Challenger, Planner, Solver, Critic) for reasoning evolution |

**Key insight:** These works use "self-play" metaphorically (model vs. older copy) for single-agent improvement. They do NOT address genuine multi-agent strategic interaction where equilibrium concepts matter.

### 1.2 LLMs Evaluated in Game-Theoretic Settings (No Training)

| Paper | Authors | Venue/Year | Key Finding |
|-------|---------|------------|-------------|
| **Playing Repeated Games with LLMs** | Akata, Schulz, Coda-Forno et al. (2305.16867) | NeurIPS 2023 Workshop | LLMs cooperate in PD but fail at coordination games; first systematic evaluation |
| **Can LLMs Serve as Rational Players?** | Fan, Chen, Jin, He (2312.05488) | 2023 | Systematic analysis of LLM rationality boundaries in games; shows consistent failures |
| **GTBench** | Duan, Zhang, Diffenderfer, Kailkhura (2402.12348) | NeurIPS 2024 | 10-game benchmark; LLMs show poor strategic reasoning across board/card games |
| **What Suppresses Nash Equilibrium Play in LLMs** | Lekeas, Stamatopoulos (2604.27167) | 2026 | **Critical finding:** LLMs compute Nash internally but a prosocial override in final layers suppresses it; probing shows 96% history encoding but only 56% Nash encoding |
| **Tacit Coordination of LLMs** | Aharon, La Malfa, Wooldridge (2601.22184) | 2026 | LLMs coordinate via focal points; outperform humans in some coordination games |
| **The Illusion of Rationality** | Rios, Manrique, Quijano (2512.09254) | 2025 | LLMs show tacit biases in negotiation games; strategic behavior is not well understood |
| **Counterparty Modeling is Not Strategy** | Cosentino et al. (2605.16575) | 2026 | LLMs can model counterparty preferences but fail to convert this into strategic advantage |

**Key insight:** Extensive evaluation literature, but these papers diagnose problems without solving them through training. Lekeas & Stamatopoulos (2604.27167) is particularly relevant -- they show LLMs have Nash competence but suppress it, and use Qwen2.5 + Llama-3.

### 1.3 Training LLMs for Strategic Multi-Agent Interaction

| Paper | Authors | Venue/Year | Key Contribution |
|-------|---------|------------|------------------|
| **GameTalk** | Vendrell, Luyten, van der Schaar (2601.16276) | 2026 | **Most relevant.** Trains LLMs for strategic conversation via GRPO/DPO/STaR across multi-turn games. DPO works best. Focus: coordination and negotiation games |
| **MindGames Arena (In2AI)** | Korshuk, Buyantuev, Makarov (2606.00017) | NeurIPS 2025 | 8B model trained with delayed reward attribution beats GPT-5 in multi-agent strategic games; uses vLLM + curriculum opponent sampling |
| **Safe Equilibrium Policy Optimization (SEPO)** | Arumugam et al. (2605.30854) | 2026 | GRPO + explicit equilibrium penalties (exploitability, collusion, externality cost) on Qwen-3.5-4B; trains for equilibrium play in PD, auctions, Kuhn Poker |
| **Stronger-MAS (AT-GRPO)** | Zhao, Hu, Wang (2510.11062) | 2025 | Agent- and turn-wise GRPO for multi-agent systems; massive gains on planning/reasoning |
| **ToMPO** | Zhang, Chen, Kong (2509.21134) | 2025 | Theory of Mind policy optimization; rollouts conditioned on others' strategies; outperforms GRPO by 35% |
| **LSPO (Werewolf)** | Xu, Gu, Yu (2502.04686) | 2025 | Maps language to latent strategy space, applies CFR, then DPO fine-tunes LLM; iterative self-play in Werewolf |
| **MaKTO (Werewolf)** | Ye, Zhang, Zhang (2501.14225) | 2025 | Multi-agent KTO on Werewolf; 61% win rate; beats GPT-4o by 23% |
| **TRACER** | Li, Liu, Zhou (2605.28699) | 2026 | Turn-level regret matching for cooperative multi-LLM reasoning; game-theoretic convergence guarantees |
| **COvolve** | Sygkounas, Hazra, Persson (2603.28386) | 2026 | Co-evolutionary LLM framework; computes mixed-strategy Nash equilibrium for meta-policy |
| **Toward Optimal LLM Alignment as Two-Player Games** | Zheng, Guo, Liu (2406.10977) | 2024 | Adversarial-defensive agent game for alignment; proves convergence to Nash; improves generalization |
| **ALSO** | Li, Yi, Kong (2605.15768) | 2026 | Adversarial online strategy optimization; formulates multi-turn social interaction as adversarial bandits |

### 1.4 Diplomacy and Complex Communication Games

| Paper | Authors | Venue/Year | Key Contribution |
|-------|---------|------------|------------------|
| **Diplodocus (No-Press Diplomacy)** | Bakhtin, Wu, Lerer (Meta) (2210.05492) | NeurIPS 2022 | Self-play RL regularized toward human policy; ranked #1 among humans |
| **Welfare Diplomacy** | Mukobi, Erlebach, Lauffer (2310.08901) | 2023 | General-sum variant of Diplomacy; benchmarks cooperation |
| **Dynamic Coalition Detection in Diplomacy** | Kulkarni et al. (2502.16339) | 2025 | LLM + game theory for predicting coalition formation from natural language |
| **AgenticPay** | Liu, Gu, Song (2602.06008) | 2026 | Multi-agent buyer-seller negotiation benchmark with private constraints |
| **Strategic Persuasion (Courtroom)** | Siedler (2604.07028) | 2026 | Trait-conditioned LLM agents in adversarial legal argumentation |

### 1.5 Emergent Communication (Classical + LLM Era)

| Paper | Authors | Venue/Year | Key Contribution |
|-------|---------|------------|------------------|
| **Multi-Agent Cooperation and Emergence of Language** | Lazaridou, Peysakhovich, Baroni (1612.07182) | ICLR 2017 | Foundational: agents develop communication protocols in referential games |
| **Emergent Communication of Generalizations** | Mu, Goodman (2106.02668) | NeurIPS 2021 | Communicating abstract concepts improves systematicity and compositionality |
| **Learning to Communicate with Strangers** | Cope, Schoots (2104.09557) | 2021 | Channel randomization for zero-shot communication with novel partners |
| **Policy Learning with a Language Bottleneck (PLLB)** | Srivastava, Colas, Sadigh (2405.04118) | NeurIPS 2024 | **Critical bridge paper.** Agents generate linguistic rules capturing strategies; alternates rule generation + policy learning; tested on signaling games |
| **NomicLaw** | Hota, Jokinen (2508.05344) | 2025 | LLMs spontaneously form alliances, betray trust, adapt rhetoric in collaborative lawmaking |
| **LLM Policy Synthesis (Social Dilemmas)** | Gallego (2603.19453) | 2026 | LLM generates Python policies for social dilemmas; dense social metrics as coordination signal |
| **Incentivizing Truthful LMs via Peer Elicitation Games** | Chen, Zhu, Han (2505.13636) | 2025 | Game-theoretic framework for truthful LLM alignment without ground truth |

### 1.6 Additional Relevant Context

| Paper | Authors | Year | Relevance |
|-------|---------|------|-----------|
| **Strategist** | Light, Cai, Chen (2408.10635) | 2024 | LLM bi-level tree search for self-improvement in strategic games |
| **YOLO-MARL** | Zhuang, Shen, Zhang (2410.03997) | 2024 | LLM generates high-level coordination once, then MARL executes |
| **SCAR (Shapley for RLHF)** | Cao, Zhang, Chang (2505.20417) | 2025 | Shapley values for credit assignment in cooperative RL for LLMs |
| **Efficient Exploration for Nash Preference Optimization** | Nan, Li, Kroer (2606.01382) | 2026 | Theoretical foundations for iterative Nash learning in LLM alignment |
| **Liar's Poker (Solly)** | Dewey, Botyanszki, Moallemi (2511.03724) | 2025 | Self-play RL (not LLM-based) masters multi-player poker with deception |

---

## 2. Gap Analysis

### Gap 1: Self-Play for Equilibrium Learning (Not Just Winning)

**Current state:** GameTalk, MindGames Arena, and SEPO train LLMs to win at games. But:
- GameTalk optimizes global reward, not equilibrium concepts specifically
- MindGames Arena focuses on win rate, not strategic understanding  
- SEPO adds equilibrium penalties but as reward shaping, not as an internalized capability

**The gap:** No work trains LLMs to *understand and apply* equilibrium concepts (Nash, correlated equilibrium, Pareto optimality) as transferable reasoning skills across game families.

### Gap 2: Emergent Communication in LLM Self-Play

**Current state:** 
- Classical emergent communication (Lazaridou 2017, Mu & Goodman 2021) uses small neural networks, not LLMs
- LLM multi-agent work (Werewolf, Diplomacy, negotiation) uses pre-existing natural language, not emergent protocols
- PLLB (Srivastava 2024) is closest -- but generates rules post-hoc, not through interactive self-play

**The gap:** No work studies how LLMs develop novel communication strategies (signaling, commitment, deception detection) through multi-agent self-play training. When two LLMs play a game with a communication channel, what protocols emerge? Do they converge to theoretically optimal signaling equilibria?

### Gap 3: Communication as a Mechanism for Equilibrium Selection

**Current state:** Lekeas (2604.27167) shows LLMs suppress Nash play due to prosocial biases. Aharon (2601.22184) shows LLMs use focal points for coordination. But:

**The gap:** Nobody has studied whether allowing LLMs to communicate during self-play enables them to reach better equilibria (e.g., correlated equilibria via pre-play communication, or Pareto-optimal Nash through signaling). This connects to Aumann's correlated equilibrium theorem -- cheap talk can expand the equilibrium set.

### Gap 4: Transfer of Game-Theoretic Reasoning Across Games

**Current state:** Most training is game-specific (Werewolf, negotiation, specific matrix games). 

**The gap:** Can self-play in a curriculum of games with communication produce a model that generalizes game-theoretic reasoning to unseen game structures?

---

## 3. Concrete Idea Formulation

### Title: **"Cheap Talk and Equilibrium Emergence: Training LLMs via Communicative Self-Play in Strategic Games"**

### Core Idea

Train two LLM agents to play a curriculum of strategic games (matrix games, sequential games, social dilemmas) where they can communicate before/during play. Use multi-agent GRPO to optimize both game outcomes AND communication quality simultaneously. Analyze:
1. What communication protocols emerge (signaling? commitment? bluffing?)
2. Do agents converge to equilibria that are unreachable without communication (correlated equilibria)?
3. Does the emergent strategic communication transfer across game types?

### Theoretical Grounding

From game theory:
- **Aumann's Correlated Equilibrium** (1974): Pre-play communication can enable payoff profiles outside the Nash equilibrium convex hull
- **Crawford & Sobel Signaling** (1982): Strategic information transmission with aligned/misaligned incentives  
- **Cheap Talk Equilibria** (Farrell & Rabin, 1996): When can talk be credible? When does it improve outcomes?

**Research Questions:**
1. Can LLM self-play with communication channels discover correlated equilibria without explicit game-theoretic supervision?
2. Do emergent communication protocols exhibit properties predicted by signaling game theory (partial pooling, babbling equilibria vs. informative equilibria)?
3. Does communicative self-play produce better game-theoretic reasoners than non-communicative self-play?
4. Does strategic communication transfer across game families?

---

## 4. Why This Is Novel and Publishable

### Novelty Arguments

1. **First to combine emergent communication + LLM self-play training + game-theoretic analysis.** The three communities (emergent communication, LLM RL, computational game theory) have not intersected at this point.

2. **Theoretical connection to classical game theory.** Unlike pure engineering papers (GameTalk, MaKTO), this connects to Aumann's correlated equilibrium and Crawford-Sobel signaling -- giving it theoretical depth valued at ICML/ICLR.

3. **Going beyond evaluation to training.** The Lekeas (2604.27167) paper shows LLMs suppress Nash play. We provide a *training solution* that uses communication as the mechanism to overcome this suppression.

4. **Emergent communication at LLM scale.** All prior emergent communication work uses tiny networks. We show what happens when pre-trained LLMs (with existing language competence) develop strategic communication protocols -- a qualitatively different regime.

5. **Timely.** The MindGames Arena (NeurIPS 2025), GameTalk (2026), SEPO (2026) papers show the community is moving toward multi-agent LLM training for games RIGHT NOW. Adding the communication dimension is the natural next step.

### Positioning for Top Venues

- **ICML:** Emphasize the theoretical connection (convergence to correlated equilibria, regret bounds)
- **NeurIPS:** Emphasize the empirical novelty (emergent protocols, transfer learning, scaling)
- **ICLR:** Emphasize the representation learning angle (what strategic communication representations emerge)

---

## 5. Experimental Design

### Phase 1: Game Environment Suite

Design a suite of games with varying properties:

| Game Type | Communication Benefit | Equilibrium Concept | Examples |
|-----------|----------------------|---------------------|----------|
| Pure coordination | High (focal point selection) | Pareto-dominant Nash | Battle of Sexes, Stag Hunt |
| Mixed-motive with alignment | Medium (trust building) | Correlated equilibrium | Chicken, Commons Dilemma |
| Pure conflict | Low-None (only deception) | Minimax | Matching Pennies, Zero-sum |
| Sequential with signaling | High (commitment/threat) | Subgame-perfect via signaling | Ultimatum, Entry Deterrence |
| Incomplete information | Critical (type revelation) | Bayesian Nash / PBE | Auction, Market for Lemons |

### Phase 2: Training Framework

```
Architecture:
- Base: Qwen2.5-3B-Instruct (primary), Qwen2.5-7B-Instruct (scaling)
- Framework: veRL + GRPO (already in researcher's codebase)
- Communication: Free-form text channel (1-3 messages before action)

Training Loop (per iteration):
1. Sample game from curriculum
2. Agent A and Agent B (same model, different roles) communicate
3. Both select actions  
4. Compute rewards based on game outcome
5. Multi-agent GRPO update (AT-GRPO style, per Stronger-MAS)

Reward Design:
- r_outcome: Game-theoretic outcome (payoff achieved)
- r_equilibrium: Bonus for reaching Pareto-optimal equilibrium (optional ablation)
- r_communication: Mutual information between messages and actions (intrinsic reward for meaningful communication)
```

### Phase 3: Analysis and Metrics

1. **Equilibrium convergence:** Track what fraction of game outcomes correspond to Nash, correlated equilibrium, or Pareto-optimal outcomes across training
2. **Communication analysis:**
   - Message informativeness (mutual information with game state/intended action)
   - Protocol emergence (does a consistent signaling convention develop?)
   - Deception rate (in misaligned games, do agents learn to bluff?)
3. **Transfer evaluation:** Train on game set A, evaluate on held-out game set B
4. **Comparison baselines:**
   - No-communication self-play (same GRPO, no message channel)
   - Fixed communication (pre-defined protocol, e.g., "announce your intended action")
   - Prompted (no training, just instruction to communicate strategically)
5. **Mechanistic analysis:** Probe trained models for equilibrium concept encoding (building on Phase 1 of existing project)

### Phase 4: Scaling Experiments

- 3B vs 7B (same compute budget)
- Communication length (0, 1, 3, 5 messages)
- Curriculum vs. single-game training
- Self-play vs. cross-play (play against different model sizes)

### Resource Budget (8x H20 GPUs)

| Component | GPU Hours (est.) | Config |
|-----------|-----------------|--------|
| veRL GRPO training (3B, 10 games, 5000 steps) | 24h x 8 GPUs | Full |
| veRL GRPO training (7B, 10 games, 5000 steps) | 48h x 8 GPUs | Full |
| Ablations (no-comm, fixed-comm) | 24h x 8 GPUs each | Full |
| Evaluation + probing | 8h x 4 GPUs | Partial |
| **Total estimate** | ~200 GPU-hours on H20 | Feasible |

H20 has 96GB HBM3, sufficient for Qwen2.5-7B full training with GRPO (needs ~60GB for model + optimizer + rollout buffer with gradient checkpointing).

---

## 6. Risks and Mitigations

### Risk 1: Communication degenerates to babbling or copying
**Likelihood:** Medium  
**Mitigation:** 
- Add mutual information intrinsic reward for communication
- Use information-theoretic regularization (prevent trivial protocols)
- Curriculum: start with games where communication is necessary (coordination games), then add mixed-motive games

### Risk 2: Models converge to simple "announce my action" instead of sophisticated strategies
**Likelihood:** High for pure coordination  
**Mitigation:**
- Include incomplete information games where truthful announcement is not equilibrium
- Analyze games where communication with misaligned incentives creates the signaling tension
- Track protocol sophistication metrics (beyond simple action announcement)

### Risk 3: Training instability in multi-agent GRPO
**Likelihood:** Medium (known issue per Stronger-MAS)  
**Mitigation:**
- Use AT-GRPO (agent- and turn-wise grouping from Stronger-MAS, 2510.11062)
- Delayed reward attribution (from MindGames Arena, 2606.00017)
- Population-based training with diverse opponents to avoid cycling

### Risk 4: Results not clearly better than prompted baselines
**Likelihood:** Low-Medium  
**Mitigation:**
- Focus on games where prompted models demonstrably fail (mixed strategies, correlated equilibria)
- Use the Lekeas (2604.27167) finding: LLMs suppress Nash without training, so training SHOULD help
- Emphasize qualitative novelty (emergent protocols) even if quantitative gains are modest

### Risk 5: Scope too broad for one paper
**Likelihood:** High  
**Mitigation:**
- Core paper: Focus on 2-player matrix games with pre-play communication
- Extension: Sequential games and N-player as future work
- Keep game suite small (5-6 canonical games) but analysis deep

### Risk 6: Reviewers ask "why not just prompt better?"
**Mitigation:**
- Include strong prompting baselines (CoT, role-play, game-theory expert persona)
- Show training produces qualitatively different behavior (novel protocols not in training data)
- Demonstrate transfer: trained model generalizes to new games without game-specific prompting

---

## 7. Concrete Timeline (3-4 months)

| Week | Milestone |
|------|-----------|
| 1-2 | Design game suite; implement communication channel in veRL |
| 3-4 | Implement multi-agent GRPO with communication; baseline experiments |
| 5-6 | Main experiments: communicative vs. non-communicative self-play |
| 7-8 | Analysis: equilibrium convergence, protocol emergence |
| 9-10 | Transfer experiments; scaling (3B vs 7B) |
| 11-12 | Ablations; mechanistic probing of trained models |
| 13-14 | Writing; additional experiments from reviewer perspective |

---

## 8. Related Work Comparison Table

| Dimension | GameTalk | SEPO | MaKTO | SPAG | **Ours (proposed)** |
|-----------|----------|------|-------|------|---------------------|
| Multi-agent training | Yes | Yes | Yes | No (self-play) | Yes |
| Communication channel | Implicit (conversation IS the game) | No | Yes (Werewolf speech) | No | **Explicit, analyzable** |
| Game-theoretic analysis | Minimal | Strong (exploitability) | No | No | **Strong (equilibrium theory)** |
| Emergent protocols | No (fixed language) | No | No | No | **Yes (core contribution)** |
| Equilibrium as objective | No (just reward) | Partial (penalty) | No | No | **Yes (correlated eq.)** |
| Transfer across games | No | No | No | Yes (reasoning) | **Yes (game families)** |
| Theoretical grounding | Minimal | Some | None | None | **Aumann, Crawford-Sobel** |

---

## 9. Key References to Cite

1. Akata et al. (2023) "Playing Repeated Games with LLMs" - Foundation evaluation
2. Lekeas & Stamatopoulos (2026) "What Suppresses Nash Equilibrium Play in LLMs" - Motivation (Nash suppression)
3. Vendrell et al. (2026) "GameTalk" - Closest training work
4. Cheng et al. (2024) "SPAG" - Self-play for LLM improvement
5. Arumugam et al. (2026) "SEPO" - Equilibrium-aware training
6. Zhao et al. (2025) "Stronger-MAS (AT-GRPO)" - Multi-agent GRPO algorithm
7. Korshuk et al. (2026) "MindGames Arena" - State-of-art in multi-agent LLM game training
8. Lazaridou et al. (2017) "Emergence of Language in Multi-Agent Communication" - Classical emergent communication
9. Mu & Goodman (2021) "Emergent Communication of Generalizations" - Protocol analysis methodology
10. Srivastava et al. (2024) "PLLB" - Language bottleneck for strategy learning
11. Xu et al. (2025) "LSPO" - Latent strategy space + game theory + LLM fine-tuning
12. Wu et al. (2024) "SPPO" - Nash equilibrium framing for LLM alignment
13. Bakhtin et al. (2022) "Diplodocus" - Human-regularized self-play for cooperative games
14. Aumann (1974) "Subjectivity and Correlation in Randomized Strategies" - Correlated equilibrium theory
15. Zhang et al. (2025) "ToMPO" - Theory of mind for strategic decisions

---

## 10. One-Paragraph Elevator Pitch

Large language models demonstrably fail at game-theoretic reasoning, and recent mechanistic work shows they *compute* Nash equilibria internally but *suppress* them. We propose training LLMs through communicative self-play: two copies of the model play strategic games while exchanging messages through a natural language channel, optimized via multi-agent GRPO. We show that (1) emergent communication protocols spontaneously arise that resemble theoretically predicted signaling equilibria, (2) communication enables agents to coordinate on correlated equilibria unreachable without communication, and (3) strategic reasoning transfers across game families. This work bridges three communities -- emergent communication, LLM RL, and computational game theory -- with strong theoretical grounding in Aumann's framework and practical feasibility on standard academic hardware.
