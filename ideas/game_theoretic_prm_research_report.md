# Game-Theoretic Process Reward Models: Using Formal Game Solutions as Dense Supervision for LLM Reasoning

## Deep Research Report for ICML/ICLR/NeurIPS 2025-2026 Submission

---

## Executive Summary

Process Reward Models (PRMs) have emerged as one of the most impactful techniques for improving LLM reasoning, but they suffer from two fundamental limitations: (1) expensive human annotation of step-level correctness, and (2) domain restriction to mathematics. This report proposes **Game-Theoretic Process Reward Models (GT-PRM)** — leveraging the formal, algorithmically verifiable structure of game theory to generate dense, step-level reward signals automatically. Game theory is uniquely suited because every intermediate reasoning step (identify game structure, eliminate dominated strategies, compute best responses, find Nash equilibria) is formally verifiable without human annotation. This creates an ideal testbed for studying process supervision while also advancing LLM strategic reasoning capabilities.

---

## 1. Literature Review: Process Reward Models and Dense Supervision

### 1.1 Foundational PRM Work

**"Let's Verify Step by Step"**
- Authors: Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe
- Venue: ICLR 2024 (originally arXiv 2023)
- Key contribution: Trained a Process Reward Model (PRM800K) using 800K human-labeled step-level correctness judgments on math solutions. Demonstrated that process-based supervision significantly outperforms outcome-based supervision for best-of-N selection on MATH benchmark. Achieved 78.2% on MATH with majority voting.
- Limitation: Requires expensive human annotation (~$25/hour labelers marking each step as correct/neutral/incorrect).

**"Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"**
- Authors: Wang, Li, Shi, Lu, Chen, Yan, Lu (Microsoft Research)
- Venue: ACL 2024
- Key contribution: Automated process reward labeling by using Monte Carlo Tree Search (MCTS)-style rollouts. For each intermediate step, they complete multiple rollouts to final answers and estimate step correctness as the fraction that reach correct final answers. Eliminates human annotation entirely.
- Method: Given a partial solution $s_1, ..., s_k$, perform N completions. Step $s_k$ is labeled correct if $\geq \theta$ fraction of completions reach the correct answer.
- Results: Matches or exceeds human-annotated PRM on GSM8K and MATH while being fully automated.
- Limitation: Computationally expensive (many rollouts per step); requires an outcome verifier (correct final answer); only applicable to domains with verifiable final answers.

**"OmegaPRM: Improve Mathematical Reasoning in Language Models by Automated Process Supervision"**
- Authors: Luo, Sun, Huang, et al. (Google DeepMind / Tsinghua)
- Venue: arXiv 2024 (submitted to top venue)
- Key contribution: Extends Math-Shepherd with a more efficient tree search strategy. Uses a divide-and-conquer approach to binary search for the first incorrect step, reducing rollout costs by ~5x. Trains PRM on 1.5M automatically labeled step-level annotations.
- Results: Combined with Best-of-N selection, achieves state-of-the-art on MATH (>90% with Gemini backbone).

**"GenPRM: Scaling Test-Time Compute with Generative Process Reward Models"**
- Authors: Google DeepMind, 2025
- Key contribution: Instead of scalar step rewards, trains a generative model that produces natural language critiques of each step. Enables chain-of-thought verification. Combines PRM with generative critique, allowing more nuanced step evaluation.

### 1.2 Outcome-Based vs Process-Based Reward

**"Outcome-Based vs Process-Based Reward for RL Training"**
- The debate centers on ORM (Outcome Reward Model) vs PRM (Process Reward Model):
  - **ORM**: Binary signal — was the final answer correct? Sparse, delayed reward.
  - **PRM**: Per-step signal — was each reasoning step correct? Dense, immediate reward.
- Key findings from the literature:
  - Uesato et al. (2022), "Solving Math Word Problems with Process- and Outcome-Based Feedback" (DeepMind): First rigorous comparison. PRM significantly outperforms ORM for best-of-N selection, but the gap narrows when used for RL training.
  - Wang et al. (2024), "Math-Shepherd": Process rewards are especially beneficial for longer reasoning chains where credit assignment is harder.
  - Havrilla et al. (2024), "Teaching Models to Verify their Own Solutions with Dense Rewards": Dense step-level rewards accelerate RL convergence by 3-5x compared to sparse outcome rewards.

**"Reinforcement Learning from Process Feedback" (various groups, 2024)**
- Multiple groups showed that using PRM scores as dense rewards in PPO/GRPO training outperforms outcome-only rewards:
  - Faster convergence (fewer RL steps needed)
  - Better sample efficiency
  - Reduced reward hacking (model learns correct reasoning, not just answer-matching)

### 1.3 Formal Verification as Reward Signal

**"Lean-STaR / AlphaProof" (DeepMind, 2024)**
- Uses formal theorem provers (Lean 4) to verify mathematical proofs step-by-step
- Each proof step is formally verified: provides perfect binary reward (proof accepted or rejected)
- Limitation: Requires formalizing problems in a theorem prover language; not scalable to arbitrary domains

**"MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data" (2024)**
- Generates formal math problems with verifiable proof steps
- Uses the formal verification as training signal

**"Formal Verification as Reward" general paradigm:**
- Code execution (pass/fail unit tests) — used in CodeRL, RLTF
- Mathematical proofs (Lean/Coq/Isabelle verification)
- Symbolic computation (SymPy verification of algebraic steps)
- **Gap identified**: No equivalent formal verification framework for strategic/game-theoretic reasoning

### 1.4 Dense Reward Signals for Reasoning

**"RLHF with Step-Level Rewards" / "Process Reward RLHF"**
- Setlur et al. (2024), "Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning" (CMU): Proposes automated process reward that measures "progress" toward the correct answer at each step.
- Sun et al. (2024), "Easy-to-Hard Generalization with Process Reward" (Berkeley): Shows PRMs trained on easy problems generalize to verify hard problems.

**"Reinforced Self-Training (ReST)" and variants:**
- ReST-EM (Google, 2024): Iterative self-training with outcome filtering
- STaR (Zelikman et al., 2022): Self-taught reasoner, but outcome-only
- These lack step-level signal — GT-PRM would address this

### 1.5 Game Theory + LLM Reasoning

**"GTBench: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations"**
- Authors: Duan et al.
- Venue: NeurIPS 2024
- Evaluates LLMs on 10 game-theoretic tasks; shows GPT-4 achieves only ~60% on simple normal-form games

**"TMGBench" (Text-based Matrix Game Benchmark)**
- 144 game scenarios with story-based formulations
- Reveals LLMs struggle with strategic reasoning even in simple settings

**"Can LLMs Reason about Game Theory?" (various 2024 papers)**
- Consistent finding: LLMs fail at:
  - Iterated dominance elimination
  - Mixed strategy computation  
  - Backward induction
  - Even identifying Nash equilibria in 3x3+ games

**"CoRY: Cognitive Reasoning of LLMs in Strategic Games" (NeurIPS 2024)**
- Multi-agent self-play fine-tuning
- Different approach: uses game playing experience, not formal step supervision

---

## 2. Gap Analysis

### 2.1 The PRM Annotation Bottleneck

| Method | Domain | Annotation Source | Cost | Scalability |
|--------|--------|-------------------|------|-------------|
| PRM800K (OpenAI) | Math | Human labelers | Very high ($) | Low |
| Math-Shepherd | Math | MCTS rollouts | High (compute) | Medium |
| OmegaPRM | Math | Efficient tree search | Medium (compute) | Medium-High |
| Code verification | Code | Unit test execution | Low | High |
| Lean/formal proofs | Proofs | Theorem prover | Low (per-step) | Low (formalization effort) |
| **GT-PRM (proposed)** | **Game theory** | **Algorithmic verification** | **Near-zero** | **Very high** |

### 2.2 Identified Gaps

**Gap 1: PRMs are domain-restricted to mathematics**
- All major PRM work (OpenAI, Math-Shepherd, OmegaPRM, GenPRM) focuses exclusively on mathematical reasoning
- No PRM exists for strategic/game-theoretic reasoning
- This is surprising because game theory offers *richer* intermediate structure than math

**Gap 2: No formally verifiable step-level rewards for strategic reasoning**
- Game theory has a formal, algorithmic solution procedure:
  1. Parse game structure (identify players, strategies, payoffs) -- verifiable
  2. Identify dominated strategies -- verifiable via payoff comparison
  3. Perform iterated elimination -- verifiable step-by-step
  4. Compute best responses -- verifiable via argmax of expected utility
  5. Find Nash equilibria -- verifiable via best-response intersection
- Each step is individually verifiable *without* human annotation and *without* expensive rollouts
- This is strictly easier than Math-Shepherd (which needs rollouts) and cheaper than PRM800K (which needs humans)

**Gap 3: No study of process rewards in formally structured multi-step reasoning beyond math**
- Math reasoning steps are often ad-hoc (many valid proof paths)
- Game-theoretic solution has a *canonical* step sequence — making step-level verification unambiguous
- This canonical structure enables studying PRM properties that are confounded in math (where step correctness is path-dependent)

**Gap 4: Automated dense reward generation without rollouts**
- Math-Shepherd and OmegaPRM still require expensive rollouts to estimate step correctness
- In game theory, step correctness can be computed *directly* from the game matrix in O(1) per step
- This eliminates the computational overhead of existing automated PRM methods

**Gap 5: RL for game-theoretic reasoning is unexplored with dense rewards**
- GRPO and PPO have been applied to math reasoning with outcome rewards
- The researcher's own work shows GRPO training on game theory with outcome-only reward
- Dense game-theoretic process rewards for RL training have never been explored

---

## 3. Concrete Idea Formulation: GT-PRM

### 3.1 Core Proposal

**Game-Theoretic Process Reward Model (GT-PRM)**: A framework that automatically generates step-level reward signals from the formal structure of game-theoretic problems, using these as dense supervision for training LLMs via reinforcement learning (GRPO).

### 3.2 Key Insight

For a normal-form game, the solution procedure decomposes into a sequence of formally verifiable steps:

```
Step 1: Parse the payoff matrix correctly
        -> Verify: extracted payoffs match ground truth matrix

Step 2: Identify game properties (zero-sum? symmetric?)
        -> Verify: compare against ground truth game_type label

Step 3: Check for dominated strategies
        -> Verify: for each strategy s_i, check if there exists s_j 
           such that u(s_j, s_-i) >= u(s_i, s_-i) for all s_-i

Step 4: Perform iterated elimination of dominated strategies (IEDS)
        -> Verify: each elimination step is valid (removed strategy 
           is indeed dominated in the reduced game)

Step 5: Compute best response functions
        -> Verify: BR(s_-i) = argmax_{s_i} u(s_i, s_-i)

Step 6: Find Nash Equilibria (pure strategy)
        -> Verify: check that each player's strategy is a best response
           to the other's (mutual best response condition)

Step 7: Find Nash Equilibria (mixed strategy, if applicable)
        -> Verify: check indifference conditions and support conditions
```

**Each step has a formally computable correctness criterion that requires zero human judgment.**

### 3.3 Reward Function Design

For a generated reasoning trace $\tau = (s_1, s_2, ..., s_K)$, the GT-PRM assigns rewards:

$$r(s_k) = \begin{cases} +1 & \text{if step } s_k \text{ is formally correct} \\ -1 & \text{if step } s_k \text{ is formally incorrect} \\ 0 & \text{if step } s_k \text{ is not verifiable (e.g., natural language connector)} \end{cases}$$

**Process reward for GRPO**: The dense reward at each step enables credit assignment:

$$R_{\text{process}}(\tau) = \sum_{k=1}^{K} \gamma^{K-k} \cdot r(s_k)$$

where $\gamma$ is a discount factor emphasizing later (harder) steps.

**Comparison to outcome-only reward**:
$$R_{\text{outcome}}(\tau) = \mathbb{1}[\text{final answer is correct}]$$

### 3.4 Automatic Step Extraction and Verification

The system consists of:

1. **Step Parser**: Segment LLM-generated reasoning traces into discrete steps using structural markers (e.g., "First, ...", "Next, ...", "The dominated strategy is...", "The Nash equilibrium is...")

2. **Step Classifier**: Classify each step into a verification category:
   - `PARSE`: Extracting payoff values -> verify against ground truth
   - `DOMINANCE`: Claiming strategy X dominates Y -> verify via payoff comparison
   - `ELIMINATION`: Removing dominated strategy -> verify it was indeed dominated
   - `BEST_RESPONSE`: Computing BR -> verify via argmax
   - `EQUILIBRIUM`: Claiming NE -> verify mutual best response condition
   - `ARITHMETIC`: Numerical computation -> verify via symbolic computation
   - `CONNECTOR`: Natural language transition -> not verified

3. **Step Verifier**: For each classified step, apply the formal verification criterion:
   ```python
   def verify_dominance_claim(step, game_matrix):
       """Verify: 'strategy i dominates strategy j for player p'"""
       claimed_dominant = parse_strategy(step, "dominant")
       claimed_dominated = parse_strategy(step, "dominated")
       player = parse_player(step)
       # Check: u_p(dominant, s_{-p}) >= u_p(dominated, s_{-p}) for all s_{-p}
       for opponent_strategy in game_matrix.opponent_strategies(player):
           if payoff(player, claimed_dominant, opponent_strategy) < \
              payoff(player, claimed_dominated, opponent_strategy):
               return -1  # Incorrect dominance claim
       return +1  # Valid dominance claim
   ```

### 3.5 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    GT-PRM Training Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Generate game instances (automated, scalable)                │
│     - Random normal-form games (2x2, 3x3, 4x4, asymmetric)     │
│     - Compute ground truth: dominated strategies, BR, NE         │
│                                                                   │
│  2. Generate reasoning traces (policy model rollouts)            │
│     - LLM generates step-by-step solution attempts               │
│     - Parse into structured steps                                │
│                                                                   │
│  3. Compute step-level rewards (GT-PRM verifier)                 │
│     - Each step verified against formal game-theoretic criteria   │
│     - No human annotation, no rollouts needed                    │
│                                                                   │
│  4. Train with GRPO using dense rewards                          │
│     - Per-step rewards enable fine-grained credit assignment     │
│     - Advantage computed at step level, not trajectory level     │
│                                                                   │
│  5. Iterate: improved policy -> better traces -> better training │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Novelty and Publishability Analysis

### 4.1 Why This Is Novel

| Dimension | Prior Work | GT-PRM (Ours) |
|-----------|-----------|----------------|
| PRM domain | Math only | Game theory (new domain) |
| Step verification | Rollouts (expensive) or human | Direct algorithmic verification (free) |
| Reward density | Per-step (binary) | Per-step with rich type information |
| Step structure | Ad-hoc (many valid paths) | Canonical procedure (unique correct path) |
| Scalability | Limited by rollout/annotation cost | Unlimited (game generation is trivial) |
| RL integration | Best-of-N selection mostly | Direct GRPO integration with step rewards |
| Formal verification | Theorem provers (Lean) | Game-theoretic algorithms (much simpler) |

### 4.2 Contribution Claims (for paper)

1. **GT-PRM Framework**: First process reward model for game-theoretic reasoning with zero-cost automated step-level verification
2. **Dense Reward for Strategic RL**: First application of step-level process rewards to GRPO training for strategic reasoning
3. **Canonical Verification**: Demonstrate that domains with canonical solution procedures are ideal PRM testbeds (implications beyond game theory)
4. **Empirical Analysis**: Comprehensive comparison of outcome-only vs. process rewards for game-theoretic reasoning, ablating reward density, verification granularity, and step types
5. **Scalability Demonstration**: Show that GT-PRM scales to unlimited training data (unlike math PRMs bounded by problem availability)

### 4.3 Venue Fit

| Venue | Fit | Angle to Emphasize |
|-------|-----|-------------------|
| ICLR 2026 | Excellent | Learning + formal methods + reasoning |
| NeurIPS 2025 | Strong | Dense rewards + RL for reasoning |
| ICML 2025 | Strong | RL theory + process supervision |

### 4.4 Why Reviewers Would Accept

- **Timeliness**: PRMs are the hottest topic in LLM reasoning (2024-2025)
- **Formal rigor**: Game theory provides unambiguous ground truth (unlike math where multiple proof paths exist)
- **No expensive annotation**: Addresses the main limitation of PRM800K
- **No expensive rollouts**: Addresses the main limitation of Math-Shepherd/OmegaPRM
- **Builds on strong prior work**: Extends PRM paradigm to a new, well-motivated domain
- **Clean experimental setup**: Researcher already has GameSolve-Bench, GRPO infrastructure, 8xH20

---

## 5. Experimental Design

### 5.1 Research Questions

- **RQ1**: Does step-level process reward improve GRPO training for game-theoretic reasoning over outcome-only reward?
- **RQ2**: Which verification granularity is most effective? (coarse: correct/incorrect trace; medium: per-step binary; fine: per-step with type-specific reward)
- **RQ3**: Does GT-PRM training generalize to harder/unseen game types better than outcome-only training?
- **RQ4**: How does GT-PRM compare to the auxiliary probing approach (researcher's existing work)?
- **RQ5**: Does the canonical step structure of game theory make process rewards more effective than in math (where steps are non-canonical)?

### 5.2 Experimental Setup

**Models**: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct (primary), Qwen2.5-7B-Instruct (scale test)

**Training Data**: 
- GameSolve-Bench (existing 2,400 samples) + scaled to 20K via automated game generation
- Games: 2x2, 3x3, 4x4, 2x3, 3x2 normal-form games
- Properties: zero-sum, symmetric, general; pure/mixed/both equilibria

**Training Framework**: verl (researcher's existing GRPO infrastructure)

**Hardware**: 8x H20 96GB GPUs

### 5.3 Conditions (Main Experiments)

| Condition | Reward Type | Verification | Description |
|-----------|-------------|--------------|-------------|
| Baseline-ORM | Outcome only | Final answer | Standard GRPO with binary reward |
| GT-PRM-Binary | Process | Per-step correct/incorrect | Each step gets +1/-1 |
| GT-PRM-Typed | Process | Type-specific | Different weights for dominance/BR/NE steps |
| GT-PRM-Progressive | Process | Curriculum | Reward only later steps as training progresses |
| GT-PRM-Hybrid | Outcome + Process | Both | Weighted combination |
| SFT-CoT | Supervised | N/A | Fine-tune on correct CoT traces (baseline) |
| Aux-Probe (existing) | Auxiliary loss | Probing heads | Researcher's current method (comparison) |

### 5.4 Step-Level Reward Design Variants

**Variant A: Uniform step reward**
$$r_k = \mathbb{1}[\text{step } k \text{ is correct}]$$

**Variant B: Difficulty-weighted step reward**
$$r_k = w_{\text{type}(k)} \cdot \mathbb{1}[\text{step } k \text{ is correct}]$$
where $w_{\text{NE}} > w_{\text{BR}} > w_{\text{dominance}} > w_{\text{parse}}$

**Variant C: Progress-based step reward**  
$$r_k = \frac{k}{K} \cdot \mathbb{1}[\text{step } k \text{ is correct}]$$
(Later steps receive higher reward — finding NE is worth more than parsing)

**Variant D: Penalty-augmented**
$$r_k = \begin{cases} +w_k & \text{correct step} \\ -\alpha w_k & \text{incorrect step} \\ 0 & \text{non-verifiable step} \end{cases}$$

### 5.5 Evaluation Metrics

**In-Distribution (ID)**:
- GameSolve-Bench test set accuracy (by game size, type, difficulty)
- Step-level reasoning correctness (what fraction of intermediate steps are correct)
- Reasoning chain quality metrics (length, coherence, correctness of each type of step)

**Out-of-Distribution (OOD)** (critical for showing RL advantage):
- Larger games (5x5, 6x6) not seen in training
- Games with multiple equilibria (harder credit assignment)
- Sequential games (backward induction — structural generalization)
- Bayesian games (incomplete information)
- TMGBench (story-based game descriptions)
- GTBench (diverse game-theoretic tasks)

**Reward Model Quality**:
- Step-level F1 of the verifier (how accurately does it classify steps)
- Correlation between step rewards and final answer correctness

### 5.6 Ablation Studies

| Ablation | Question |
|----------|----------|
| Reward density (1, 3, 5, all steps) | How many verified steps are needed? |
| Step types verified (only NE, only dominance, all) | Which step types contribute most to learning? |
| Training data scale (2K, 5K, 10K, 20K) | How does GT-PRM benefit from scale? |
| Reward noise injection (5%, 10%, 20% random) | How robust is GT-PRM to verification errors? |
| GRPO vs PPO vs DPO with step rewards | Which RL algorithm pairs best with dense rewards? |
| Step reward vs token reward | Granularity: per-step vs per-token credit assignment |

### 5.7 Compute Budget (8x H20 96GB)

| Experiment | Estimated Time | GPU Usage |
|-----------|---------------|-----------|
| Data generation (20K games + solutions) | 2-3 hours | 8 GPUs (vLLM generation) |
| GRPO Baseline-ORM (3B model) | 4-6 hours | 8 GPUs (verl) |
| GT-PRM-Binary training (3B) | 4-6 hours | 8 GPUs (verl) |
| GT-PRM-Typed training (3B) | 4-6 hours | 8 GPUs (verl) |
| Full ablation suite (~12 conditions) | 48-72 hours | 8 GPUs |
| Evaluation (all checkpoints, ID + OOD) | 6-8 hours | 8 GPUs (vLLM) |
| **Total estimated** | **~4-5 days** | |

---

## 6. Technical Implementation Details

### 6.1 Step Parser Design

```python
class GameReasoningStepParser:
    """Parse LLM reasoning into verifiable steps."""
    
    STEP_PATTERNS = {
        'PARSE': r'payoff.*?(?:matrix|table)|player \d.*?(?:gets|receives|utility)',
        'DOMINANCE': r'(?:dominat|strictly better|weakly dominat)',
        'ELIMINATION': r'(?:eliminat|remov|cross out|discard)',
        'BEST_RESPONSE': r'(?:best response|BR|optimal.*?given|maximize)',
        'EQUILIBRIUM': r'(?:Nash|equilibri|NE|mutual best response)',
        'MIXED': r'(?:mixed strategy|probability|indifferen|randomiz)',
        'ARITHMETIC': r'(?:\d+\s*[+\-*/]\s*\d+|calculate|compute)',
    }
    
    def parse(self, reasoning_text: str) -> List[VerifiableStep]:
        """Segment reasoning into typed, verifiable steps."""
        ...
```

### 6.2 Step Verifier Design

```python
class GameTheoreticVerifier:
    """Verify each reasoning step against ground truth game structure."""
    
    def verify_step(self, step: VerifiableStep, game: NormalFormGame) -> float:
        if step.type == 'DOMINANCE':
            return self._verify_dominance(step, game)
        elif step.type == 'BEST_RESPONSE':
            return self._verify_best_response(step, game)
        elif step.type == 'EQUILIBRIUM':
            return self._verify_equilibrium(step, game)
        elif step.type == 'ELIMINATION':
            return self._verify_elimination(step, game)
        ...
    
    def _verify_dominance(self, step, game):
        """Check if claimed dominance relation is valid."""
        claimed = step.extract_dominance_claim()
        # Formally verify: u(s_dom, s_{-i}) >= u(s_dominated, s_{-i}) for all s_{-i}
        return game.check_dominance(
            player=claimed.player,
            dominant=claimed.dominant_strategy,
            dominated=claimed.dominated_strategy,
            strict=claimed.is_strict
        )
```

### 6.3 Integration with GRPO (verl framework)

The key modification to standard GRPO is replacing the scalar outcome reward with step-level rewards:

```python
# Standard GRPO reward (outcome only):
# reward = 1.0 if final_answer_correct else 0.0

# GT-PRM reward (process):
def compute_gt_prm_reward(response, game, verifier, parser):
    steps = parser.parse(response)
    step_rewards = []
    for step in steps:
        r = verifier.verify_step(step, game)
        step_rewards.append(r)
    
    # Aggregate: weighted sum with later steps worth more
    total_reward = sum(
        (i + 1) / len(steps) * r 
        for i, r in enumerate(step_rewards)
    )
    # Optionally add outcome bonus
    if final_answer_correct(response, game):
        total_reward += outcome_bonus
    
    return total_reward, step_rewards  # step_rewards for per-token credit assignment
```

### 6.4 Per-Token Advantage Estimation with Step Rewards

For GRPO, we can assign advantages at the step level rather than trajectory level:

$$A(s_k) = r(s_k) + \gamma V(s_{k+1}) - V(s_k)$$

Or more practically (GAE-style with step rewards):
$$\hat{A}_k = \sum_{t=k}^{K} (\gamma\lambda)^{t-k} \delta_t, \quad \delta_t = r_t + \gamma V_{t+1} - V_t$$

This provides much finer credit assignment than trajectory-level advantage.

---

## 7. Risks and Mitigations

### 7.1 Technical Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Step parsing accuracy**: LLM outputs may not neatly decompose into verifiable steps | High | Train a lightweight step parser (or use few-shot prompting to force structured output format); design prompt templates that encourage step-by-step output with clear markers |
| **Verification ambiguity**: Some steps may be partially correct or span multiple verification types | Medium | Use soft rewards (partial credit) instead of binary; classify steps conservatively (skip ambiguous ones) |
| **Reward hacking**: Model learns to game the step verifier (e.g., stating obviously correct but uninformative steps) | Medium | Add length penalty; require that steps make "progress" (new information); add diversity bonus |
| **GRPO instability with dense rewards**: More reward signal may cause training instability | Medium | Start with outcome-only, gradually increase process reward weight (reward curriculum); use reward normalization |
| **Limited improvement over SFT**: If SFT on CoT already achieves high accuracy, RL may show marginal gains | Medium | Focus on OOD generalization (where RL should excel); use harder games (5x5+) where SFT plateaus |

### 7.2 Conceptual Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **"Just another PRM paper"**: Reviewers may see this as incremental extension of Math-Shepherd to new domain | High | Emphasize unique properties: (1) zero-cost verification (no rollouts), (2) canonical step structure enables cleaner study of process rewards, (3) automated scaling, (4) comparison reveals when PRMs help most |
| **Game theory too niche**: Reviewers may not see broad impact | Medium | Frame as studying *formal verification as process reward* — game theory is the testbed, but insights generalize to any domain with verifiable intermediate steps (code, logic, planning) |
| **Researcher's probing work overlaps**: Need to differentiate from existing proposal | Low | GT-PRM is complementary: probing studies *what* the model encodes; GT-PRM provides *training signal* to improve reasoning. Can combine: GT-PRM rewards + probing-guided auxiliary loss |

### 7.3 Experimental Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Outcome reward is "good enough"**: Step rewards may not improve over outcome-only GRPO | High | Focus on (a) harder games where outcome reward has credit assignment problems, (b) sample efficiency (fewer training steps needed), (c) reasoning quality (not just final answer) |
| **Results don't transfer to larger models**: Effects may be scale-dependent | Medium | Run core experiments on 1.5B and 3B; validate key finding on 7B; report scaling trends |
| **Comparison to Math-Shepherd unfair**: Different domain makes direct comparison impossible | Low | Don't claim superiority over Math-Shepherd; position as a *study of when/why process rewards help*, using game theory as an ideal controlled testbed |

---

## 8. Relationship to Researcher's Existing Work

### 8.1 Synergies with Current Project

The researcher's existing work provides strong foundations:

| Existing Asset | How GT-PRM Uses It |
|----------------|-------------------|
| GameSolve-Bench (2,400 games) | Training/evaluation data |
| Game generator (automated) | Unlimited scaling of training games |
| GRPO training pipeline (verl) | Direct integration of step rewards |
| Probing results (Phase 1) | Understand which layers encode step-level concepts |
| OOD evaluation data | Already generated for generalization testing |
| vLLM evaluation scripts | Efficient batch evaluation |

### 8.2 GT-PRM vs. Auxiliary Probing (Differentiation)

| Aspect | Auxiliary Probing (current) | GT-PRM (proposed) |
|--------|---------------------------|-------------------|
| Signal type | Representation-level labels | Reasoning step correctness |
| When applied | During forward pass (internal) | After generation (reward) |
| Training algorithm | LoRA + aux loss | GRPO with dense rewards |
| Inference overhead | Zero (probes discarded) | Zero (verifier not needed at inference) |
| Information source | Game metadata (static labels) | Step-by-step solution quality (dynamic) |
| Motivation | Encode concepts better | Generate better reasoning |

**Key insight**: These are complementary. GT-PRM provides *behavioral* training signal (reason correctly step-by-step), while auxiliary probing provides *representational* training signal (encode concepts linearly). A combined approach could be a strong follow-up.

### 8.3 Potential Combined Paper Story

If the current probing project doesn't yield strong results alone, GT-PRM can be combined:

**Story**: "We first show that LLMs poorly encode game-theoretic concepts (probing analysis). We then show that step-level process rewards, derived from formal game structure, improve both the internal representations AND the external reasoning behavior. This provides evidence that process supervision shapes internal concept encoding."

---

## 9. Paper Outline (Draft)

### Title Options
1. "Game-Theoretic Process Reward Models: Zero-Cost Dense Supervision for Strategic Reasoning"
2. "Formally Verified Step Rewards: Game Theory as an Ideal Testbed for Process Supervision"
3. "GT-PRM: Leveraging Formal Game Solutions as Automated Process Rewards for LLM Training"

### Structure

1. **Introduction** (1.5 pages)
   - PRMs are transformative but expensive (human annotation or rollouts)
   - Game theory offers free, perfect step-level verification
   - We propose GT-PRM: first process reward model for strategic reasoning

2. **Related Work** (1.5 pages)
   - Process Reward Models (PRM800K, Math-Shepherd, OmegaPRM)
   - RL for LLM reasoning (GRPO, PPO, DPO)
   - Game-theoretic reasoning in LLMs
   - Formal verification as reward

3. **Method: GT-PRM** (2.5 pages)
   - Problem formulation
   - Step parsing and classification
   - Formal verification functions
   - Integration with GRPO training
   - Reward design variants

4. **Experimental Setup** (1.5 pages)
   - Data generation and benchmarks
   - Models and training details
   - Baselines and conditions
   - Evaluation metrics

5. **Results** (2.5 pages)
   - RQ1: Process vs outcome reward
   - RQ2: Reward granularity analysis
   - RQ3: OOD generalization
   - RQ4: Comparison to auxiliary probing
   - Ablation studies

6. **Analysis** (1 page)
   - When do step rewards help most? (game size, complexity)
   - What step types are most valuable for learning?
   - Scaling properties (data efficiency, model size)

7. **Discussion and Conclusion** (1 page)
   - Implications for PRM design in other domains
   - Limitations and future work

---

## 10. Timeline and Next Steps

### Immediate Actions (Week 1)
1. Implement `GameReasoningStepParser` — segment LLM outputs into typed steps
2. Implement `GameTheoreticVerifier` — formal verification for each step type
3. Design prompt templates that encourage structured step-by-step reasoning
4. Generate pilot data: 1K games with step-level annotations

### Short-term (Weeks 2-3)
5. Integrate GT-PRM reward into verl GRPO pipeline
6. Run pilot experiment: Outcome-only vs GT-PRM-Binary on Qwen2.5-3B
7. Iterate on step parser accuracy (test on 100 manually inspected traces)

### Medium-term (Weeks 3-5)
8. Full experimental suite (all conditions in Section 5.3)
9. Ablation studies
10. OOD evaluation
11. Scale to 7B model

### Paper Writing (Weeks 5-7)
12. Results analysis and visualization
13. Paper draft
14. Internal review and revision

---

## 11. Key References (Consolidated)

### Process Reward Models
1. Lightman et al. "Let's Verify Step by Step." ICLR 2024.
2. Wang et al. "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations." ACL 2024.
3. Luo et al. "Improve Mathematical Reasoning in Language Models by Automated Process Supervision (OmegaPRM)." arXiv 2024.
4. Zhang et al. "GenPRM: Scaling Test-Time Compute with Generative Process Reward Models." arXiv 2025.
5. Uesato et al. "Solving Math Word Problems with Process- and Outcome-Based Feedback." arXiv 2022.

### RL for LLM Reasoning
6. Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv 2024. (GRPO)
7. Havrilla et al. "Teaching Models to Verify their Own Solutions." arXiv 2024.
8. Setlur et al. "Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning." arXiv 2024.
9. Sun et al. "Easy-to-Hard Generalization with Process Reward." arXiv 2024.
10. Zelikman et al. "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022.

### Game Theory + LLMs
11. Duan et al. "GTBench: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations." NeurIPS 2024.
12. Gemp et al. "States as Strings as Strategies: Steering Language Models with Game-Theoretic Solvers." arXiv 2024.
13. Xu et al. "Magic: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration (CoRY)." NeurIPS 2024.

### Formal Verification + LLMs
14. Trinh et al. "Solving olympiad geometry without human demonstrations (AlphaGeometry)." Nature 2024.
15. DeepMind. "AlphaProof and AlphaGeometry 2." 2024.
16. Xin et al. "DeepSeek-Prover: Advancing Theorem Proving via Large Language Models." arXiv 2024.
17. Polu et al. "Formal Mathematics Statement Curriculum Learning." ICLR 2023.

### Dense Reward / Credit Assignment
18. Rafailov et al. "Direct Preference Optimization (DPO)." NeurIPS 2023.
19. Ahmadian et al. "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback." arXiv 2024.
20. Chan et al. "Dense Reward for Free in Reinforcement Learning from Human Feedback." ICML 2024.

---

## 12. Summary: Why This Paper Should Be Written

**The 30-second pitch:**

> Process Reward Models revolutionized math reasoning but require expensive human annotation (PRM800K) or Monte Carlo rollouts (Math-Shepherd). We observe that game theory — unlike math — provides *formally verifiable* intermediate reasoning steps at zero cost. Every step in game-theoretic reasoning (dominance elimination, best response computation, equilibrium finding) can be verified algorithmically without human judgment or completion rollouts. We build GT-PRM, the first process reward model for strategic reasoning, and show that dense, formally-verified step rewards dramatically outperform outcome-only rewards in GRPO training, especially for out-of-distribution generalization. This establishes game theory as an ideal controlled testbed for studying process supervision and demonstrates a scalable path for automated PRM construction in formally structured domains.

**Why it will get accepted:**
- Hot topic (PRMs, reward models, RL for reasoning)
- Clean theoretical motivation (formal verification = perfect reward)
- Strong experimental infrastructure (researcher already has everything)
- Clear novelty (first PRM for game theory; first zero-cost automated PRM without rollouts)
- Broad implications (methodology applies to any formally structured domain)
- Addresses a real limitation of existing PRM work (annotation cost)
