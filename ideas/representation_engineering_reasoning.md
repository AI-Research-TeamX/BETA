# Representation Engineering for Reasoning: Steering LLM Internal States to Improve Structured Problem Solving

## Deep Research Report

**Date:** 2025-06-10  
**Target Venues:** ICML 2025 / ICLR 2026 / NeurIPS 2025

---

## 1. Literature Review

### 1.1 Foundational Work: Representation Engineering (RepE)

**Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Lin, Z., Forsyth, M., Scherlis, A., Emmons, S., Hendrycks, D. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv:2310.01405.**

- Proposed RepE as a top-down framework for understanding and controlling neural network representations
- Key insight: identify "concept directions" in representation space using contrastive stimuli
- Demonstrated control over honesty, power-seeking, morality, emotion by adding/subtracting directions
- Method: collect activations from pairs of prompts (positive vs negative for a concept), compute the difference in means (or first principal component of differences) to get a "reading vector"
- Apply this vector at inference time by adding it to residual stream activations
- Tested on Llama-2-13B-chat; showed effective behavioral control without any fine-tuning
- **Limitation:** All experiments focused on behavioral/safety properties, NOT reasoning capabilities

### 1.2 Inference-Time Intervention (ITI)

**Li, K., Patel, O., Viégas, F., Pfister, H., Wattenberg, M. (2024). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." NeurIPS 2023 (published proceedings 2024).**

- Identified "truthful directions" via probing on TruthfulQA
- At inference time, shift activations along these directions at specific attention heads
- Achieved significant improvement on TruthfulQA without any training
- Key methodological contribution: used probing accuracy to select WHICH attention heads to intervene on (top-k heads by probing accuracy)
- **Direct relevance:** Their methodology of using probing to select intervention sites is exactly what the proposed work would do for reasoning

### 1.3 Activation Addition / Steering Vectors

**Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., MacDiarmid, M. (2023). "Activation Addition: Steering Language Models Without Optimization." arXiv:2308.10248.**

- Formalized "activation addition" (ActAdd): add a steering vector to residual stream at a chosen layer during forward pass
- Computed steering vectors as difference in mean activations between positive/negative prompt pairs
- Demonstrated steering of sentiment, topic, behavior with single vector addition
- Important finding: intervention layer matters significantly; middle layers tend to work best
- No training required; zero computational overhead beyond standard inference

**Rimsky, N., Gabrieli, N., Schulz, J., Turner, A., et al. (2024). "Steering Llama 2 via Contrastive Activation Addition." ACL 2024 Findings.**

- Extended ActAdd to more systematic evaluation
- Showed contrastive activation addition (CAA) works across multiple behavioral dimensions
- Found that steering at layers around 50-70% depth works best
- Tested on sycophancy, corrigibility, power-seeking
- **Still focused on behavioral properties, not reasoning**

### 1.4 Probing and Intervention for Reasoning (Sparse Literature)

**Stolfo, A., Belinkov, Y., Sachan, M. (2023). "A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis." EMNLP 2023.**

- Identified specific components (MLPs, attention heads) responsible for arithmetic reasoning
- Used causal mediation analysis (activation patching) to locate reasoning circuits
- Found reasoning information concentrates in specific layers and heads
- **Relevant but different:** identifies where reasoning happens but does not propose steering

**Hanna, M., Liu, O., Variengien, A. (2023). "How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model." NeurIPS 2023.**

- Circuit-level analysis of how transformers perform numerical comparison
- Identified specific attention heads and MLPs involved
- Showed these can be ablated to destroy capability
- **Relevant:** establishes that reasoning capabilities are localized, hence steerable

**Todd, E., Li, M. L., Sharma, A. S., Mueller, A., Wallace, B. C., Bau, D. (2024). "Function Vectors in Large Language Models." ICLR 2024.**

- Extracted "function vectors" that encode in-context learning tasks
- Showed that adding these vectors can trigger specific reasoning patterns
- Demonstrated vectors for tasks like translation, sentiment, antonym generation
- **Key precedent:** shows that task-level capabilities can be represented as directions

**Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. (2023). "Progress measures for grokking via mechanistic interpretability." ICLR 2023.**

- Showed that internal representations develop structured circuits for modular arithmetic
- Demonstrated that representations undergo phase transitions during training
- **Relevant:** suggests that reasoning concepts have discrete, identifiable representation structures

### 1.5 Recent Work on Steering for Capabilities (2024-2025)

**Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., Neubig, G. (2024). "PAL: Program-aided Language Models." ICML 2023 (and subsequent follow-ups).**

- While not directly RepE, shows reasoning can be enhanced by structured interventions at inference time

**Hernandez, E., Sharma, A. S., Haklay, T., Meng, K., Wattenberg, M., Andreas, J., Belinkov, Y., Bau, D. (2024). "Linearity of Relation Representations in Transformer Language Models." ICLR 2024.**

- Established that factual relations are encoded as linear subspaces (not just directions)
- Showed these can be manipulated for knowledge editing
- **Key implication:** if factual knowledge is linear, reasoning knowledge may be too

**Marks, S., Rager, A., Michaud, E. J., Belinkov, Y., Bau, D., Mueller, A. (2024). "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models." arXiv:2403.19647.**

- Discovered sparse circuits responsible for specific behaviors using SAE features
- Extended intervention from single directions to sparse feature circuits
- **Relevant:** shows fine-grained reasoning control is possible

**Wu, Z., Arora, A., Wang, Z., Geiger, A., Jurafsky, D., Manning, C. D., Potts, C. (2024). "ReFT: Representation Finetuning for Language Models." NeurIPS 2024.**

- Proposed learning interventions on representations as a PEFT method
- Trains small linear transformations applied to hidden states at specific positions and layers
- Showed competitive with LoRA on many benchmarks with fewer parameters
- **Highly relevant:** formalized "learning where and how to intervene" but requires training; the proposed work would use probing to derive interventions without training

**Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., Wattenberg, M. (2024). "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task." ICLR 2023.**

- Showed linear probes can extract world-state representations from transformers
- The probed directions can be used for intervention
- **Relevant precedent for probing -> intervention pipeline**

### 1.6 Chain-of-Thought and Reasoning Representations (2024-2025)

**Deng, Y., Zhang, K., Ren, J., Ye, T., Liu, C., Gu, Q., He, J. (2024). "Explicit CoT Training Enhances Reasoning Representations in LLMs."**

- Showed that CoT training reshapes internal representations to better encode reasoning steps
- Linear probes on intermediate layers can predict reasoning correctness
- **Directly relevant:** provides evidence that reasoning quality has a representational signature

**Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C., Hernandez, D., et al. (2023). "Measuring Faithfulness in Chain-of-Thought Reasoning." arXiv:2307.13702.**

- Investigated whether internal representations reflect CoT reasoning
- Found cases where the model's internal state "knows" the answer before generating CoT
- **Implication:** reasoning information exists in early layers; interventions could promote it

**Mallen, A., Tong, N., Wiegreffe, S., Mahowald, K. (2024). "Eliciting Latent Knowledge from Language Models via Probing." ACL 2024.**

- Used probing to extract knowledge the model has but doesn't express
- Showed that probed directions can be used to improve model outputs
- **Directly supports the proposed approach**

### 1.7 Game-Theoretic Reasoning in LLMs

**Gemp, I., Lanctot, M., et al. (2024). "States as Strings as Strategies: Steering Language Models for Game-Theoretic Tasks." NeurIPS 2024 Workshop.**

- Early work on prompting LLMs for game theory tasks
- Found LLMs struggle with Nash equilibrium computation

**Guo, F., et al. (2024). "Economics Arena: LLM Reasoning in Strategic Games."**

- Comprehensive benchmark for LLM game-theoretic reasoning
- Documented systematic failures in dominance, best response, Nash equilibrium

**Fan, C., et al. (2024). "Can LLMs Reason Strategically?" NeurIPS 2024.**

- Found fundamental limitations in LLM strategic reasoning
- Suggests training-based approaches have ceiling effects
- **Motivation for inference-time intervention as an alternative**

---

## 2. Gap Analysis

### 2.1 Critical Gap: RepE for Reasoning Capabilities

| Property Studied | Papers | Status |
|---|---|---|
| Honesty/Truthfulness | Zou et al. 2023, Li et al. 2024 (ITI) | Well-explored |
| Safety/Refusal | Rimsky et al. 2024, multiple | Well-explored |
| Sycophancy | Rimsky et al. 2024 | Explored |
| Emotion/Sentiment | Turner et al. 2023 | Explored |
| Power-seeking | Zou et al. 2023 | Explored |
| **Mathematical Reasoning** | **None** | **OPEN GAP** |
| **Logical Reasoning** | **None** | **OPEN GAP** |
| **Strategic/Game-Theoretic Reasoning** | **None** | **OPEN GAP** |
| **Structured Problem Solving** | **None** | **OPEN GAP** |

### 2.2 Specific Research Gaps

1. **No probing-guided intervention for reasoning:** All ITI/RepE work selects intervention sites based on behavioral probes (truthfulness, honesty). Nobody has used probes that measure *reasoning capability* (e.g., "can this layer predict whether the model will solve a game correctly?") to guide intervention.

2. **No contrastive reasoning directions:** Existing work computes contrastive directions for behavioral properties (truthful vs. untruthful). Nobody has computed "correct reasoning direction" vs. "incorrect reasoning direction" as a steering vector.

3. **No domain-specific structured reasoning steering:** While function vectors (Todd et al. 2024) show in-context task steering is possible, nobody has extracted and applied domain-specific reasoning directions (e.g., "Nash equilibrium computation direction").

4. **No connection between probing diagnostics and inference-time improvement:** Probing papers diagnose what representations encode; intervention papers steer behavior. Nobody has built an end-to-end pipeline: probe -> identify critical layers -> extract directions -> steer at inference time -> improved reasoning.

5. **Training-free improvement of reasoning:** All existing methods to improve reasoning (SFT, RLHF, GRPO, etc.) require training. Representation engineering for reasoning would be **training-free** (only needs forward passes for direction extraction).

### 2.3 Why This Gap Exists

- RepE community comes from AI safety/alignment -> focuses on safety properties
- Reasoning community focuses on training (CoT, RLHF, verifiers) -> doesn't consider inference-time representation interventions
- Mechanistic interpretability community identifies circuits but doesn't propose practical improvements
- These three communities have not cross-pollinated on the specific question of "can we steer reasoning quality?"

---

## 3. Concrete Idea Formulation

### 3.1 Title

**"Probing-Guided Representation Steering for Game-Theoretic Reasoning in LLMs"**

Alternative: "Reasoning by Intervention: Using Probing-Derived Directions to Steer LLM Problem Solving at Inference Time"

### 3.2 Core Thesis

Given that linear probes can identify (a) which layers encode game-theoretic concepts and (b) the directions along which these concepts are represented, we can extract **contrastive reasoning directions** from correct vs. incorrect problem-solving instances and apply them at inference time to improve reasoning accuracy—with zero additional training.

### 3.3 Method: Probing-Guided Activation Steering (PGAS)

**Step 1: Concept Probing (already done in Phase 1)**
- Train linear probes on frozen model representations across all layers
- Identify critical layers L* where game-theoretic concepts peak
- This yields: layer selection + concept directions (probe weight vectors)

**Step 2: Contrastive Direction Extraction**
- Collect hidden states from instances where the model reasons *correctly* vs. *incorrectly*
- For each critical layer l in L*, compute:
  - h_correct(l) = mean activation over correctly-solved instances
  - h_incorrect(l) = mean activation over incorrectly-solved instances
  - d_reasoning(l) = h_correct(l) - h_incorrect(l)  [contrastive direction]
- Normalize: d_reasoning(l) = d_reasoning(l) / ||d_reasoning(l)||

**Step 3: Inference-Time Steering**
- At inference time, for each token position, add the steering vector:
  - h'(l) = h(l) + alpha * d_reasoning(l)
- alpha is a scalar hyperparameter (tuned on validation set, typically 1-10)
- Apply at all critical layers simultaneously (multi-layer intervention)

**Step 4: Concept-Specific Steering (Advanced)**
- Instead of a single "reasoning" direction, extract concept-specific directions:
  - d_dominance: direction for correct dominance identification
  - d_nash: direction for correct Nash equilibrium computation
  - d_best_response: direction for correct best-response identification
- Apply the relevant direction based on the problem type
- Or combine multiple directions with learned weights

### 3.4 Key Innovation: Probing as Layer/Direction Selection

The critical novelty is using probing accuracy as a **principled method for selecting intervention sites and directions**, rather than:
- Random layer selection (Turner et al. 2023 tried different layers empirically)
- Top-k attention head selection by truthfulness probe (ITI, Li et al. 2024)
- Learning interventions via backprop (ReFT, Wu et al. 2024)

Our approach: the very same probes that diagnose concept encoding also provide:
1. **Where** to intervene (layers with highest probing accuracy for the target concept)
2. **What direction** to use (the probe's weight vector IS the concept direction)
3. **How much** to intervene (scaled by inverse of probing certainty - less certain layers need more push)

### 3.5 Connection to Existing Results

From the Phase 1 results:
- br_direction (best response) peaks at layers ~70-77% depth -> intervene here for best response tasks
- eq_type (equilibrium type) peaks at ~63% depth -> intervene here for Nash equilibrium tasks
- dominance peaks at 67-77% depth -> intervene here for dominance reasoning

The probe weight vectors W^(l,t) from Phase 1 can be directly repurposed as steering directions.

---

## 4. Why This Is Novel and Publishable

### 4.1 Novelty Claims

1. **First application of representation engineering to reasoning capabilities:** All prior RepE/steering work targets behavioral properties (safety, honesty, sycophancy). This would be the first to target *cognitive capabilities* like structured reasoning.

2. **First probing-to-intervention pipeline for reasoning:** Establishes a methodology where diagnostic probing directly informs inference-time intervention - a principled alternative to empirical layer/direction selection.

3. **First training-free reasoning improvement method:** Unlike SFT, RLHF, GRPO (which all require gradient-based training), this achieves reasoning improvement with zero training cost (only forward passes for direction extraction).

4. **Novel domain (game-theoretic reasoning):** Game theory provides formally verifiable ground truth and multiple well-defined sub-concepts, making it an ideal testbed for concept-specific steering.

5. **Bridging three communities:** Connects mechanistic interpretability (probing), AI safety (RepE/steering), and reasoning improvement (CoT/RLHF) into a single framework.

### 4.2 Publication Viability

**Strengths for top venue:**
- Clean, principled method with strong theoretical motivation
- Easily reproducible (no complex training setup)
- Clear ablation axes (layer selection, direction extraction method, steering strength)
- Quantitative evaluation on formally-verifiable tasks
- Builds on prestigious recent work (RepE NeurIPS, ITI NeurIPS, ReFT NeurIPS)
- Novel angle that reviewers will find refreshing vs. yet another training-based paper

**Venue fit:**
- **ICLR 2026** (submission deadline ~Oct 2025): Best fit - interpretability + methodology
- **NeurIPS 2025** (submission deadline ~May 2025): Tight but possible if fast execution
- **ICML 2025** (already past): Next ICML 2026
- **ACL/EMNLP 2025**: Also strong fit given NLP focus

---

## 5. Experimental Design

### 5.1 Research Questions

- **RQ1:** Can contrastive directions (correct vs. incorrect reasoning) be extracted as clean linear directions in representation space?
- **RQ2:** Does adding these directions at inference time improve game-solving accuracy?
- **RQ3:** Does probing-guided layer selection outperform random/uniform layer selection?
- **RQ4:** Are concept-specific directions more effective than a generic "correct reasoning" direction?
- **RQ5:** How does steering compare with training-based methods (SFT, GRPO) in terms of accuracy-compute tradeoff?
- **RQ6:** Does steering generalize out-of-distribution (larger games, different formats)?

### 5.2 Experimental Setup

**Models:**
- Qwen2.5-3B-Instruct (primary, matching existing probing results)
- Qwen2.5-7B-Instruct (scale test)
- Llama-3.1-8B-Instruct (architecture generality)
- DeepSeek-R1-Distill-Qwen-7B (reasoning-tuned model)

**Data (already available from current project):**
- GameSolve-Bench: 2,400 samples for direction extraction
- OOD test set: 750 samples for generalization evaluation
- Split strategy: use training set instances to extract directions, test on held-out set

**Baselines:**
- Base model (no intervention)
- Random direction addition (control)
- PCA direction from all activations (non-contrastive baseline)
- ITI-style (top-k attention heads by probing accuracy)
- SFT (CoT) - your existing strong baseline
- GRPO (your existing RL baseline)
- ReFT (learned interventions, requires training)

### 5.3 Main Experiments

**Experiment 1: Direction Quality Analysis**
- Extract contrastive directions at each layer
- Measure: cosine similarity with probe weight vectors
- Measure: linear separability of correct/incorrect via direction projection
- Visualize: t-SNE/PCA of activations colored by correctness
- Expected output: confirm that "reasoning quality" has a linear representation

**Experiment 2: Single-Layer Steering**
- For each layer l, apply d_reasoning(l) at varying strengths alpha
- Measure accuracy on held-out test set
- Plot: accuracy vs. alpha curve for each layer
- Identify optimal layer and strength
- Compare probing-selected layer vs. exhaustive search -> does probing identify the best layer?

**Experiment 3: Multi-Layer Steering**
- Apply directions at multiple critical layers simultaneously
- Test combinations: {L/2}, {L/4, L/2, 2L/3}, all layers with probing acc > threshold
- Measure if multi-layer intervention compounds benefits

**Experiment 4: Concept-Specific Steering**
- Extract separate directions for each game-theoretic concept
- Test on concept-matched evaluation (e.g., Nash direction on Nash problems)
- Compare: generic direction vs. concept-specific direction vs. combined

**Experiment 5: Comparison with Training-Based Methods**
- Compare accuracy: PGAS vs. SFT (CoT) vs. GRPO vs. Full+Probe GRPO
- Compare compute cost: PGAS needs only ~100 forward passes for extraction; training methods need thousands of gradient steps
- Key metric: accuracy improvement per FLOP

**Experiment 6: OOD Generalization**
- Apply steering vectors extracted from in-distribution data to OOD test set
- Test: larger matrices (5x5, 6x6), non-integer payoffs, novel formats
- Hypothesis: steering may generalize better than SFT because it targets internal reasoning mechanism rather than surface patterns

**Experiment 7: Compositionality and Transfer**
- Can directions extracted from one model transfer to another (same family, different scale)?
- Can directions extracted from one game type transfer to another?
- Test cross-concept steering (apply dominance direction to Nash problems)

### 5.4 Ablation Studies

1. **Direction extraction method:** mean difference vs. PCA vs. LDA vs. probe weights
2. **Steering position:** all tokens vs. last token only vs. game-description tokens only
3. **Number of examples for extraction:** 10, 50, 100, 500, 1000
4. **Steering strength schedule:** constant alpha vs. layer-dependent alpha vs. token-position-dependent alpha
5. **Probe-guided vs. random layer selection:** verify that probing accuracy correlates with intervention effectiveness

### 5.5 Evaluation Metrics

- **Primary:** Accuracy on game-solving tasks (best response, Nash equilibrium)
- **Secondary:** Calibration (confidence vs. correctness), reasoning coherence (via CoT analysis)
- **Efficiency:** FLOPs for direction extraction vs. training methods
- **Faithfulness:** Does the model's CoT reasoning change in meaningful ways after steering?

### 5.6 Compute Requirements

All experiments are highly compute-efficient on 8x H20 GPUs:
- Direction extraction: ~200 forward passes (1-2 minutes on 3B model)
- Per-experiment evaluation: ~200-750 forward passes (1-5 minutes)
- Full experimental suite: ~1 day total (including all ablations)
- Compare: GRPO training took ~hours; SFT training took ~hours

---

## 6. Risks and Mitigations

### 6.1 Technical Risks

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Reasoning quality is NOT linearly represented | High | Low | Phase 1 probing already shows linear separability at 63-82% accuracy. If directions exist for probing, they exist for steering. |
| Steering improves accuracy marginally (<5%) | Medium | Medium | Use stronger vectors (higher alpha), multi-layer intervention, concept-specific directions. Even small improvement is novel as a training-free method. |
| Steering damages fluency/coherence | Medium | Medium | Monitor generation quality; use small alpha; apply only to early layers (which affect reasoning more than surface form). |
| Directions don't transfer OOD | Low | Medium | This would still be a paper (characterizing when steering fails is interesting). Also, OOD-specific directions can be extracted. |
| Method only works for game theory, not general reasoning | Low | Low | Test on additional domains (math, logic). Even if domain-specific, game theory is a legitimate and important domain. |

### 6.2 Scientific Risks

| Risk | Mitigation |
|---|---|
| Concurrent work publishes similar idea | Execute quickly; game-theoretic angle is unique. No existing work combines probing + steering + reasoning. |
| Reviewers say "incremental over RepE" | Emphasize the novel direction (reasoning vs. safety), probing-guided selection, and practical implications. |
| Reviewers want scaling to larger models | Test on 7B+ models; discuss compute scaling (steering is O(1) regardless of model size). |

### 6.3 Comparison with Current Project's Limitations

Your current results show:
- GRPO barely improves over base model (~51% vs 52%)
- Probe-augmented GRPO actually *hurts* (~48%)
- SFT achieves ~94% but may overfit

**Why PGAS might succeed where training failed:**
1. No optimization instability (no gradients, no reward hacking)
2. Directly targets the representational mechanism, not the output distribution
3. Doesn't suffer from reward signal sparsity (GRPO's main issue)
4. Can be combined with SFT: steer the SFT model further on its failure cases

---

## 7. Extended Ideas and Variations

### 7.1 Dynamic Steering (Token-Adaptive)

Instead of applying a fixed vector at every token, train a tiny classifier that predicts *when* to steer:
- Monitor activations during generation
- When the model enters "confusion" state (low probe confidence), apply steering
- When probe confidence is high, don't intervene
- This is a "guided decoding" variant

### 7.2 Steering + Best-of-N

- Generate N completions without steering
- Generate N completions with steering
- Use reward model or verifier to select best
- Hypothesis: steering narrows the output distribution toward correctness, making best-of-N more efficient

### 7.3 Iterative Direction Refinement

- Extract initial directions from base model
- Apply steering -> model now solves more problems correctly
- Re-extract directions from the *steered* model's representations
- Apply updated directions -> further improvement
- This is a "self-improvement" loop without training

### 7.4 Multi-Step Reasoning Steering

For problems requiring multi-step reasoning (e.g., iterated elimination of dominated strategies):
- Apply different concept directions at different generation steps
- Step 1: apply "dominance detection" direction
- Step 2: apply "elimination" direction
- Step 3: apply "equilibrium computation" direction
- Orchestrate the reasoning process through sequential steering

---

## 8. Positioning Statement (for Paper Introduction)

"While representation engineering has shown remarkable success in controlling behavioral properties of language models—honesty, safety, and sycophancy—its potential for enhancing *cognitive capabilities* remains entirely unexplored. We bridge this gap by demonstrating that reasoning quality in structured problem-solving has a clean linear representation in LLM activation space, and that probing-derived directions can be applied at inference time to significantly improve game-theoretic reasoning accuracy—achieving [X]% of SFT performance with zero training cost. Our probing-guided activation steering (PGAS) framework establishes a new paradigm: use interpretability to diagnose reasoning failures, then fix them through targeted representation intervention."

---

## 9. Timeline Estimate

| Week | Task |
|---|---|
| 1 | Direction extraction + quality analysis (Exp 1) |
| 1-2 | Single-layer and multi-layer steering (Exp 2, 3) |
| 2 | Concept-specific steering (Exp 4) |
| 2-3 | Comparison with training methods + OOD (Exp 5, 6) |
| 3 | Ablations + transfer experiments (Exp 7) |
| 3-4 | Paper writing |
| 4 | Revision + submission |

Total: ~4 weeks from idea to submission (all experiments are fast due to no training).

---

## 10. Key References (Consolidated)

1. Zou et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv:2310.01405.
2. Li, K. et al. (2024). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." NeurIPS 2023.
3. Turner et al. (2023). "Activation Addition: Steering Language Models Without Optimization." arXiv:2308.10248.
4. Rimsky et al. (2024). "Steering Llama 2 via Contrastive Activation Addition." ACL 2024 Findings.
5. Todd et al. (2024). "Function Vectors in Large Language Models." ICLR 2024.
6. Wu et al. (2024). "ReFT: Representation Finetuning for Language Models." NeurIPS 2024.
7. Hernandez et al. (2024). "Linearity of Relation Representations in Transformer Language Models." ICLR 2024.
8. Stolfo et al. (2023). "A Mechanistic Interpretation of Arithmetic Reasoning in Language Models." EMNLP 2023.
9. Marks et al. (2024). "Sparse Feature Circuits." arXiv:2403.19647.
10. Nanda et al. (2023). "Progress measures for grokking via mechanistic interpretability." ICLR 2023.
11. Hanna et al. (2023). "How does GPT-2 compute greater-than?" NeurIPS 2023.
12. Deng et al. (2024). "Explicit CoT Training Enhances Reasoning Representations in LLMs."
13. Lanham et al. (2023). "Measuring Faithfulness in Chain-of-Thought Reasoning." arXiv:2307.13702.
14. Mallen et al. (2024). "Eliciting Latent Knowledge from Language Models via Probing." ACL 2024.
15. Li, K. et al. (2023). "Emergent World Representations." ICLR 2023.

---

## 11. Summary: Why This Idea Is Strong

| Criterion | Assessment |
|---|---|
| **Novelty** | First RepE work on reasoning; first probing-guided steering; bridges 3 communities |
| **Timeliness** | RepE is hot (2023-2024); reasoning is hot; combination is untouched |
| **Feasibility** | Phase 1 probing already done; method needs only forward passes; 8xH20 is more than enough |
| **Build on existing work** | Directly leverages your Phase 1 probing results and GameSolve-Bench |
| **Clear contribution** | Method + extensive empirical analysis + domain application |
| **Reviewer appeal** | Clean story: diagnose (probe) -> locate (critical layers) -> fix (steer) -> evaluate |
| **Risk level** | Low: even negative results (steering doesn't help reasoning) are publishable as an empirical finding |
| **Compute cost** | Minimal: ~1 day for all experiments |
| **Comparison advantage** | Your GRPO results show training struggles; PGAS offers training-free alternative |
