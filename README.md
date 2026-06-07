# GameSolve-Bench

A benchmark for evaluating LLM reasoning on **game theory** tasks — specifically Nash Equilibrium computation and Best Response identification in two-player normal-form games.

## Overview

GameSolve-Bench tests whether language models can perform precise game-theoretic reasoning. It includes two tasks:

| Task | Description | What the model must do |
|------|-------------|----------------------|
| **Task A: Nash Equilibrium** | Find all NE (pure + mixed) of a 2-player game | Identify equilibrium strategies, classify equilibrium type |
| **Task B: Best Response** | Compute optimal play against a given opponent mixed strategy | Calculate expected payoffs, identify BR actions and value |

The benchmark contains **2,400 samples** (1,350 Nash + 1,050 Best Response) with verified ground truth computed by exact solvers ([nashpy](https://github.com/drvinceknight/Nashpy)).

## Dataset

### Game Configurations

| Dimensions | Game Types | Task A (Nash) | Task B (BR) |
|:----------:|:----------:|:-------------:|:-----------:|
| 2×2 | general, zero-sum, symmetric | 650 | 400 |
| 3×3 | general, zero-sum, symmetric | 380 | 280 |
| 4×4 | general | 100 | 150 |
| 2×3 | general | 120 | 120 |
| 3×2 | general | 100 | 100 |

### Sample Structure

Each sample in `gamesolve_bench.jsonl` includes:

- **Payoff matrices** — row and column player payoffs
- **Three description variants** — `abstract` (formal), `story` (narrative with roles), `compact` (minimal enumeration)
- **Ground truth** — exact equilibria / best response computed by nashpy
- **Chain-of-thought** — step-by-step reference solution
- **Difficulty label** — `easy`, `medium`, or `hard` (based on matrix size and equilibrium class)

### Equilibrium Class Distribution (Task A)

| Class | Count |
|:-----:|:-----:|
| Pure only | 978 |
| Both pure + mixed | 202 |
| Mixed only | 170 |

## Evaluation

### Metrics

**Task A (Nash Equilibrium):**
- Pure NE F1 / Precision / Recall
- Mixed NE L1 distance (strategy profile error)
- Equilibrium class accuracy (pure / mixed / both / none)

**Task B (Best Response):**
- Action accuracy (correct BR action identification)
- Expected utility MAE (payoff calculation error)
- BR value error (optimal payoff error)

All metrics are reported overall and broken down by difficulty level and equilibrium class.

### Running Evaluation

**Single model** (requires a running vLLM / OpenAI-compatible server):

```bash
# Start a vLLM server first, then:
python eval_gamesolve.py --model Qwen2.5-7B-Instruct

# Options
python eval_gamesolve.py --model Qwen2.5-7B-Instruct --max-samples 100   # quick test
python eval_gamesolve.py --model Qwen2.5-7B-Instruct --task nash          # Task A only
python eval_gamesolve.py --model Qwen2.5-7B-Instruct --task br            # Task B only
python eval_gamesolve.py --model Qwen2.5-7B-Instruct --variant story      # use story descriptions
python eval_gamesolve.py --model Qwen2.5-7B-Instruct --temperature 0.0    # greedy (default)
```

**All models** (automatically starts/stops vLLM for each model):

```bash
bash eval_all.sh                          # full eval
bash eval_all.sh --max-samples 50         # quick test
bash eval_all.sh --task br --temperature 0.2
```

Results are saved to `eval_results/<model_name>/`.

### Evaluated Models

The default configuration evaluates the Qwen2.5-Instruct family:

- Qwen2.5-3B-Instruct
- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct
- Qwen2.5-32B-Instruct

## Regenerating the Dataset

```bash
python gamesolve_gen.py
```

This produces `gamesolve_bench.jsonl` (2,400 samples) and `gamesolve_stats.json`. Generation is seeded (`SEED=42`) for reproducibility.

## Project Structure

```
├── gamesolve_gen.py          # Dataset generator (game sampling, solving, NL descriptions)
├── gamesolve_bench.jsonl     # Generated benchmark dataset (2,400 samples)
├── gamesolve_stats.json      # Dataset statistics
├── eval_gamesolve.py         # Evaluation script (API calls, parsing, scoring)
├── eval_all.sh               # Batch evaluation across multiple models via vLLM
└── verl/                     # verl — RL training library for LLMs (ByteDance Seed)
```

## Dependencies

- Python 3.10+
- `numpy`
- `nashpy` (Nash equilibrium solver)
- `openai` (API client for vLLM-served models)
- [vLLM](https://github.com/vllm-project/vllm) (model serving)

