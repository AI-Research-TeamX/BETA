"""
Preprocess GameSolve-Bench for GRPO + Probing training.
Outputs train/val JSON files with prompts, ground truth, and concept labels.
"""
import json
import random
import numpy as np
from pathlib import Path

BENCH_PATH = "gamesolve_bench.jsonl"
OUTPUT_DIR = Path("data/phase2")
SEED = 42
TRAIN_RATIO = 0.8

LABEL_SPECS = {
    "eq_type": {"classes": ["pure", "mixed", "both"], "task_filter": "nash_equilibrium"},
    "difficulty": {"classes": ["easy", "medium", "hard"], "task_filter": None},
    "dominance": {"classes": ["no", "yes"], "task_filter": None},
    "br_direction": {"classes": ["row_0", "row_1", "row_2", "row_3", "mixed"], "task_filter": "best_response"},
    "eq_uniqueness": {"classes": ["one", "multiple"], "task_filter": "nash_equilibrium"},
}


def has_dominant_strategy(payoff_matrix):
    m = np.array(payoff_matrix)
    n_rows = m.shape[0]
    for i in range(n_rows):
        dominates_all = True
        for j in range(n_rows):
            if i == j:
                continue
            if not np.all(m[i] > m[j]):
                dominates_all = False
                break
        if dominates_all:
            return True
    return False


def extract_labels(sample):
    labels = {}
    task = sample["task"]

    if task == "nash_equilibrium":
        labels["eq_type"] = sample["ground_truth"]["equilibrium_class"]
        n_eq = sample["ground_truth"]["n_equilibria"]
        labels["eq_uniqueness"] = "one" if n_eq == 1 else "multiple"
    else:
        labels["eq_type"] = None
        labels["eq_uniqueness"] = None

    labels["difficulty"] = sample["metadata"]["difficulty"]
    labels["dominance"] = "yes" if has_dominant_strategy(sample["payoff_matrix_row"]) else "no"

    if task == "best_response":
        actions = sample["ground_truth"]["best_response_actions"]
        if len(actions) == 1:
            labels["br_direction"] = f"row_{actions[0]}"
        else:
            labels["br_direction"] = "mixed"
    else:
        labels["br_direction"] = None

    return labels


def encode_labels(labels):
    encoded = {}
    for name, spec in LABEL_SPECS.items():
        val = labels[name]
        if val is None:
            encoded[name] = -1
        else:
            encoded[name] = spec["classes"].index(val)
    return encoded


def format_prompt(sample):
    desc_style = random.choice(["abstract", "story", "compact"])
    description = sample["descriptions"][desc_style]

    if sample["task"] == "nash_equilibrium":
        instruction = (
            "Find all Nash Equilibria of this game. For each equilibrium, state whether it is "
            "pure or mixed. For pure NE, give the strategy pair. For mixed NE, give the "
            "probability distributions over strategies for each player."
        )
    else:
        instruction = (
            "Compute the best response for the row player given the column player's strategy. "
            "Report: (1) the expected payoff for each row action, (2) the best response action(s), "
            "and (3) the best response expected payoff value."
        )

    prompt = f"{description}\n\n{instruction}"
    return prompt, desc_style


def format_ground_truth(sample):
    return json.dumps(sample["ground_truth"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    samples = [json.loads(l) for l in open(BENCH_PATH)]
    random.shuffle(samples)

    split_idx = int(len(samples) * TRAIN_RATIO)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    for split_name, split_data in [("train", train_samples), ("val", val_samples)]:
        processed = []
        for sample in split_data:
            prompt_text, desc_style = format_prompt(sample)
            labels = extract_labels(sample)
            encoded = encode_labels(labels)

            entry = {
                "id": sample["id"],
                "task": sample["task"],
                "prompt": prompt_text,
                "ground_truth": sample["ground_truth"],
                "game_type": sample["game_type"],
                "dimensions": sample["dimensions"],
                "row_labels": sample["row_labels"],
                "col_labels": sample["col_labels"],
                "desc_style": desc_style,
                "concept_labels": encoded,
            }
            processed.append(entry)

        out_path = OUTPUT_DIR / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(processed, f, indent=2)
        print(f"Wrote {len(processed)} samples to {out_path}")

    print(f"\nLabel specs:")
    for name, spec in LABEL_SPECS.items():
        print(f"  {name}: {len(spec['classes'])} classes = {spec['classes']}")


if __name__ == "__main__":
    main()
