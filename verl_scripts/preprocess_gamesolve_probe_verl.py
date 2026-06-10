"""
Convert GameSolve-Bench data to verl Parquet format WITH probing labels.
Same train/val split as grpo_verl (seed=42, 80/20).
Probing labels stored in extra_info for use by auxiliary probing loss.
"""
import json
import random
import os

import pandas as pd


SYSTEM_PROMPT = """You are an expert in game theory. Analyze the given game carefully and provide precise answers.

Always end your response with a clearly marked ANSWER section using this exact format:

For Nash Equilibrium tasks:
ANSWER:
Pure NE: [(action_row, action_col), ...]   or "none"
Mixed NE: [(sigma_row, sigma_col), ...]    or "none"
(sigma = probability vector, e.g. [0.4, 0.6])

For Best Response tasks:
ANSWER:
Best response action(s): [action_name, ...]
Expected payoffs: [eu_a1, eu_a2, ...]
Best response value: <number>
"""

LABEL_SPECS = {
    "eq_type":       {"num_classes": 3, "task_filter": "nash_equilibrium"},
    "difficulty":    {"num_classes": 3, "task_filter": None},
    "dominance":     {"num_classes": 2, "task_filter": None},
    "br_direction":  {"num_classes": 5, "task_filter": "best_response"},
    "eq_uniqueness": {"num_classes": 2, "task_filter": "nash_equilibrium"},
}

EQ_TYPE_MAP = {"pure": 0, "mixed": 1, "both": 2}
DIFF_MAP = {"easy": 0, "medium": 1, "hard": 2}


def has_dominant_strategy(payoff_matrix_row):
    n_rows = len(payoff_matrix_row)
    if n_rows < 2:
        return 0
    for i in range(n_rows):
        dominates_all = True
        for j in range(n_rows):
            if i == j:
                continue
            if not all(payoff_matrix_row[i][c] > payoff_matrix_row[j][c]
                       for c in range(len(payoff_matrix_row[i]))):
                dominates_all = False
                break
        if dominates_all:
            return 1
    return 0


def get_br_direction(ground_truth, task):
    if task != "best_response":
        return -1
    actions = ground_truth.get("best_response_actions", [])
    if len(actions) == 0:
        return -1
    if len(actions) > 1:
        return 4  # mixed
    action = actions[0]
    try:
        idx = int(action.split("_")[-1]) if "_" in str(action) else int(action)
        return min(idx, 3)
    except (ValueError, IndexError):
        return 0


def get_eq_uniqueness(ground_truth, task):
    if task != "nash_equilibrium":
        return -1
    n_eq = ground_truth.get("n_equilibria", 0)
    if n_eq == 0:
        return 0
    elif n_eq == 1:
        return 1
    else:
        return 1  # binary: 0=zero, 1=one_or_more


def compute_concept_labels(sample):
    task = sample["task"]
    gt = sample["ground_truth"]
    labels = {}

    # eq_type
    if task == "nash_equilibrium":
        eq_class = gt.get("equilibrium_class", "")
        labels["eq_type"] = EQ_TYPE_MAP.get(eq_class, -1)
    else:
        labels["eq_type"] = -1

    # difficulty
    diff = sample.get("metadata", {}).get("difficulty", "medium")
    labels["difficulty"] = DIFF_MAP.get(diff, 1)

    # dominance
    labels["dominance"] = has_dominant_strategy(sample.get("payoff_matrix_row", []))

    # br_direction
    labels["br_direction"] = get_br_direction(gt, task)

    # eq_uniqueness
    labels["eq_uniqueness"] = get_eq_uniqueness(gt, task)

    return labels


def main():
    bench_path = "gamesolve_bench.jsonl"
    output_dir = "data/verl_gamesolve_probe"
    os.makedirs(output_dir, exist_ok=True)

    with open(bench_path) as f:
        samples = [json.loads(line) for line in f]

    random.seed(42)
    random.shuffle(samples)

    n_train = int(len(samples) * 0.8)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    print(f"Total: {len(samples)}, Train: {len(train_samples)}, Val: {len(val_samples)}")

    for split_name, split_data in [("train", train_samples), ("val", val_samples)]:
        records = []
        for sample in split_data:
            desc_keys = list(sample["descriptions"].keys())
            random.seed(hash(sample["id"]))
            desc_style = random.choice(desc_keys)
            desc = sample["descriptions"][desc_style]

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

            prompt_text = f"{desc}\n\n{instruction}"
            concept_labels = compute_concept_labels(sample)

            record = {
                "data_source": "gamesolve",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "ability": "game_theory",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(sample["ground_truth"]),
                },
                "extra_info": {
                    "id": sample["id"],
                    "task": sample["task"],
                    "row_labels": sample["row_labels"],
                    "col_labels": sample["col_labels"],
                    "dimensions": sample["dimensions"],
                    "game_type": sample.get("game_type", "unknown"),
                    "split": split_name,
                    "concept_labels": concept_labels,
                },
            }
            records.append(record)

        df = pd.DataFrame(records)
        out_path = os.path.join(output_dir, f"{split_name}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(records)} samples to {out_path}")

        label_stats = {}
        for name in LABEL_SPECS:
            vals = [r["extra_info"]["concept_labels"][name] for r in records]
            valid = [v for v in vals if v >= 0]
            label_stats[name] = {"total": len(vals), "valid": len(valid),
                                 "distribution": {str(v): vals.count(v) for v in set(vals)}}
        print(f"  Label stats: {json.dumps(label_stats, indent=2)}")


if __name__ == "__main__":
    main()
