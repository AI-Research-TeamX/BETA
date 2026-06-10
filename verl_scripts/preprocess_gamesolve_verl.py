"""
Convert GameSolve-Bench data to verl Parquet format.
Matches the same train/val split as the previous GRPO experiment (seed=42, 80/20).
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


def main():
    bench_path = "gamesolve_bench.jsonl"
    output_dir = "data/verl_gamesolve"
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
                },
            }
            records.append(record)

        df = pd.DataFrame(records)
        out_path = os.path.join(output_dir, f"{split_name}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(records)} samples to {out_path}")


if __name__ == "__main__":
    main()
