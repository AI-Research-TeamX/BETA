"""
Preprocess GameSolve-Bench for SFT training on chain-of-thought solutions.
Creates train/val JSON in chat format.
"""
import json
import random
from pathlib import Path

BENCH_PATH = "gamesolve_bench.jsonl"
OUTPUT_DIR = Path("data/phase2_sft")
SEED = 42
TRAIN_RATIO = 0.8

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


def format_answer_block(sample):
    gt = sample["ground_truth"]
    task = sample["task"]
    row_labels = sample["row_labels"]
    col_labels = sample["col_labels"]

    if task == "nash_equilibrium":
        eqs = gt["equilibria"]
        pure_ne = []
        mixed_ne = []
        for eq in eqs:
            if eq["is_pure"]:
                r_idx = eq["sigma_row"].index(max(eq["sigma_row"]))
                c_idx = eq["sigma_col"].index(max(eq["sigma_col"]))
                pure_ne.append(f"({row_labels[r_idx]}, {col_labels[c_idx]})")
            else:
                sigma_r = "[" + ", ".join(f"{p:.4f}" for p in eq["sigma_row"]) + "]"
                sigma_c = "[" + ", ".join(f"{p:.4f}" for p in eq["sigma_col"]) + "]"
                mixed_ne.append(f"({sigma_r}, {sigma_c})")

        answer = "ANSWER:\n"
        answer += f"Pure NE: {', '.join(pure_ne) if pure_ne else 'none'}\n"
        answer += f"Mixed NE: {', '.join(mixed_ne) if mixed_ne else 'none'}"
        return answer
    else:
        actions = [row_labels[i] for i in gt["best_response_actions"]]
        payoffs = gt["expected_payoffs"]
        br_val = gt["best_response_value"]

        answer = "ANSWER:\n"
        answer += f"Best response action(s): {', '.join(actions)}\n"
        answer += f"Expected payoffs: [{', '.join(f'{p:.6f}' for p in payoffs)}]\n"
        answer += f"Best response value: {br_val:.6f}"
        return answer


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    samples = [json.loads(l) for l in open(BENCH_PATH)]
    random.shuffle(samples)

    split_idx = int(len(samples) * TRAIN_RATIO)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    for split_name, split_data in [("train", train_samples), ("val", val_samples)]:
        processed = []
        for sample in split_data:
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

            user_content = f"{description}\n\n{instruction}"

            cot = sample["chain_of_thought"]
            answer_block = format_answer_block(sample)
            response = f"{cot}\n\n{answer_block}"

            entry = {
                "id": sample["id"],
                "task": sample["task"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response},
                ],
                "ground_truth": sample["ground_truth"],
                "row_labels": sample["row_labels"],
                "col_labels": sample["col_labels"],
                "desc_style": desc_style,
            }
            processed.append(entry)

        out_path = OUTPUT_DIR / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(processed, f, indent=2)
        print(f"Wrote {len(processed)} samples to {out_path}")

        # Stats
        resp_lens = [len(e["messages"][2]["content"]) for e in processed]
        print(f"  Response length: mean={sum(resp_lens)/len(resp_lens):.0f}, max={max(resp_lens)}, min={min(resp_lens)}")


if __name__ == "__main__":
    main()
