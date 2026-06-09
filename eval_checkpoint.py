"""
Evaluate a checkpoint on GameSolve-Bench using local inference (no vLLM).
Runs on a single GPU for simplicity. Reports structured metrics.
"""
import json
import argparse
import random
import time
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gamesolve_reward import compute_reward


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


def load_benchmark(bench_path, max_samples=None, seed=42):
    samples = [json.loads(l) for l in open(bench_path)]
    random.seed(seed)
    random.shuffle(samples)
    if max_samples:
        samples = samples[:max_samples]
    return samples


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

    return f"{description}\n\n{instruction}"


@torch.no_grad()
def generate(model, tokenizer, prompt_text, max_new_tokens=1024, temperature=0.1):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--bench_path", default="gamesolve_bench.jsonl")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    if args.output_dir is None:
        model_name = Path(args.model_path).name
        args.output_dir = f"eval_results/{model_name}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    print(f"Model loaded on {device}")

    samples = load_benchmark(args.bench_path, args.max_samples)
    print(f"Evaluating {len(samples)} samples...")

    results = []
    metrics = defaultdict(list)

    for i, sample in enumerate(samples):
        t0 = time.time()
        prompt = format_prompt(sample)
        response = generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature)

        reward = compute_reward(
            response, sample["ground_truth"],
            sample["task"], sample["row_labels"], sample["col_labels"]
        )

        results.append({
            "id": sample["id"],
            "task": sample["task"],
            "difficulty": sample["metadata"]["difficulty"],
            "reward": reward,
            "response_length": len(response),
        })

        metrics["all"].append(reward)
        metrics[f"task/{sample['task']}"].append(reward)
        metrics[f"diff/{sample['metadata']['difficulty']}"].append(reward)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            avg_r = sum(metrics["all"]) / len(metrics["all"])
            print(f"  [{i+1}/{len(samples)}] reward={reward:.3f}  avg={avg_r:.3f}  time={elapsed:.1f}s")

    summary = {}
    for key, values in sorted(metrics.items()):
        summary[key] = {
            "mean": float(sum(values) / len(values)),
            "count": len(values),
        }

    summary["model_path"] = args.model_path
    summary["n_samples"] = len(samples)

    out_path = Path(args.output_dir) / "eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    details_path = Path(args.output_dir) / "eval_details.json"
    with open(details_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results ===")
    for key, val in sorted(summary.items()):
        if isinstance(val, dict):
            print(f"  {key}: {val['mean']:.4f} (n={val['count']})")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
