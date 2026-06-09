"""
vLLM-based evaluation for GameSolve benchmarks (ID and OOD).
Uses vLLM offline batch inference for maximum throughput.
Supports tensor parallelism across multiple GPUs.
"""
import json
import argparse
import random
import time
import sys
from pathlib import Path
from collections import defaultdict

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
    desc_keys = list(sample["descriptions"].keys())
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

    return f"{desc}\n\n{instruction}"


def build_chat_prompt(tokenizer, prompt_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser(description="vLLM batch evaluation for GameSolve")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--bench_path", default="gamesolve_bench.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_mem", type=float, default=0.85, help="GPU memory utilization")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    print(f"Loading vLLM model from {args.model_path} (tp={args.tp})...")
    t_load = time.time()
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
    )
    tokenizer = llm.get_tokenizer()
    print(f"Model loaded in {time.time() - t_load:.1f}s")

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=0.95,
        seed=args.seed,
    )

    samples = load_benchmark(args.bench_path, args.max_samples, args.seed)
    print(f"Evaluating {len(samples)} samples...")

    prompts = [build_chat_prompt(tokenizer, format_prompt(s)) for s in samples]

    t_gen = time.time()
    all_outputs = []
    for batch_start in range(0, len(prompts), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(outputs)
        elapsed = time.time() - t_gen
        print(f"  Generated {batch_end}/{len(prompts)} in {elapsed:.1f}s")

    gen_time = time.time() - t_gen
    print(f"Generation complete: {len(all_outputs)} responses in {gen_time:.1f}s "
          f"({len(all_outputs)/gen_time:.1f} samples/s)")

    results = []
    metrics = defaultdict(list)

    for i, (sample, output) in enumerate(zip(samples, all_outputs)):
        response = output.outputs[0].text

        reward = compute_reward(
            response, sample["ground_truth"],
            sample["task"], sample["row_labels"], sample["col_labels"]
        )

        ood_cat = sample.get("metadata", {}).get("ood_category", None)
        difficulty = sample.get("metadata", {}).get("difficulty", "unknown")

        entry = {
            "id": sample["id"],
            "task": sample["task"],
            "dimensions": sample["dimensions"],
            "reward": reward,
            "response_length": len(response),
        }
        if ood_cat:
            entry["ood_category"] = ood_cat

        results.append(entry)

        metrics["all"].append(reward)
        metrics[f"task/{sample['task']}"].append(reward)

        if ood_cat:
            metrics[f"ood/{ood_cat}"].append(reward)
            dim_key = f"dim/{sample['dimensions'][0]}x{sample['dimensions'][1]}"
            metrics[dim_key].append(reward)
        else:
            metrics[f"diff/{difficulty}"].append(reward)

    summary = {}
    for key, values in sorted(metrics.items()):
        summary[key] = {
            "mean": float(sum(values) / len(values)),
            "count": len(values),
        }

    summary["model_path"] = args.model_path
    summary["n_samples"] = len(samples)
    summary["gen_time_s"] = round(gen_time, 1)
    summary["throughput"] = round(len(samples) / gen_time, 1)

    out_name = "ood_eval_summary.json" if any(
        s.get("metadata", {}).get("ood_category") for s in samples
    ) else "eval_summary.json"

    out_path = Path(args.output_dir) / out_name
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    details_name = out_name.replace("summary", "details")
    details_path = Path(args.output_dir) / details_name
    with open(details_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results ===")
    for key, val in sorted(summary.items()):
        if isinstance(val, dict):
            print(f"  {key}: {val['mean']:.4f} (n={val['count']})")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
