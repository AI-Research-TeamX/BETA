"""
Phase 1, Step 1: Hidden State Extraction
=========================================
Extract hidden states from all transformer layers for GameSolve-Bench samples.
Saves per-layer tensors for downstream probing.

Usage:
    python extract_representations.py --model_path ./Qwen/Qwen2.5-1.5B-Instruct
    python extract_representations.py --model_path ./Qwen/Qwen2.5-3B-Instruct --batch_size 8
    python extract_representations.py --model_path ./Qwen/Qwen2.5-1.5B-Instruct --max_samples 10  # quick test
"""

import argparse, json, os, gc, time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

BENCH_PATH = "./gamesolve_bench.jsonl"
OUTPUT_ROOT = "./results/representations"


def load_samples(path, max_samples=None):
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if max_samples:
        samples = samples[:max_samples]
    return samples


def build_prompts(samples, variant="abstract"):
    prompts = []
    for s in samples:
        desc = s["descriptions"][variant]
        if s["task"] == "nash_equilibrium":
            prompt = f"{desc}\n\nTask: Find ALL Nash equilibria of this game (pure-strategy and mixed-strategy).\nShow your reasoning step by step."
        else:
            prompt = f"{desc}\n\nTask: Compute the expected payoff for each action and identify your best response(s).\nShow your reasoning step by step."
        prompts.append(prompt)
    return prompts


def extract(model_path, batch_size=4, max_samples=None, variant="abstract", pooling="last"):
    model_name = Path(model_path).name
    out_dir = Path(OUTPUT_ROOT) / model_name / pooling
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"Model: {model_name}, layers={num_layers}, hidden_size={hidden_size}")

    samples = load_samples(BENCH_PATH, max_samples)
    prompts = build_prompts(samples, variant)
    n = len(prompts)
    print(f"Samples: {n}, batch_size={batch_size}, pooling={pooling}")

    layer_reps = {l: [] for l in range(num_layers)}
    hooks = []
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            captured[layer_idx] = hidden.detach()
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    t0 = time.time()
    with torch.no_grad():
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_prompts = prompts[batch_start:batch_end]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            attention_mask = inputs["attention_mask"]
            model(**inputs)

            for l in range(num_layers):
                h = captured[l].float().cpu()
                if pooling == "last":
                    seq_lens = attention_mask.cpu().sum(dim=1) - 1
                    reps = h[torch.arange(h.size(0)), seq_lens]
                elif pooling == "mean":
                    mask = attention_mask.unsqueeze(-1).float().cpu()
                    reps = (h * mask).sum(dim=1) / mask.sum(dim=1)
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")
                layer_reps[l].append(reps)

            captured.clear()

            if (batch_start // batch_size) % 10 == 0:
                elapsed = time.time() - t0
                progress = batch_end / n * 100
                print(f"  [{batch_end:4d}/{n}] {progress:.1f}% — {elapsed:.1f}s elapsed")

    for h in hooks:
        h.remove()

    print(f"Saving representations to {out_dir}/")
    for l in range(num_layers):
        tensor = torch.cat(layer_reps[l], dim=0)
        assert tensor.shape == (n, hidden_size), f"Layer {l}: expected ({n}, {hidden_size}), got {tensor.shape}"
        torch.save(tensor, out_dir / f"layer_{l:02d}.pt")

    meta = {
        "model_name": model_name,
        "model_path": str(model_path),
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_samples": n,
        "variant": variant,
        "pooling": pooling,
        "batch_size": batch_size,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {num_layers} layers × {n} samples × {hidden_size}d saved in {meta['elapsed_seconds']}s")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--variant", default="abstract", choices=["abstract", "story", "compact"])
    parser.add_argument("--pooling", default="last", choices=["last", "mean"])
    args = parser.parse_args()

    extract(args.model_path, args.batch_size, args.max_samples, args.variant, args.pooling)
