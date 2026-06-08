"""
Phase 1, Step 1: Multi-Pooling Hidden State Extraction
=======================================================
Extract hidden states from all transformer layers using multiple pooling methods
in a SINGLE forward pass. Saves per-layer .pt files for each pooling method.

Pooling methods:
  - last:     Last non-padding token (standard for decoder-only LMs)
  - mean:     Average over all non-padding tokens
  - sum:      Sum over all non-padding tokens
  - max:      Element-wise max over all non-padding tokens
  - first:    First token position
  - weighted: Exponentially weighted average (later tokens get higher weight)

Usage:
    python extract_representations_multi.py --model_path ./Qwen/Qwen2.5-1.5B-Instruct
    python extract_representations_multi.py --model_path ./Qwen/Qwen2.5-3B-Instruct --batch_size 4
"""

import argparse, json, gc, time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

BENCH_PATH = "./gamesolve_bench.jsonl"
OUTPUT_ROOT = "./results/representations"

POOLING_METHODS = ["last", "mean", "sum", "max", "first", "weighted"]


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


def apply_pooling(h, attention_mask):
    """Apply all pooling methods on GPU. h [B, T, D], attention_mask [B, T]. Returns CPU tensors."""
    B, T, D = h.shape
    device = h.device
    mask_f = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
    seq_lens = attention_mask.sum(dim=1) - 1        # [B]
    h_masked_sum = h * mask_f                       # reuse for mean/sum

    results = {}
    results["last"] = h[torch.arange(B, device=device), seq_lens].cpu()
    results["first"] = h[:, 0, :].cpu()

    token_sum = h_masked_sum.sum(dim=1)
    token_count = mask_f.sum(dim=1).clamp(min=1)
    results["mean"] = (token_sum / token_count).cpu()
    results["sum"] = token_sum.cpu()

    h_for_max = h.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))
    results["max"] = h_for_max.max(dim=1).values.cpu()

    positions = torch.arange(T, device=device).float().unsqueeze(0)
    norm_pos = positions / seq_lens.unsqueeze(1).clamp(min=1).float()
    weights = torch.exp(2.0 * norm_pos) * attention_mask.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    results["weighted"] = (h * weights.unsqueeze(-1)).sum(dim=1).cpu()

    return results


def extract(model_path, batch_size=4, max_samples=None, variant="abstract", skip_existing=True):
    model_name = Path(model_path).name

    # Check which pooling methods still need extraction
    methods_to_run = []
    for method in POOLING_METHODS:
        out_dir = Path(OUTPUT_ROOT) / model_name / method
        if skip_existing and (out_dir / "meta.json").exists():
            print(f"  Skipping {method} — already exists")
        else:
            methods_to_run.append(method)

    if not methods_to_run:
        print(f"All pooling methods already extracted for {model_name}")
        return

    print(f"Pooling methods to extract: {methods_to_run}")

    # Create output directories
    out_dirs = {}
    for method in methods_to_run:
        out_dirs[method] = Path(OUTPUT_ROOT) / model_name / method
        out_dirs[method].mkdir(parents=True, exist_ok=True)

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
    print(f"Samples: {n}, batch_size={batch_size}")

    # Per-layer per-method storage
    layer_reps = {method: {l: [] for l in range(num_layers)} for method in methods_to_run}

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
                h = captured[l].float()  # keep on whichever GPU this layer lives on
                pooled = apply_pooling(h, attention_mask.to(h.device))

                for method in methods_to_run:
                    layer_reps[method][l].append(pooled[method])

            captured.clear()

            if (batch_start // batch_size) % 10 == 0:
                elapsed = time.time() - t0
                progress = batch_end / n * 100
                print(f"  [{batch_end:4d}/{n}] {progress:.1f}% — {elapsed:.1f}s elapsed")

    for h in hooks:
        h.remove()

    # Save results
    for method in methods_to_run:
        print(f"Saving {method} representations to {out_dirs[method]}/")
        for l in range(num_layers):
            tensor = torch.cat(layer_reps[method][l], dim=0)
            assert tensor.shape == (n, hidden_size), \
                f"{method} layer {l}: expected ({n}, {hidden_size}), got {tensor.shape}"
            torch.save(tensor, out_dirs[method] / f"layer_{l:02d}.pt")

        meta = {
            "model_name": model_name,
            "model_path": str(model_path),
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_samples": n,
            "variant": variant,
            "pooling": method,
            "batch_size": batch_size,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        with open(out_dirs[method] / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {len(methods_to_run)} pooling methods × {num_layers} layers × {n} samples saved in {elapsed:.1f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--variant", default="abstract", choices=["abstract", "story", "compact"])
    parser.add_argument("--no_skip", action="store_true", help="Re-extract even if exists")
    args = parser.parse_args()

    extract(args.model_path, args.batch_size, args.max_samples, args.variant,
            skip_existing=not args.no_skip)
