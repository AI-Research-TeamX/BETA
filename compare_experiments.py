"""
Compare all experiment results and produce summary table.
"""
import json
from pathlib import Path

EVAL_DIRS = {
    "Base (Qwen2.5-3B)": "eval_results/qwen3b_base",
    "GRPO-only": "eval_results/grpo_only_best",
    "Full+Probe GRPO": "eval_results/full_probe_grpo_best",
    "SFT (CoT)": "eval_results/sft_cot_best",
    "SFT → GRPO": "eval_results/sft_then_grpo_best",
}


def load_result(eval_dir):
    path = Path(eval_dir) / "eval_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPARISON: GameSolve-Bench Reward")
    print("=" * 70)

    header = f"{'Method':<25} {'Overall':>8} {'Nash':>8} {'BR':>8} {'Easy':>8} {'Med':>8} {'Hard':>8}"
    print(header)
    print("-" * 70)

    all_results = {}
    for name, eval_dir in EVAL_DIRS.items():
        result = load_result(eval_dir)
        if result is None:
            print(f"{name:<25} {'(no data)':>8}")
            continue

        overall = result.get("all", {}).get("mean", 0)
        nash = result.get("task/nash_equilibrium", {}).get("mean", 0)
        br = result.get("task/best_response", {}).get("mean", 0)
        easy = result.get("diff/easy", {}).get("mean", 0)
        med = result.get("diff/medium", {}).get("mean", 0)
        hard = result.get("diff/hard", {}).get("mean", 0)

        print(f"{name:<25} {overall:>8.4f} {nash:>8.4f} {br:>8.4f} {easy:>8.4f} {med:>8.4f} {hard:>8.4f}")
        all_results[name] = result

    print("-" * 70)

    # Save comparison
    out_path = Path("results/phase2/comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull comparison saved to {out_path}")


if __name__ == "__main__":
    main()
