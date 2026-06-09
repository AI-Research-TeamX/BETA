"""
Compare OOD evaluation results across all methods.
Also compare with ID (in-distribution) results.
"""
import json
from pathlib import Path

METHODS = {
    "Base (Qwen2.5-3B)": {
        "ood": "eval_results/ood_base",
        "id": "eval_results/qwen3b_base",
    },
    "GRPO-only": {
        "ood": "eval_results/ood_grpo_only",
        "id": "eval_results/grpo_only_best",
    },
    "Full+Probe GRPO": {
        "ood": "eval_results/ood_full_probe",
        "id": "eval_results/full_probe_grpo_best",
    },
    "SFT (CoT)": {
        "ood": "eval_results/ood_sft_cot",
        "id": "eval_results/sft_cot_best",
    },
    "SFT → GRPO": {
        "ood": "eval_results/ood_sft_grpo",
        "id": "eval_results/sft_then_grpo_best",
    },
}

OOD_CATEGORIES = [
    "large_matrix", "non_integer", "wide_range",
    "novel_format_math", "novel_format_json", "novel_format_table",
    "asymmetric", "combined_hard",
]


def load_result(eval_dir, filename):
    path = Path(eval_dir) / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    # ── Table 1: Overall ID vs OOD ──
    print("\n" + "=" * 80)
    print("  TABLE 1: In-Distribution vs Out-of-Distribution Overall Performance")
    print("=" * 80)
    header = f"{'Method':<22} {'ID':>8} {'OOD':>8} {'Δ':>8} {'Δ%':>8} {'OOD Nash':>10} {'OOD BR':>10}"
    print(header)
    print("-" * 80)

    all_ood = {}
    for name, dirs in METHODS.items():
        id_r = load_result(dirs["id"], "eval_summary.json")
        ood_r = load_result(dirs["ood"], "ood_eval_summary.json")

        if id_r is None or ood_r is None:
            print(f"{name:<22} {'(missing)':>8}")
            continue

        id_overall = id_r.get("all", {}).get("mean", 0)
        ood_overall = ood_r.get("all", {}).get("mean", 0)
        delta = ood_overall - id_overall
        delta_pct = (delta / id_overall * 100) if id_overall > 0 else 0
        ood_nash = ood_r.get("task/nash_equilibrium", {}).get("mean", 0)
        ood_br = ood_r.get("task/best_response", {}).get("mean", 0)

        print(f"{name:<22} {id_overall:>8.4f} {ood_overall:>8.4f} {delta:>+8.4f} {delta_pct:>+7.1f}% {ood_nash:>10.4f} {ood_br:>10.4f}")
        all_ood[name] = ood_r

    # ── Table 2: OOD breakdown by category ──
    print("\n" + "=" * 100)
    print("  TABLE 2: Performance by OOD Category")
    print("=" * 100)

    cat_header = f"{'OOD Category':<22}"
    for name in METHODS:
        cat_header += f" {name[:12]:>12}"
    print(cat_header)
    print("-" * 100)

    for cat in OOD_CATEGORIES:
        row = f"{cat:<22}"
        for name in METHODS:
            ood_r = all_ood.get(name)
            if ood_r is None:
                row += f" {'--':>12}"
                continue
            val = ood_r.get(f"ood/{cat}", {}).get("mean", 0)
            n = ood_r.get(f"ood/{cat}", {}).get("count", 0)
            row += f" {val:>12.4f}"
        print(row)

    # ── Table 3: Dimension breakdown ──
    print("\n" + "=" * 100)
    print("  TABLE 3: Performance by Matrix Dimension (OOD)")
    print("=" * 100)

    dims = set()
    for name, ood_r in all_ood.items():
        if ood_r:
            dims.update(k.replace("dim/", "") for k in ood_r if k.startswith("dim/"))
    dims = sorted(dims, key=lambda x: (int(x.split("x")[0]), int(x.split("x")[1])))

    dim_header = f"{'Dimension':<12}"
    for name in METHODS:
        dim_header += f" {name[:12]:>12}"
    print(dim_header)
    print("-" * 100)

    for dim in dims:
        row = f"{dim:<12}"
        for name in METHODS:
            ood_r = all_ood.get(name)
            if ood_r is None:
                row += f" {'--':>12}"
                continue
            val = ood_r.get(f"dim/{dim}", {}).get("mean", 0)
            row += f" {val:>12.4f}"
        print(row)

    # ── Table 4: Generalization gap analysis ──
    print("\n" + "=" * 80)
    print("  TABLE 4: Generalization Gap (ID - OOD) by Category")
    print("=" * 80)

    gap_header = f"{'Method':<22} {'ID':>8} {'OOD All':>8} {'Gap':>8}"
    for cat in ["large_matrix", "non_integer", "wide_range", "novel_format", "asymmetric", "combined"]:
        gap_header += f" {cat[:10]:>10}"
    print(gap_header)
    print("-" * (22 + 8*3 + 10*6 + 10))

    for name, dirs in METHODS.items():
        id_r = load_result(dirs["id"], "eval_summary.json")
        ood_r = all_ood.get(name)
        if id_r is None or ood_r is None:
            continue

        id_overall = id_r.get("all", {}).get("mean", 0)
        ood_overall = ood_r.get("all", {}).get("mean", 0)
        gap = id_overall - ood_overall

        row = f"{name:<22} {id_overall:>8.4f} {ood_overall:>8.4f} {gap:>+8.4f}"

        # Aggregate novel_format categories
        for cat_group in ["large_matrix", "non_integer", "wide_range", "novel_format", "asymmetric", "combined"]:
            vals = []
            for k, v in ood_r.items():
                if k.startswith(f"ood/") and cat_group in k and isinstance(v, dict):
                    vals.extend([v["mean"]] * v["count"])
            if vals:
                cat_mean = sum(vals) / len(vals)
                cat_gap = id_overall - cat_mean
                row += f" {cat_gap:>+10.4f}"
            else:
                row += f" {'--':>10}"
        print(row)

    # ── Save combined results ──
    combined = {}
    for name, dirs in METHODS.items():
        id_r = load_result(dirs["id"], "eval_summary.json")
        ood_r = all_ood.get(name)
        combined[name] = {"id": id_r, "ood": ood_r}

    out_path = Path("results/phase2/ood_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nFull comparison saved to {out_path}")


if __name__ == "__main__":
    main()
