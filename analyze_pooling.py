"""
Cross-Pooling Analysis
=======================
Compare probing results across all pooling methods for each model.
Generates comparison tables, heatmaps, and identifies the best pooling per concept.

Usage:
    python analyze_pooling.py
    python analyze_pooling.py --models Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct
"""

import argparse, json
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

PROBE_ROOT = Path("./results/probing")
OUT_ROOT = Path("./results/analysis")
LABELS = ["eq_type", "game_type", "difficulty", "dominance", "br_direction", "eq_uniqueness"]


def discover_pooling_methods(model_name):
    model_dir = PROBE_ROOT / model_name
    methods = []
    if model_dir.exists():
        for d in sorted(model_dir.iterdir()):
            if d.is_dir() and (d / "probe_summary.json").exists():
                methods.append(d.name)
    return methods


def load_summary(model_name, pooling):
    path = PROBE_ROOT / model_name / pooling / "probe_summary.json"
    with open(path) as f:
        return json.load(f)


def plot_pooling_comparison_per_model(model_name, methods, summaries, out_dir):
    if not HAS_PLOT:
        return

    # 1. Bar chart: peak accuracy per label per pooling
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(LABELS))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        s = summaries[method]
        peaks = []
        for label in LABELS:
            if label in s["per_label"]:
                peaks.append(s["per_label"][label]["peak_accuracy"])
            else:
                peaks.append(0)
        ax.bar(x + i * width - 0.4 + width / 2, peaks, width, label=method, alpha=0.85)

    ax.set_xlabel("Concept Label")
    ax.set_ylabel("Peak Test Accuracy")
    ax.set_title(f"Peak Probing Accuracy by Pooling Method — {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / f"pooling_comparison_{model_name}.png", dpi=150)
    plt.close()
    print(f"  Saved pooling_comparison_{model_name}.png")

    # 2. Heatmap: pooling × label (peak accuracy)
    matrix = np.zeros((len(methods), len(LABELS)))
    for i, method in enumerate(methods):
        s = summaries[method]
        for j, label in enumerate(LABELS):
            if label in s["per_label"]:
                matrix[i, j] = s["per_label"][label]["peak_accuracy"]

    fig, ax = plt.subplots(figsize=(10, max(3, len(methods) * 0.6)))
    sns.heatmap(matrix, annot=True, fmt=".3f",
                xticklabels=LABELS, yticklabels=methods,
                cmap="YlOrRd", vmin=0.3, vmax=1, ax=ax,
                cbar_kws={"label": "Peak Test Accuracy"})
    ax.set_title(f"Pooling × Concept Peak Accuracy — {model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"pooling_heatmap_{model_name}.png", dpi=150)
    plt.close()
    print(f"  Saved pooling_heatmap_{model_name}.png")

    # 3. Layer-wise curves for each label, one subplot per label, one line per pooling
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    for idx, label in enumerate(LABELS):
        ax = axes[idx]
        for method in methods:
            s = summaries[method]
            if label not in s["per_label"]:
                continue
            acc_by_layer = s["per_label"][label]["accuracy_by_layer"]
            layers = sorted([int(k) for k in acc_by_layer.keys()])
            accs = [acc_by_layer.get(str(l), acc_by_layer.get(l, 0)) for l in layers]
            ax.plot(layers, accs, marker=".", markersize=2, label=method, linewidth=1.2)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Layer")
        if idx % 3 == 0:
            ax.set_ylabel("Test Accuracy")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Layer-wise Probing by Pooling — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / f"pooling_layerwise_{model_name}.png", dpi=150)
    plt.close()
    print(f"  Saved pooling_layerwise_{model_name}.png")


def plot_cross_model_pooling(all_data, model_names, out_dir):
    if not HAS_PLOT or len(model_names) < 2:
        return

    # Best pooling per label per model comparison
    fig, axes = plt.subplots(1, len(LABELS), figsize=(4 * len(LABELS), 5), sharey=True)
    if len(LABELS) == 1:
        axes = [axes]

    for idx, label in enumerate(LABELS):
        ax = axes[idx]
        for model_name in model_names:
            best_method = None
            best_peak = 0
            for method, s in all_data[model_name].items():
                if label in s["per_label"]:
                    peak = s["per_label"][label]["peak_accuracy"]
                    if peak > best_peak:
                        best_peak = peak
                        best_method = method

            if best_method and label in all_data[model_name][best_method]["per_label"]:
                s = all_data[model_name][best_method]
                acc_by_layer = s["per_label"][label]["accuracy_by_layer"]
                layers = sorted([int(k) for k in acc_by_layer.keys()])
                accs = [acc_by_layer.get(str(l), acc_by_layer.get(l, 0)) for l in layers]
                short = model_name.replace("Qwen2.5-", "Q").replace("-Instruct", "")
                ax.plot(layers, accs, marker=".", markersize=3,
                        label=f"{short} ({best_method})")

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Layer")
        if idx == 0:
            ax.set_ylabel("Test Accuracy")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Best Pooling per Concept — Cross-Model", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "best_pooling_cross_model.png", dpi=150)
    plt.close()
    print(f"  Saved best_pooling_cross_model.png")


def run(model_names):
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    full_results = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Cross-Pooling Analysis: {model_name}")
        print(f"{'='*60}")

        methods = discover_pooling_methods(model_name)
        print(f"Found pooling methods: {methods}")

        summaries = {}
        for method in methods:
            summaries[method] = load_summary(model_name, method)
        all_data[model_name] = summaries

        # Build comparison table
        print(f"\n{'Label':<16}", end="")
        for m in methods:
            print(f"  {m:>10}", end="")
        print(f"  {'BEST':>10}")
        print("-" * (16 + 12 * (len(methods) + 1)))

        model_results = {"methods": methods, "per_label": {}}

        for label in LABELS:
            print(f"{label:<16}", end="")
            best_acc = 0
            best_method = ""
            peaks = {}
            peak_layers = {}
            for m in methods:
                s = summaries[m]
                if label in s["per_label"]:
                    acc = s["per_label"][label]["peak_accuracy"]
                    layer = s["per_label"][label]["peak_layer"]
                    peaks[m] = acc
                    peak_layers[m] = layer
                    print(f"  {acc:>10.4f}", end="")
                    if acc > best_acc:
                        best_acc = acc
                        best_method = m
                else:
                    peaks[m] = 0
                    print(f"  {'N/A':>10}", end="")
            print(f"  {best_method:>10}")

            model_results["per_label"][label] = {
                "peaks": peaks,
                "peak_layers": peak_layers,
                "best_method": best_method,
                "best_accuracy": best_acc,
            }

        full_results[model_name] = model_results

        model_out = out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        plot_pooling_comparison_per_model(model_name, methods, summaries, model_out)

    # Cross-model best-pooling comparison
    if len(model_names) > 1:
        plot_cross_model_pooling(all_data, model_names, out_dir)

    # Save consolidated results
    with open(out_dir / "pooling_comparison.json", "w") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    # Print final summary
    print(f"\n{'='*60}")
    print("BEST POOLING PER CONCEPT")
    print(f"{'='*60}")
    for model_name in model_names:
        print(f"\n{model_name}:")
        for label in LABELS:
            r = full_results[model_name]["per_label"].get(label)
            if r:
                print(f"  {label:<18}: {r['best_method']:<10} (acc={r['best_accuracy']:.4f})")

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    if args.models is None:
        args.models = sorted([d.name for d in PROBE_ROOT.iterdir() if d.is_dir()])
        print(f"Auto-detected models: {args.models}")

    run(args.models)
