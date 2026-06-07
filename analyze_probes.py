"""
Phase 1, Step 3: Probing Analysis
==================================
Produce heatmaps, critical layer sets, linearity coefficients, cross-label interference.

Usage:
    python analyze_probes.py
    python analyze_probes.py --models Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct
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
    print("WARNING: matplotlib/seaborn not available, skipping plots")

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

BENCH_PATH = "./gamesolve_bench.jsonl"
REP_ROOT = "./results/representations"
PROBE_ROOT = "./results/probing"
OUT_ROOT = "./results/analysis"
LABELS = ["eq_type", "game_type", "difficulty", "dominance", "br_direction", "eq_uniqueness"]
PCA_COMPONENTS = [10, 50, 100]


def load_probe_summary(model_name, pooling="last"):
    path = Path(PROBE_ROOT) / model_name / pooling / "probe_summary.json"
    with open(path) as f:
        return json.load(f)


def load_probe_results(model_name, pooling="last"):
    path = Path(PROBE_ROOT) / model_name / pooling / "probe_results.json"
    with open(path) as f:
        return json.load(f)


def plot_accuracy_heatmap(summary, model_name, out_dir):
    if not HAS_PLOT:
        return
    num_layers = summary["num_layers"]
    labels_with_data = [l for l in LABELS if l in summary["per_label"]]

    matrix = np.zeros((len(labels_with_data), num_layers))
    for i, label in enumerate(labels_with_data):
        acc_by_layer = summary["per_label"][label]["accuracy_by_layer"]
        for l in range(num_layers):
            matrix[i, l] = acc_by_layer.get(str(l), acc_by_layer.get(l, 0))

    fig, ax = plt.subplots(figsize=(max(14, num_layers * 0.5), max(4, len(labels_with_data) * 0.8)))
    sns.heatmap(matrix, annot=(num_layers <= 32), fmt=".2f",
                xticklabels=range(num_layers), yticklabels=labels_with_data,
                cmap="YlOrRd", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Test Accuracy"})
    ax.set_xlabel("Layer")
    ax.set_ylabel("Concept Label")
    ax.set_title(f"Probing Accuracy Heatmap — {model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"heatmap_{model_name}.png", dpi=150)
    plt.close()
    print(f"  Saved heatmap_{model_name}.png")


def plot_accuracy_curves(summary, model_name, out_dir):
    if not HAS_PLOT:
        return
    num_layers = summary["num_layers"]
    fig, ax = plt.subplots(figsize=(12, 6))

    for label in LABELS:
        if label not in summary["per_label"]:
            continue
        acc_by_layer = summary["per_label"][label]["accuracy_by_layer"]
        layers = sorted([int(k) for k in acc_by_layer.keys()])
        accs = [acc_by_layer[str(l)] if str(l) in acc_by_layer else acc_by_layer.get(l, 0) for l in layers]
        ax.plot(layers, accs, marker="o", markersize=3, label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Probing Accuracy by Layer — {model_name}")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"accuracy_curves_{model_name}.png", dpi=150)
    plt.close()
    print(f"  Saved accuracy_curves_{model_name}.png")


def compute_linearity_coefficients(model_name, pooling="last"):
    rep_dir = Path(REP_ROOT) / model_name / pooling
    with open(rep_dir / "meta.json") as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]
    n_samples = meta["num_samples"]

    samples = []
    with open(BENCH_PATH) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    samples = samples[:n_samples]

    from train_probes import extract_labels
    all_labels = extract_labels(samples)

    results = {}
    for label in LABELS:
        y = np.array(all_labels[label])
        valid = y != "N/A"
        if valid.sum() < 50:
            continue

        le = LabelEncoder()
        y_enc = le.fit_transform(y[valid])
        if len(np.unique(y_enc)) < 2:
            continue

        peak_summary = load_probe_summary(model_name, pooling)
        if label not in peak_summary["per_label"]:
            continue
        peak_layer = peak_summary["per_label"][label]["peak_layer"]

        X = torch.load(rep_dir / f"layer_{peak_layer:02d}.pt", weights_only=True).numpy()
        X_valid = X[valid]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X_valid, y_enc))

        label_results = {"peak_layer": peak_layer, "full_dim_accuracy": None, "pca_accuracies": {}}

        clf_full = LogisticRegression(C=100, solver="lbfgs", max_iter=3000, random_state=42)
        clf_full.fit(X_valid[train_idx], y_enc[train_idx])
        label_results["full_dim_accuracy"] = round(accuracy_score(y_enc[test_idx], clf_full.predict(X_valid[test_idx])), 4)

        for k in PCA_COMPONENTS:
            actual_k = min(k, X_valid.shape[1], X_valid.shape[0])
            pca = PCA(n_components=actual_k, random_state=42)
            X_pca = pca.fit_transform(X_valid)

            clf = LogisticRegression(C=100, solver="lbfgs", max_iter=3000, random_state=42)
            clf.fit(X_pca[train_idx], y_enc[train_idx])
            acc = round(accuracy_score(y_enc[test_idx], clf.predict(X_pca[test_idx])), 4)
            variance_explained = round(float(pca.explained_variance_ratio_.sum()), 4)

            label_results["pca_accuracies"][k] = {
                "accuracy": acc,
                "variance_explained": variance_explained,
                "actual_components": actual_k,
            }

        results[label] = label_results
        print(f"  {label}: full={label_results['full_dim_accuracy']}, "
              f"PCA-10={label_results['pca_accuracies'].get(10, {}).get('accuracy', 'N/A')}, "
              f"PCA-50={label_results['pca_accuracies'].get(50, {}).get('accuracy', 'N/A')}")

    return results


def compute_cross_label_interference(model_name, pooling="last"):
    probe_results = load_probe_results(model_name, pooling)
    summary = load_probe_summary(model_name, pooling)

    interference = {}
    for label in LABELS:
        if label not in summary["per_label"]:
            continue
        peak_layer = summary["per_label"][label]["peak_layer"]
        layer_key = f"layer_{peak_layer:02d}"
        if layer_key not in probe_results or label not in probe_results[layer_key]:
            continue

    rep_dir = Path(REP_ROOT) / model_name / pooling
    with open(rep_dir / "meta.json") as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]
    n_samples = meta["num_samples"]

    samples = []
    with open(BENCH_PATH) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    samples = samples[:n_samples]

    from train_probes import extract_labels
    all_labels = extract_labels(samples)

    best_layer = max(
        [summary["per_label"][l]["peak_layer"] for l in LABELS if l in summary["per_label"]],
        key=lambda x: x
    )

    X = torch.load(rep_dir / f"layer_{best_layer:02d}.pt", weights_only=True).numpy()
    weight_vectors = {}

    for label in LABELS:
        y = np.array(all_labels[label])
        valid = y != "N/A"
        if valid.sum() < 50:
            continue
        le = LabelEncoder()
        y_enc = le.fit_transform(y[valid])
        if len(np.unique(y_enc)) < 2:
            continue

        clf = LogisticRegression(C=100, solver="lbfgs", max_iter=3000, random_state=42)
        clf.fit(X[valid], y_enc)
        w = clf.coef_.mean(axis=0)
        w = w / (np.linalg.norm(w) + 1e-10)
        weight_vectors[label] = w

    cosine_sim = {}
    label_pairs = [(a, b) for a in weight_vectors for b in weight_vectors if a < b]
    for a, b in label_pairs:
        sim = float(np.dot(weight_vectors[a], weight_vectors[b]))
        cosine_sim[f"{a}_vs_{b}"] = round(sim, 4)

    return {"layer_used": best_layer, "cosine_similarities": cosine_sim}


def plot_cross_model_comparison(summaries, model_names, out_dir):
    if not HAS_PLOT or len(summaries) < 2:
        return

    fig, axes = plt.subplots(1, len(LABELS), figsize=(4 * len(LABELS), 5), sharey=True)
    if len(LABELS) == 1:
        axes = [axes]

    for i, label in enumerate(LABELS):
        ax = axes[i]
        for model_name, summary in zip(model_names, summaries):
            if label not in summary["per_label"]:
                continue
            acc_by_layer = summary["per_label"][label]["accuracy_by_layer"]
            layers = sorted([int(k) for k in acc_by_layer.keys()])
            accs = [acc_by_layer.get(str(l), acc_by_layer.get(l, 0)) for l in layers]
            short_name = model_name.replace("Qwen2.5-", "Q").replace("-Instruct", "")
            ax.plot(layers, accs, marker=".", markersize=3, label=short_name)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Layer")
        if i == 0:
            ax.set_ylabel("Test Accuracy")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Cross-Model Probing Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "cross_model_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved cross_model_comparison.png")


def run(model_names, pooling="last"):
    out_dir = Path(OUT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    full_analysis = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*60}")

        summary = load_probe_summary(model_name, pooling)
        all_summaries.append(summary)

        model_out = out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        plot_accuracy_heatmap(summary, model_name, model_out)
        plot_accuracy_curves(summary, model_name, model_out)

        print("\nLinearity coefficients (PCA):")
        linearity = compute_linearity_coefficients(model_name, pooling)

        print("\nCross-label interference:")
        interference = compute_cross_label_interference(model_name, pooling)
        for pair, sim in interference["cosine_similarities"].items():
            print(f"  {pair}: {sim}")

        critical_layers = {}
        for label in LABELS:
            if label in summary["per_label"]:
                critical_layers[label] = {
                    "peak_layer": summary["per_label"][label]["peak_layer"],
                    "peak_accuracy": summary["per_label"][label]["peak_accuracy"],
                    "critical_layers": summary["per_label"][label]["critical_layers"],
                }

        model_analysis = {
            "model_name": model_name,
            "num_layers": summary["num_layers"],
            "critical_layers": critical_layers,
            "linearity_coefficients": linearity,
            "cross_label_interference": interference,
            "per_label_summary": summary["per_label"],
        }
        full_analysis[model_name] = model_analysis

        with open(model_out / "full_analysis.json", "w") as f:
            json.dump(model_analysis, f, indent=2, ensure_ascii=False)
        print(f"\nSaved analysis to {model_out}/")

    if len(model_names) > 1:
        plot_cross_model_comparison(all_summaries, model_names, out_dir)

    with open(out_dir / "phase1_results.json", "w") as f:
        json.dump(full_analysis, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("PHASE 1 SUMMARY")
    print(f"{'='*60}")
    for model_name in model_names:
        a = full_analysis[model_name]
        print(f"\n{model_name}:")
        for label in LABELS:
            if label in a["critical_layers"]:
                cl = a["critical_layers"][label]
                print(f"  {label:20s}: peak_acc={cl['peak_accuracy']:.4f} @ layer {cl['peak_layer']}, "
                      f"critical={cl['critical_layers']}")

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--pooling", default="last", choices=["last", "mean"])
    args = parser.parse_args()

    if args.models is None:
        rep_root = Path(REP_ROOT)
        args.models = sorted([d.name for d in rep_root.iterdir() if d.is_dir()])
        print(f"Auto-detected models: {args.models}")

    run(args.models, args.pooling)
