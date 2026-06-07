"""
Phase 1, Step 2: Probing Classifier Training (8-GPU parallel)
==============================================================
Each GPU processes one layer at a time from a shared work queue.
8 layers train simultaneously across 8 GPUs.

Usage:
    python train_probes_parallel.py --model_name Qwen2.5-1.5B-Instruct
    python train_probes_parallel.py --model_name Qwen2.5-3B-Instruct
"""

import argparse, json, time, os
from pathlib import Path
from multiprocessing import Process, Queue, Manager
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

BENCH_PATH = "./gamesolve_bench.jsonl"
REP_ROOT = "./results/representations"
OUT_ROOT = "./results/probing"

LABELS = ["eq_type", "game_type", "difficulty", "dominance", "br_direction", "eq_uniqueness"]
REG_VALUES = [1e-4, 1e-3, 1e-2, 1e-1]
NUM_GPUS = min(8, torch.cuda.device_count())


def load_samples(path):
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def has_dominant_strategy(payoff_matrix_row):
    R = np.array(payoff_matrix_row)
    m = R.shape[0]
    for i in range(m):
        if all(np.all(R[i] > R[j]) for j in range(m) if j != i):
            return True
    return False


def get_br_direction(sample):
    gt = sample.get("ground_truth", {})
    br_actions = gt.get("best_response_actions")
    if br_actions is None:
        return None
    if isinstance(br_actions, list):
        return f"row_{br_actions[0]}" if len(br_actions) == 1 else "mixed"
    return None


def get_eq_uniqueness(sample):
    gt = sample.get("ground_truth", {})
    n_eq = gt.get("n_equilibria")
    if n_eq is None:
        eqs = gt.get("equilibria")
        n_eq = len(eqs) if eqs is not None else None
    if n_eq is None:
        return None
    if n_eq == 0:
        return "zero"
    return "one" if n_eq == 1 else "multiple"


def extract_labels(samples):
    labels = {k: [] for k in LABELS}
    for s in samples:
        gt = s.get("ground_truth", {})
        eq_type = gt.get("equilibrium_class") or "N/A"
        game_type = s.get("game_type") or "N/A"
        difficulty = s.get("metadata", {}).get("difficulty") or "N/A"
        dominance = "yes" if has_dominant_strategy(s["payoff_matrix_row"]) else "no"
        br_dir = get_br_direction(s) or "N/A"
        eq_uniq = get_eq_uniqueness(s) or "N/A"

        labels["eq_type"].append(eq_type)
        labels["game_type"].append(game_type)
        labels["difficulty"].append(difficulty)
        labels["dominance"].append(dominance)
        labels["br_direction"].append(br_dir)
        labels["eq_uniqueness"].append(eq_uniq)
    return labels


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_probe_gpu(X_tensor, y_tensor, n_classes, device, weight_decay=1e-3, lr=0.1, max_epochs=500):
    model = LinearProbe(X_tensor.shape[1], n_classes).to(device)
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    X_dev = X_tensor.to(device)
    y_dev = y_tensor.to(device)

    def closure():
        optimizer.zero_grad()
        output = model(X_dev)
        loss = criterion(output, y_dev)
        l2 = sum(p.pow(2).sum() for p in model.parameters()) * weight_decay
        total = loss + l2
        total.backward()
        return total

    for epoch in range(max_epochs):
        prev_loss = closure().item()
        optimizer.step(closure)
        curr_loss = closure().item()
        if abs(prev_loss - curr_loss) < 1e-7:
            break

    return model


def predict_gpu(model, X_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor.to(device))
        return logits.argmax(dim=1).cpu().numpy()


def train_probe_for_label(X_all, y_all, label_name, device):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)
    classes = le.classes_.tolist()
    n_classes = len(classes)

    if n_classes < 2:
        return None

    valid = np.array([y != "N/A" for y in y_all])
    X = X_all[valid]
    y_enc = y_encoded[valid]

    unique, counts = np.unique(y_enc, return_counts=True)
    if len(unique) < 2:
        return None
    if counts.min() < 3:
        keep = np.isin(y_enc, unique[counts >= 3])
        X, y_enc = X[keep], y_enc[keep]
        unique, counts = np.unique(y_enc, return_counts=True)
        if len(unique) < 2 or len(y_enc) < 20:
            return None

    if counts.min() < 5 or len(y_enc) < 20:
        return None

    le2 = LabelEncoder()
    y_enc = le2.fit_transform(y_enc)
    n_classes = len(np.unique(y_enc))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(sss1.split(X, y_enc))
    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y_enc[train_val_idx], y_enc[test_idx]

    if len(np.unique(y_train_val)) < 2 or len(y_train_val) < 10:
        return None
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    train_idx, val_idx = next(sss2.split(X_train_val, y_train_val))
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_trainval_t = torch.tensor(X_train_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_trainval_t = torch.tensor(y_train_val, dtype=torch.long)

    best_acc = -1
    best_wd = REG_VALUES[0]
    for wd in REG_VALUES:
        model = train_probe_gpu(X_train_t, y_train_t, n_classes, device, weight_decay=wd)
        val_pred = predict_gpu(model, X_val_t, device)
        val_acc = accuracy_score(y_val, val_pred)
        if val_acc > best_acc:
            best_acc = val_acc
            best_wd = wd

    final_model = train_probe_gpu(X_trainval_t, y_trainval_t, n_classes, device, weight_decay=best_wd)

    y_pred = predict_gpu(final_model, X_test_t, device)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred).tolist()

    w = final_model.linear.weight.detach().cpu().numpy()

    return {
        "label": label_name,
        "classes": classes,
        "n_classes": n_classes,
        "n_valid_samples": int(valid.sum()),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "best_reg_lambda": best_wd,
        "val_accuracy": round(best_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
        "confusion_matrix": cm,
        "weight_norm": round(float(np.linalg.norm(w)), 4),
    }


def gpu_worker(gpu_id, task_queue, result_dict, rep_dir, all_labels, n_samples, hidden_size, t0):
    device = torch.device(f"cuda:{gpu_id}")
    while True:
        try:
            layer_idx = task_queue.get_nowait()
        except Exception:
            break

        X = torch.load(rep_dir / f"layer_{layer_idx:02d}.pt", weights_only=True).numpy()
        assert X.shape == (n_samples, hidden_size)

        layer_results = {}
        for label_name in LABELS:
            y = all_labels[label_name]
            result = train_probe_for_label(X, y, label_name, device)
            if result:
                layer_results[label_name] = result

        layer_key = f"layer_{layer_idx:02d}"
        result_dict[layer_key] = layer_results

        accs = {k: v["test_accuracy"] for k, v in layer_results.items()}
        elapsed = time.time() - t0
        print(f"  [GPU {gpu_id}] Layer {layer_idx:2d}: {accs} ({elapsed:.1f}s)", flush=True)


def build_summary(results, num_layers, model_name, pooling, elapsed):
    summary = {
        "model_name": model_name,
        "pooling": pooling,
        "num_layers": num_layers,
        "elapsed_seconds": round(elapsed, 1),
        "per_label": {},
    }

    for label in LABELS:
        accs = []
        for l in range(num_layers):
            lk = f"layer_{l:02d}"
            if lk in results and label in results[lk]:
                accs.append((l, results[lk][label]["test_accuracy"]))

        if not accs:
            continue

        layers, acc_vals = zip(*accs)
        peak_idx = int(np.argmax(acc_vals))
        peak_layer = layers[peak_idx]
        peak_acc = acc_vals[peak_idx]

        tau = max(0.5, peak_acc - 0.1)
        critical_layers = [l for l, a in accs if a >= tau]

        summary["per_label"][label] = {
            "peak_accuracy": round(peak_acc, 4),
            "peak_layer": peak_layer,
            "mean_accuracy": round(float(np.mean(acc_vals)), 4),
            "accuracy_by_layer": {l: round(a, 4) for l, a in accs},
            "critical_layers": critical_layers,
            "tau": round(tau, 4),
        }

    return summary


def run(model_name, pooling="last"):
    rep_dir = Path(REP_ROOT) / model_name / pooling
    out_dir = Path(OUT_ROOT) / model_name / pooling
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(rep_dir / "meta.json") as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]
    n_samples = meta["num_samples"]
    hidden_size = meta["hidden_size"]

    print(f"Probing {model_name} ({pooling}): {num_layers} layers, {n_samples} samples, d={hidden_size}")
    print(f"Parallel across {NUM_GPUS} GPUs")

    samples = load_samples(BENCH_PATH)[:n_samples]
    all_labels = extract_labels(samples)

    task_queue = Queue()
    for i in range(num_layers):
        task_queue.put(i)

    manager = Manager()
    result_dict = manager.dict()

    t0 = time.time()

    workers = []
    for gpu_id in range(NUM_GPUS):
        p = Process(target=gpu_worker, args=(
            gpu_id, task_queue, result_dict, rep_dir, all_labels, n_samples, hidden_size, t0
        ))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    results = dict(result_dict)

    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = build_summary(results, num_layers, model_name, pooling, time.time() - t0)
    with open(out_dir / "probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_dir}/")
    print(f"Total time: {time.time() - t0:.1f}s")
    return results, summary


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--pooling", default="last", choices=["last", "mean"])
    args = parser.parse_args()
    run(args.model_name, args.pooling)
