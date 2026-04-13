"""
GameSolve-Bench Evaluator
=========================
Calls a local OpenAI-compatible API (e.g. vLLM / LMStudio serving Qwen2.5-7B-Instruct)
and evaluates Task A (Nash Equilibrium) and Task B (Best Response).

Usage:
    python eval_gamesolve.py --model Qwen2.5-7B-Instruct
    python eval_gamesolve.py --model Qwen2.5-72B-Instruct --max-tokens 2048 --temperature 0.0
    python eval_gamesolve.py --model Qwen2.5-7B-Instruct --max-samples 100   # quick smoke-test
    python eval_gamesolve.py --model Qwen2.5-7B-Instruct --task br
"""

import json, re, time, argparse, sys, math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────
BASE_URL      = "http://localhost:8000/v1"
API_KEY       = "my-token"
DEFAULT_MODEL = "Qwen2.5-7B-Instruct"

DATA_PATH     = Path("./gamesolve_bench.jsonl")
OUT_DIR       = Path("./eval_results")

VARIANT       = "abstract"
MAX_TOKENS    = 1024
TEMPERATURE   = 0.0
REQUEST_DELAY = 0.05     # seconds between calls

# ── Prompts ───────────────────────────────────────────────────────

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

NASH_USER_TEMPLATE = """{description}

Task: Find ALL Nash equilibria of this game (pure-strategy and mixed-strategy).
Show your reasoning step by step, then provide the ANSWER section."""

BR_USER_TEMPLATE = """{description}

Task: Given the opponent's mixed strategy above, compute the expected payoff for each of your actions and identify your best response(s).
Show your reasoning step by step, then provide the ANSWER section."""


# ── API Client ────────────────────────────────────────────────────

def make_client():
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def call_model(client: OpenAI, user_msg: str,
               model: str = DEFAULT_MODEL,
               max_tokens: int = MAX_TOKENS,
               temperature: float = TEMPERATURE) -> tuple[str, dict]:
    """Returns (response_text, usage_dict)."""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    latency = time.time() - t0
    text = resp.choices[0].message.content or ""
    usage = {
        "prompt_tokens":     resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "latency_s":         round(latency, 3),
    }
    return text, usage


# ── Parsers ───────────────────────────────────────────────────────

def extract_answer_block(text: str) -> str:
    """Pull out everything after the last 'ANSWER:' marker."""
    idx = text.upper().rfind("ANSWER:")
    if idx == -1:
        return text   # fallback: try to parse the whole response
    return text[idx + 7:].strip()


def parse_numbers(s: str) -> list[float]:
    """Extract all floats/ints from a string."""
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]


def parse_nash_response(text: str, row_labels: list, col_labels: list) -> dict:
    """
    Parse model output for Task A.
    Returns:
        {
          "pure_ne":  [(row_label, col_label), ...],   # parsed pure NEs
          "mixed_ne": [(sigma_r, sigma_c), ...],        # parsed mixed NEs
          "raw_answer": str
        }
    """
    ans = extract_answer_block(text)
    result = {"pure_ne": [], "mixed_ne": [], "raw_answer": ans}

    # ── Pure NE parsing ──────────────────────────────────────────
    # Look for patterns like (A, B) or (Cooperate, Defect) or "none"
    pure_section = ""
    m = re.search(r"Pure\s*NE\s*:(.+?)(?:Mixed\s*NE|$)", ans,
                  re.IGNORECASE | re.DOTALL)
    if m:
        pure_section = m.group(1).strip()

    if pure_section and "none" not in pure_section.lower():
        # try to find action pairs
        for row_a in row_labels:
            for col_a in col_labels:
                patterns = [
                    rf"\({re.escape(row_a)}\s*,\s*{re.escape(col_a)}\)",
                    rf"\({re.escape(row_a)}\s*,\s*{re.escape(col_a)}\s*\)",
                ]
                for pat in patterns:
                    if re.search(pat, pure_section, re.IGNORECASE):
                        pair = (row_a, col_a)
                        if pair not in result["pure_ne"]:
                            result["pure_ne"].append(pair)

    # ── Mixed NE parsing ─────────────────────────────────────────
    mixed_section = ""
    m = re.search(r"Mixed\s*NE\s*:(.+?)$", ans,
                  re.IGNORECASE | re.DOTALL)
    if m:
        mixed_section = m.group(1).strip()

    if mixed_section and "none" not in mixed_section.lower():
        # find bracket-enclosed probability vectors
        vecs = re.findall(r"\[([^\]]+)\]", mixed_section)
        parsed_vecs = []
        for v in vecs:
            nums = parse_numbers(v)
            if nums and abs(sum(nums) - 1.0) < 0.05:  # looks like a prob vector
                parsed_vecs.append(nums)
        # pair up vecs (sigma_r, sigma_c)
        if len(parsed_vecs) >= 2:
            result["mixed_ne"].append((parsed_vecs[0], parsed_vecs[1]))

    return result


def parse_br_response(text: str, row_labels: list) -> dict:
    """
    Parse model output for Task B.
    Returns:
        {
          "br_actions":      [label, ...],
          "expected_payoffs": [float, ...],
          "br_value":        float or None,
          "raw_answer":      str
        }
    """
    ans = extract_answer_block(text)
    result = {
        "br_actions":       [],
        "expected_payoffs": [],
        "br_value":         None,
        "raw_answer":       ans,
    }

    # Best response actions
    br_line = re.search(r"Best\s*response\s*action[s]?\s*:(.+?)(?:\n|$)",
                        ans, re.IGNORECASE)
    if br_line:
        seg = br_line.group(1)
        for lbl in row_labels:
            if re.search(re.escape(lbl), seg, re.IGNORECASE):
                result["br_actions"].append(lbl)

    # Expected payoffs
    eu_line = re.search(r"Expected\s*payoffs?\s*:(.+?)(?:\n|$)",
                        ans, re.IGNORECASE)
    if eu_line:
        result["expected_payoffs"] = parse_numbers(eu_line.group(1))

    # Best response value
    val_line = re.search(r"Best\s*response\s*value\s*:(.+?)(?:\n|$)",
                         ans, re.IGNORECASE)
    if val_line:
        nums = parse_numbers(val_line.group(1))
        if nums:
            result["br_value"] = nums[0]

    return result


# ── Metrics ───────────────────────────────────────────────────────

def eval_nash(parsed: dict, gt: dict, row_labels: list, col_labels: list) -> dict:
    """
    Returns per-sample metrics for Task A.
    """
    gt_eqs   = gt["equilibria"]
    gt_pure  = [(row_labels[int(round(e["sigma_row"].index(max(e["sigma_row"]))))],
                 col_labels[int(round(e["sigma_col"].index(max(e["sigma_col"]))))])
                for e in gt_eqs if e["is_pure"]]
    gt_mixed = [(e["sigma_row"], e["sigma_col"])
                for e in gt_eqs if not e["is_pure"]]

    # Pure NE accuracy
    pred_pure = parsed["pure_ne"]
    if not gt_pure:
        pure_precision = 1.0 if not pred_pure else 0.0
        pure_recall    = 1.0
    else:
        correct_pure = [p for p in pred_pure if p in gt_pure]
        pure_precision = len(correct_pure) / max(len(pred_pure), 1)
        pure_recall    = len(correct_pure) / len(gt_pure)
    pure_f1 = _f1(pure_precision, pure_recall)

    # Mixed NE: best-match L1 distance
    mixed_l1 = None
    if gt_mixed and parsed["mixed_ne"]:
        best = min(
            _mixed_l1(pm, gm)
            for pm in parsed["mixed_ne"]
            for gm in gt_mixed
        )
        mixed_l1 = round(best, 4)

    # Did model correctly identify equilibrium class?
    pred_class = _predict_eq_class(pred_pure, parsed["mixed_ne"])
    class_correct = (pred_class == gt["equilibrium_class"])

    return {
        "pure_precision":  round(pure_precision, 4),
        "pure_recall":     round(pure_recall, 4),
        "pure_f1":         round(pure_f1, 4),
        "mixed_l1":        mixed_l1,
        "class_correct":   class_correct,
        "pred_class":      pred_class,
        "gt_class":        gt["equilibrium_class"],
        "n_gt_pure":       len(gt_pure),
        "n_pred_pure":     len(pred_pure),
    }


def eval_br(parsed: dict, gt: dict, row_labels: list) -> dict:
    """
    Returns per-sample metrics for Task B.
    """
    gt_br_idx   = gt["best_response_actions"]
    gt_br_lbls  = [row_labels[i] for i in gt_br_idx]
    gt_eu       = gt["expected_payoffs"]
    gt_val      = gt["best_response_value"]

    pred_lbls   = parsed["br_actions"]
    pred_eu     = parsed["expected_payoffs"]
    pred_val    = parsed["br_value"]

    # Action accuracy
    if not pred_lbls:
        action_acc = 0.0
    else:
        correct = [l for l in pred_lbls if l in gt_br_lbls]
        action_acc = len(correct) / max(len(gt_br_lbls), len(pred_lbls))

    # Expected payoff MAE (if model provided EUs)
    eu_mae = None
    if pred_eu and len(pred_eu) == len(gt_eu):
        eu_mae = round(float(np.mean(np.abs(
            np.array(pred_eu) - np.array(gt_eu)))), 4)

    # Best response value error
    val_err = None
    if pred_val is not None:
        val_err = round(abs(pred_val - gt_val), 4)

    return {
        "action_accuracy": round(action_acc, 4),
        "eu_mae":          eu_mae,
        "val_error":       val_err,
        "gt_br_actions":   gt_br_lbls,
        "pred_br_actions": pred_lbls,
        "gt_is_unique":    gt["is_unique"],
    }


def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _mixed_l1(pm, gm):
    """L1 distance between two (sigma_r, sigma_c) pairs."""
    try:
        pr, pc = np.array(pm[0]), np.array(pm[1])
        gr, gc = np.array(gm[0]), np.array(gm[1])
        if len(pr) != len(gr) or len(pc) != len(gc):
            return 2.0  # incompatible dims → max penalty
        return float(np.sum(np.abs(pr - gr)) + np.sum(np.abs(pc - gc)))
    except Exception:
        return 2.0


def _predict_eq_class(pure_ne, mixed_ne):
    has_pure  = len(pure_ne)  > 0
    has_mixed = len(mixed_ne) > 0
    if has_pure and has_mixed: return "both"
    if has_pure:               return "pure"
    if has_mixed:              return "mixed"
    return "none"


# ── Main Evaluation Loop ──────────────────────────────────────────

@dataclass
class RunConfig:
    model:        str = DEFAULT_MODEL
    max_samples:  Optional[int] = None
    max_tokens:   int = MAX_TOKENS
    temperature:  float = TEMPERATURE
    task_filter:  Optional[str] = None   # "nash" | "br" | None
    variant:      str = VARIANT
    sample_seed:  int = 0


def run_evaluation(cfg: RunConfig):
    # Per-model output directory
    model_out = OUT_DIR / cfg.model
    model_out.mkdir(parents=True, exist_ok=True)

    client = make_client()

    # Load data
    samples = [json.loads(l) for l in DATA_PATH.read_text().splitlines() if l.strip()]

    # Filter by task
    if cfg.task_filter == "nash":
        samples = [s for s in samples if s["task"] == "nash_equilibrium"]
    elif cfg.task_filter == "br":
        samples = [s for s in samples if s["task"] == "best_response"]

    # Sub-sample if requested (stratified by task + difficulty)
    if cfg.max_samples and len(samples) > cfg.max_samples:
        rng = np.random.default_rng(cfg.sample_seed)
        idx = rng.choice(len(samples), cfg.max_samples, replace=False)
        samples = [samples[i] for i in sorted(idx)]

    print(f"Evaluating {len(samples)} samples on {cfg.model}")
    print(f"  Base URL : {BASE_URL}")
    print(f"  Variant  : {cfg.variant}")
    print(f"  Tasks    : {set(s['task'] for s in samples)}")
    print()

    results = []
    nash_metrics_all, br_metrics_all = [], []
    errors = 0

    for i, sample in enumerate(samples):
        task     = sample["task"]
        desc     = sample["descriptions"][cfg.variant]
        row_lbl  = sample["row_labels"]
        col_lbl  = sample["col_labels"]
        gt       = sample["ground_truth"]

        # Build prompt
        if task == "nash_equilibrium":
            user_msg = NASH_USER_TEMPLATE.format(description=desc)
        else:
            user_msg = BR_USER_TEMPLATE.format(description=desc)

        # Call model
        try:
            response_text, usage = call_model(
                client, user_msg,
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] ERROR: {e}")
            errors += 1
            results.append({
                "id": sample["id"], "task": task,
                "error": str(e), "metrics": None,
            })
            time.sleep(REQUEST_DELAY * 10)
            continue

        # Parse & score
        if task == "nash_equilibrium":
            parsed  = parse_nash_response(response_text, row_lbl, col_lbl)
            metrics = eval_nash(parsed, gt, row_lbl, col_lbl)
            nash_metrics_all.append(metrics)
        else:
            parsed  = parse_br_response(response_text, row_lbl)
            metrics = eval_br(parsed, gt, row_lbl)
            br_metrics_all.append(metrics)

        result_entry = {
            "id":           sample["id"],
            "task":         task,
            "game_type":    sample["game_type"],
            "dimensions":   sample["dimensions"],
            "difficulty":   sample["metadata"]["difficulty"],
            "eq_class":     gt.get("equilibrium_class"),
            "response":     response_text,
            "parsed":       parsed,
            "metrics":      metrics,
            "usage":        usage,
        }
        results.append(result_entry)

        # Progress log
        if task == "nash_equilibrium":
            status = (f"pure_f1={metrics['pure_f1']:.2f} "
                      f"class={'✓' if metrics['class_correct'] else '✗'}")
        else:
            status = (f"action_acc={metrics['action_accuracy']:.2f} "
                      f"val_err={metrics['val_error']}")
        print(f"  [{i+1:4d}/{len(samples)}] {sample['id'][:30]:<30} {status}  "
              f"({usage['latency_s']}s)")

        time.sleep(REQUEST_DELAY)

    # ── Aggregate metrics ────────────────────────────────────────
    summary = _aggregate(nash_metrics_all, br_metrics_all, results, errors,
                         model_name=cfg.model)
    _print_summary(summary)

    # ── Save outputs ─────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = model_out / f"results_{ts}.jsonl"
    summary_path = model_out / f"summary_{ts}.json"

    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults  → {results_path}")
    print(f"Summary  → {summary_path}")
    return results, summary


def _aggregate(nash_all, br_all, results, errors, model_name=DEFAULT_MODEL):
    def mean(vals): return round(float(np.mean(vals)), 4) if vals else None
    def pct(vals):  return round(float(np.mean(vals)) * 100, 2) if vals else None

    summary = {
        "model":          model_name,
        "total_samples":  len(results),
        "errors":         errors,
        "task_A_nash": None,
        "task_B_br":   None,
    }

    if nash_all:
        # by difficulty
        by_diff = _split_by(results, "nash_equilibrium", "difficulty")
        by_class = _split_by(results, "nash_equilibrium", "eq_class")
        by_dim  = _split_by(results, "nash_equilibrium", "dimensions")

        summary["task_A_nash"] = {
            "n": len(nash_all),
            "pure_f1":         mean([m["pure_f1"]       for m in nash_all]),
            "pure_precision":  mean([m["pure_precision"] for m in nash_all]),
            "pure_recall":     mean([m["pure_recall"]    for m in nash_all]),
            "class_accuracy":  pct( [m["class_correct"]  for m in nash_all]),
            "mixed_l1_mean":   mean([m["mixed_l1"] for m in nash_all
                                     if m["mixed_l1"] is not None]),
            "by_difficulty": {
                d: {
                    "n": len(ms),
                    "pure_f1":      mean([m["metrics"]["pure_f1"]     for m in ms]),
                    "class_acc_%":  pct( [m["metrics"]["class_correct"] for m in ms]),
                }
                for d, ms in by_diff.items()
            },
            "by_eq_class": {
                c: {
                    "n": len(ms),
                    "pure_f1":      mean([m["metrics"]["pure_f1"]     for m in ms]),
                    "class_acc_%":  pct( [m["metrics"]["class_correct"] for m in ms]),
                }
                for c, ms in by_class.items()
            },
        }

    if br_all:
        by_diff = _split_by(results, "best_response", "difficulty")
        by_dim  = _split_by(results, "best_response", "dimensions")

        summary["task_B_br"] = {
            "n": len(br_all),
            "action_accuracy":  mean([m["action_accuracy"] for m in br_all]),
            "eu_mae":           mean([m["eu_mae"]   for m in br_all
                                      if m["eu_mae"] is not None]),
            "val_error_mean":   mean([m["val_error"] for m in br_all
                                      if m["val_error"] is not None]),
            "by_difficulty": {
                d: {
                    "n": len(ms),
                    "action_acc":   mean([m["metrics"]["action_accuracy"] for m in ms]),
                    "val_err_mean": mean([m["metrics"]["val_error"]
                                         for m in ms if m["metrics"]["val_error"] is not None]),
                }
                for d, ms in by_diff.items()
            },
        }

    return summary


def _split_by(results, task, key):
    out = {}
    for r in results:
        if r["task"] != task or r.get("metrics") is None:
            continue
        v = str(r.get(key, "unknown"))
        out.setdefault(v, []).append(r)
    return out


def _print_summary(s):
    print("\n" + "═" * 60)
    print("  EVALUATION SUMMARY")
    print("═" * 60)
    print(f"  Model   : {s['model']}")
    print(f"  Samples : {s['total_samples']}  (errors: {s['errors']})")

    if (A := s.get("task_A_nash")):
        print(f"\n  ── Task A: Nash Equilibrium  (n={A['n']}) ──")
        print(f"     Pure NE  F1        : {A['pure_f1']}")
        print(f"     Pure NE  Precision : {A['pure_precision']}")
        print(f"     Pure NE  Recall    : {A['pure_recall']}")
        print(f"     Class accuracy %   : {A['class_accuracy']}")
        if A['mixed_l1_mean'] is not None:
            print(f"     Mixed NE L1 (mean) : {A['mixed_l1_mean']}")
        if A.get("by_difficulty"):
            print("     By difficulty:")
            for d, m in sorted(A["by_difficulty"].items()):
                print(f"       {d:8s}: n={m['n']:4d}  "
                      f"pure_f1={m['pure_f1']}  class_acc={m['class_acc_%']}%")
        if A.get("by_eq_class"):
            print("     By equilibrium class:")
            for c, m in sorted(A["by_eq_class"].items()):
                print(f"       {c:8s}: n={m['n']:4d}  "
                      f"pure_f1={m['pure_f1']}  class_acc={m['class_acc_%']}%")

    if (B := s.get("task_B_br")):
        print(f"\n  ── Task B: Best Response  (n={B['n']}) ──")
        print(f"     Action accuracy    : {B['action_accuracy']}")
        if B['eu_mae'] is not None:
            print(f"     EU MAE             : {B['eu_mae']}")
        if B['val_error_mean'] is not None:
            print(f"     BR value error     : {B['val_error_mean']}")
        if B.get("by_difficulty"):
            print("     By difficulty:")
            for d, m in sorted(B["by_difficulty"].items()):
                print(f"       {d:8s}: n={m['n']:4d}  "
                      f"action_acc={m['action_acc']}  val_err={m['val_err_mean']}")

    print("═" * 60)


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GameSolve-Bench evaluator")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model name as served by vLLM (--served-model-name)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help="Max tokens per completion (default: 1024)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="Sampling temperature (default: 0.0 = greedy)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit total samples (for quick tests)")
    parser.add_argument("--task", choices=["nash", "br"], default=None,
                        help="Only evaluate one task")
    parser.add_argument("--variant", default=VARIANT,
                        choices=["abstract", "story", "compact"],
                        help="NL description variant to use")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sub-sampling")
    args = parser.parse_args()

    cfg = RunConfig(
        model        = args.model,
        max_tokens   = args.max_tokens,
        temperature  = args.temperature,
        max_samples  = args.max_samples,
        task_filter  = args.task,
        variant      = args.variant,
        sample_seed  = args.seed,
    )
    run_evaluation(cfg)