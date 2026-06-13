"""
verl reward function for GT-PRM experiments.

Mode is selected by env var GT_PRM_MODE:
  outcome  : answer-quality reward only (ORM baseline = existing gamesolve_reward)
  process  : pure GT-PRM step-verification score
  hybrid   : 0.5*process + 0.5*outcome

Variant (process/hybrid) by GT_PRM_VARIANT:
  binary   : all step types weighted equally
  typed    : NE > BR > exp_payoff > dominance

Requires payoff matrices in extra_info (see preprocess_gtprm_verl.py).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gamesolve_reward import compute_reward
from gt_prm_verifier import verify_trace, TYPE_WEIGHTS, BINARY_WEIGHTS

MODE = os.environ.get("GT_PRM_MODE", "outcome")
VARIANT = os.environ.get("GT_PRM_VARIANT", "binary")
HYBRID_W = float(os.environ.get("GT_PRM_HYBRID_W", "0.5"))  # weight on process in hybrid


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "gamesolve":
        raise NotImplementedError(f"Unsupported data_source: {data_source}")
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    extra_info = extra_info or {}
    task = extra_info.get("task", "best_response")
    row_labels = list(extra_info.get("row_labels", []))
    col_labels = list(extra_info.get("col_labels", []))

    outcome = compute_reward(solution_str, ground_truth, task, row_labels, col_labels)
    if MODE == "outcome":
        return float(outcome)

    # process / hybrid need payoff matrices
    A = extra_info.get("payoff_matrix_row")
    B = extra_info.get("payoff_matrix_col")
    if A is None or B is None:
        return float(outcome)  # fail safe → outcome
    # parquet may store nested lists as numpy arrays
    A = [list(map(float, r)) for r in A]
    B = [list(map(float, r)) for r in B]

    weights = TYPE_WEIGHTS if VARIANT == "typed" else BINARY_WEIGHTS
    game = {"payoff_matrix_row": A, "payoff_matrix_col": B,
            "row_labels": row_labels, "col_labels": col_labels,
            "task": task, "ground_truth": ground_truth}
    proc = verify_trace(solution_str, game, weights=weights)["process_score"]

    if MODE == "process":
        return float(proc)
    if MODE == "hybrid":
        return float(HYBRID_W * proc + (1 - HYBRID_W) * outcome)
    raise ValueError(f"Unknown GT_PRM_MODE: {MODE}")
