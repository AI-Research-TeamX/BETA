"""
Custom reward function for verl's GameSolve GRPO training.
Wraps gamesolve_reward.compute_reward to match verl's compute_score interface.

verl calls: compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
  - ground_truth: JSON string of the ground truth dict
  - extra_info: dict with task, row_labels, col_labels, etc.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gamesolve_reward import compute_reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "gamesolve":
        raise NotImplementedError(f"Unsupported data_source: {data_source}")

    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    task = extra_info.get("task", "best_response")
    row_labels = extra_info.get("row_labels", [])
    col_labels = extra_info.get("col_labels", [])

    score = compute_reward(solution_str, ground_truth, task, row_labels, col_labels)
    return float(score)
