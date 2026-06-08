"""
Reward function for GameSolve-Bench GRPO training.
Scores model-generated responses against ground truth.
Returns float in [0, 1].
"""
import re
import json
import numpy as np


def parse_numbers(s: str) -> list:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]


def extract_answer_block(text: str) -> str:
    idx = text.upper().rfind("ANSWER:")
    if idx == -1:
        return text
    return text[idx + 7:].strip()


def parse_nash_response(text: str, row_labels: list, col_labels: list) -> dict:
    ans = extract_answer_block(text)
    result = {"pure_ne": [], "mixed_ne": []}

    pure_section = ""
    m = re.search(r"Pure\s*NE\s*:(.+?)(?:Mixed\s*NE|$)", ans, re.IGNORECASE | re.DOTALL)
    if m:
        pure_section = m.group(1).strip()

    if pure_section and "none" not in pure_section.lower():
        for row_a in row_labels:
            for col_a in col_labels:
                pat = rf"\({re.escape(row_a)}\s*,\s*{re.escape(col_a)}\)"
                if re.search(pat, pure_section, re.IGNORECASE):
                    pair = (row_a, col_a)
                    if pair not in result["pure_ne"]:
                        result["pure_ne"].append(pair)

    mixed_section = ""
    m = re.search(r"Mixed\s*NE\s*:(.+?)$", ans, re.IGNORECASE | re.DOTALL)
    if m:
        mixed_section = m.group(1).strip()

    if mixed_section and "none" not in mixed_section.lower():
        vecs = re.findall(r"\[([^\]]+)\]", mixed_section)
        parsed_vecs = []
        for v in vecs:
            nums = parse_numbers(v)
            if nums and abs(sum(nums) - 1.0) < 0.05:
                parsed_vecs.append(nums)
        if len(parsed_vecs) >= 2:
            result["mixed_ne"].append((parsed_vecs[0], parsed_vecs[1]))

    return result


def parse_br_response(text: str, row_labels: list) -> dict:
    ans = extract_answer_block(text)
    result = {"br_actions": [], "expected_payoffs": [], "br_value": None}

    br_line = re.search(r"Best\s*response\s*action\(?s?\)?\s*:(.+?)(?:\n|$)", ans, re.IGNORECASE)
    if br_line:
        seg = br_line.group(1)
        for lbl in row_labels:
            if re.search(r'(?<![a-zA-Z])' + re.escape(lbl) + r'(?![a-zA-Z])', seg, re.IGNORECASE):
                result["br_actions"].append(lbl)

    eu_line = re.search(r"Expected\s*payoffs?\s*:(.+?)(?:\n|$)", ans, re.IGNORECASE)
    if eu_line:
        result["expected_payoffs"] = parse_numbers(eu_line.group(1))

    val_line = re.search(r"Best\s*response\s*value\s*:(.+?)(?:\n|$)", ans, re.IGNORECASE)
    if val_line:
        nums = parse_numbers(val_line.group(1))
        if nums:
            result["br_value"] = nums[0]

    return result


def compute_reward(response_text: str, ground_truth: dict, task: str,
                   row_labels: list, col_labels: list) -> float:
    """
    Compute scalar reward in [0, 1] for a game-solving response.
    Combines structural correctness and numerical accuracy.
    """
    if not response_text or not response_text.strip():
        return 0.0

    try:
        if task == "nash_equilibrium":
            return _reward_nash(response_text, ground_truth, row_labels, col_labels)
        else:
            return _reward_br(response_text, ground_truth, row_labels)
    except Exception:
        return 0.0


def _reward_nash(text, gt, row_labels, col_labels):
    parsed = parse_nash_response(text, row_labels, col_labels)

    gt_eqs = gt["equilibria"]
    gt_pure = [
        (row_labels[e["sigma_row"].index(max(e["sigma_row"]))],
         col_labels[e["sigma_col"].index(max(e["sigma_col"]))])
        for e in gt_eqs if e["is_pure"]
    ]

    score = 0.0
    total_weight = 0.0

    # Pure NE F1 (weight 0.5)
    if gt_pure:
        correct = [p for p in parsed["pure_ne"] if p in gt_pure]
        precision = len(correct) / max(len(parsed["pure_ne"]), 1)
        recall = len(correct) / len(gt_pure)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        f1 = 1.0 if not parsed["pure_ne"] else 0.0
    score += 0.5 * f1
    total_weight += 0.5

    # Equilibrium class identification (weight 0.3)
    pred_has_pure = len(parsed["pure_ne"]) > 0
    pred_has_mixed = len(parsed["mixed_ne"]) > 0
    gt_class = gt["equilibrium_class"]
    if gt_class == "pure":
        class_score = 1.0 if (pred_has_pure and not pred_has_mixed) else 0.0
    elif gt_class == "mixed":
        class_score = 1.0 if (pred_has_mixed and not pred_has_pure) else 0.0
    elif gt_class == "both":
        class_score = 1.0 if (pred_has_pure and pred_has_mixed) else (0.5 if (pred_has_pure or pred_has_mixed) else 0.0)
    else:
        class_score = 0.0
    score += 0.3 * class_score
    total_weight += 0.3

    # Mixed NE accuracy (weight 0.2)
    gt_mixed = [(e["sigma_row"], e["sigma_col"]) for e in gt_eqs if not e["is_pure"]]
    if gt_mixed and parsed["mixed_ne"]:
        best_l1 = min(
            np.sum(np.abs(np.array(pm[0]) - np.array(gm[0]))) +
            np.sum(np.abs(np.array(pm[1]) - np.array(gm[1])))
            for pm in parsed["mixed_ne"] for gm in gt_mixed
            if len(pm[0]) == len(gm[0]) and len(pm[1]) == len(gm[1])
        ) if any(len(pm[0]) == len(gm[0]) and len(pm[1]) == len(gm[1])
                 for pm in parsed["mixed_ne"] for gm in gt_mixed) else 2.0
        mixed_score = max(0.0, 1.0 - best_l1)
    elif not gt_mixed:
        mixed_score = 1.0 if not parsed["mixed_ne"] else 0.0
    else:
        mixed_score = 0.0
    score += 0.2 * mixed_score
    total_weight += 0.2

    return score / total_weight if total_weight > 0 else 0.0


def _reward_br(text, gt, row_labels):
    parsed = parse_br_response(text, row_labels)

    gt_br_lbls = [row_labels[i] for i in gt["best_response_actions"]]
    gt_eu = gt["expected_payoffs"]
    gt_val = gt["best_response_value"]

    score = 0.0

    # Action accuracy (weight 0.5)
    if parsed["br_actions"]:
        correct = [l for l in parsed["br_actions"] if l in gt_br_lbls]
        action_acc = len(correct) / max(len(gt_br_lbls), len(parsed["br_actions"]))
    else:
        action_acc = 0.0
    score += 0.5 * action_acc

    # Expected payoff MAE (weight 0.3)
    if parsed["expected_payoffs"] and len(parsed["expected_payoffs"]) == len(gt_eu):
        mae = float(np.mean(np.abs(np.array(parsed["expected_payoffs"]) - np.array(gt_eu))))
        eu_score = max(0.0, 1.0 - mae / 10.0)
    else:
        eu_score = 0.0
    score += 0.3 * eu_score

    # BR value error (weight 0.2)
    if parsed["br_value"] is not None:
        val_err = abs(parsed["br_value"] - gt_val)
        val_score = max(0.0, 1.0 - val_err / 10.0)
    else:
        val_score = 0.0
    score += 0.2 * val_score

    return score
