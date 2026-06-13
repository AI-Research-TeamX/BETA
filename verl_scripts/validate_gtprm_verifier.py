"""Offline validation of the GT-PRM verifier — the gate before any training.

Checks:
 1. Gold CoT traces should score HIGH (parser+verifier find the correct claims).
 2. Empty / shuffled-wrong traces should score LOW.
 3. process_score should correlate with answer-quality (outcome) reward on a mix,
    i.e. the verifier rewards genuinely-correct reasoning.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gt_prm_verifier import verify_trace, TYPE_WEIGHTS, BINARY_WEIGHTS
from gamesolve_reward import compute_reward

rows = [json.loads(l) for l in open("gamesolve_bench.jsonl")]
random.seed(0)
random.shuffle(rows)
sample = rows[:400]


def make_game(s):
    return {
        "payoff_matrix_row": s["payoff_matrix_row"],
        "payoff_matrix_col": s["payoff_matrix_col"],
        "row_labels": s["row_labels"], "col_labels": s["col_labels"],
        "task": s["task"], "ground_truth": s["ground_truth"],
    }


def stats(xs):
    import statistics as st
    return f"mean={st.mean(xs):.3f} std={st.stdev(xs):.3f} min={min(xs):.3f} max={max(xs):.3f}"


gold_scores, empty_scores, wrong_scores = [], [], []
gold_proc, gold_out = [], []
nash_gold, br_gold = [], []
for s in sample:
    g = make_game(s)
    gold = s["chain_of_thought"]
    r = verify_trace(gold, g)
    gold_scores.append(r["process_score"])
    (nash_gold if s["task"] == "nash_equilibrium" else br_gold).append(r["process_score"])
    gold_proc.append(r["process_score"])
    gold_out.append(compute_reward(gold, s["ground_truth"], s["task"], s["row_labels"], s["col_labels"]))
    # empty
    empty_scores.append(verify_trace("I don't know.", g)["process_score"])
    # wrong: gold CoT from a DIFFERENT random game (labels may mismatch → mostly unverifiable/incorrect)
    other = random.choice(sample)
    wrong_scores.append(verify_trace(other["chain_of_thought"], g)["process_score"])

# correlation
import statistics as st
def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da * db > 0 else 0.0

print(f"GOLD  CoT  process_score: {stats(gold_scores)}   (n={len(gold_scores)})")
print(f"  └ nash: {stats(nash_gold)}")
print(f"  └ br  : {stats(br_gold)}")
print(f"EMPTY      process_score: {stats(empty_scores)}")
print(f"WRONG-game process_score: {stats(wrong_scores)}")
print(f"corr(gold process, gold outcome reward) = {corr(gold_proc, gold_out):.3f}")
print()
# claim coverage: how often does the verifier find ANY claim on gold CoT?
any_claim = sum(1 for s in sample if verify_trace(s['chain_of_thought'], make_game(s))['n_total'] > 0)
print(f"gold CoT with >=1 verifiable claim: {any_claim}/{len(sample)} ({any_claim/len(sample):.0%})")
