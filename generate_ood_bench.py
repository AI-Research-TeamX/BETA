"""
Generate Out-of-Distribution (OOD) GameSolve benchmark data.

OOD dimensions vs original GameSolve-Bench:
  1. large_matrix:  5×5, 6×6 (training max: 4×4)
  2. non_integer:   continuous payoffs with decimals (training: integers)
  3. wide_range:    payoff in [-50, 50] (training: [-5, 5])
  4. novel_format:  mathematical / JSON / markdown-table descriptions
  5. asymmetric:    4×2, 2×5, 3×5, 5×3 (novel dimension combos)
  6. combined_hard: large + non-integer + wide range + novel format
"""

import json, random, hashlib, itertools
import numpy as np
import nashpy as nash

SEED = 2024
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── Reuse solvers from gamesolve_gen.py ──

def find_pure_nash(R, C):
    R, C = np.array(R), np.array(C)
    m, n = R.shape
    eqs = []
    for i, j in itertools.product(range(m), range(n)):
        if R[i, j] == R[:, j].max() and C[i, j] == C[i, :].max():
            eqs.append((i, j))
    return eqs

def find_all_nash(R, C, tol=1e-6):
    R, C = np.array(R, dtype=float), np.array(C, dtype=float)
    game = nash.Game(R, C)
    eqs = []
    try:
        for eq in game.support_enumeration():
            s1, s2 = eq
            if (s1 >= -tol).all() and (s2 >= -tol).all():
                s1 = np.clip(s1, 0, None); s1 /= s1.sum()
                s2 = np.clip(s2, 0, None); s2 /= s2.sum()
                eqs.append((np.round(s1, 6).tolist(), np.round(s2, 6).tolist()))
    except Exception:
        pass
    unique = []
    for eq in eqs:
        if not any(_eq_close(eq, u) for u in unique):
            unique.append(eq)
    return unique

def _eq_close(a, b, tol=1e-4):
    return np.allclose(a[0], b[0], atol=tol) and np.allclose(a[1], b[1], atol=tol)

def _is_pure(sigma, tol=1e-4):
    return any(abs(p - 1.0) < tol for p in sigma)

def classify_equilibria(eqs, m, n):
    pure = [e for e in eqs if _is_pure(e[0]) and _is_pure(e[1])]
    mixed = [e for e in eqs if not (_is_pure(e[0]) and _is_pure(e[1]))]
    if pure and mixed: return "both"
    if pure: return "pure"
    if mixed: return "mixed"
    return "none"

def compute_best_response(R, opponent_sigma):
    R = np.array(R, dtype=float)
    sigma = np.array(opponent_sigma, dtype=float)
    eu = R @ sigma
    max_val = eu.max()
    br_actions = np.where(np.abs(eu - max_val) < 1e-9)[0].tolist()
    br_mixed = np.zeros(len(eu))
    for a in br_actions:
        br_mixed[a] = 1.0 / len(br_actions)
    return {
        "expected_payoffs": np.round(eu, 6).tolist(),
        "best_response_actions": br_actions,
        "best_response_value": round(float(max_val), 6),
        "is_unique": len(br_actions) == 1,
        "br_mixed_strategy": np.round(br_mixed, 6).tolist(),
    }

def compute_nash_payoffs(R, C, sigma_r, sigma_c):
    R, C = np.array(R, dtype=float), np.array(C, dtype=float)
    s1, s2 = np.array(sigma_r), np.array(sigma_c)
    return round(float(s1 @ R @ s2), 6), round(float(s1 @ C @ s2), 6)

def sample_opponent_mixed(n_actions):
    x = rng.exponential(1.0, size=n_actions)
    x = x / x.sum()
    return np.round(x, 4).tolist()

# ── Action labels ──

def action_labels(n):
    if n <= 26:
        return [chr(65 + i) for i in range(n)]
    return [f"A{i+1}" for i in range(n)]

# ── OOD Description Formats ──

def desc_math_notation(R, C, row_labels, col_labels, role_r="Player 1", role_c="Player 2"):
    """Mathematical notation with matrices written out."""
    m, n = len(R), len(R[0])
    lines = [f"Let Γ = (N, S, u) be a two-player normal-form game where:"]
    lines.append(f"  N = {{{role_r}, {role_c}}}")
    lines.append(f"  S₁ = {{{', '.join(row_labels)}}}, S₂ = {{{', '.join(col_labels)}}}")
    lines.append(f"  u₁ (row player payoff matrix):")
    for i in range(m):
        lines.append(f"    [{', '.join(f'{R[i][j]:>7}' for j in range(n))}]")
    lines.append(f"  u₂ (column player payoff matrix):")
    for i in range(m):
        lines.append(f"    [{', '.join(f'{C[i][j]:>7}' for j in range(n))}]")
    return "\n".join(lines)

def desc_json_format(R, C, row_labels, col_labels, role_r="Player 1", role_c="Player 2"):
    """JSON-like structured format."""
    data = {
        "game": {
            "players": [role_r, role_c],
            "strategies": {role_r: row_labels, role_c: col_labels},
            "outcomes": {}
        }
    }
    for i, a in enumerate(row_labels):
        for j, b in enumerate(col_labels):
            data["game"]["outcomes"][f"({a},{b})"] = {
                role_r: R[i][j], role_c: C[i][j]
            }
    return f"Game specification:\n{json.dumps(data, indent=2)}"

def desc_markdown_table(R, C, row_labels, col_labels, role_r="Player 1", role_c="Player 2"):
    """Markdown table format."""
    m, n = len(R), len(R[0])
    lines = [f"# {m}×{n} Strategic Game"]
    lines.append(f"**{role_r}** (rows) vs **{role_c}** (columns)\n")
    header = "| | " + " | ".join(col_labels) + " |"
    sep = "|---|" + "|".join(["---"] * n) + "|"
    lines.append(header)
    lines.append(sep)
    for i in range(m):
        row = f"| {row_labels[i]} | " + " | ".join(
            f"({R[i][j]}, {C[i][j]})" for j in range(n)) + " |"
        lines.append(row)
    return "\n".join(lines)

def desc_enumerated(R, C, row_labels, col_labels, role_r="Player 1", role_c="Player 2"):
    """Enumerated outcomes format (flat list)."""
    m, n = len(R), len(R[0])
    lines = [f"Strategic interaction between {role_r} and {role_c}."]
    lines.append(f"{role_r} can choose from: {', '.join(row_labels)}")
    lines.append(f"{role_c} can choose from: {', '.join(col_labels)}")
    lines.append("Payoff outcomes:")
    for i in range(m):
        for j in range(n):
            lines.append(f"  - {row_labels[i]} vs {col_labels[j]}: "
                         f"{role_r} receives {R[i][j]}, {role_c} receives {C[i][j]}")
    return "\n".join(lines)

NOVEL_FORMATS = [desc_math_notation, desc_json_format, desc_markdown_table, desc_enumerated]
ORIGINAL_FORMATS = ["abstract", "story", "compact"]

# ── Reuse original description builders for some categories ──
GAME_CONTEXTS = [
    ("Market competition", "Firm 1", "Firm 2", "chooses output level"),
    ("Negotiation", "Buyer", "Seller", "sets offer"),
    ("Resource allocation", "Agent 1", "Agent 2", "selects resource"),
    ("Abstract game", "Row player", "Column player", "selects action"),
]

def format_matrix_text(R, C, row_labels, col_labels):
    m, n = len(R), len(R[0])
    w = max(8, max(len(str(R[i][j])) + len(str(C[i][j])) + 3
                   for i in range(m) for j in range(n)) + 1)
    header = "         " + "  ".join(f"{c:>{w}}" for c in col_labels)
    lines = [header]
    for i in range(m):
        row_str = f"{row_labels[i]:>8} " + "  ".join(
            f"({R[i][j]},{C[i][j]})".rjust(w) for j in range(n))
        lines.append(row_str)
    return "\n".join(lines)

def build_abstract_desc(R, C, row_labels, col_labels, context):
    _, role_r, role_c, _ = context
    m, n = len(R), len(R[0])
    matrix_text = format_matrix_text(R, C, row_labels, col_labels)
    return (
        f"Consider a two-player strategic-form game.\n"
        f"{role_r} has {m} actions: {{{', '.join(row_labels)}}}.\n"
        f"{role_c} has {n} actions: {{{', '.join(col_labels)}}}.\n"
        f"The payoff matrix (row player, col player) is:\n\n"
        f"{matrix_text}\n\n"
        f"Each cell shows (payoff to {role_r}, payoff to {role_c})."
    )

def build_br_abstract_desc(R, C, row_labels, col_labels, context, opponent_sigma):
    _, role_r, role_c, _ = context
    m, n = len(R), len(R[0])
    matrix_text = format_matrix_text(R, C, row_labels, col_labels)
    sigma_str = ", ".join(
        f"P({col_labels[j]})={opponent_sigma[j]:.4f}" for j in range(n))
    return (
        f"Consider a two-player strategic-form game.\n"
        f"{role_r} has {m} actions: {{{', '.join(row_labels)}}}.\n"
        f"{role_c} has {n} actions: {{{', '.join(col_labels)}}}.\n"
        f"The payoff matrix (row player, col player) is:\n\n"
        f"{matrix_text}\n\n"
        f"{role_c} plays the following mixed strategy: {sigma_str}.\n"
        f"What is {role_r}'s best response?"
    )

# ── OOD Sample Generators ──

def gen_payoffs(m, n, low=-5, high=5, integer=True):
    if integer:
        R = rng.integers(low, high + 1, size=(m, n)).tolist()
        C = rng.integers(low, high + 1, size=(m, n)).tolist()
    else:
        R = np.round(rng.uniform(low, high, size=(m, n)), 2).tolist()
        C = np.round(rng.uniform(low, high, size=(m, n)), 2).tolist()
    return R, C

def build_ood_nash_sample(m, n, ood_category, payoff_low=-5, payoff_high=5,
                          integer=True, desc_fn=None):
    R, C = gen_payoffs(m, n, payoff_low, payoff_high, integer)
    eqs = find_all_nash(R, C)
    if not eqs:
        return None

    pure_eqs = find_pure_nash(R, C)
    eq_class = classify_equilibria(eqs, m, n)
    row_labels = action_labels(m)
    col_labels = action_labels(n)
    context = random.choice(GAME_CONTEXTS)
    _, role_r, role_c, _ = context

    if desc_fn is not None:
        description = desc_fn(R, C, row_labels, col_labels, role_r, role_c)
    else:
        description = build_abstract_desc(R, C, row_labels, col_labels, context)

    gt_eqs = []
    for s1, s2 in eqs:
        eu_r, eu_c = compute_nash_payoffs(R, C, s1, s2)
        gt_eqs.append({
            "sigma_row": s1, "sigma_col": s2,
            "is_pure": _is_pure(s1) and _is_pure(s2),
            "payoffs": [eu_r, eu_c],
        })

    sample_id = hashlib.md5(
        json.dumps({"R": R, "C": C, "ood": ood_category}).encode()).hexdigest()[:10]

    return {
        "id": f"ood_nash_{m}x{n}_{ood_category}_{sample_id}",
        "task": "nash_equilibrium",
        "game_type": "general",
        "dimensions": [m, n],
        "row_labels": row_labels,
        "col_labels": col_labels,
        "payoff_matrix_row": R,
        "payoff_matrix_col": C,
        "descriptions": {"abstract": description},
        "ground_truth": {
            "equilibria": gt_eqs,
            "n_equilibria": len(eqs),
            "equilibrium_class": eq_class,
            "pure_ne_count": len(pure_eqs),
            "mixed_ne_count": len(eqs) - len(pure_eqs),
        },
        "metadata": {
            "difficulty": "ood",
            "ood_category": ood_category,
        },
    }

def build_ood_br_sample(m, n, ood_category, payoff_low=-5, payoff_high=5,
                        integer=True, desc_fn=None):
    R, C = gen_payoffs(m, n, payoff_low, payoff_high, integer)
    opponent_sigma = sample_opponent_mixed(n)
    row_labels = action_labels(m)
    col_labels = action_labels(n)
    context = random.choice(GAME_CONTEXTS)
    _, role_r, role_c, _ = context

    br = compute_best_response(R, opponent_sigma)

    if desc_fn is not None:
        sigma_str = ", ".join(
            f"P({col_labels[j]})={opponent_sigma[j]:.4f}" for j in range(n))
        base_desc = desc_fn(R, C, row_labels, col_labels, role_r, role_c)
        description = (
            f"{base_desc}\n\n"
            f"{role_c} plays the following mixed strategy: {sigma_str}.\n"
            f"What is {role_r}'s best response?"
        )
    else:
        description = build_br_abstract_desc(
            R, C, row_labels, col_labels, context, opponent_sigma)

    sample_id = hashlib.md5(
        json.dumps({"R": R, "sigma": opponent_sigma, "ood": ood_category}).encode()
    ).hexdigest()[:10]

    return {
        "id": f"ood_br_{m}x{n}_{ood_category}_{sample_id}",
        "task": "best_response",
        "game_type": "general",
        "dimensions": [m, n],
        "row_labels": row_labels,
        "col_labels": col_labels,
        "payoff_matrix_row": R,
        "payoff_matrix_col": C,
        "opponent_sigma": opponent_sigma,
        "descriptions": {"abstract": description},
        "ground_truth": br,
        "metadata": {
            "difficulty": "ood",
            "ood_category": ood_category,
        },
    }


# ── OOD Generation Plan ──

OOD_PLAN = [
    # (task, m, n, count, ood_category, payoff_low, payoff_high, integer, desc_fn)

    # Category 1: Large matrices (5×5, 6×6) — never seen in training
    ("nash", 5, 5, 40, "large_matrix", -5, 5, True, None),
    ("nash", 6, 6, 30, "large_matrix", -5, 5, True, None),
    ("br",   5, 5, 40, "large_matrix", -5, 5, True, None),
    ("br",   6, 6, 30, "large_matrix", -5, 5, True, None),

    # Category 2: Non-integer payoffs (continuous values)
    ("nash", 2, 2, 30, "non_integer", -5, 5, False, None),
    ("nash", 3, 3, 30, "non_integer", -5, 5, False, None),
    ("br",   2, 2, 30, "non_integer", -5, 5, False, None),
    ("br",   3, 3, 30, "non_integer", -5, 5, False, None),

    # Category 3: Wide payoff range [-50, 50]
    ("nash", 2, 2, 30, "wide_range", -50, 50, True, None),
    ("nash", 3, 3, 30, "wide_range", -50, 50, True, None),
    ("br",   2, 2, 30, "wide_range", -50, 50, True, None),
    ("br",   3, 3, 30, "wide_range", -50, 50, True, None),

    # Category 4: Novel description formats (same game sizes as training)
    ("nash", 2, 2, 15, "novel_format_math", -5, 5, True, desc_math_notation),
    ("nash", 3, 3, 15, "novel_format_math", -5, 5, True, desc_math_notation),
    ("nash", 2, 2, 15, "novel_format_json", -5, 5, True, desc_json_format),
    ("nash", 3, 3, 15, "novel_format_json", -5, 5, True, desc_json_format),
    ("nash", 2, 2, 15, "novel_format_table", -5, 5, True, desc_markdown_table),
    ("nash", 3, 3, 15, "novel_format_table", -5, 5, True, desc_markdown_table),
    ("br",   2, 2, 15, "novel_format_math", -5, 5, True, desc_math_notation),
    ("br",   3, 3, 15, "novel_format_math", -5, 5, True, desc_math_notation),
    ("br",   2, 2, 15, "novel_format_json", -5, 5, True, desc_json_format),
    ("br",   3, 3, 15, "novel_format_json", -5, 5, True, desc_json_format),

    # Category 5: Asymmetric dimensions (novel combos)
    ("nash", 4, 2, 25, "asymmetric", -5, 5, True, None),
    ("nash", 2, 5, 25, "asymmetric", -5, 5, True, None),
    ("nash", 3, 5, 25, "asymmetric", -5, 5, True, None),
    ("br",   4, 2, 25, "asymmetric", -5, 5, True, None),
    ("br",   2, 5, 25, "asymmetric", -5, 5, True, None),
    ("br",   5, 3, 25, "asymmetric", -5, 5, True, None),

    # Category 6: Combined hard (large + non-integer + wide range + novel format)
    ("nash", 5, 5, 20, "combined_hard", -50, 50, False, desc_math_notation),
    ("nash", 6, 6, 15, "combined_hard", -50, 50, False, desc_json_format),
    ("br",   5, 5, 20, "combined_hard", -50, 50, False, desc_math_notation),
    ("br",   6, 6, 15, "combined_hard", -50, 50, False, desc_json_format),
]


def generate_ood_dataset():
    samples = []
    stats = {}

    for task, m, n, count, ood_cat, low, high, integer, desc_fn in OOD_PLAN:
        key = f"{task}_{m}x{n}_{ood_cat}"
        generated = 0
        attempts = 0
        max_attempts = count * 10

        while generated < count and attempts < max_attempts:
            attempts += 1
            try:
                if task == "nash":
                    s = build_ood_nash_sample(m, n, ood_cat, low, high, integer, desc_fn)
                    if s is None:
                        continue
                else:
                    s = build_ood_br_sample(m, n, ood_cat, low, high, integer, desc_fn)
                samples.append(s)
                generated += 1
            except Exception as e:
                continue

        stats[key] = {"target": count, "generated": generated, "attempts": attempts}
        print(f"  {key}: {generated}/{count}")

    out_path = "gamesolve_ood_bench.jsonl"
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Summary stats
    from collections import Counter
    cats = Counter(s["metadata"]["ood_category"] for s in samples)
    tasks = Counter(s["task"] for s in samples)

    summary = {
        "total": len(samples),
        "by_task": dict(tasks),
        "by_ood_category": dict(cats),
        "by_config": stats,
    }

    with open("gamesolve_ood_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal OOD samples: {len(samples)}")
    print(f"By task: {dict(tasks)}")
    print(f"By OOD category: {dict(cats)}")

    return samples, summary


if __name__ == "__main__":
    generate_ood_dataset()
