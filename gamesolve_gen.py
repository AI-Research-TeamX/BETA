"""
GameSolve-Bench  —  Dataset Generator
======================================
Task A: Nash Equilibrium  (pure + mixed, 2-player normal-form)
Task B: Best Response      (given opponent mixed strategy)

Output: JSONL  —  one sample per line
"""

import json, random, itertools, hashlib, textwrap
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np
import nashpy as nash

# ── reproducibility ───────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)
random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
# 1.  GAME GENERATOR
# ═══════════════════════════════════════════════════════════════════

def sample_payoffs(m: int, n: int,
                   low: int = -5, high: int = 5,
                   integer: bool = True) -> tuple:
    """Sample payoff matrices for row and col player."""
    if integer:
        R = rng.integers(low, high + 1, size=(m, n))
        C = rng.integers(low, high + 1, size=(m, n))
    else:
        R = np.round(rng.uniform(low, high, size=(m, n)), 2)
        C = np.round(rng.uniform(low, high, size=(m, n)), 2)
    return R.tolist(), C.tolist()


def sample_zero_sum(m: int, n: int, low=-5, high=5) -> tuple:
    R = rng.integers(low, high + 1, size=(m, n))
    return R.tolist(), (-R).tolist()


def sample_symmetric(n: int, low=-5, high=5) -> tuple:
    """Symmetric game: C = R.T"""
    R = rng.integers(low, high + 1, size=(n, n))
    return R.tolist(), R.T.tolist()


def sample_opponent_mixed(n_actions: int) -> list:
    """Sample a random mixed strategy (probability simplex)."""
    x = rng.exponential(1.0, size=n_actions)
    x = x / x.sum()
    return np.round(x, 4).tolist()


# ═══════════════════════════════════════════════════════════════════
# 2.  SOLVERS  (exact, verifiable)
# ═══════════════════════════════════════════════════════════════════

def find_pure_nash(R, C) -> list:
    """Enumerate all pure-strategy Nash equilibria."""
    R, C = np.array(R), np.array(C)
    m, n = R.shape
    eqs = []
    for i, j in itertools.product(range(m), range(n)):
        # row player: no profitable deviation
        if R[i, j] == R[:, j].max():
            # col player: no profitable deviation
            if C[i, j] == C[i, :].max():
                eqs.append((i, j))
    return eqs


def find_all_nash(R, C, tol=1e-6) -> list:
    """
    Find all Nash equilibria (pure + mixed) via nashpy support enumeration.
    Returns list of (sigma_row, sigma_col) as numpy arrays.
    """
    R, C = np.array(R, dtype=float), np.array(C, dtype=float)
    game = nash.Game(R, C)
    eqs = []
    try:
        for eq in game.support_enumeration():
            s1, s2 = eq
            if (s1 >= -tol).all() and (s2 >= -tol).all():
                s1 = np.clip(s1, 0, None); s1 /= s1.sum()
                s2 = np.clip(s2, 0, None); s2 /= s2.sum()
                eqs.append((np.round(s1, 6).tolist(),
                             np.round(s2, 6).tolist()))
    except Exception:
        pass
    # deduplicate
    unique = []
    for eq in eqs:
        if not any(_eq_close(eq, u) for u in unique):
            unique.append(eq)
    return unique


def _eq_close(a, b, tol=1e-4):
    return (np.allclose(a[0], b[0], atol=tol) and
            np.allclose(a[1], b[1], atol=tol))


def classify_equilibria(eqs: list, m: int, n: int) -> str:
    """pure / mixed / both / none"""
    pure  = [e for e in eqs if _is_pure(e[0]) and _is_pure(e[1])]
    mixed = [e for e in eqs if not (_is_pure(e[0]) and _is_pure(e[1]))]
    if pure and mixed: return "both"
    if pure:           return "pure"
    if mixed:          return "mixed"
    return "none"


def _is_pure(sigma, tol=1e-4):
    return any(abs(p - 1.0) < tol for p in sigma)


def compute_best_response(R, opponent_sigma: list) -> dict:
    """
    Task B: given row player's payoff matrix R and opponent's
    mixed strategy sigma (col player), find row player's best response.

    Returns: {
        "expected_payoffs": [EU for each row action],
        "best_response_actions": [list of BR pure actions],
        "best_response_value": float,
        "is_unique": bool,
        "br_mixed_strategies": list of valid BR mixed strategies
    }
    """
    R = np.array(R, dtype=float)
    sigma = np.array(opponent_sigma, dtype=float)
    eu = R @ sigma                          # expected utility per row action
    max_val = eu.max()
    br_actions = np.where(np.abs(eu - max_val) < 1e-9)[0].tolist()
    is_unique = (len(br_actions) == 1)

    # Any mixture over BR actions is also a BR
    # Return canonical: uniform over BR actions
    br_mixed = np.zeros(len(eu))
    for a in br_actions:
        br_mixed[a] = 1.0 / len(br_actions)

    return {
        "expected_payoffs": np.round(eu, 6).tolist(),
        "best_response_actions": br_actions,
        "best_response_value": round(float(max_val), 6),
        "is_unique": is_unique,
        "br_mixed_strategy": np.round(br_mixed, 6).tolist(),
    }


def compute_nash_payoffs(R, C, sigma_r, sigma_c) -> tuple:
    R, C = np.array(R, dtype=float), np.array(C, dtype=float)
    s1, s2 = np.array(sigma_r), np.array(sigma_c)
    return round(float(s1 @ R @ s2), 6), round(float(s1 @ C @ s2), 6)


# ═══════════════════════════════════════════════════════════════════
# 3.  NATURAL LANGUAGE DESCRIPTION GENERATOR
# ═══════════════════════════════════════════════════════════════════

ACTION_LABELS = {
    2: [["A", "B"], ["Up", "Down"], ["Left", "Right"],
        ["Cooperate", "Defect"], ["High", "Low"]],
    3: [["A", "B", "C"], ["Rock", "Paper", "Scissors"],
        ["Low", "Mid", "High"]],
    4: [["A", "B", "C", "D"], ["Q1", "Q2", "Q3", "Q4"]],
}

GAME_CONTEXTS = [
    # (name, row_role, col_role, action_verb)
    ("Market competition",
     "Firm 1", "Firm 2", "chooses output level"),
    ("Negotiation",
     "Buyer",  "Seller", "sets offer"),
    ("Political strategy",
     "Party A", "Party B", "selects campaign strategy"),
    ("Network routing",
     "Router 1", "Router 2", "selects path"),
    ("Abstract game",
     "Row player", "Column player", "selects action"),
    ("Auction",
     "Bidder 1", "Bidder 2", "places bid"),
    ("Arms race",
     "Country A", "Country B", "chooses defense level"),
]


def action_label_set(n: int) -> list:
    pool = ACTION_LABELS.get(n, [])
    if pool:
        return random.choice(pool)
    return [f"A{i+1}" for i in range(n)]


def format_matrix_text(R, C, row_labels, col_labels) -> str:
    """Human-readable payoff table."""
    m, n = len(R), len(R[0])
    header = "         " + "  ".join(f"{c:>8}" for c in col_labels)
    lines = [header]
    for i in range(m):
        row_str = f"{row_labels[i]:>8} " + "  ".join(
            f"({R[i][j]:>3},{C[i][j]:>3})" for j in range(n))
        lines.append(row_str)
    return "\n".join(lines)


def build_game_description(R, C, row_labels, col_labels,
                           context: tuple, variant: str = "abstract") -> str:
    name, role_r, role_c, verb = context
    m, n = len(R), len(R[0])
    matrix_text = format_matrix_text(R, C, row_labels, col_labels)

    if variant == "abstract":
        desc = (
            f"Consider a two-player strategic-form game.\n"
            f"{role_r} has {m} actions: {{{', '.join(row_labels)}}}.\n"
            f"{role_c} has {n} actions: {{{', '.join(col_labels)}}}.\n"
            f"The payoff matrix (row player, col player) is:\n\n"
            f"{matrix_text}\n\n"
            f"Each cell shows (payoff to {role_r}, payoff to {role_c})."
        )
    elif variant == "story":
        desc = (
            f"Scenario: {name}.\n"
            f"{role_r} and {role_c} simultaneously {verb}.\n"
            f"Available strategies — {role_r}: {{{', '.join(row_labels)}}}; "
            f"{role_c}: {{{', '.join(col_labels)}}}.\n"
            f"Payoffs (first number = {role_r}'s gain, "
            f"second = {role_c}'s gain):\n\n"
            f"{matrix_text}"
        )
    elif variant == "compact":
        # Minimal format: just enumerate outcomes
        lines = []
        for i, a in enumerate(row_labels):
            for j, b in enumerate(col_labels):
                lines.append(
                    f"  If {role_r} plays {a} and {role_c} plays {b}: "
                    f"payoffs = ({R[i][j]}, {C[i][j]})")
        desc = (
            f"Two-player game ({name}).\n"
            + "\n".join(lines)
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return desc


def build_br_description(R, C, row_labels, col_labels,
                         context: tuple,
                         opponent_sigma: list,
                         variant: str = "abstract") -> str:
    name, role_r, role_c, verb = context
    m, n = len(R), len(R[0])
    matrix_text = format_matrix_text(R, C, row_labels, col_labels)

    sigma_str = ", ".join(
        f"P({col_labels[j]})={opponent_sigma[j]:.4f}"
        for j in range(n))

    if variant == "abstract":
        desc = (
            f"Consider a two-player strategic-form game.\n"
            f"{role_r} has {m} actions: {{{', '.join(row_labels)}}}.\n"
            f"{role_c} has {n} actions: {{{', '.join(col_labels)}}}.\n"
            f"The payoff matrix (row player, col player) is:\n\n"
            f"{matrix_text}\n\n"
            f"{role_c} plays the following mixed strategy: {sigma_str}.\n"
            f"What is {role_r}'s best response?"
        )
    elif variant == "story":
        desc = (
            f"Scenario: {name}.\n"
            f"You are {role_r}. Your opponent ({role_c}) randomizes: {sigma_str}.\n"
            f"Payoff table:\n\n{matrix_text}\n\n"
            f"Which action (or mixture) maximizes your expected payoff?"
        )
    elif variant == "compact":
        lines = []
        for i, a in enumerate(row_labels):
            for j, b in enumerate(col_labels):
                lines.append(
                    f"  ({a}, {b}): ({R[i][j]}, {C[i][j]})")
        desc = (
            f"Game outcomes:\n" + "\n".join(lines) + "\n\n"
            f"{role_c}'s mixed strategy: {sigma_str}.\n"
            f"Find {role_r}'s best response."
        )
    return desc


# ═══════════════════════════════════════════════════════════════════
# 4.  CHAIN-OF-THOUGHT GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_nash_cot(R, C, row_labels, col_labels, eqs, role_r, role_c) -> str:
    R_arr, C_arr = np.array(R), np.array(C)
    m, n = R_arr.shape

    steps = []
    steps.append("## Step 1: Identify game structure")
    steps.append(
        f"This is a {m}×{n} two-player normal-form game. "
        f"{role_r} has {m} strategies, {role_c} has {n} strategies.")

    steps.append("\n## Step 2: Check for pure-strategy Nash equilibria")
    steps.append(
        "A pure-strategy NE requires: no player can gain by unilaterally deviating.")
    pure_eqs = []
    for i, j in itertools.product(range(m), range(n)):
        row_max = R_arr[:, j].max()
        col_max = C_arr[i, :].max()
        is_row_br = abs(R_arr[i, j] - row_max) < 1e-9
        is_col_br = abs(C_arr[i, j] - col_max) < 1e-9
        verdict = "✓ NE" if (is_row_br and is_col_br) else "✗"
        steps.append(
            f"  ({row_labels[i]}, {col_labels[j]}): "
            f"{role_r} payoff={R_arr[i,j]} (col-max={row_max}, BR={is_row_br}), "
            f"{role_c} payoff={C_arr[i,j]} (row-max={col_max}, BR={is_col_br}) → {verdict}")
        if is_row_br and is_col_br:
            pure_eqs.append((i, j))

    if pure_eqs:
        eq_strs = [f"({row_labels[i]}, {col_labels[j]})" for i,j in pure_eqs]
        steps.append(f"Pure-strategy NE found: {', '.join(eq_strs)}")
    else:
        steps.append("No pure-strategy NE exists.")

    # Mixed strategy NE check
    mixed_eqs = [e for e in eqs if not (_is_pure(e[0]) and _is_pure(e[1]))]
    if m == 2 and n == 2 and mixed_eqs:
        steps.append("\n## Step 3: Solve for mixed-strategy Nash equilibrium")
        steps.append(
            "For a 2×2 game, the mixed NE requires each player to be indifferent "
            "across their support actions.")
        eq = mixed_eqs[0]
        p = eq[0][0]  # row prob of action 0
        q = eq[1][0]  # col prob of action 0
        steps.append(
            f"  Let {role_r} mix with p = P({row_labels[0]}) = {p:.4f}, "
            f"P({row_labels[1]}) = {1-p:.4f}")
        steps.append(
            f"  Let {role_c} mix with q = P({col_labels[0]}) = {q:.4f}, "
            f"P({col_labels[1]}) = {1-q:.4f}")

        # Show indifference condition
        eu_r0 = C_arr[0, 0] * p + C_arr[1, 0] * (1 - p)
        eu_r1 = C_arr[0, 1] * p + C_arr[1, 1] * (1 - p)
        steps.append(
            f"  Verify col player indifference: "
            f"EU({col_labels[0]}) = {eu_r0:.4f}, EU({col_labels[1]}) = {eu_r1:.4f}")
    elif not mixed_eqs and not pure_eqs:
        steps.append("\n## Step 3: No equilibrium found via support enumeration")
        steps.append("(Game may require larger support or numerical methods.)")
    elif not mixed_eqs:
        steps.append(
            "\n## Step 3: Mixed-strategy NE check\n"
            "All Nash equilibria are pure-strategy. No interior mixed NE exists.")

    steps.append("\n## Step 4: Summary")
    eq_summaries = []
    for s1, s2 in eqs:
        eu_r, eu_c = compute_nash_payoffs(R, C, s1, s2)
        if _is_pure(s1) and _is_pure(s2):
            a = row_labels[s1.index(max(s1))]
            b = col_labels[s2.index(max(s2))]
            eq_summaries.append(f"Pure NE ({a}, {b}): payoffs = ({eu_r}, {eu_c})")
        else:
            eq_summaries.append(
                f"Mixed NE: {role_r} plays {s1}, {role_c} plays {s2}; "
                f"payoffs = ({eu_r}, {eu_c})")
    if eq_summaries:
        steps.append("Nash Equilibria:\n" + "\n".join(f"  {s}" for s in eq_summaries))
    else:
        steps.append("No Nash equilibrium was found.")

    return "\n".join(steps)


def generate_br_cot(R, C, row_labels, col_labels,
                    opponent_sigma, br_result,
                    role_r, role_c) -> str:
    R_arr = np.array(R, dtype=float)
    eu = br_result["expected_payoffs"]
    br_actions = br_result["best_response_actions"]
    br_val = br_result["best_response_value"]

    steps = []
    steps.append("## Step 1: Setup")
    n = len(opponent_sigma)
    sigma_str = ", ".join(
        f"P({col_labels[j]})={opponent_sigma[j]:.4f}" for j in range(n))
    steps.append(
        f"{role_c} plays mixed strategy: {sigma_str}.")
    steps.append(
        f"We compute the expected payoff to {role_r} for each of their actions.")

    steps.append("\n## Step 2: Compute expected payoffs")
    for i, a in enumerate(row_labels):
        terms = " + ".join(
            f"{R_arr[i,j]:.2f}×{opponent_sigma[j]:.4f}"
            for j in range(n))
        steps.append(f"  EU({a}) = {terms} = {eu[i]:.6f}")

    steps.append("\n## Step 3: Identify best response")
    steps.append(f"Maximum expected payoff = {br_val:.6f}")
    br_names = [row_labels[a] for a in br_actions]
    if br_result["is_unique"]:
        steps.append(
            f"Unique best response: {br_names[0]} "
            f"(strictly dominates all other actions).")
    else:
        steps.append(
            f"Multiple best responses: {{{', '.join(br_names)}}} "
            f"all yield EU = {br_val:.6f}.")
        steps.append(
            "Any mixture over these actions is also a best response. "
            f"Canonical BR (uniform): {br_result['br_mixed_strategy']}")

    steps.append("\n## Step 4: Conclusion")
    steps.append(
        f"Best response to {sigma_str}:\n"
        f"  Action(s): {{{', '.join(br_names)}}}\n"
        f"  Expected payoff: {br_val:.6f}")

    return "\n".join(steps)


# ═══════════════════════════════════════════════════════════════════
# 5.  SAMPLE BUILDERS
# ═══════════════════════════════════════════════════════════════════

def build_nash_sample(m: int, n: int,
                      game_type: str = "general",
                      variants: list = None) -> Optional[dict]:
    """Build one Task A sample. Returns None if no NE found."""
    if variants is None:
        variants = ["abstract", "story", "compact"]

    # Generate payoffs
    if game_type == "zero_sum":
        R, C = sample_zero_sum(m, n)
    elif game_type == "symmetric" and m == n:
        R, C = sample_symmetric(n)
    else:
        R, C = sample_payoffs(m, n)

    # Solve
    eqs = find_all_nash(R, C)
    if not eqs:
        return None  # skip degenerate cases

    pure_eqs  = find_pure_nash(R, C)
    eq_class  = classify_equilibria(eqs, m, n)

    # Labels & context
    row_labels = action_label_set(m)
    col_labels = action_label_set(n)
    context    = random.choice(GAME_CONTEXTS)
    _, role_r, role_c, _ = context

    # Build descriptions
    descs = {v: build_game_description(R, C, row_labels, col_labels, context, v)
             for v in variants}

    # CoT
    cot = generate_nash_cot(R, C, row_labels, col_labels, eqs, role_r, role_c)

    # Ground truth
    gt_eqs = []
    for s1, s2 in eqs:
        eu_r, eu_c = compute_nash_payoffs(R, C, s1, s2)
        gt_eqs.append({
            "sigma_row": s1,
            "sigma_col": s2,
            "is_pure": _is_pure(s1) and _is_pure(s2),
            "payoffs": [eu_r, eu_c],
        })

    sample_id = hashlib.md5(
        json.dumps({"R": R, "C": C}).encode()).hexdigest()[:10]

    return {
        "id": f"nash_{m}x{n}_{game_type}_{sample_id}",
        "task": "nash_equilibrium",
        "game_type": game_type,
        "dimensions": [m, n],
        "row_labels": row_labels,
        "col_labels": col_labels,
        "payoff_matrix_row": R,
        "payoff_matrix_col": C,
        "context": context[0],
        "role_row": role_r,
        "role_col": role_c,
        "descriptions": descs,
        "ground_truth": {
            "equilibria": gt_eqs,
            "n_equilibria": len(eqs),
            "equilibrium_class": eq_class,
            "pure_ne_count": len(pure_eqs),
            "mixed_ne_count": len(eqs) - len(pure_eqs),
        },
        "chain_of_thought": cot,
    }


def build_br_sample(m: int, n: int,
                    game_type: str = "general",
                    variants: list = None) -> dict:
    """Build one Task B sample."""
    if variants is None:
        variants = ["abstract", "story", "compact"]

    if game_type == "zero_sum":
        R, C = sample_zero_sum(m, n)
    elif game_type == "symmetric" and m == n:
        R, C = sample_symmetric(n)
    else:
        R, C = sample_payoffs(m, n)

    opponent_sigma = sample_opponent_mixed(n)

    row_labels = action_label_set(m)
    col_labels = action_label_set(n)
    context    = random.choice(GAME_CONTEXTS)
    _, role_r, role_c, _ = context

    br = compute_best_response(R, opponent_sigma)

    descs = {v: build_br_description(
                    R, C, row_labels, col_labels, context, opponent_sigma, v)
             for v in variants}

    cot = generate_br_cot(
        R, C, row_labels, col_labels, opponent_sigma, br, role_r, role_c)

    sample_id = hashlib.md5(
        json.dumps({"R": R, "sigma": opponent_sigma}).encode()).hexdigest()[:10]

    return {
        "id": f"br_{m}x{n}_{game_type}_{sample_id}",
        "task": "best_response",
        "game_type": game_type,
        "dimensions": [m, n],
        "row_labels": row_labels,
        "col_labels": col_labels,
        "payoff_matrix_row": R,
        "payoff_matrix_col": C,
        "opponent_sigma": opponent_sigma,
        "context": context[0],
        "role_row": role_r,
        "role_col": role_c,
        "descriptions": descs,
        "ground_truth": br,
        "chain_of_thought": cot,
    }


# ═══════════════════════════════════════════════════════════════════
# 6.  GENERATION PLAN
# ═══════════════════════════════════════════════════════════════════

GENERATION_PLAN = [
    # Task, m, n, game_type, count
    # ── Task A: Nash Equilibrium ──────────────────────────────────
    ("nash", 2, 2, "general",   400),
    ("nash", 2, 2, "zero_sum",  150),
    ("nash", 2, 2, "symmetric", 100),
    ("nash", 3, 3, "general",   200),
    ("nash", 3, 3, "zero_sum",  100),
    ("nash", 3, 3, "symmetric",  80),
    ("nash", 4, 4, "general",   100),
    ("nash", 2, 3, "general",   120),
    ("nash", 3, 2, "general",   100),
    # ── Task B: Best Response ────────────────────────────────────
    ("br",   2, 2, "general",   300),
    ("br",   2, 2, "zero_sum",  100),
    ("br",   3, 3, "general",   200),
    ("br",   3, 3, "zero_sum",   80),
    ("br",   4, 4, "general",   150),
    ("br",   2, 3, "general",   120),
    ("br",   3, 2, "general",   100),
]


def generate_dataset(plan=GENERATION_PLAN,
                     out_path="./gamesolve_bench.jsonl",
                     stats_path="./gamesolve_stats.json"):
    samples = []
    skipped = 0
    counters = {}

    for task, m, n, gtype, count in plan:
        key = f"{task}_{m}x{n}_{gtype}"
        counters[key] = {"target": count, "generated": 0, "skipped": 0}
        attempts = 0
        generated = 0
        max_attempts = count * 5

        while generated < count and attempts < max_attempts:
            attempts += 1
            try:
                if task == "nash":
                    s = build_nash_sample(m, n, gtype)
                    if s is None:
                        counters[key]["skipped"] += 1
                        skipped += 1
                        continue
                else:
                    s = build_br_sample(m, n, gtype)

                # add difficulty metadata
                s["metadata"] = {
                    "difficulty": _difficulty(task, m, n, s),
                    "generation_attempt": attempts,
                }
                samples.append(s)
                generated += 1
                counters[key]["generated"] += 1

            except Exception as e:
                counters[key]["skipped"] += 1
                skipped += 1

        print(f"  {key}: {generated}/{count} generated, "
              f"{counters[key]['skipped']} skipped")

    # Write JSONL
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Statistics
    stats = {
        "total_samples": len(samples),
        "total_skipped": skipped,
        "task_breakdown": {
            "nash_equilibrium": sum(1 for s in samples if s["task"] == "nash_equilibrium"),
            "best_response":    sum(1 for s in samples if s["task"] == "best_response"),
        },
        "by_config": counters,
        "equilibrium_class_dist": {},
        "br_uniqueness": {},
    }

    ne_classes = [s["ground_truth"].get("equilibrium_class")
                  for s in samples if s["task"] == "nash_equilibrium"]
    for c in set(ne_classes):
        stats["equilibrium_class_dist"][c] = ne_classes.count(c)

    br_unique = [s["ground_truth"]["is_unique"]
                 for s in samples if s["task"] == "best_response"]
    stats["br_uniqueness"] = {
        "unique": sum(br_unique),
        "non_unique": len(br_unique) - sum(br_unique),
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return samples, stats


def _difficulty(task, m, n, sample) -> str:
    if task == "nash":
        ec = sample["ground_truth"]["equilibrium_class"]
        if m <= 2 and n <= 2 and ec == "pure":   return "easy"
        if m <= 2 and n <= 2:                     return "medium"
        if ec == "mixed":                         return "hard"
        return "medium"
    else:  # br
        if m <= 2 and n <= 2:   return "easy"
        if m <= 3 and n <= 3:   return "medium"
        return "hard"


# ═══════════════════════════════════════════════════════════════════
# 7.  QUICK DEMO  (run standalone)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GameSolve-Bench  —  Sample Previews")
    print("=" * 60)

    # Task A demo
    print("\n>>> TASK A: Nash Equilibrium (2×2 general)\n")
    s = None
    while s is None:
        s = build_nash_sample(2, 2, "general")
    print("Description (abstract):")
    print(textwrap.indent(s["descriptions"]["abstract"], "  "))
    print("\nGround Truth:")
    for eq in s["ground_truth"]["equilibria"]:
        print(f"  sigma_row={eq['sigma_row']}  sigma_col={eq['sigma_col']}"
              f"  payoffs={eq['payoffs']}  pure={eq['is_pure']}")
    print(f"\nEquilibrium class: {s['ground_truth']['equilibrium_class']}")
    print("\nChain of Thought:")
    print(textwrap.indent(s["chain_of_thought"], "  "))

    # Task B demo
    print("\n" + "=" * 60)
    print(">>> TASK B: Best Response (3×3 general)\n")
    s2 = build_br_sample(3, 3, "general")
    print("Description (story):")
    print(textwrap.indent(s2["descriptions"]["story"], "  "))
    print("\nOpponent sigma:", s2["opponent_sigma"])
    print("\nGround Truth:")
    gt = s2["ground_truth"]
    print(f"  Expected payoffs: {gt['expected_payoffs']}")
    print(f"  Best response actions: {gt['best_response_actions']}")
    print(f"  Best response value: {gt['best_response_value']}")
    print(f"  Unique: {gt['is_unique']}")
    print("\nChain of Thought:")
    print(textwrap.indent(s2["chain_of_thought"], "  "))

    print("\n" + "=" * 60)
    print("Generating full dataset...")
    samples, stats = generate_dataset()
    print(f"\nDone! {stats['total_samples']} samples written.")
    print(f"  Nash: {stats['task_breakdown']['nash_equilibrium']}")
    print(f"  BR:   {stats['task_breakdown']['best_response']}")
    print(f"  Skipped: {stats['total_skipped']}")
    print(f"\nEquilibrium class distribution: {stats['equilibrium_class_dist']}")
    print(f"BR uniqueness: {stats['br_uniqueness']}")
