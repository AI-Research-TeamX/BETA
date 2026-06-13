"""
Game-Theoretic Process Reward (GT-PRM) verifier.

Extracts formally-verifiable intermediate reasoning claims from an LLM's free-form
game-solving trace and checks each against the ground-truth game structure (payoff
matrices + solver ground truth). Zero human annotation, zero rollouts.

Verifiable claim types:
  - DOMINANCE   : "<X> dominates <Y>"  /  "<Y> is dominated"   → check payoff matrix
  - EQUILIBRIUM : "(<row>, <col>) is a Nash equilibrium"        → mutual best response
  - BR_ACTION   : "best response is <action>"                   → vs ground-truth BR
  - EXP_PAYOFF  : "expected payoff for <action> is <v>"         → vs ground-truth EUs

Returns per-trace statistics and a scalar process score in [0, 1] that rewards
BOTH accuracy (fraction of claims correct) AND coverage (covering real solution facts),
so it cannot be trivially gamed by emitting one correct claim or many wrong ones.
"""
import re
import numpy as np

# weights for the "typed" variant (NE hardest → highest); binary uses all 1.0
TYPE_WEIGHTS = {"EQUILIBRIUM": 3.0, "BR_ACTION": 2.0, "EXP_PAYOFF": 1.5,
                "DOMINANCE": 1.0}
BINARY_WEIGHTS = {k: 1.0 for k in TYPE_WEIGHTS}


# ───────────────────────── game-theory primitives ─────────────────────────
def _row_weakly_dominates(A, i, k):
    """Row strategy i weakly dominates k for the ROW player (chooses rows)."""
    ge = all(A[i][j] >= A[k][j] for j in range(len(A[0])))
    gt = any(A[i][j] > A[k][j] for j in range(len(A[0])))
    return ge and gt


def _col_weakly_dominates(B, j, l):
    """Col strategy j weakly dominates l for the COL player (chooses cols)."""
    ge = all(B[i][j] >= B[i][l] for i in range(len(B)))
    gt = any(B[i][j] > B[i][l] for i in range(len(B)))
    return ge and gt


def _is_pure_ne(A, B, i, j):
    nrow, ncol = len(A), len(A[0])
    row_br = A[i][j] >= max(A[r][j] for r in range(nrow)) - 1e-9
    col_br = B[i][j] >= max(B[i][c] for c in range(ncol)) - 1e-9
    return row_br and col_br


def _count_dominated_strategies(A, B):
    nrow, ncol = len(A), len(A[0])
    rows = sum(1 for k in range(nrow) if any(_row_weakly_dominates(A, i, k)
                                             for i in range(nrow) if i != k))
    cols = sum(1 for l in range(ncol) if any(_col_weakly_dominates(B, j, l)
                                             for j in range(ncol) if j != l))
    return rows + cols


# ───────────────────────── claim extraction ─────────────────────────
def _label_alt(labels):
    return "|".join(re.escape(l) for l in labels)


def extract_dominance_claims(text, row_labels, col_labels):
    """Return list of (dominant_label, dominated_label)."""
    claims = []
    alll = _label_alt(row_labels + col_labels)
    # "X (strictly|weakly) dominates Y"
    for m in re.finditer(rf"\b({alll})\b\s+(?:strictly\s+|weakly\s+)?dominat\w*\s+(?:over\s+)?\b({alll})\b",
                         text, re.IGNORECASE):
        claims.append((m.group(1), m.group(2)))
    return claims


def extract_ne_claims(text, row_labels, col_labels):
    """Return list of (row_label, col_label) claimed as a (pure) Nash equilibrium.
    Only counts pairs appearing within ~60 chars of an NE keyword."""
    claims = []
    rl, cl = _label_alt(row_labels), _label_alt(col_labels)
    pair_pat = rf"\(\s*({rl})\s*,\s*({cl})\s*\)"
    for m in re.finditer(pair_pat, text, re.IGNORECASE):
        window = text[max(0, m.start() - 60): m.end() + 60].lower()
        if "nash" in window or re.search(r"\bne\b", window) or "equilibri" in window:
            claims.append((m.group(1), m.group(2)))
    return claims


def extract_br_action_claims(text, row_labels):
    """Return list of claimed best-response action labels (row player).
    Handles: 'best response is/: X', 'best response: X', 'Action(s): {X}'."""
    claims = []
    rl = _label_alt(row_labels)
    # 'best response' (optionally 'action(s)'), then a connector, then a label within a short window
    for m in re.finditer(r"best\s+response\b[^\n]{0,40}", text, re.IGNORECASE):
        seg = m.group(0)
        lm = re.search(rf"\b({rl})\b", seg[len("best response"):], re.IGNORECASE)
        if lm:
            claims.append(lm.group(1))
    # 'Action(s): {X}' restatement
    for m in re.finditer(rf"Action\(?s?\)?\s*[:=]\s*\{{?\s*({rl})\b", text, re.IGNORECASE):
        claims.append(m.group(1))
    return claims


def extract_exp_payoff_claims(text, row_labels):
    """Return list of (action_label, value) for per-action expected-payoff claims.
    Handles 'EU(X) = ... = v' and 'expected payoff for X ... v'."""
    claims = []
    rl = _label_alt(row_labels)
    # EU(X) = .... = value   (take the LAST number on the line as the result)
    for m in re.finditer(rf"(?:EU|expected\s+payoff)\s*(?:for\s+|\(\s*)?({rl})\b[^\n]*",
                         text, re.IGNORECASE):
        line = m.group(0)
        nums = re.findall(r"-?\d+\.?\d*", line)
        if nums:
            claims.append((m.group(1), float(nums[-1])))
    return claims


def _label_idx(label, labels):
    for i, l in enumerate(labels):
        if l.lower() == label.lower():
            return i
    return None


# ───────────────────────── verification ─────────────────────────
def verify_trace(response, game, weights=None):
    """
    game: dict with payoff_matrix_row (A), payoff_matrix_col (B), row_labels,
          col_labels, task, ground_truth(dict).
    Returns dict with counts and process_score in [0,1].
    """
    weights = weights or BINARY_WEIGHTS
    A = game["payoff_matrix_row"]
    B = game["payoff_matrix_col"]
    row_labels = game["row_labels"]
    col_labels = game["col_labels"]
    task = game.get("task", "nash_equilibrium")
    gt = game.get("ground_truth", {})

    correct_w = 0.0
    incorrect_w = 0.0
    by_type = {k: [0, 0] for k in TYPE_WEIGHTS}  # type -> [correct, incorrect]

    def record(t, ok):
        by_type[t][0 if ok else 1] += 1
        return weights[t] if ok else 0.0, 0.0 if ok else weights[t]

    # DOMINANCE
    for dom, dnt in extract_dominance_claims(response, row_labels, col_labels):
        if dom in row_labels and dnt in row_labels:
            i, k = _label_idx(dom, row_labels), _label_idx(dnt, row_labels)
            ok = i is not None and k is not None and i != k and _row_weakly_dominates(A, i, k)
        elif dom in col_labels and dnt in col_labels:
            j, l = _label_idx(dom, col_labels), _label_idx(dnt, col_labels)
            ok = j is not None and l is not None and j != l and _col_weakly_dominates(B, j, l)
        else:
            continue  # cross-set claim is ill-typed; skip
        c, w = record("DOMINANCE", ok)
        correct_w += c; incorrect_w += w

    # EQUILIBRIUM (Nash task)
    if task == "nash_equilibrium":
        for (rlab, clab) in extract_ne_claims(response, row_labels, col_labels):
            i, j = _label_idx(rlab, row_labels), _label_idx(clab, col_labels)
            ok = i is not None and j is not None and _is_pure_ne(A, B, i, j)
            c, w = record("EQUILIBRIUM", ok)
            correct_w += c; incorrect_w += w

    # BEST RESPONSE (BR task)
    if task == "best_response":
        gt_actions = set(gt.get("best_response_actions", []))
        for lab in extract_br_action_claims(response, row_labels):
            idx = _label_idx(lab, row_labels)
            ok = idx is not None and idx in gt_actions
            c, w = record("BR_ACTION", ok)
            correct_w += c; incorrect_w += w
        # per-action expected payoff claims: 'EU(label) = ... = v'
        eus = gt.get("expected_payoffs", [])
        if eus:
            for lab, v in extract_exp_payoff_claims(response, row_labels):
                idx = _label_idx(lab, row_labels)
                if idx is None or idx >= len(eus):
                    continue
                ok = abs(v - eus[idx]) < 0.10 * (abs(eus[idx]) + 1)
                c, w = record("EXP_PAYOFF", ok)
                correct_w += c; incorrect_w += w

    # ── aggregate: accuracy × coverage ──
    n_correct = sum(v[0] for v in by_type.values())
    n_incorrect = sum(v[1] for v in by_type.values())
    n_total = n_correct + n_incorrect
    accuracy = correct_w / (correct_w + incorrect_w) if (correct_w + incorrect_w) > 0 else 0.0

    # coverage target: real verifiable facts available in this game
    if task == "nash_equilibrium":
        target = gt.get("pure_ne_count", 1) + _count_dominated_strategies(A, B)
    else:
        target = 1 + len(gt.get("expected_payoffs", []))  # BR action + EUs
    target = max(1.0, float(target))
    coverage = min(1.0, correct_w / (target * np.mean(list((weights or BINARY_WEIGHTS).values()))))

    process_score = 0.5 * accuracy + 0.5 * coverage
    return {
        "process_score": float(process_score),
        "accuracy": float(accuracy),
        "coverage": float(coverage),
        "n_correct": n_correct, "n_incorrect": n_incorrect, "n_total": n_total,
        "by_type": by_type,
    }
