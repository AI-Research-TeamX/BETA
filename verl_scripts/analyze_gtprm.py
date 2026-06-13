"""Aggregate GT-PRM eval results (common answer-quality metric) by condition.
Reports ID + OOD overall/Nash/BR, mean±std over seeds, Welch t vs ORM."""
import json, glob, statistics as st
from pathlib import Path

W = Path("/root/storage/cuisijia.csj/vlmarch/BETA")
EVAL = W / "eval_results/gtprm"
CONDS = {"orm": "ORM (outcome)", "proc": "GT-PRM-Process", "hyb": "GT-PRM-Hybrid", "typed": "GT-PRM-Typed"}
SEEDS = [0, 1, 2]


def load(name, ood=False):
    f = EVAL / (f"{name}_ood" if ood else name) / ("ood_eval_summary.json" if ood else "eval_summary.json")
    if not f.exists():
        return None
    d = json.load(open(f))
    return {"overall": d["all"]["mean"],
            "nash": d.get("task/nash_equilibrium", {}).get("mean"),
            "br": d.get("task/best_response", {}).get("mean")}


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2: return None
    va, vb = st.variance(a), st.variance(b); na, nb = len(a), len(b)
    se = (va/na + vb/nb) ** 0.5
    return (st.mean(a) - st.mean(b)) / se if se else None


def agg(cond, ood=False):
    vals = [load(f"{cond}_s{s}", ood) for s in SEEDS]
    vals = [v for v in vals if v]
    if not vals: return None
    return {k: (st.mean([v[k] for v in vals if v[k] is not None]),
                st.stdev([v[k] for v in vals if v[k] is not None]) if len(vals) > 1 else 0.0)
            for k in ["overall", "nash", "br"]}, [v["overall"] for v in vals]


for ood in [False, True]:
    print(f"\n{'='*66}\n{'OOD (750)' if ood else 'ID (200)'}  — common answer-quality reward (mean±std, n=3 seeds)\n{'='*66}")
    print(f"{'condition':<18}{'overall':>16}{'nash':>12}{'br':>12}")
    orm_overall = None
    summary = {}
    for c, label in CONDS.items():
        a = agg(c, ood)
        if not a: print(f"{label:<18}  [no data]"); continue
        stats, overalls = a
        summary[c] = overalls
        if c == "orm": orm_overall = overalls
        row = f"{label:<18}"
        for k in ["overall", "nash", "br"]:
            m, sd = stats[k]; row += f"{m:>10.3f}±{sd:.3f}"[:16].rjust(16) if k == "overall" else f"{m:>12.3f}"
        print(row)
    if orm_overall:
        print(f"\n  vs ORM (overall, Welch t):")
        for c in ["proc", "hyb", "typed"]:
            if c in summary:
                d = st.mean(summary[c]) - st.mean(orm_overall)
                t = welch_t(summary[c], orm_overall)
                print(f"    {CONDS[c]:<18} Δ={d:+.4f}  t={t:.2f}" if t is not None else f"    {CONDS[c]}: Δ={d:+.4f}")

out = {}
for ood in [False, True]:
    key = "ood" if ood else "id"
    out[key] = {}
    for c in CONDS:
        a = agg(c, ood)
        if a: out[key][c] = {"overall_mean": st.mean(a[1]), "overall_vals": a[1]}
json.dump(out, open(W / "results/gtprm/gtprm_eval_summary.json", "w"), indent=1)
print("\nsaved results/gtprm/gtprm_eval_summary.json")
