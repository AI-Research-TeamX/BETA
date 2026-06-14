"""Aggregate Nash-GRPO results. All conditions share the same reward, so BOTH the
training val reward AND the external ID/OOD eval are comparable across conditions.
Reports mean±std over seeds, Welch t vs GRPO. Also runs the offline intransitive-
preference illustration (Nash Condorcet-consistency vs mean-baseline)."""
import json, glob, statistics as st, sys
from pathlib import Path
W = Path("/root/storage/cuisijia.csj/vlmarch/BETA")
EVAL = W / "eval_results/nashgrpo"
CONDS = {"grpo": "GRPO (mean)", "nash4": "Nash-GRPO (β=4)", "rank": "rank baseline",
         "nash1": "Nash-GRPO (β=1)", "nash16": "Nash-GRPO (β=16)"}
SEEDS = [0, 1, 2]


def load_eval(name, ood=False):
    f = EVAL / (f"{name}_ood" if ood else name) / ("ood_eval_summary.json" if ood else "eval_summary.json")
    if not f.exists():
        return None
    d = json.load(open(f))
    return {"overall": d["all"]["mean"], "nash": d.get("task/nash_equilibrium", {}).get("mean"),
            "br": d.get("task/best_response", {}).get("mean")}


def load_valpeak(name):
    for p in [W / f"nashgrpo/{name}.jsonl", W / f"{name}/{name}.jsonl"]:
        if p.exists():
            vals = [d["data"]["val-aux/gamesolve/reward/mean@1"] for d in (json.loads(l) for l in open(p))
                    if "val-aux/gamesolve/reward/mean@1" in d.get("data", {})]
            return max(vals) if vals else None
    return None


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2: return None
    va, vb = st.variance(a), st.variance(b); se = (va/len(a)+vb/len(b))**0.5
    return (st.mean(a)-st.mean(b))/se if se else None


def report():
    # val peak (free, comparable)
    print(f"\n{'='*60}\nVAL peak reward (480 val, comparable across conditions)\n{'='*60}")
    valg = {}
    for c, lab in CONDS.items():
        vs = [load_valpeak(f"{c}_s{s}") for s in SEEDS]; vs = [v for v in vs if v]
        valg[c] = vs
        if vs: print(f"  {lab:<20} {st.mean(vs):.4f} ± {st.stdev(vs) if len(vs)>1 else 0:.4f}  {[round(x,3) for x in vs]}")
    for est in [False, True]:
        tag = "OOD (750)" if est else "ID (200)"
        print(f"\n{'='*60}\n{tag} external eval (mean±std, n=3)\n{'='*60}")
        print(f"{'condition':<20}{'overall':>16}{'nash':>10}{'br':>10}")
        base = None
        for c, lab in CONDS.items():
            vals = [load_eval(f"{c}_s{s}", est) for s in SEEDS]; vals = [v for v in vals if v]
            if not vals: print(f"{lab:<20}  [no data]"); continue
            ov = [v["overall"] for v in vals]
            if c == "grpo": base = ov
            m, sd = st.mean(ov), st.stdev(ov) if len(ov) > 1 else 0
            na = st.mean([v["nash"] for v in vals if v["nash"] is not None])
            br = st.mean([v["br"] for v in vals if v["br"] is not None])
            print(f"{lab:<20}{m:>10.3f}±{sd:.3f}"[:26].ljust(36)[:36] + f"{na:>10.3f}{br:>10.3f}")
        if base:
            print(f"  vs GRPO (overall):")
            for c, lab in CONDS.items():
                if c == "grpo": continue
                vals = [load_eval(f"{c}_s{s}", est) for s in SEEDS]; ov = [v["overall"] for v in vals if v]
                if len(ov) >= 2:
                    t = welch_t(ov, base); print(f"    {lab:<20} Δ={st.mean(ov)-st.mean(base):+.4f}  t={t:.2f}")


def intransitive_illustration():
    sys.path.insert(0, str(W / "verl"))
    from verl.trainer.ppo.core_algos import _nash_minimax_mixture
    import numpy as np
    print(f"\n{'='*60}\nINTRANSITIVE-PREFERENCE ILLUSTRATION (offline)\n{'='*60}")
    # cyclic A>B>C>A tournament
    M = np.array([[0, .4, -.4], [-.4, 0, .4], [.4, -.4, 0]])
    p = _nash_minimax_mixture(M)
    print(f"  Cyclic A>B>C>A: Nash mixture = {np.round(p,3)} (uniform → no arbitrary winner)")
    print("    A mean/argmax baseline would pick whichever response happens to have the")
    print("    highest raw score, ignoring the cycle; Nash spreads credit correctly.")
    M2 = np.array([[0, .5, .5], [-.5, 0, .1], [-.5, -.1, 0]])
    print(f"  Condorcet winner present: Nash mixture = {np.round(_nash_minimax_mixture(M2),3)} (mass on winner)")


if __name__ == "__main__":
    report()
    intransitive_illustration()
