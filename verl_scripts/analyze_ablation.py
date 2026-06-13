"""Aggregate the probe layer-ablation / multi-seed runs.

Reads each run's file-logger jsonl (val reward trajectory), computes peak val
reward per run, aggregates mean±std per condition, runs Welch t-test for
probe@17 vs no-probe, and prints the layer-ablation curve.
"""
import glob
import json
import os
import statistics as st
from collections import defaultdict

WORK = "/root/storage/cuisijia.csj/vlmarch/BETA"
VAL_KEYS = ["val-core/gamesolve/reward/mean@1", "val-aux/gamesolve/reward/mean@1"]

# experiment_name -> (condition, seed)
MATRIX = {}
for s in (0, 1, 2):
    MATRIX[f"np_s{s}"] = ("no_probe", s)
    for L in (6, 17, 30, 35):
        MATRIX[f"p{L}_s{s}"] = (f"probe@{L}", s)


def find_jsonl(exp):
    cands = [
        f"{WORK}/probe_ablation/{exp}.jsonl",
        f"{WORK}/{exp}/{exp}.jsonl",
        f"{WORK}/probe_ablation/probe_ablation.jsonl",
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    hits = glob.glob(f"{WORK}/**/{exp}.jsonl", recursive=True)
    return hits[0] if hits else None


def traj(path):
    pts = []
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        step, data = d.get("step"), d.get("data", {})
        v = next((data[k] for k in VAL_KEYS if k in data), None)
        if v is not None:
            pts.append((step, v))
    return sorted(pts)


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return None, None
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    na, nb = len(a), len(b)
    se = (va / na + vb / nb) ** 0.5
    if se == 0:
        return None, None
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df


def main():
    runs = {}  # exp -> {peak, last3, n_evals}
    for exp in MATRIX:
        p = find_jsonl(exp)
        if not p:
            continue
        t = traj(p)
        if not t:
            continue
        vals = [v for _, v in t]
        runs[exp] = {
            "peak": max(vals),
            "last3": st.mean(vals[-3:]) if len(vals) >= 3 else st.mean(vals),
            "final": vals[-1],
            "n_evals": len(vals),
            "max_step": t[-1][0],
        }

    by_cond = defaultdict(list)   # condition -> [peak per seed]
    by_cond_last3 = defaultdict(list)
    for exp, (cond, seed) in MATRIX.items():
        if exp in runs:
            by_cond[cond].append(runs[exp]["peak"])
            by_cond_last3[cond].append(runs[exp]["last3"])

    print(f"\n{'='*70}\nPER-RUN peak val reward ({len(runs)}/{len(MATRIX)} runs found)\n{'='*70}")
    for exp, (cond, seed) in MATRIX.items():
        r = runs.get(exp)
        if r:
            print(f"  {exp:<10} {cond:<10} seed{seed}  peak={r['peak']:.4f}  last3={r['last3']:.4f}  (evals={r['n_evals']}, step{r['max_step']})")
        else:
            print(f"  {exp:<10} {cond:<10} seed{seed}  [missing]")

    print(f"\n{'='*70}\nPER-CONDITION (mean ± std over seeds), metric = peak val reward\n{'='*70}")
    order = ["no_probe", "probe@6", "probe@17", "probe@30", "probe@35"]
    summ = {}
    for cond in order:
        vals = by_cond.get(cond, [])
        if not vals:
            print(f"  {cond:<10} [no data]")
            continue
        m = st.mean(vals)
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        summ[cond] = {"mean_peak": m, "std_peak": sd, "n": len(vals),
                      "vals": vals, "last3_mean": st.mean(by_cond_last3[cond])}
        print(f"  {cond:<10} peak {m:.4f} ± {sd:.4f}  (n={len(vals)}, last3_mean={st.mean(by_cond_last3[cond]):.4f})  {[round(x,4) for x in vals]}")

    # significance: probe@17 vs no_probe
    if "probe@17" in by_cond and "no_probe" in by_cond:
        a, b = by_cond["probe@17"], by_cond["no_probe"]
        t, df = welch_t(a, b)
        delta = st.mean(a) - st.mean(b)
        print(f"\n{'='*70}\nSIGNIFICANCE: probe@17 vs no_probe (peak val reward)\n{'='*70}")
        print(f"  Δ = {delta:+.4f}  (probe@17 {st.mean(a):.4f} vs no_probe {st.mean(b):.4f})")
        if t is not None:
            print(f"  Welch t = {t:.3f}, df ≈ {df:.1f}  (|t|>~2.8 ⇒ p<0.05 at these df)")

    # layer ablation: best non-critical vs critical
    print(f"\n{'='*70}\nLAYER ABLATION (does critical layer 17 beat non-critical?)\n{'='*70}")
    crit = summ.get("probe@17", {}).get("mean_peak")
    for cond in ["probe@6", "probe@30", "probe@35"]:
        if cond in summ and crit is not None:
            d = crit - summ[cond]["mean_peak"]
            print(f"  probe@17 - {cond:<9} = {d:+.4f}")

    json.dump({"runs": runs, "by_condition": summ},
              open(f"{WORK}/results/phase2/ablation_summary.json", "w"), indent=1)
    print(f"\nsaved: results/phase2/ablation_summary.json")


if __name__ == "__main__":
    main()
