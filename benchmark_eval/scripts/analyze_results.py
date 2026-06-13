"""Aggregate GTBench + TextArena results for the 4 checkpoints into
benchmark_eval/results/summary.json and a markdown report.

GTBench: candidate (PromptAgent) vs RandomAgent, win rate over Normal matches;
also reports completion rate (share of matches that ended Normally and the
candidate was not at fault).
TextArena: candidate vs fixed qwen3b-base opponent, win/draw/loss + score.
"""
import json
import glob
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["qwen3b-base", "qwen3b-sft", "qwen3b-grpo", "qwen3b-grpo-probe"]
MODEL_LABELS = {
    "qwen3b-base": "Base (Qwen2.5-3B)",
    "qwen3b-sft": "SFT (CoT)",
    "qwen3b-grpo": "GRPO (verl)",
    "qwen3b-grpo-probe": "GRPO+Probe (verl)",
}


def analyze_gtbench():
    out = defaultdict(dict)
    for model in MODELS:
        for path in glob.glob(str(ROOT / "results/gtbench" / model / "*" / "*.jsonl")):
            game = Path(path).parent.name
            wins = draws = losses = 0
            normal = 0
            total = 0
            fault = 0
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for m in record.get("matches", []):
                    total += 1
                    if m.get("status") != "Normal":
                        if any(model in str(a) for a in m.get("agents_at_fault", [])):
                            fault += 1
                        continue
                    normal += 1
                    w = m.get("winner", "")
                    if not w:
                        draws += 1
                    elif model in w:
                        wins += 1
                    else:
                        losses += 1
            if total == 0:
                continue
            out[model][game] = {
                "matches_total": total, "matches_normal": normal,
                "wins": wins, "draws": draws, "losses": losses,
                "win_rate": wins / normal if normal else None,
                "score": (wins + 0.5 * draws) / normal if normal else None,
                "completion_rate": normal / total,
            }
    return dict(out)


def analyze_textarena():
    out = defaultdict(dict)
    for path in glob.glob(str(ROOT / "results/textarena" / "*.json")):
        name = Path(path).stem
        model, env_id = name.split("__", 1)
        data = json.load(open(path))
        out[model][env_id] = data["summary"]
    return dict(out)


def fmt(v, pct=True):
    if v is None:
        return "—"
    return f"{v:.0%}" if pct else f"{v:.3f}"


def write_report(gt, ta):
    lines = ["# Benchmark Evaluation: GTBench & TextArena", "",
             "4 个 checkpoint:Base (Qwen2.5-3B-Instruct)、SFT (CoT)、GRPO (verl, step280)、GRPO+Probe (verl, step320)。", ""]

    # GTBench table
    MIN_N = 10
    games = sorted({g for m in gt.values() for g in m})
    # games where every model has enough valid matches -> used for avg
    common_games = [g for g in games
                    if all(gt.get(m, {}).get(g, {}).get("matches_normal", 0) >= MIN_N for m in MODELS)]
    lines += ["## GTBench — LLM (prompt agent) vs Random agent", "",
              "指标:score = (win + 0.5*draw)/normal;`n` 为有效(Normal)对局数。",
              f"avg score 仅在所有模型 n≥{MIN_N} 的游戏上计算(排除:" +
              (", ".join(g for g in games if g not in common_games) or "无") + ")。", "",
              "| Game | " + " | ".join(MODEL_LABELS[m] for m in MODELS) + " |",
              "|:-----|" + "----:|" * len(MODELS)]
    for g in games:
        row = [g]
        for m in MODELS:
            s = gt.get(m, {}).get(g)
            if not s or s["score"] is None:
                row.append("—")
            else:
                cell = f"{s['score']:.0%} (n={s['matches_normal']})"
                row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    # averages over common games
    row = ["**avg score**"]
    avg_scores = {}
    for m in MODELS:
        ss = [gt[m][g]["score"] for g in common_games if gt.get(m, {}).get(g, {}).get("score") is not None]
        avg = sum(ss) / len(ss) if ss else None
        avg_scores[m] = avg
        row.append(fmt(avg))
    lines.append("| " + " | ".join(row) + " |")
    row = ["**avg completion**"]
    for m in MODELS:
        cs = [s["completion_rate"] for s in gt.get(m, {}).values()]
        row.append(fmt(sum(cs) / len(cs)) if cs else "—")
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # TextArena table
    envs = sorted({e for m in ta.values() for e in m})
    lines += ["## TextArena — candidate vs fixed Base opponent", "",
              "指标:score = (win + 0.5*draw)/valid。Base vs Base 行为对照(期望 ~50%)。", "",
              "| Env | " + " | ".join(MODEL_LABELS[m] for m in MODELS) + " |",
              "|:----|" + "----:|" * len(MODELS)]
    for e in envs:
        row = [e]
        for m in MODELS:
            s = ta.get(m, {}).get(e)
            row.append(fmt(s["score"]) if s and s.get("score") is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    row = ["**avg score**"]
    ta_avg = {}
    for m in MODELS:
        ss = [s["score"] for s in ta.get(m, {}).values() if s.get("score") is not None]
        avg = sum(ss) / len(ss) if ss else None
        ta_avg[m] = avg
        row.append(fmt(avg))
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines), avg_scores, ta_avg


if __name__ == "__main__":
    gt = analyze_gtbench()
    ta = analyze_textarena()
    report, gt_avg, ta_avg = write_report(gt, ta)
    summary_path = ROOT / "results/summary.json"
    json.dump({"gtbench": gt, "textarena": ta,
               "gtbench_avg_score": gt_avg, "textarena_avg_score": ta_avg},
              open(summary_path, "w"), indent=1)
    report_path = ROOT / "results/REPORT.md"
    open(report_path, "w").write(report)
    print(report)
    print(f"\nsaved: {summary_path}\n       {report_path}")
