"""Decisive validity gate: on REAL base-model generations (natural quality spread),
does GT-PRM process_score correlate POSITIVELY with answer-quality (outcome) reward?
If yes, process reward is a sane training signal. If negative, abort & fix.
"""
import json, random, sys, statistics as st
from pathlib import Path
from vllm import LLM, SamplingParams
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gt_prm_verifier import verify_trace, BINARY_WEIGHTS
from gamesolve_reward import compute_reward

SYSTEM_PROMPT = """You are an expert in game theory. Analyze the given game carefully and provide precise answers."""
NASH_INSTR = ("Find all Nash Equilibria of this game. For each equilibrium, state whether it is "
              "pure or mixed. For pure NE, give the strategy pair. For mixed NE, give the "
              "probability distributions over strategies for each player.")
BR_INSTR = ("Compute the best response for the row player given the column player's strategy. "
            "Report: (1) the expected payoff for each row action, (2) the best response action(s), "
            "and (3) the best response expected payoff value.")

rows = [json.loads(l) for l in open("gamesolve_bench.jsonl")]
random.seed(1); random.shuffle(rows)
sample = rows[:300]

llm = LLM(model="./Qwen/Qwen2.5-3B-Instruct", gpu_memory_utilization=0.85, max_model_len=4096)
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512, seed=0)

prompts = []
for s in sample:
    desc_keys = list(s["descriptions"].keys())
    random.seed(hash(s["id"]))
    desc = s["descriptions"][random.choice(desc_keys)]
    instr = NASH_INSTR if s["task"] == "nash_equilibrium" else BR_INSTR
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{desc}\n\n{instr}"}]
    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

outs = llm.generate(prompts, sp)
proc, out = [], []
for s, o in zip(sample, outs):
    resp = o.outputs[0].text
    g = {"payoff_matrix_row": s["payoff_matrix_row"], "payoff_matrix_col": s["payoff_matrix_col"],
         "row_labels": s["row_labels"], "col_labels": s["col_labels"], "task": s["task"],
         "ground_truth": s["ground_truth"]}
    proc.append(verify_trace(resp, g)["process_score"])
    out.append(compute_reward(resp, s["ground_truth"], s["task"], s["row_labels"], s["col_labels"]))

def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    da = sum((x-ma)**2 for x in a)**0.5; db = sum((y-mb)**2 for y in b)**0.5
    return num/(da*db) if da*db else 0.0

print(f"n={len(sample)}")
print(f"process: mean={st.mean(proc):.3f} std={st.stdev(proc):.3f}")
print(f"outcome: mean={st.mean(out):.3f} std={st.stdev(out):.3f}")
print(f"CORR(process, outcome) on base-model gens = {corr(proc, out):.3f}")
# bucketed: mean outcome for low/high process
pairs = sorted(zip(proc, out))
lo = [o for p, o in pairs if p <= 0.33]; hi = [o for p, o in pairs if p >= 0.66]
if lo and hi:
    print(f"mean outcome | process<=0.33: {st.mean(lo):.3f} (n={len(lo)})")
    print(f"mean outcome | process>=0.66: {st.mean(hi):.3f} (n={len(hi)})")
