"""TextArena evaluation: each candidate model plays a fixed opponent
(qwen3b-base) on a selection of two-player games.

Each (model, game) pair plays N episodes, alternating which side the candidate
plays to cancel first-player advantage. Results -> JSON per (model, game).

Models are served by local vLLM OpenAI-compatible servers (see serve script).
"""
import argparse
import json
import os

# local vLLM endpoints must not go through any user-level proxy
for _v in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
import re
import time
import traceback
from pathlib import Path

import textarena as ta
from openai import OpenAI

ENDPOINTS = {
    "qwen3b-base": "http://127.0.0.1:8001/v1",
    "qwen3b-sft": "http://127.0.0.1:8002/v1",
    "qwen3b-grpo": "http://127.0.0.1:8003/v1",
    "qwen3b-grpo-probe": "http://127.0.0.1:8004/v1",
}

SYSTEM_PROMPT = (
    "You are a competitive game player. Read the game instructions and the "
    "current observation carefully, reason briefly, and then output your "
    "chosen action in EXACTLY the format the game asks for (e.g. moves in "
    "square brackets like '[4]' or '[a1]'). Your reply MUST contain the "
    "bracketed action."
)


class VLLMAgent:
    def __init__(self, served_name, temperature=0.3, max_tokens=512):
        self.served_name = served_name
        self.client = OpenAI(base_url=ENDPOINTS[served_name], api_key="EMPTY", timeout=120)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, observation: str) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.served_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": observation},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)


def play_episode(env_id, agents, seed):
    """agents: dict player_id -> agent. Returns rewards dict or None on error."""
    env = ta.make(env_id=env_id)
    env.reset(num_players=2, seed=seed)
    done = False
    turns = 0
    while not done:
        player_id, observation = env.get_observation()
        action = agents[player_id](observation)
        done, _ = env.step(action=action)
        turns += 1
        if turns > 200:  # hard stop
            break
    rewards, game_info = env.close()
    return rewards, game_info, turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ENDPOINTS))
    parser.add_argument("--opponent", default="qwen3b-base", choices=list(ENDPOINTS))
    parser.add_argument("--games", nargs="+", default=[
        "TicTacToe-v0", "ConnectFour-v0", "Nim-v0",
        "KuhnPoker-v0", "IteratedPrisonersDilemma-v0", "SimpleNegotiation-v0",
    ])
    parser.add_argument("--episodes", type=int, default=20, help="episodes per game (half each side)")
    parser.add_argument("--output_dir", default="benchmark_eval/results/textarena")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    candidate = VLLMAgent(args.model)
    opponent = VLLMAgent(args.opponent)

    for env_id in args.games:
        out_path = out_root / f"{args.model}__{env_id}.json"
        if out_path.exists():
            print(f"[skip] {out_path} exists")
            continue
        episodes = []
        for ep in range(args.episodes):
            cand_pid = ep % 2  # alternate sides
            agents = {cand_pid: candidate, 1 - cand_pid: opponent}
            seed = args.seed + ep
            try:
                rewards, game_info, turns = play_episode(env_id, agents, seed)
                cand_r = rewards.get(cand_pid) if rewards else None
                opp_r = rewards.get(1 - cand_pid) if rewards else None
                if cand_r is None:
                    outcome = "error"
                elif cand_r > opp_r:
                    outcome = "win"
                elif cand_r < opp_r:
                    outcome = "loss"
                else:
                    outcome = "draw"
                episodes.append({
                    "episode": ep, "candidate_player": cand_pid, "seed": seed,
                    "rewards": {str(k): v for k, v in (rewards or {}).items()},
                    "candidate_reward": cand_r, "outcome": outcome, "turns": turns,
                })
                print(f"[{args.model}][{env_id}] ep{ep}: {outcome} (r={cand_r}, turns={turns})")
            except Exception as e:
                episodes.append({"episode": ep, "candidate_player": cand_pid,
                                 "seed": seed, "outcome": "error", "error": str(e)})
                print(f"[{args.model}][{env_id}] ep{ep}: ERROR {e}")
                traceback.print_exc()

        valid = [e for e in episodes if e["outcome"] in ("win", "loss", "draw")]
        n = len(valid)
        wins = sum(e["outcome"] == "win" for e in valid)
        draws = sum(e["outcome"] == "draw" for e in valid)
        losses = sum(e["outcome"] == "loss" for e in valid)
        rewards_list = [e["candidate_reward"] for e in valid if e["candidate_reward"] is not None]
        summary = {
            "model": args.model, "opponent": args.opponent, "env_id": env_id,
            "episodes": args.episodes, "valid": n,
            "wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / n if n else None,
            "score": (wins + 0.5 * draws) / n if n else None,  # draw counts half
            "mean_reward": sum(rewards_list) / len(rewards_list) if rewards_list else None,
        }
        json.dump({"summary": summary, "episodes": episodes}, open(out_path, "w"), indent=1)
        print(f"[done] {env_id}: win {wins}/{n}, draw {draws}, loss {losses} -> {out_path}")


if __name__ == "__main__":
    main()
