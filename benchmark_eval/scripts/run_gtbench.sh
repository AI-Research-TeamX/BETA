#!/bin/bash
# GTBench: <model> (prompt_agent) vs random_agent across games.
# Usage: bash run_gtbench.sh <model_name> [num_matches] [games...]
set -u

MODEL=${1:?model name required, e.g. qwen3b-base}
NUM_MATCHES=${2:-20}
shift 2 2>/dev/null || shift 1
GAMES=${@:-"tictactoe connect4 kuhn_poker nim pig prisoners_dilemma liars_dice first_sealed_auction"}

W=/root/storage/cuisijia.csj/vlmarch/BETA
cd $W/benchmark_eval/GTBench

# bypass any user-level proxy for local vLLM endpoints
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

export LOCAL_LLM_ENDPOINTS='{"qwen3b-base":"http://127.0.0.1:8001/v1","qwen3b-sft":"http://127.0.0.1:8002/v1","qwen3b-grpo":"http://127.0.0.1:8003/v1","qwen3b-grpo-probe":"http://127.0.0.1:8004/v1"}'

python3 -m gamingbench.main \
    --num-matches ${NUM_MATCHES} \
    --exp-root $W/benchmark_eval/results/gtbench/${MODEL} \
    --seed 0 \
    --game-names ${GAMES} \
    --agent-configs gamingbench/configs/agent_configs/prompt_agent.yaml gamingbench/configs/agent_configs/random_agent.yaml \
    --model-configs gamingbench/configs/model_configs/${MODEL}.yaml gamingbench/configs/model_configs/dummy-random.yaml \
    --exchange-first-player \
    --num-workers 8 \
    --threshold-matches 30
