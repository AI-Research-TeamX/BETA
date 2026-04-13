#!/usr/bin/env bash
# eval_all.sh
# Iterates every model directory under MODELS_ROOT and:
#   1. Starts a vLLM server for that model
#   2. Waits until the server is ready
#   3. Runs eval_gamesolve.py
#   4. Kills the server and moves to the next model
#
# Usage:
#   bash eval_all.sh                          # full eval, all models
#   bash eval_all.sh --max-samples 50         # quick test
#   bash eval_all.sh --task br --temperature 0.2

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────
MODELS_ROOT="/primus_datasets/cuisijia.csj/models/Qwen"
EVAL_SCRIPT="$(cd "$(dirname "$0")" && pwd)/eval_gamesolve.py"
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/eval_logs"
mkdir -p "$LOG_DIR"

# ── vLLM server settings ──────────────────────────────────────────
VLLM_PORT=8000
TENSOR_PARALLEL=4
GPU_MEM_UTIL=0.95
VLLM_STARTUP_TIMEOUT=600   # seconds to wait for server ready

# ── eval_gamesolve.py pass-through args ───────────────────────────
# All extra arguments to this script are forwarded to eval_gamesolve.py
# e.g.:  bash eval_all.sh --max-samples 100 --variant story
EVAL_EXTRA_ARGS=("$@")

# ── Helpers ───────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
info() { echo ""; log ">>> $*"; }

wait_for_server() {
    local url="http://localhost:${VLLM_PORT}/health"
    local deadline=$(( $(date +%s) + VLLM_STARTUP_TIMEOUT ))
    log "Waiting for vLLM server at $url ..."
    while [[ $(date +%s) -lt $deadline ]]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            log "Server is ready."
            return 0
        fi
        sleep 3
    done
    log "ERROR: Server did not start within ${VLLM_STARTUP_TIMEOUT}s"
    return 1
}

kill_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        log "Stopping vLLM server (PID $pid) ..."
        kill "$pid"
        # Give it up to 30 s to release the GPU memory
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [[ $waited -lt 30 ]]; do
            sleep 2; (( waited+=2 ))
        done
        kill -9 "$pid" 2>/dev/null || true
        sleep 5   # extra pause so GPU memory is fully freed
    fi
}

# ── Manually specify models ───────────────────────────────────────
MODEL_DIRS=(
    "$MODELS_ROOT/Qwen2.5-3B-Instruct"
    "$MODELS_ROOT/Qwen2.5-7B-Instruct"
    "$MODELS_ROOT/Qwen2.5-14B-Instruct"
    "$MODELS_ROOT/Qwen2.5-32B-Instruct"
)

if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
    echo "No model directories specified in MODEL_DIRS"
    exit 1
fi

info "Found ${#MODEL_DIRS[@]} model(s) to evaluate:"
for d in "${MODEL_DIRS[@]}"; do echo "  - $(basename "$d")"; done
echo ""

# ── Main loop ─────────────────────────────────────────────────────
FAILED_MODELS=()

for MODEL_PATH in "${MODEL_DIRS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    VLLM_LOG="$LOG_DIR/${MODEL_NAME}_vllm.log"
    EVAL_LOG="$LOG_DIR/${MODEL_NAME}_eval.log"

    info "===== Starting: $MODEL_NAME ====="

    # 1. Start vLLM server in background
    log "Launching vLLM for $MODEL_NAME ..."
    vllm serve \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --tensor-parallel-size $TENSOR_PARALLEL \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --port $VLLM_PORT \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    log "vLLM PID: $VLLM_PID  (log: $VLLM_LOG)"

    # 2. Wait for server to be ready
    if ! wait_for_server; then
        log "ERROR: $MODEL_NAME — server failed to start. Skipping."
        kill_server $VLLM_PID
        FAILED_MODELS+=("$MODEL_NAME (server_start_failed)")
        continue
    fi

    # 3. Run evaluation
    log "Running evaluation for $MODEL_NAME ..."
    log "  Extra args: ${EVAL_EXTRA_ARGS[*]:-<none>}"
    if python3 "$EVAL_SCRIPT" \
        --model "$MODEL_NAME" \
        "${EVAL_EXTRA_ARGS[@]}" \
        2>&1 | tee "$EVAL_LOG"; then
        log "$MODEL_NAME — evaluation complete."
    else
        log "WARNING: eval_gamesolve.py exited with non-zero status for $MODEL_NAME"
        FAILED_MODELS+=("$MODEL_NAME (eval_error)")
    fi

    # 4. Kill server, free GPU
    kill_server $VLLM_PID

    info "===== Done: $MODEL_NAME =====\n"
done

# ── Final report ─────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  eval_all.sh complete"
echo "  Models processed : ${#MODEL_DIRS[@]}"
echo "  Failed           : ${#FAILED_MODELS[@]}"
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo "  Failed list:"
    for m in "${FAILED_MODELS[@]}"; do echo "    - $m"; done
fi
echo "  Results saved to : $(python3 -c "
import json, pathlib
base = pathlib.Path('eval_results')
dirs = sorted(base.iterdir()) if base.exists() else []
print(str(base))
")"
echo "  Logs saved to    : $LOG_DIR"
echo "════════════════════════════════════════"