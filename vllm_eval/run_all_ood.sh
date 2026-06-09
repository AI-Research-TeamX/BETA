#!/bin/bash
# Evaluate all 5 checkpoints on OOD data using vLLM.
# Each model uses 1 GPU (tp=1), runs sequentially but fast with vLLM batch inference.
# For parallel execution: run separate scripts on different GPU sets.
set -e

BENCH="gamesolve_ood_bench.jsonl"
EVAL_SCRIPT="vllm_eval/eval.py"

declare -A MODELS
MODELS=(
    ["ood_base"]="./Qwen/Qwen2.5-3B-Instruct"
    ["ood_grpo_only"]="results/phase2/full_grpo_only/checkpoint-best/model"
    ["ood_full_probe"]="results/phase2/full_probe_grpo/checkpoint-best/model"
    ["ood_sft_cot"]="results/phase2/sft_cot/checkpoint-best"
    ["ood_sft_grpo"]="results/phase2/sft_then_grpo/checkpoint-best/model"
)

GPU=${1:-0}

for name in ood_base ood_grpo_only ood_full_probe ood_sft_cot ood_sft_grpo; do
    model_path="${MODELS[$name]}"
    echo ""
    echo "=========================================="
    echo "  Evaluating: $name"
    echo "  Model: $model_path"
    echo "  GPU: $GPU"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU python3 "$EVAL_SCRIPT" \
        --model_path "$model_path" \
        --bench_path "$BENCH" \
        --output_dir "eval_results/$name" \
        --tp 1 \
        --gpu_mem 0.85

    echo "Done: $name"
done

echo ""
echo "All OOD evaluations complete."
