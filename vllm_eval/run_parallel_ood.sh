#!/bin/bash
# Run all 5 OOD evaluations in parallel, each on a separate GPU.
set -e

BENCH="gamesolve_ood_bench.jsonl"
SCRIPT="vllm_eval/eval.py"

echo "Launching 5 parallel OOD evaluations on GPU 0-4..."

CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT \
    --model_path ./Qwen/Qwen2.5-3B-Instruct \
    --bench_path $BENCH --output_dir eval_results/ood_base \
    --tp 1 --gpu_mem 0.85 > eval_results/ood_base/vllm.log 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python3 $SCRIPT \
    --model_path results/phase2/full_grpo_only/checkpoint-best/model \
    --bench_path $BENCH --output_dir eval_results/ood_grpo_only \
    --tp 1 --gpu_mem 0.85 > eval_results/ood_grpo_only/vllm.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT \
    --model_path results/phase2/full_probe_grpo/checkpoint-best/model \
    --bench_path $BENCH --output_dir eval_results/ood_full_probe \
    --tp 1 --gpu_mem 0.85 > eval_results/ood_full_probe/vllm.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT \
    --model_path results/phase2/sft_cot/checkpoint-best \
    --bench_path $BENCH --output_dir eval_results/ood_sft_cot \
    --tp 1 --gpu_mem 0.85 > eval_results/ood_sft_cot/vllm.log 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=4 python3 $SCRIPT \
    --model_path results/phase2/sft_then_grpo/checkpoint-best/model \
    --bench_path $BENCH --output_dir eval_results/ood_sft_grpo \
    --tp 1 --gpu_mem 0.85 > eval_results/ood_sft_grpo/vllm.log 2>&1 &
PID4=$!

echo "PIDs: base=$PID0 grpo=$PID1 probe=$PID2 sft=$PID3 sft_grpo=$PID4"
echo "Waiting for all to finish..."
wait $PID0 $PID1 $PID2 $PID3 $PID4
echo "All OOD evaluations complete!"
