#!/bin/bash
# Serve the 4 checkpoints with vLLM on GPUs 0-3 (ports 8001-8004).
set -u
W=/root/storage/cuisijia.csj/vlmarch/BETA
mkdir -p $W/benchmark_eval/logs

CUDA_VISIBLE_DEVICES=0 nohup vllm serve $W/Qwen/Qwen2.5-3B-Instruct \
    --served-model-name qwen3b-base --port 8001 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 --disable-log-requests \
    > $W/benchmark_eval/logs/serve_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup vllm serve $W/results/phase2/sft_cot/checkpoint-best \
    --served-model-name qwen3b-sft --port 8002 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 --disable-log-requests \
    > $W/benchmark_eval/logs/serve_sft.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup vllm serve $W/results/phase2/grpo_verl/merged_step_280 \
    --served-model-name qwen3b-grpo --port 8003 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 --disable-log-requests \
    > $W/benchmark_eval/logs/serve_grpo.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup vllm serve $W/results/phase2/grpo_verl_probe/merged_step_320 \
    --served-model-name qwen3b-grpo-probe --port 8004 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 --disable-log-requests \
    > $W/benchmark_eval/logs/serve_probe.log 2>&1 &

echo "waiting for servers..."
for p in 8001 8002 8003 8004; do
  until curl -sf http://127.0.0.1:$p/health >/dev/null 2>&1; do sleep 5; done
  echo "port $p ready"
done
