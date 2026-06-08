#!/bin/bash
# Phase 2: Full+Probe GRPO Training on 8×H20 GPUs
# Qwen2.5-3B-Instruct, Layer 18 (0-indexed: 17), Mean Pooling

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="results/phase2/full_probe_grpo"
mkdir -p "$OUTPUT_DIR"

echo "=== Phase 2: Full+Probe GRPO Training ==="
echo "Model: Qwen2.5-3B-Instruct"
echo "Probe layer: 17 (L/2 = 18, 0-indexed)"
echo "Pooling: mean"
echo "GPUs: 8"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

torchrun --nproc_per_node=8 \
    train_grpo_probe.py \
    --model_path ./Qwen/Qwen2.5-3B-Instruct \
    --train_data data/phase2/train.json \
    --val_data data/phase2/val.json \
    --output_dir "$OUTPUT_DIR" \
    --probe_layer 17 \
    --n_rollouts 5 \
    --batch_size_per_gpu 2 \
    --micro_batch_size 4 \
    --max_prompt_length 1024 \
    --max_response_length 512 \
    --max_seq_length 1536 \
    --lr 1e-6 \
    --probe_lr 1e-3 \
    --probe_lambda 0.1 \
    --gen_temperature 0.7 \
    --gen_top_p 0.9 \
    --num_epochs 3 \
    --grad_clip 1.0 \
    --warmup_ratio 0.05 \
    --save_steps 40 \
    --log_steps 5 \
    --eval_steps 40 \
    --seed 42 \
    --gradient_checkpointing \
    --bf16 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Training complete. Results in $OUTPUT_DIR"
