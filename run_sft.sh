#!/bin/bash
# Phase 2: SFT on Chain-of-Thought Solutions (Qwen2.5-3B-Instruct)
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="results/phase2/sft_cot"
mkdir -p "$OUTPUT_DIR"

echo "=== Phase 2: SFT on Chain-of-Thought ==="
echo "Model: Qwen2.5-3B-Instruct"
echo "Data: GameSolve-Bench CoT solutions (1920 train)"
echo "GPUs: 8, Effective batch: 2*8*4=64"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

torchrun --nproc_per_node=8 \
    train_sft.py \
    --model_path ./Qwen/Qwen2.5-3B-Instruct \
    --train_data data/phase2_sft/train.json \
    --val_data data/phase2_sft/val.json \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length 2048 \
    --batch_size_per_gpu 2 \
    --gradient_accumulation_steps 4 \
    --lr 2e-5 \
    --num_epochs 3 \
    --grad_clip 1.0 \
    --warmup_ratio 0.05 \
    --save_steps 50 \
    --log_steps 10 \
    --eval_steps 50 \
    --seed 42 \
    --gradient_checkpointing \
    --bf16 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "SFT Training complete. Results in $OUTPUT_DIR"
