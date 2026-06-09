#!/bin/bash
# Phase 2: GRPO fine-tuning on SFT checkpoint (no probe)
# This runs AFTER run_sft.sh completes
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth
export TOKENIZERS_PARALLELISM=false

SFT_CHECKPOINT="results/phase2/sft_cot/checkpoint-best"
OUTPUT_DIR="results/phase2/sft_then_grpo"
mkdir -p "$OUTPUT_DIR"

echo "=== Phase 2: GRPO on SFT Checkpoint ==="
echo "Model: SFT-trained Qwen2.5-3B (from $SFT_CHECKPOINT)"
echo "Algorithm: GRPO (no probe)"
echo "GPUs: 8"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

torchrun --nproc_per_node=8 \
    train_grpo_probe.py \
    --model_path "$SFT_CHECKPOINT" \
    --train_data data/phase2/train.json \
    --val_data data/phase2/val.json \
    --output_dir "$OUTPUT_DIR" \
    --probe_layer 17 \
    --n_rollouts 5 \
    --batch_size_per_gpu 2 \
    --micro_batch_size 4 \
    --max_prompt_length 1024 \
    --max_response_length 1024 \
    --max_seq_length 2048 \
    --lr 5e-7 \
    --probe_lr 1e-3 \
    --probe_lambda 0.0 \
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

echo "GRPO Training complete. Results in $OUTPUT_DIR"
