#!/bin/bash
# Master orchestration: Run all Phase 2 experiments sequentially
# 1. SFT on CoT
# 2. Eval SFT model
# 3. GRPO on SFT checkpoint
# 4. Eval GRPO model
# 5. Eval baseline model for comparison
# 6. Generate comparison report
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth
export TOKENIZERS_PARALLELISM=false

echo "========================================"
echo "  PHASE 2: Full Experiment Pipeline"
echo "  Starting at: $(date)"
echo "========================================"

# ─── Step 1: SFT Training ───
echo ""
echo ">>> [1/6] SFT on Chain-of-Thought..."
bash run_sft.sh

# ─── Step 2: Eval SFT ───
echo ""
echo ">>> [2/6] Evaluating SFT model..."
python3 eval_checkpoint.py \
    --model_path results/phase2/sft_cot/checkpoint-best \
    --output_dir eval_results/sft_cot_best \
    --max_samples 200 \
    --gpu 0

# ─── Step 3: GRPO on SFT checkpoint ───
echo ""
echo ">>> [3/6] GRPO on SFT checkpoint..."
bash run_sft_then_grpo.sh

# ─── Step 4: Eval SFT+GRPO ───
echo ""
echo ">>> [4/6] Evaluating SFT+GRPO model..."
python3 eval_checkpoint.py \
    --model_path results/phase2/sft_then_grpo/checkpoint-best \
    --output_dir eval_results/sft_then_grpo_best \
    --max_samples 200 \
    --gpu 0

# ─── Step 5: Eval baselines ───
echo ""
echo ">>> [5/6] Evaluating baseline and existing models..."
# Base model (no training)
python3 eval_checkpoint.py \
    --model_path ./Qwen/Qwen2.5-3B-Instruct \
    --output_dir eval_results/qwen3b_base \
    --max_samples 200 \
    --gpu 0

# GRPO-only best checkpoint
python3 eval_checkpoint.py \
    --model_path results/phase2/full_grpo_only/checkpoint-best \
    --output_dir eval_results/grpo_only_best \
    --max_samples 200 \
    --gpu 0

# Full+Probe GRPO best checkpoint
python3 eval_checkpoint.py \
    --model_path results/phase2/full_probe_grpo/checkpoint-best/model \
    --output_dir eval_results/full_probe_grpo_best \
    --max_samples 200 \
    --gpu 0

# ─── Step 6: Comparison ───
echo ""
echo ">>> [6/6] Generating comparison report..."
python3 compare_experiments.py

echo ""
echo "========================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Finished at: $(date)"
echo "========================================"
