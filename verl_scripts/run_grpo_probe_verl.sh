#!/bin/bash
# GRPO + Probe training via verl framework on GameSolve-Bench
# Same GRPO hyperparameters as grpo_verl, with auxiliary probing loss enabled
# probe_lambda=0.1, probe_layer=17 (layer L/2 for Qwen2.5-3B with 36 layers)

set -x

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth

# Probe configuration
export VERL_PROBE_ENABLED=1
export VERL_PROBE_LAMBDA=0.1
export VERL_PROBE_LAYER=17

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORK_DIR"

REWARD_FN="$WORK_DIR/verl_scripts/gamesolve_reward_verl.py"
TRAIN_DATA="$WORK_DIR/data/verl_gamesolve_probe/train.parquet"
VAL_DATA="$WORK_DIR/data/verl_gamesolve_probe/val.parquet"
MODEL_PATH="$WORK_DIR/Qwen/Qwen2.5-3B-Instruct"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name=grpo_verl_probe \
    trainer.experiment_name=grpo_verl_probe \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=40 \
    trainer.total_epochs=3 \
    trainer.val_before_train=True \
    trainer.default_local_dir=results/phase2/grpo_verl_probe \
    "$@"
