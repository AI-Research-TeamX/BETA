#!/bin/bash
# Parameterized GT-PRM GRPO run.
# Env args: SEED, GT_PRM_MODE(outcome|process|hybrid), GT_PRM_VARIANT(binary|typed),
#           GPUS, N_GPUS, RAY_TMP, EXP_NAME, OUT_DIR, STEPS_EPOCHS(default 2 = 240 steps)
set -x
set -u
SEED=${SEED:-0}
GT_PRM_MODE=${GT_PRM_MODE:-outcome}
GT_PRM_VARIANT=${GT_PRM_VARIANT:-binary}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
N_GPUS=${N_GPUS:-8}
RAY_TMP=${RAY_TMP:-/tmp/ray_default}
STEPS_EPOCHS=${STEPS_EPOCHS:-2}
EXP_NAME=${EXP_NAME:?}
OUT_DIR=${OUT_DIR:?}

export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth
export CUDA_VISIBLE_DEVICES=$GPUS
export GT_PRM_MODE GT_PRM_VARIANT

W="$(cd "$(dirname "$0")/.." && pwd)"; cd "$W"
REWARD_FN="$W/verl_scripts/gt_prm_reward.py"
TRAIN_DATA="$W/data/verl_gtprm/train.parquet"
VAL_DATA="$W/data/verl_gtprm/val.parquet"
MODEL_PATH="$W/Qwen/Qwen2.5-3B-Instruct"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" data.val_files="$VAL_DATA" \
    data.train_batch_size=16 data.max_prompt_length=1024 data.max_response_length=512 \
    data.filter_overlong_prompts=True data.truncation=error data.trust_remote_code=True data.seed=$SEED \
    actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.7 actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$REWARD_FN" reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 trainer.logger='["console","file"]' \
    trainer.project_name=gtprm trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=$N_GPUS trainer.nnodes=1 \
    trainer.save_freq=240 trainer.test_freq=40 trainer.total_epochs=$STEPS_EPOCHS \
    trainer.val_before_train=False trainer.default_local_dir="$OUT_DIR" \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMP" +ray_kwargs.ray_init.include_dashboard=False \
    "$@"
