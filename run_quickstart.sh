#!/bin/bash

# NOTE: This script is modified for INTERACTIVE execution only.
# For production runs, remove --cleanenv and submit with sbatch.

#SBATCH --job-name=verl_quickstart
#SBATCH --partition=gpuA100
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=quickstart_job_%j.out
#SBATCH --error=quickstart_job_%j.err
#SBATCH --chdir=/work/hdd/bcrn/yyu69/verl

# Path to your downloaded Apptainer container image
IMAGE_PATH="./verl_ngc-th2.8.0-cu12.9-vllm0.11.0.sif"

TRAINING_COMMAND="python3 -m verl.trainer.main_ppo \
 data.return_raw_chat=True \
 data.train_files="/work/hdd/bcrn/yyu69/verl/data/stack-exchange-preprocessed-filtered/train.parquet" \
 data.val_files="/work/hdd/bcrn/yyu69/verl/data/stack-exchange-preprocessed-filtered/train.parquet" \
 data.prompt_key=prompt \
 data.truncation=left \
 data.max_prompt_length=512 \
 data.max_response_length=1000 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
 critic.model.path=Qwen/Qwen2.5-3B-Instruct \
 data.train_batch_size=64 \
 actor_rollout_ref.actor.ppo_epochs=4 \
 critic.ppo_epochs=4 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 critic.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.optim.lr=1.41e-5 \
 critic.optim.lr=1.41e-5 \
 actor_rollout_ref.actor.clip_ratio=0.2 \
 critic.cliprange_value=0.2 \
 algorithm.gamma=1.0 \
 algorithm.lam=0.95 \
 algorithm.kl_ctrl.type=adaptive \
 algorithm.kl_ctrl.kl_coef=0.2 \
 algorithm.kl_ctrl.target_kl=6.0 \
 algorithm.kl_ctrl.horizon=10000 \
 reward_model.enable=True \
 reward_model.model.path=KaiZhuo/Qwen_3B_RM \
 reward_model.n_gpus_per_node=1 \
 reward_model.micro_batch_size_per_gpu=8 \
 trainer.n_gpus_per_node=4 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
 critic.ppo_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 trainer.logger=[console,wandb] \
 trainer.project_name=verl \
 trainer.experiment_name=stack_exchange_ppo \
 trainer.val_before_train=False \
 trainer.resume_mode=disable \
 trainer.total_epochs=15"

# ... rest of the script ...

# --- Execution ---
echo "Starting Verl quickstart training job on 4 GPUs (Interactive Mode)..."
apptainer run --nv \
  --cleanenv \
  --env PYTHONPATH=. \
  "$IMAGE_PATH" \
  /bin/bash -c "$TRAINING_COMMAND" 2>&1 | tee verl_4gpu_interactive_demo.log

echo "Job finished."