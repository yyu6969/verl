#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for the log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/training_${timestamp}.log"

# Enable command printing and redirect all output to both terminal and log file
exec 1> >(tee -a "$log_file") 2>&1
set -x

# Use the medium datasets we just created
train_files=/work/hdd/bdkz/yyu69/data/filtered-stack-exchange-paired-512/train.parquet
test_files=/work/hdd/bdkz/yyu69/data/filtered-stack-exchange-paired-512/test.parquet

# Memory management settings
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS
export OMP_NUM_THREADS=1  # Prevent thread oversubscription
export RAY_memory_usage_threshold=0.95  # Kill at 95% memory

# Ray resource configuration
export RAY_num_cpus=64  # Total CPUs available to Ray
export RAY_num_gpus=4   # Total GPUs available to Ray

echo "Starting training at $(date)"
echo "Log file: $log_file"

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=False \
    data.truncation=right \
    data.return_raw_chat=True \
    data.return_raw_input_ids=True \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
    +actor_rollout_ref.actor.param_offload=True \
    +actor_rollout_ref.actor.use_shared_weights=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.sequence_parallel=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    critic.strategy=megatron \
    critic.megatron.tensor_model_parallel_size=2 \
    critic.megatron.pipeline_model_parallel_size=1 \
    critic.megatron.sequence_parallel=True \
    critic.megatron.use_distributed_optimizer=True \
    critic.optim.lr=1e-5 \
    critic.model.path=meta-llama/Llama-3.2-3B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    +critic.param_offload=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.use_dynamic_bsz=True \
    reward_model.enable=True \
    reward_model.strategy=megatron \
    reward_model.megatron.tensor_model_parallel_size=2 \
    reward_model.megatron.pipeline_model_parallel_size=1 \
    reward_model.megatron.sequence_parallel=True \
    reward_model.megatron.use_distributed_optimizer=True \
    reward_model.megatron.pad_vocab_size_to=128258 \
    reward_model.model.path=Ray2333/GRM-Llama3.2-3B-rewardmodel-ft \
    +reward_model.model.model_type=llama \
    +reward_model.model.trust_remote_code=True \
    +reward_model.model.architectures=["LlamaForSequenceClassification"] \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.use_dynamic_bsz=True \
    reward_model.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger="['console', 'wandb']" \
    trainer.resume_mode=disable \
    +trainer.wandb_entity="yyu69-university-of-illinois-urbana-champaign" \
    +trainer.wandb_project="rlhf" \
    trainer.project_name=llama3_rlhf_training \
    trainer.experiment_name=llama3_3b_megatron \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=4

echo "Training finished at $(date)"