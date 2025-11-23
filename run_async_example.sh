#!/bin/bash

# --- Resource Allocation ---
NNODES_TRAIN=1
NNODES_ROLLOUT=1

# --- Async Config ---
rollout_mode="async"
rollout_name="vllm"
return_raw_chat="True"
staleness_threshold=0.5
trigger_parameter_sync_step=4
partial_rollout=True

# In async, this must be zero for the rollouter.
train_batch_size=0
ppo_mini_batch_size=80
total_rollout_steps=200000

# --- Command ---
TRAINING_COMMAND="python3 -m recipe.fully_async_policy.fully_async_main \
    data.train_batch_size=${train_batch_size} \
    data.return_raw_chat=${return_raw_chat} \
    data.train_files='/home/jovyan/data/stack-exchange/train.parquet' \
    data.val_files='/home/jovyan/data/stack-exchange/train.parquet' \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    critic.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    critic.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.actor.strategy=fsdp \
    critic.strategy=fsdp \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=1 \
    reward_model.n_gpus_per_node=0.5 \
    rollout.total_rollout_steps=${total_rollout_steps} \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.partial_rollout=${partial_rollout} \
    trainer.val_before_train=False \
    rollout.test_freq=-1 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    critic.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.enable=True \
    reward_model.model.path=KaiZhuo/Qwen_3B_RM \
    reward_model.model.trust_remote_code=True \
    reward_model.micro_batch_size_per_gpu=8 \
    trainer.project_name=verl_async_test \
    trainer.total_training_steps=50 \
    trainer.experiment_name=stack_exchange_async_test"

# --- Execution ---
echo "Cleaning up previous Ray processes..."
ray stop --force
pkill -9 -f "python3 -m recipe.fully_async_policy"
sleep 5

echo "Starting Verl async test job on 4 GPUs"
export PYTHONPATH=.
export RAY_IP_ADDRESS=192.168.41.88
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

$TRAINING_COMMAND 2>&1 | tee verl_4gpu_async_test.log

echo "Job finished."