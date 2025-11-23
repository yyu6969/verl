#!/bin/bash

# --- Resource Allocation (8 GPUs Total) ---
NNODES_TRAIN=1
NNODES_ROLLOUT=1
NGPUS_TRAIN=2
NGPUS_ROLLOUT=4

# --- Async Config ---
rollout_mode="async"
rollout_name="vllm"
return_raw_chat="True"
staleness_threshold=0.8
trigger_parameter_sync_step=6
partial_rollout=True

# --- Batching and Steps ---
train_batch_size=0 # Must be 0 in async mode
ppo_mini_batch_size=32
total_rollout_steps=26000 # Large number to ensure rollouter doesn't stop early

# --- Command ---
TRAINING_COMMAND="python3 -m recipe.fully_async_policy.fully_async_main \
    data.train_batch_size=${train_batch_size} \
    data.return_raw_chat=${return_raw_chat} \
    data.train_files='/home/jovyan/data/stack-exchange/train.parquet' \
    data.val_files='/home/jovyan/data/stack-exchange/train.parquet' \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=512 \
    data.max_response_length=4000 \
    data.seed=42 \
    actor_rollout_ref.model.use_remove_padding=True \
    critic.model.use_remove_padding=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.model.trust_remote_code=True \
    critic.model.path=Qwen/Qwen2.5-3B-Instruct \
    critic.model.trust_remote_code=True \
    reward_model.enable=True \
    reward_model.model.path=KaiZhuo/Qwen_3B_RM \
    reward_model.n_gpus_per_node=0.5 \
    reward_model.model.trust_remote_code=True \
    reward_model.micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    critic.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_epochs=4 \
    critic.ppo_epochs=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    critic.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    critic.ppo_micro_batch_size_per_gpu=16 \
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
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.temperature=0 \
    trainer.project_name=verl \
    trainer.experiment_name=stack_exchange_ppo_async \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.total_training_steps=50 \
    trainer.nnodes=${NNODES_TRAIN} \
    trainer.n_gpus_per_node=${NGPUS_TRAIN} \
    rollout.nnodes=${NNODES_ROLLOUT} \
    rollout.n_gpus_per_node=${NGPUS_ROLLOUT} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    rollout.test_freq=-1 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.partial_rollout=${partial_rollout} \
    async_training.use_rollout_log_probs=True"

# --- Execution ---
echo "Cleaning up previous Ray processes..."
ray stop --force
pkill -9 -f "python3 -m recipe.fully_async_policy"
sleep 5

echo "Starting Verl async test job on 8 GPUs"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

$TRAINING_COMMAND 2>&1 | tee verl_8gpu_async_test.log

echo "Job finished."