#!/bin/bash

TRAINING_COMMAND="python3 -m recipe.fully_async_policy.fully_async_main \
 data.return_raw_chat=True \
 data.train_files="/home/jovyan/data/stack-exchange/train.parquet" \
 data.val_files="/home/jovyan/data/stack-exchange/train.parquet" \
 data.prompt_key=prompt \
 data.truncation=left \
 data.max_prompt_length=512 \
 data.max_response_length=4000 \
 data.seed=42 \
 actor_rollout_ref.actor.strategy=fsdp \
 actor_rollout_ref.ref.strategy=fsdp \
 critic.strategy=fsdp \
 actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
 actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
 critic.ulysses_sequence_parallel_size=1 \
 actor_rollout_ref.model.use_remove_padding=True \
 critic.model.use_remove_padding=True \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
 actor_rollout_ref.model.trust_remote_code=True \
 critic.model.path=Qwen/Qwen2.5-3B-Instruct \
 critic.model.trust_remote_code=True \
 data.train_batch_size=0 \
 actor_rollout_ref.actor.ppo_epochs=4 \
 critic.ppo_epochs=4 \
 actor_rollout_ref.actor.ppo_mini_batch_size=80 \
 critic.ppo_mini_batch_size=80 \
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
 reward_model.model.trust_remote_code=True \
 reward_model.micro_batch_size_per_gpu=8 \
 actor_rollout_ref.hybrid_engine=False \
 trainer.n_gpus_per_node=2 \
 rollout.n_gpus_per_node=2 \
 async_training.staleness_threshold=0.5 \
 async_training.require_batches=1 \
 async_training.trigger_parameter_sync_step=1 \
 async_training.partial_rollout=True \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
 critic.ppo_micro_batch_size_per_gpu=20 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.mode=async \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.rollout.enforce_eager=True \
 actor_rollout_ref.rollout.calculate_log_probs=True \
 actor_rollout_ref.rollout.enable_prefix_caching=False \
 actor_rollout_ref.rollout.enable_chunked_prefill=False \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.temperature=0 \
 trainer.logger=[console,wandb] \
 trainer.project_name=verl \
 trainer.experiment_name=stack_exchange_ppo_async \
 trainer.val_before_train=False \
 trainer.resume_mode=disable \
 trainer.total_training_steps=50"

# ... rest of the script ...

# --- Execution ---
echo "Starting Verl quickstart fully async training job on 4 GPUs"
export PYTHONPATH=.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_IP_ADDRESS=192.168.41.88
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
$TRAINING_COMMAND 2>&1 | tee verl_4gpu_async_demo.log

echo "Job finished."