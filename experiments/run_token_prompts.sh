#!/bin/bash
# experiments/run_token_prompts.sh

# Memory management settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

# Create logs directory
mkdir -p logs

# Make sure the experiment results directory exists
mkdir -p /work/hdd/bdkz/yyu69/verl/experiment_results/hf

# Create the token-controlled prompts if they don't exist
if [ ! -f "/work/hdd/bdkz/yyu69/verl/experiment_results/prompts/prompts_8192/token_prompts.json" ]; then
    echo "Generating token-controlled prompts..."
    python -u experiments/save_token_prompts.py
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/token_gpu_monitor_${timestamp}.log"

# Output file in the experiment results directory
output_file="gpu_utilization_tokens_${timestamp}.png"

# Modify hf_gpu_monitor.py to add support for token prompts
sed -i 's/choices=\["fibonacci", "long"\]/choices=\["fibonacci", "long", "tokens"\]/g' experiments/hf_gpu_monitor.py

# Execute the Python script with token prompts
python -u experiments/hf_gpu_monitor.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output $output_file \
    --prompt-type tokens \
    2>&1 | tee $log_file

echo "Monitoring complete. Output saved to /work/hdd/bdkz/yyu69/verl/experiment_results/hf/$output_file"