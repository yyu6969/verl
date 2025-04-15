# experiments/run_direct_monitor.sh
#!/bin/bash

# Memory management settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

# Create logs directory
mkdir -p logs

# Make sure the experiment results directory exists
mkdir -p /work/hdd/bdkz/yyu69/verl/experiment_results/hf

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/hf_gpu_monitor_${timestamp}.log"

# Output file in the experiment results directory
output_file="gpu_utilization_long_${timestamp}.png"

# Execute the Python script with long prompts
python -u experiments/hf_gpu_monitor.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output $output_file \
    --prompt-type long \
    2>&1 | tee $log_file

echo "Monitoring complete. Output saved to /work/hdd/bdkz/yyu69/verl/experiment_results/hf/$output_file"