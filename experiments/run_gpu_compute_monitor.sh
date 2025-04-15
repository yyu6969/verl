#!/bin/bash
# experiments/run_gpu_compute_monitor.sh

# Memory management settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

# Create logs directory
mkdir -p logs

# Make sure the experiment results directories exist
mkdir -p /work/hdd/bdkz/yyu69/verl/experiment_results/compute

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/gpu_compute_${timestamp}.log"

# Output file in the experiment results directory
output_file="gpu_compute_utilization_${timestamp}.png"

# Execute the Python script with token prompts
python -u experiments/gpu_compute_monitor.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output $output_file \
    2>&1 | tee $log_file

echo "Monitoring complete. Output saved to /work/hdd/bdkz/yyu69/verl/experiment_results/compute/$output_file"
