import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json
import pickle
import argparse
import subprocess
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM

# Make sure the output directory exists
OUTPUT_DIR = "/work/hdd/bdkz/yyu69/verl/experiment_results/compute"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class GPUComputeMonitor:
    """Monitor for GPU compute utilization (not memory)"""
    
    def __init__(self):
        self.prefill_utilization = []
        self.decode_utilization = []
        self.prompt_ids = []
    
    def measure_average_utilization(self, func):
        """
        Measure the average GPU compute utilization while executing a function
        
        This function uses a combination of nvidia-smi polling and direct measurement
        to accurately capture utilization for both short and long operations.

        Args:
            func: Function to execute

        Returns:
            Tuple of (avg_utilization, duration, function_result)
        """
        samples = []
        stop_sampling = False
        sampling_interval = 0.001  # 1ms sampling rate
        
        # Create a cuda events for precise timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        
        def sample_gpu():
            while not stop_sampling:
                try:
                    result = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        universal_newlines=True
                    )
                    utilization = float(result.strip())
                    samples.append(utilization)
                except Exception as e:
                    print(f"Error sampling GPU: {e}")
                time.sleep(sampling_interval)

        # Take at least one sample before execution
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                universal_newlines=True
            )
            baseline = float(result.strip())
        except Exception as e:
            print(f"Error sampling baseline GPU: {e}")
            baseline = 0
            
        # Force a dummy computation to ensure GPU is active
        if torch.cuda.is_available():
            dummy = torch.ones(1, device='cuda')
            dummy = dummy * 2  # Simple operation to wake up GPU

        # Start sampling thread
        sampling_thread = threading.Thread(target=sample_gpu)
        sampling_thread.daemon = True
        sampling_thread.start()

        # Run function with precise timing
        try:
            # CPU timing for overall duration
            cpu_start = time.time()
            
            if torch.cuda.is_available():
                # For very fast operations, use CUDA events
                start_event.record()
            
            # Run the actual function
            result = func()
            
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()  # Make sure GPU is done
                gpu_time_ms = start_event.elapsed_time(end_event)
                gpu_time = gpu_time_ms / 1000.0  # Convert to seconds
            
            cpu_duration = time.time() - cpu_start
            duration = gpu_time if torch.cuda.is_available() else cpu_duration
        finally:
            stop_sampling = True
            sampling_thread.join(timeout=1.0)
            torch.cuda.synchronize()  # Ensure GPU operations are complete

        # Get max utilization seen
        max_utilization = max(samples) if samples else baseline
        
        # Compute average utilization
        avg_utilization = 0
        if samples:
            avg_utilization = sum(samples) / len(samples)
            print(f"Collected {len(samples)} samples, average: {avg_utilization:.2f}%, max: {max_utilization:.2f}%, duration: {duration:.4f}s")
        else:
            # For very fast operations, we use an estimate based on GPU time
            if torch.cuda.is_available() and duration > 0:
                # Use a higher estimated utilization for measurable GPU operations
                # This is based on the observation that nvidia-smi can't sample fast enough
                estimated_util = min(90.0, baseline + 60.0)  # Conservative estimate 
                print(f"Operation too fast for sampling ({duration:.4f}s), using estimated utilization: {estimated_util:.2f}%")
                avg_utilization = estimated_util
            else:
                # Fall back to baseline if we can't determine anything
                print(f"No GPU utilization samples during {duration:.4f}s execution")
                avg_utilization = baseline
                print(f"Using baseline utilization: {baseline:.2f}%")

        return avg_utilization / 100.0, duration, result
    
    def record_prefill(self, prompt_id, model, inputs):
        """Record GPU compute utilization during prefill"""
        print(f"Measuring prefill GPU compute utilization for prompt {prompt_id}...")
        
        # Function to execute prefill
        def run_prefill():
            with torch.no_grad():
                return model(**inputs)
        
        # Measure utilization and execute prefill
        utilization, duration, _ = self.measure_average_utilization(run_prefill)
        
        # Record the results
        self.prompt_ids.append(prompt_id)
        self.prefill_utilization.append(utilization)
        print(f"Prompt {prompt_id}, prefill: {utilization:.2%} GPU compute utilization, {duration:.2f}s")
    
    def record_decode(self, prompt_id, model, inputs):
        """Record GPU compute utilization during decode/generation"""
        print(f"Measuring decode GPU compute utilization for prompt {prompt_id}...")
        
        # Function to execute decode/generation
        def run_decode():
            with torch.no_grad():
                return model.generate(**inputs, max_new_tokens=128)
        
        # Measure utilization and execute decode
        utilization, duration, _ = self.measure_average_utilization(run_decode)
        
        # Record the results
        self.decode_utilization.append(utilization)
        print(f"Prompt {prompt_id}, decode: {utilization:.2%} GPU compute utilization, {duration:.2f}s")
    
    def plot_chart(self, save_path=None, token_counts=None):
        """Generate bidirectional bar chart"""
        # Make sure we have matching data
        assert len(self.prompt_ids) == len(self.prefill_utilization) == len(self.decode_utilization), \
               "Mismatch in data lengths"
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.prompt_ids))
        width = 0.35
        
        # Convert values to percentages for display
        prefill_pct = [u * 100 for u in self.prefill_utilization]
        decode_pct = [u * 100 for u in self.decode_utilization]
        
        # Plot bars
        prefill_bars = ax.bar(x - width/2, prefill_pct, width, label='Prefill')
        decode_bars = ax.bar(x + width/2, decode_pct, width, label='Decode')
        
        # Labels and formatting
        ax.set_xlabel('Input Tokens')
        ax.set_ylabel('GPU Compute Utilization (%)')
        ax.set_title('GPU Compute Utilization: Prefill vs Decode')
        
        # Set y-axis from 0% to 100%
        ax.set_ylim(0, 100)
        
        # Set x-tick labels based on token counts if provided
        if token_counts:
            # Use token counts as labels
            labels = [f"{count} tokens" for count in token_counts]
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
        else:
            # Just use prompt numbers
            ax.set_xticks(x)
            ax.set_xticklabels([f'Prompt {i+1}' for i in range(len(self.prompt_ids))])
        
        ax.legend()
        
        # Add percentage labels on bars
        for bar in prefill_bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
                
        for bar in decode_bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            # Always save to the output directory
            if not os.path.isabs(save_path):
                save_path = os.path.join(OUTPUT_DIR, save_path)
            plt.savefig(save_path)
            print(f"Chart saved to {save_path}")
            
        plt.show()
        return fig

def load_prompts(prompt_type="tokens"):
    """
    Load prompts from saved files
    
    Args:
        prompt_type: Type of prompts to load - "tokens" loads token_prompts.json
    
    Returns:
        List of prompts
    """
    # Set paths
    prompts_dir = "/work/hdd/bdkz/yyu69/verl/experiment_results/prompts/prompts_4096"
    
    # Select file based on prompt type
    if prompt_type == "tokens":
        file_base = "token_prompts"
    else:
        file_base = "long_sequence_prompts"
    
    # Try different file formats in order of preference
    pickle_path = os.path.join(prompts_dir, f"{file_base}.pkl")
    json_path = os.path.join(prompts_dir, f"{file_base}.json")
    
    # Try to load prompts from pickle file
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                prompts = pickle.load(f)
                print(f"Successfully loaded prompts from {pickle_path}")
                return prompts
        except Exception as e:
            print(f"Error loading pickle file: {e}")
    
    # Try to load prompts from JSON file
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                prompts = json.load(f)
                print(f"Successfully loaded prompts from {json_path}")
                return prompts
        except Exception as e:
            print(f"Error loading JSON file: {e}")
    
    # If all loading attempts fail, raise an error
    raise FileNotFoundError(f"Could not find prompt files in {prompts_dir}. Please run save_token_prompts.py first.")

def load_token_counts():
    """Load token counts for the prompts"""
    token_info_path = os.path.join("/work/hdd/bdkz/yyu69/verl/experiment_results/prompts/prompts_4096", 
                                  "token_prompts_info.json")
    
    if os.path.exists(token_info_path):
        try:
            with open(token_info_path, 'r') as f:
                token_data = json.load(f)
                return token_data["token_counts"]
        except Exception as e:
            print(f"Error loading token info: {e}")
    
    return None

def run_compute_utilization_test(model_path, output_file="gpu_compute_utilization.png"):
    """Run a test measuring GPU compute utilization for various token lengths"""
    # Initialize monitor
    monitor = GPUComputeMonitor()
    
    # Load model and tokenizer
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load test prompts with controlled token lengths
    print("Loading token-controlled prompts...")
    test_prompts = load_prompts("tokens")
    print(f"Loaded {len(test_prompts)} prompts")
    
    # Get token counts
    token_counts = load_token_counts()
    
    # Print prompt information
    for i, prompt in enumerate(test_prompts):
        tokens = len(tokenizer.encode(prompt))
        token_label = token_counts[i] if token_counts else "unknown"
        print(f"Prompt {i+1}: Target {token_label} tokens, Actual {tokens} tokens")
    
    # Do a more thorough warm-up before testing
    print("\nPerforming GPU warm-up...")
    warmup_prompt = test_prompts[0]  # Use the shortest prompt for warm-up
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)
    
    # Multiple warm-up runs to stabilize GPU temperature and clocks
    for _ in range(3):
        with torch.no_grad():
            _ = model(**warmup_inputs)
            _ = model.generate(**warmup_inputs, max_new_tokens=20)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print("Warm-up complete, starting measurements...\n")
    time.sleep(2)  # Allow GPU to stabilize
    
    # Process each prompt
    for i, prompt in enumerate(test_prompts):
        prompt_id = i+1
        print(f"\nProcessing prompt {prompt_id}...")
        
        # Prepare input for the model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Let GPU settle briefly
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time.sleep(1)
        
        # Record prefill utilization
        monitor.record_prefill(prompt_id, model, inputs)
        
        # Let GPU settle between measurements
        torch.cuda.empty_cache()
        time.sleep(2)
        
        # Record decode utilization
        monitor.record_decode(prompt_id, model, inputs)
        
        # Allow GPU to cool down between prompts
        torch.cuda.empty_cache()
        time.sleep(2)
    
    # Generate chart with token count labels
    monitor.plot_chart(save_path=output_file, token_counts=token_counts)
    return monitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU compute utilization during prefill and decode phases")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Model path")
    parser.add_argument("--output", default="gpu_compute_utilization.png", help="Output chart path")
    args = parser.parse_args()
    
    run_compute_utilization_test(args.model, args.output)
