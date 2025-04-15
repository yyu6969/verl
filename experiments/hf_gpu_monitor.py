# experiments/direct_gpu_monitor.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pickle

# Make sure the output directory exists
OUTPUT_DIR = "/work/hdd/bdkz/yyu69/verl/experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class GPUUtilizationMonitor:
    def __init__(self):
        self.prefill_utilization = []
        self.decode_utilization = []
        self.prompt_ids = []
        
    def record_prefill(self, prompt_id):
        """Record GPU utilization for prefill phase"""
        torch.cuda.synchronize()
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem = total_mem - free_mem
        utilization = used_mem / total_mem
        
        self.prompt_ids.append(prompt_id)
        self.prefill_utilization.append(utilization)
        print(f"Prompt {prompt_id}, prefill: {utilization:.2%} GPU utilization")
        
    def record_decode(self, prompt_id):
        """Record GPU utilization for decode phase"""
        torch.cuda.synchronize()
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem = total_mem - free_mem
        utilization = used_mem / total_mem
        
        self.decode_utilization.append(utilization)
        print(f"Prompt {prompt_id}, decode: {utilization:.2%} GPU utilization")
        
    def plot_chart(self, save_path=None, prompt_type=None):
        """Generate bidirectional bar chart"""
        # Make sure we have matching data
        assert len(self.prompt_ids) == len(self.prefill_utilization) == len(self.decode_utilization), \
               "Mismatch in data lengths"
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(self.prompt_ids))
        width = 0.35
        
        # Convert values to percentages for display
        prefill_pct = [u * 100 for u in self.prefill_utilization]
        decode_pct = [u * 100 for u in self.decode_utilization]
        
        # Plot bars
        prefill_bars = ax.bar(x - width/2, prefill_pct, width, label='Prefill')
        decode_bars = ax.bar(x + width/2, decode_pct, width, label='Decode')
        
        # Labels and formatting
        ax.set_xlabel('Prompt')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Utilization: Prefill vs Decode')
        
        # Set y-axis from 0% to 100%
        ax.set_ylim(0, 100)
        
        # Set x-tick labels based on prompt type
        if prompt_type == "tokens":
            # Try to load token counts
            token_path = os.path.join("/work/hdd/bdkz/yyu69/verl/experiment_results/prompts/prompts_8192", 
                                     "token_prompts_info.json")
            if os.path.exists(token_path):
                try:
                    with open(token_path, 'r') as f:
                        token_data = json.load(f)
                        token_counts = token_data["token_counts"]
                        labels = [f"{count} tokens" for count in token_counts]
                        ax.set_xticks(x)
                        ax.set_xticklabels(labels)
                except Exception as e:
                    print(f"Error loading token info: {e}")
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'Prompt {i+1}' for i in range(len(self.prompt_ids))])
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([f'Prompt {i+1}' for i in range(len(self.prompt_ids))])
        else:
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

def load_prompts(prompt_type="long"):
    """
    Load prompts from saved files
    
    Args:
        prompt_type: Type of prompts to load - "fibonacci", "long", or "tokens"
    
    Returns:
        List of prompts
    """
    # Set paths
    prompts_dir = "/work/hdd/bdkz/yyu69/verl/experiment_results/prompts/prompts_8192"
    
    # Select file based on prompt type
    if prompt_type == "fibonacci":
        file_base = "fibonacci_prompts"
    elif prompt_type == "tokens":
        file_base = "token_prompts"
    else:
        file_base = "long_sequence_prompts"
    
    # Try different file formats in order of preference
    pickle_path = os.path.join(prompts_dir, f"{file_base}.pkl")
    json_path = os.path.join(prompts_dir, f"{file_base}.json")
    txt_path = os.path.join(prompts_dir, f"{file_base}.txt")
    
    # Try to load prompts from pickle file
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
    
    # Try to load prompts from JSON file
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
    
    # Try to load prompts from text file
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error loading text file: {e}")
    
    # If all loading attempts fail, generate prompts on the fly
    print("Could not load prompts from files, generating on the fly")
    if prompt_type == "fibonacci":
        return generate_sequence_prompts(max_length=10)
    else:
        return generate_long_sequence_prompts()

def run_direct_test(model_path, output_file="gpu_utilization.png", prompt_type="long"):
    """Run a direct test using Hugging Face transformers"""
    # Initialize monitor
    monitor = GPUUtilizationMonitor()
    
    # Load model and tokenizer directly
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load test prompts
    print(f"Loading {prompt_type} prompts...")
    test_prompts = load_prompts(prompt_type)
    print(f"Loaded {len(test_prompts)} prompts")
    
    # Print prompt information
    for i, prompt in enumerate(test_prompts):
        num_numbers = len(prompt.split(',')) - 1
        print(f"Prompt {i+1}: {num_numbers} numbers, {len(prompt)} chars")
    
    # Process each prompt
    for i, prompt in enumerate(test_prompts):
        print(f"Prompt {i+1}: {prompt}")
        
        # Prepare input for the model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Clear cache before prefill measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure prefill (forward pass without generating)
        with torch.no_grad():
            _ = model(**inputs)
        
        # Record prefill utilization
        monitor.record_prefill(prompt)
        
        # Clear cache before decode measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure decode (generate)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=128)
        
        # Record decode utilization
        monitor.record_decode(prompt)
        
        # Let the system stabilize
        time.sleep(1)
    
    # Generate chart
    monitor.plot_chart(save_path=output_file, prompt_type=prompt_type)
    return monitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU utilization during prefill and decode phases")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Model path")
    parser.add_argument("--output", default="gpu_utilization.png", help="Output chart path")
    parser.add_argument("--prompt-type", default="long", choices=["fibonacci", "long", "tokens"], 
                        help="Type of prompts to use")
    args = parser.parse_args()
    
    # Ensure output is saved to the experiment_results directory
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(OUTPUT_DIR, output_path)
    
    run_direct_test(args.model, output_path, args.prompt_type)