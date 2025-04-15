# experiments/save_token_prompts.py
import os
import json
import pickle
from transformers import AutoTokenizer

# Set output directory
OUTPUT_DIR = "/work/hdd/bdkz/yyu69/verl/experiment_results/prompts/prompts_8192"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_token_prompts(model_path="meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Generate prompts with exact token lengths
    """
    # Load tokenizer
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Target token lengths
    target_lengths = [256, 512, 1024, 2048, 4096, 8192]
    
    prompts = []
    token_counts = []
    
    for target_length in target_lengths:
        print(f"Generating prompt with exactly {target_length} tokens...")
        
        # Start with small number of elements
        sequence = []
        current_prompt = ""
        current_tokens = 0
        
        # Keep adding numbers until we reach or exceed the target token length
        i = 0
        while current_tokens < target_length:
            # Add a number to the sequence
            sequence.append(i)
            
            # Update the prompt text
            current_prompt = ', '.join(str(num) for num in sequence) + ", what is the next number?"
            
            # Check token length
            tokens = tokenizer.encode(current_prompt)
            current_tokens = len(tokens)
            
            # Move to next number
            i += 1
            
            # Print progress for long sequences
            if i % 500 == 0:
                print(f"  - Added {i} numbers, current token count: {current_tokens}")
        
        # We might have overshot the target, so trim the sequence if needed
        while current_tokens > target_length:
            sequence.pop()
            current_prompt = ', '.join(str(num) for num in sequence) + ", what is the next number?"
            tokens = tokenizer.encode(current_prompt)
            current_tokens = len(tokens)
        
        # Add the final prompt
        prompts.append(current_prompt)
        token_counts.append(current_tokens)
        
        print(f"  - Final prompt: {len(sequence)} numbers, {current_tokens} tokens, {len(current_prompt)} chars")
    
    return prompts, token_counts

def save_prompts_for_import(prompts, token_counts, filename):
    """Save the prompts in multiple formats"""
    # Save as JSON
    json_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    # Save as pickle
    pickle_path = os.path.join(OUTPUT_DIR, f"{filename}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(prompts, f)
    
    # Save token information
    token_path = os.path.join(OUTPUT_DIR, f"{filename}_info.json")
    with open(token_path, 'w') as f:
        token_data = {
            "prompts": prompts,
            "token_counts": token_counts
        }
        json.dump(token_data, f, indent=2)
    
    # Save readable info
    info_path = os.path.join(OUTPUT_DIR, f"{filename}_info.txt")
    with open(info_path, 'w') as f:
        f.write("Token-Controlled Prompts\n")
        f.write("=======================\n\n")
        for i, (prompt, count) in enumerate(zip(prompts, token_counts)):
            f.write(f"Prompt {i+1}:\n")
            f.write(f"  Target tokens: {token_counts[i]}\n")
            f.write(f"  Actual tokens: {count}\n")
            f.write(f"  Numbers: {len(prompt.split(',')) - 1}\n")
            f.write(f"  Characters: {len(prompt)}\n")
            f.write(f"  First 50 chars: {prompt[:50]}...\n\n")
    
    return json_path, pickle_path, token_path, info_path

if __name__ == "__main__":
    # Generate the prompts
    prompts, token_counts = generate_token_prompts()
    
    # Save the prompts
    json_path, pickle_path, token_path, info_path = save_prompts_for_import(
        prompts, token_counts, "token_prompts")
    
    print(f"Saved {len(prompts)} token-controlled prompts in formats:")
    print(f"  - JSON: {json_path}")
    print(f"  - Pickle: {pickle_path}")
    print(f"  - Token info: {token_path}")
    print(f"  - Info: {info_path}")
    
    # Print prompt statistics
    print("\nPrompt Statistics:")
    print("=" * 60)
    for i, (prompt, token_count) in enumerate(zip(prompts, token_counts)):
        num_numbers = len(prompt.split(',')) - 1
        print(f"Prompt {i+1}: {token_count} tokens, {num_numbers} numbers, {len(prompt)} chars")