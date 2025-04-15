# save_sequence_prompts.py
import os
import json
import pickle

# Set output directory
OUTPUT_DIR = "/work/hdd/bdkz/yyu69/verl/experiment_results/prompts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_sequence_prompts(max_length=10, step=1):
    """
    Generate a list of prompts with increasing Fibonacci sequence length
    """
    prompts = []
    
    # Generate Fibonacci sequence
    fib = [0, 1]
    for i in range(2, 1025):  # Generate up to 1025 Fibonacci numbers
        fib.append(fib[i-1] + fib[i-2])
    
    # Create prompts with increasing sequence length
    for length in range(1, max_length + 1, step):
        sequence = ', '.join(str(fib[i]) for i in range(length))
        prompt = f"{sequence}, what is the next number?"
        prompts.append(prompt)
    
    return prompts

def generate_long_sequence_prompts():
    """Generate test prompts with increasing sequence length up to 1024 numbers"""
    prompts = []
    
    # Short prompts with Fibonacci sequence
    for length in [1, 2, 4, 8, 16]:
        fib = [0, 1]
        for i in range(2, length):
            fib.append(fib[i-1] + fib[i-2])
        sequence = ', '.join(str(num) for num in fib[:length])
        prompt = f"{sequence}, what is the next number?"
        prompts.append(prompt)
    
    # Medium prompts with arithmetic sequences
    for length in [32, 64, 128]:
        sequence = ', '.join(str(i) for i in range(length))
        prompt = f"{sequence}, what is the next number?"
        prompts.append(prompt)
    
    # Long prompts with arithmetic sequences
    for length in [256, 512, 1024]:
        sequence = ', '.join(str(i) for i in range(length))
        prompt = f"{sequence}, what is the next number?"
        prompts.append(prompt)
    
    return prompts

def save_prompts_to_file(prompts, filename):
    """Save the prompts to a file, one prompt per line"""
    with open(filename, 'w') as f:
        for i, prompt in enumerate(prompts):
            f.write(f"Prompt {i+1}: {prompt}\n")
            f.write("-" * 80 + "\n")  # Add separator between prompts

def save_prompts_for_import(prompts, filename):
    """Save the prompts in a format that can be imported in Python"""
    # Save as JSON
    json_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    # Save as pickle for direct import
    pickle_path = os.path.join(OUTPUT_DIR, f"{filename}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(prompts, f)
    
    # Also save as plain text file with one prompt per line
    txt_path = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    with open(txt_path, 'w') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")
    
    return json_path, pickle_path, txt_path

if __name__ == "__main__":
    # Generate Fibonacci sequence prompts (smaller lengths)
    fib_prompts = generate_sequence_prompts(max_length=10)
    save_prompts_to_file(fib_prompts, os.path.join(OUTPUT_DIR, "fibonacci_prompts_readable.txt"))
    json_path, pickle_path, txt_path = save_prompts_for_import(fib_prompts, "fibonacci_prompts")
    print(f"Saved {len(fib_prompts)} Fibonacci prompts in formats:")
    print(f"  - JSON: {json_path}")
    print(f"  - Pickle: {pickle_path}")
    print(f"  - Text: {txt_path}")
    
    # Generate long sequence prompts (up to 1024 numbers)
    long_prompts = generate_long_sequence_prompts()
    save_prompts_to_file(long_prompts, os.path.join(OUTPUT_DIR, "long_sequence_prompts_readable.txt"))
    json_path, pickle_path, txt_path = save_prompts_for_import(long_prompts, "long_sequence_prompts")
    print(f"Saved {len(long_prompts)} sequence prompts in formats:")
    print(f"  - JSON: {json_path}")
    print(f"  - Pickle: {pickle_path}")
    print(f"  - Text: {txt_path}")
    
    # Print prompt statistics
    print("\nPrompt Statistics:")
    print("=" * 60)
    print("Fibonacci prompts:")
    for i, prompt in enumerate(fib_prompts):
        print(f"Prompt {i+1}: {len(prompt.split(',')) - 1} numbers, {len(prompt)} chars")
    
    print("\nLong sequence prompts:")
    for i, prompt in enumerate(long_prompts):
        print(f"Prompt {i+1}: {len(prompt.split(',')) - 1} numbers, {len(prompt)} chars")