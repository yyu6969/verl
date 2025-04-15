import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

# Path to your parquet files
train_file = "/work/hdd/bdkz/yyu69/data/filtered-stack-exchange-paired-512/train.parquet"
test_file = "/work/hdd/bdkz/yyu69/data/filtered-stack-exchange-paired-512/test.parquet"

# Load the tokenizer (same as in your training script)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

def analyze_lengths(file_path, split_name):
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    # Initialize lists to store lengths
    prompt_lengths = []
    
    # Process each prompt
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        prompt = row['prompt'][0]['content']  # Assuming this is the structure based on your dataset
        tokens = tokenizer(prompt, return_length=True)
        prompt_lengths.append(tokens['length'])
    
    # Convert to numpy array for easier analysis
    prompt_lengths = np.array(prompt_lengths)
    
    # Print statistics
    print(f"\n=== {split_name} Statistics ===")
    print(f"Total samples: {len(prompt_lengths)}")
    print(f"Mean length: {prompt_lengths.mean():.2f}")
    print(f"Median length: {np.median(prompt_lengths):.2f}")
    print(f"Max length: {prompt_lengths.max()}")
    print(f"Min length: {prompt_lengths.min()}")
    print(f"95th percentile: {np.percentile(prompt_lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(prompt_lengths, 99):.2f}")
    print(f"Samples > 512: {(prompt_lengths > 512).sum()}")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(prompt_lengths, bins=50, alpha=0.7)
    plt.title(f'Prompt Length Distribution - {split_name}')
    plt.xlabel('Length (tokens)')
    plt.ylabel('Count')
    plt.axvline(x=512, color='r', linestyle='--', label='Max length (512)')
    plt.legend()
    plt.savefig(f'prompt_length_dist_{split_name}.png')
    plt.close()
    
    return prompt_lengths

# Analyze both train and test sets
print("Analyzing train set...")
train_lengths = analyze_lengths(train_file, "Train")

print("\nAnalyzing test set...")
test_lengths = analyze_lengths(test_file, "Test")