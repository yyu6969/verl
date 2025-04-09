#!/bin/bash
set -x

# Make sure pandas and pyarrow are installed
pip install pandas pyarrow transformers -q

# Create filtered dataset
python -c "
import pandas as pd
import pyarrow.parquet as pq
import os
from transformers import AutoTokenizer

# Create directory for datasets if it doesn't exist
output_dir = '/work/hdd/bdkz/yyu69/data/filtered-stack-exchange-paired-512'
os.makedirs(output_dir, exist_ok=True)

# Load Llama 3 tokenizer for token counting
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')

# Function to count tokens in a prompt
def count_tokens(prompt):
    # Extract the content from the prompt which is a list of dicts
    if isinstance(prompt, list) and len(prompt) > 0 and 'content' in prompt[0]:
        text = prompt[0]['content']
    else:
        text = str(prompt)  # Fallback
    return len(tokenizer.encode(text))

# Process training data
print('Loading and filtering training data...')
train_file = '/work/hdd/bdkz/yyu69/data/stack-exchange-paired/train.parquet'
train_path = os.path.join(output_dir, 'train.parquet')

# Process in chunks to manage memory
parquet_file = pq.ParquetFile(train_file)
filtered_dfs = []
total_samples = 0
MAX_SAMPLES = 100000

for batch in parquet_file.iter_batches(batch_size=10000):
    if total_samples >= MAX_SAMPLES:
        break
        
    df = batch.to_pandas()
    print(f'Processing batch of {len(df)} samples...')
    
    # Count tokens in prompts
    token_counts = df['prompt'].apply(count_tokens)
    
    # Filter by token length <= 512
    filtered_df = df[token_counts <= 512]
    print(f'Filtered to {len(filtered_df)} samples (≤ 512 tokens)')
    
    # Add to our collection
    filtered_dfs.append(filtered_df)
    
    # Update total count
    total_samples += len(filtered_df)
    print(f'Total samples so far: {total_samples}')
    
    # If we've exceeded our limit, truncate the last dataframe
    if total_samples > MAX_SAMPLES:
        excess = total_samples - MAX_SAMPLES
        filtered_dfs[-1] = filtered_dfs[-1].iloc[:-excess]
        total_samples = MAX_SAMPLES
        print(f'Truncated to {MAX_SAMPLES} total samples')

# Combine all filtered chunks
final_df = pd.concat(filtered_dfs, ignore_index=True)
print(f'Saving {len(final_df)} training samples to {train_path}')
final_df.to_parquet(train_path)

# Process test data (with similar logic but smaller limit)
print('Loading and filtering test data...')
test_file = '/work/hdd/bdkz/yyu69/data/stack-exchange-paired/test.parquet'
test_path = os.path.join(output_dir, 'test.parquet')

# Process test data with a smaller sample size
parquet_file = pq.ParquetFile(test_file)
filtered_test_dfs = []
total_test_samples = 0
MAX_TEST_SAMPLES = 10000  # 10% of training samples

for batch in parquet_file.iter_batches(batch_size=5000):
    if total_test_samples >= MAX_TEST_SAMPLES:
        break
        
    df = batch.to_pandas()
    
    # Count tokens in prompts
    token_counts = df['prompt'].apply(count_tokens)
    
    # Filter by token length <= 512
    filtered_df = df[token_counts <= 512]
    
    # Add to our collection
    filtered_test_dfs.append(filtered_df)
    
    # Update total count
    total_test_samples += len(filtered_df)
    
    # If we've exceeded our limit, truncate the last dataframe
    if total_test_samples > MAX_TEST_SAMPLES:
        excess = total_test_samples - MAX_TEST_SAMPLES
        filtered_test_dfs[-1] = filtered_test_dfs[-1].iloc[:-excess]
        total_test_samples = MAX_TEST_SAMPLES

# Combine all filtered test chunks
final_test_df = pd.concat(filtered_test_dfs, ignore_index=True)
print(f'Saving {len(final_test_df)} test samples to {test_path}')
final_test_df.to_parquet(test_path)

print('Done creating filtered datasets!')
"

# Output file paths
train_files=/work/hdd/bdkz/yyu69/verl/examples/data/filtered-stack-exchange-paired-512/train.parquet
test_files=/work/hdd/bdkz/yyu69/verl/examples/data/filtered-stack-exchange-paired-512/test.parquet

echo "Created filtered datasets at:"
echo "- $train_files"
echo "- $test_files"
echo "Use these paths in your training script!"