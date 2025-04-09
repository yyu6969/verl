#!/bin/bash
set -x

# Make sure pandas and pyarrow are installed
pip install pandas pyarrow -q

# Create tiny training dataset (50 samples)
python -c "
import pandas as pd
import os

# Create directory for tiny datasets if it doesn't exist
tiny_dir = '/root/code/verl/examples/data/tiny'
os.makedirs(tiny_dir, exist_ok=True)

# Process training data
print('Loading training data sample...')
df = pd.read_parquet('/root/code/verl/examples/data/train.parquet')
print(f'Original training data size: {len(df)} rows')
small_df = df.head(50)  # Just take first 50 samples
tiny_train_path = os.path.join(tiny_dir, 'tiny_train.parquet')
print(f'Saving {len(small_df)} samples to {tiny_train_path}')
small_df.to_parquet(tiny_train_path)

# Process test data
print('Loading test data sample...')
test_df = pd.read_parquet('/root/code/verl/examples/data/test.parquet')
print(f'Original test data size: {len(test_df)} rows')
tiny_test = test_df.head(10)
tiny_test_path = os.path.join(tiny_dir, 'tiny_test.parquet')
print(f'Saving {len(tiny_test)} samples to {tiny_test_path}')
tiny_test.to_parquet(tiny_test_path)

print('Done creating tiny datasets!')
"

# Now run with the tiny datasets
train_files=/root/code/verl/examples/data/tiny/tiny_train.parquet
test_files=/root/code/verl/examples/data/tiny/tiny_test.parquet

echo "Created tiny datasets at:"
echo "- $train_files"
echo "- $test_files"
echo "Use these paths in your training script!"