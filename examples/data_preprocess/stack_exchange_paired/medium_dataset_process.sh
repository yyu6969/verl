#!/bin/bash
set -x

# Make sure pandas and pyarrow are installed
pip install pandas pyarrow -q

# Create dataset with 10k samples
python -c "
import pandas as pd
import pyarrow.parquet as pq
import os

# Create directory for datasets if it doesn't exist
data_dir = '/work/hdd/bdkz/yyu69/verl/examples/data/medium-processed-stack-exchange-paired'
os.makedirs(data_dir, exist_ok=True)

# Process training data
print('Loading training data sample...')
train_file = '/work/hdd/bdkz/yyu69/verl/examples/data/processed-stack_exchange_paired/train.parquet'
train_path = os.path.join(data_dir, 'train.parquet')

# Read and process in chunks
parquet_file = pq.ParquetFile(train_file)
first_chunk = next(parquet_file.iter_batches(batch_size=10000))
df = first_chunk.to_pandas()
print(f'Saving {len(df)} samples to {train_path}')
df.to_parquet(train_path)
del df, first_chunk, parquet_file

# Process test data
print('Loading test data sample...')
test_file = '/work/hdd/bdkz/yyu69/verl/examples/data/processed-stack_exchange_paired/test.parquet'
test_path = os.path.join(data_dir, 'test.parquet')

# Read and process test data in chunks
parquet_file = pq.ParquetFile(test_file)
first_chunk = next(parquet_file.iter_batches(batch_size=1000))
test_df = first_chunk.to_pandas()
print(f'Saving {len(test_df)} samples to {test_path}')
test_df.to_parquet(test_path)

print('Done creating medium-sized datasets!')
"

# Now run with the medium datasets
train_files=/work/hdd/bdkz/yyu69/verl/examples/data/medium-processed-stack-exchange-paired/train.parquet
test_files=/work/hdd/bdkz/yyu69/verl/examples/data/medium-processed-stack-exchange-paired/test.parquet

echo "Created medium-sized datasets at:"
echo "- $train_files"
echo "- $test_files"
echo "Use these paths in your training script!"