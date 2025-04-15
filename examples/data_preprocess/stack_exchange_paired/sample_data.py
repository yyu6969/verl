import pandas as pd

# Paths
train_input = "/work/hdd/bdkz/yyu69/data/filtered-stack-exchange-paired-512-100000/train.parquet"
test_input = "/work/hdd/bdkz/yyu69/data/filtered-stack-exchange-paired-512-100000/test.parquet"

train_output = "/work/hdd/bdkz/yyu69/data/filtered-stack-exchange-paired-512/train.parquet"
test_output = "/work/hdd/bdkz/yyu69/data/filtered-stack-exchange-paired-512/test.parquet"

# Read and sample train data
print("Processing train data...")
df_train = pd.read_parquet(train_input)
df_train_sampled = df_train.sample(n=50000, random_state=42)
df_train_sampled.to_parquet(train_output)
print(f"Saved {len(df_train_sampled)} train samples to {train_output}")

# Read and sample test data
print("Processing test data...")
df_test = pd.read_parquet(test_input)
df_test_sampled = df_test.sample(n=5000, random_state=42)
df_test_sampled.to_parquet(test_output)
print(f"Saved {len(df_test_sampled)} test samples to {test_output}")