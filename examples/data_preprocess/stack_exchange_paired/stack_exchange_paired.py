import os
import datasets
from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/root/code/verl/examples/data/stack_exchange')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'lvwerra/stack-exchange-paired'
    
    # Load only the train dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset(data_source, data_dir="data/rl", split="train", verification_mode="no_checks")
    
    # Split into train and test
    print("Splitting dataset...")
    dataset = dataset.train_test_split(test_size=0.1, seed=0)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    def make_map_fn(split):
        def process_fn(example, idx):
            # Each example has 'question', 'response_j', 'response_k', 'score_j', 'score_k'
            # where j is the chosen response and k is the rejected response
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": example['question']
                }],
                "ability": "qa",  # This dataset is about Q&A ability
                "reward_model": {
                    "style": "preference",  # Using preference-based reward
                    "chosen": example['response_j'],
                    "rejected": example['response_k'],
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'date': example['date'],
                    'qid': example['qid'],
                    'metadata': example['metadata']
                }
            }
            return data
            
        return process_fn

    # Process both datasets
    print("Processing train dataset...")
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    print("Processing test dataset...")
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Create directory if it doesn't exist
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir

    # Save to parquet format
    print("Saving datasets...")
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print("Done!")
