import argparse
import os
import datasets
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

def main():
    """
    Preprocesses the yvngexe/stack-exchange-paired-small dataset.
    - Filters prompts to keep lengths between 50 and 450 tokens.
    - Converts to the chat format expected by Verl's PPO trainer.
    """
    parser = argparse.ArgumentParser(description="Preprocess stack-exchange-paired dataset for Verl.")
    parser.add_argument(
        "--local_save_dir",
        default="/work/hdd/bcrn/yyu69/verl/data/stack-exchange-preprocessed-filtered",
        help="The local directory to save the preprocessed dataset."
    )
    args = parser.parse_args()

    # --- Configuration ---
    dataset_name = "yvngexe/stack-exchange-paired-small"
    config_name = "rl_deduplicate_shuffled"
    model_path = "Qwen/Qwen2.5-3B-Instruct"  # Use the same tokenizer as the model
    
    print(f"Loading tokenizer for '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading dataset '{dataset_name}' (config: '{config_name}')...")
    dataset = datasets.load_dataset(dataset_name, config_name, split="train")
    print(f"Original dataset size: {len(dataset)}")

    def filter_by_length(example):
        tokens = tokenizer(example['question'], add_special_tokens=False)
        length = len(tokens['input_ids'])
        return 50 <= length <= 450

    print("Filtering dataset by prompt length (50 to 450 tokens)...")
    # Using num_proc > 1 can significantly speed this up.
    dataset = dataset.filter(filter_by_length, num_proc=os.cpu_count() // 2)
    print(f"Filtered dataset size: {len(dataset)}")

    print("Converting dataset to Pandas DataFrame...")
    df = dataset.to_pandas()

    def format_row(row):
        question_content = row["question"]
        row['prompt'] = [{"role": "user", "content": question_content}]
        return row

    print("Applying chat format to the dataset...")
    tqdm.pandas(desc="Preprocessing rows")
    df = df.progress_apply(format_row, axis=1)
    
    print("Creating final DataFrame...")
    final_df = df[["prompt", "response_j", "response_k"]]

    os.makedirs(args.local_save_dir, exist_ok=True)
    output_path = os.path.join(args.local_save_dir, "train.parquet")
    
    print(f"Saving preprocessed data to '{output_path}'...")
    final_df.to_parquet(output_path)
    
    print("\nPreprocessing complete.")
    print(f"New dataset saved at: {args.local_save_dir}")

if __name__ == "__main__":
    main()