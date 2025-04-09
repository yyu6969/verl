import pandas as pd
import os

# Path to the data
data_dir = '/work/hdd/bdkz/yyu69/verl/examples/data/processed-stack_exchange_paired'
output_file = 'longest_prompt.txt'

# Function to find the longest prompt in a dataframe
def find_longest_prompt(df):
    # Assuming the prompt column is named 'prompt', 'question', or similar
    # Adjust this based on your actual column name
    prompt_column = None
    for col in df.columns:
        if col.lower() in ['prompt', 'question', 'input', 'context']:
            prompt_column = col
            break
    
    if prompt_column is None:
        print(f"Column names: {df.columns}")
        raise ValueError("Could not identify the prompt column")
    
    # Get the lengths of all prompts
    prompt_lengths = df[prompt_column].str.len()
    
    # Find the index of the longest prompt
    max_idx = prompt_lengths.idxmax()
    
    # Get the longest prompt and its length
    longest_prompt = df.loc[max_idx, prompt_column]
    longest_length = prompt_lengths[max_idx]
    
    return longest_prompt, longest_length, max_idx

# Process both files and find the overall longest prompt
longest_prompt = ""
longest_length = 0
file_with_longest = ""
idx_with_longest = -1

for file in ['train.parquet', 'test.parquet']:
    file_path = os.path.join(data_dir, file)
    print(f"Processing {file}...")
    
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    # Find the longest prompt in this file
    prompt, length, idx = find_longest_prompt(df)
    
    # Check if this is the longest overall
    if length > longest_length:
        longest_prompt = prompt
        longest_length = length
        file_with_longest = file
        idx_with_longest = idx

# Print information about the longest prompt
print(f"\nLongest prompt found in {file_with_longest}, index {idx_with_longest}")
print(f"Length: {longest_length} characters")

# Save the longest prompt to a file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(longest_prompt)

print(f"\nLongest prompt saved to {output_file}")