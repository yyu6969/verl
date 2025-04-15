import json
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Load the prompts
with open("experiment_results/prompts/prompts_4096/token_prompts.json", "r") as f:
    prompts = json.load(f)

# Count tokens for each prompt
results = []
for i, prompt in enumerate(prompts):
    tokens = tokenizer.encode(prompt)
    token_count = len(tokens)
    results.append({
        "prompt_index": i,
        "token_count": token_count,
        "prompt_ending": prompt[-50:]  # Show the last 50 chars of prompt for reference
    })
    print(f"Prompt {i}: {token_count} tokens")

# Save results
with open("experiment_results/prompts/prompts_4096/token_counts.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to experiment_results/prompts/prompts_4096/token_counts.json")