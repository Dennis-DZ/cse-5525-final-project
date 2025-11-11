from transformers import AutoTokenizer
import json

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./models/qwen-coder-3b",
    trust_remote_code=True
)

# Load ONE prepared training example
with open("./prepared_data/train_formatted.jsonl", 'r') as f:
    first_example = json.loads(f.readline())

# This example contains: schema + prompt + JSON output
full_text = first_example['text']

# Tokenize it
tokens = tokenizer.encode(full_text)

print("=" * 60)
print("ACTUAL TRAINING EXAMPLE TOKEN ANALYSIS")
print("=" * 60)
print(f"Total tokens: {len(tokens)}")
print(f"Total characters: {len(full_text)}")
print("\nBreakdown:")

# Try to estimate parts
parts = full_text.split("<|im_end|>")
if len(parts) >= 3:
    system_part = parts[0]  # Schema + instructions
    user_part = parts[1]     # User prompt
    
    system_tokens = len(tokenizer.encode(system_part))
    user_tokens = len(tokenizer.encode(user_part))
    output_tokens = len(tokens) - system_tokens - user_tokens
    
    print(f"  System (schema + instructions): {system_tokens} tokens")
    print(f"  User prompt: {user_tokens} tokens")
    print(f"  Assistant output (JSON): {output_tokens} tokens")
    print(f"  Total: {len(tokens)} tokens")

print("\n" + "=" * 60)