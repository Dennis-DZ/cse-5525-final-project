from transformers import AutoTokenizer
import json

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./models/qwen-coder-3b",
    trust_remote_code=True
)

# Load YOUR actual schema
with open("comprehensive_fea_schema.json", 'r') as f:
    schema = json.load(f)

# Convert to string (how it appears in prompts)
schema_str = json.dumps(schema, indent=2)

# Count tokens
tokens = tokenizer.encode(schema_str)

print("=" * 60)
print("YOUR SCHEMA TOKEN ANALYSIS")
print("=" * 60)
print(f"Schema file size: {len(schema_str)} characters")
print(f"Schema tokens: {len(tokens)} tokens")
print(f"Chars per token: {len(schema_str)/len(tokens):.2f}")
print(f"Est. reading time: {len(tokens)/750:.1f} seconds")
print("=" * 60)

# Show breakdown
print(f"\nFirst 50 tokens: {tokenizer.decode(tokens[:50])}")
print(f"\nLast 50 tokens: {tokenizer.decode(tokens[-50:])}")