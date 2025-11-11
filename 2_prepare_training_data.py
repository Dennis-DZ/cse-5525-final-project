"""
Step 2: Prepare Training Data with Schema Prefix
=================================================

This script will:
1. Load your FEA schema (comprehensive_fea_schema.json)
2. Load training examples (TrainingExamples_900.jsonl)
3. Format each example with:
   - System message with schema
   - User prompt
   - Assistant response (JSON)
4. Split into train/validation sets (80/20)
5. Save prepared datasets

This ensures the model learns the schema structure!
"""

import json
import os
from pathlib import Path
from datasets import Dataset
import random

def load_schema(schema_path="comprehensive_fea_schema.json"):
    """Load and format the FEA schema"""
    print("=" * 70)
    print("Loading FEA Schema")
    print("=" * 70)
    
    if not os.path.exists(schema_path):
        print(f"❌ Schema file not found: {schema_path}")
        return None
    
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Convert schema to readable string
    schema_str = json.dumps(schema, indent=2)
    
    # Create condensed version for context (keep it under 2000 tokens)
    # Extract key structure without all the verbose descriptions
    condensed_schema = {
        "type": "object",
        "required": schema.get("required", []),
        "properties": {
            key: {
                "type": value.get("type", "object"),
                "description": value.get("description", "")[:100]  # Truncate long descriptions
            }
            for key, value in schema.get("properties", {}).items()
        }
    }
    
    condensed_str = json.dumps(condensed_schema, indent=2)
    
    print(f"✅ Schema loaded successfully")
    print(f"   Full schema size: {len(schema_str)} characters")
    print(f"   Condensed schema size: {len(condensed_str)} characters")
    print(f"   Required fields: {', '.join(schema.get('required', []))}")
    
    return condensed_str

def load_training_examples(data_path="TrainingExamples_900.jsonl"):
    """Load training examples from JSONL file"""
    print("\n" + "=" * 70)
    print("Loading Training Examples")
    print("=" * 70)
    
    if not os.path.exists(data_path):
        print(f"❌ Training data not found: {data_path}")
        return None
    
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipped invalid JSON at line {line_num}: {e}")
    
    print(f"✅ Loaded {len(examples)} training examples")
    
    # Show sample
    if examples:
        print(f"\n📋 Sample example:")
        print(f"   Prompt: {examples[0]['prompt'][:80]}...")
        print(f"   Completion length: {len(examples[0]['completion'])} chars")
    
    return examples

def create_formatted_prompt(schema_str, user_prompt, assistant_response):
    """
    Create formatted prompt with Qwen's chat template
    
    Format:
    <|im_start|>system
    You are an FEA expert. Convert natural language to JSON following this schema:
    {schema}
    <|im_end|>
    <|im_start|>user
    {user_prompt}
    <|im_end|>
    <|im_start|>assistant
    {assistant_response}<|im_end|>
    """
    
    system_message = f"""You are an expert FEA (Finite Element Analysis) engineer assistant. Your task is to convert natural language descriptions of mechanical engineering problems into structured JSON format.

Follow this JSON schema exactly:

{schema_str}

Instructions:
- Generate valid JSON matching the schema
- All dimensions in millimeters (mm)
- Forces in Newtons (N)
- Pressures in Pascals (Pa)
- Include all required fields: geometry, material, boundary_conditions, loads
- Use exact material type names from the schema
- Use proper geometry type and dimensions"""

    formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>"""
    
    return formatted_prompt

def prepare_dataset(examples, schema_str):
    """Prepare dataset with schema prefix"""
    print("\n" + "=" * 70)
    print("Formatting Dataset")
    print("=" * 70)
    
    formatted_examples = []
    
    for i, example in enumerate(examples):
        prompt = example.get('prompt', '')
        completion = example.get('completion', '')
        
        if not prompt or not completion:
            print(f"⚠️  Warning: Skipped incomplete example {i+1}")
            continue
        
        # Create formatted prompt with schema
        formatted_text = create_formatted_prompt(
            schema_str,
            prompt,
            completion
        )
        
        formatted_examples.append({
            'text': formatted_text,
            'original_prompt': prompt,  # Keep for reference
            'original_completion': completion
        })
        
        if (i + 1) % 100 == 0:
            print(f"   Formatted {i+1}/{len(examples)} examples...")
    
    print(f"✅ Formatted {len(formatted_examples)} examples")
    
    return formatted_examples

def split_dataset(examples, train_ratio=0.8, seed=42):
    """Split into train and validation sets"""
    print("\n" + "=" * 70)
    print("Splitting Dataset")
    print("=" * 70)
    
    random.seed(seed)
    random.shuffle(examples)
    
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f"✅ Train set: {len(train_examples)} examples ({train_ratio*100:.0f}%)")
    print(f"✅ Validation set: {len(val_examples)} examples ({(1-train_ratio)*100:.0f}%)")
    
    return train_examples, val_examples

def save_datasets(train_examples, val_examples, output_dir="./prepared_data"):
    """Save prepared datasets"""
    print("\n" + "=" * 70)
    print("Saving Datasets")
    print("=" * 70)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_dict({
        'text': [ex['text'] for ex in train_examples]
    })
    
    val_dataset = Dataset.from_dict({
        'text': [ex['text'] for ex in val_examples]
    })
    
    # Save as parquet (efficient format)
    train_path = os.path.join(output_dir, "train_dataset")
    val_path = os.path.join(output_dir, "val_dataset")
    
    train_dataset.save_to_disk(train_path)
    val_dataset.save_to_disk(val_path)
    
    print(f"✅ Saved training dataset to: {train_path}")
    print(f"✅ Saved validation dataset to: {val_path}")
    
    # Also save as JSONL for inspection
    train_jsonl = os.path.join(output_dir, "train_formatted.jsonl")
    val_jsonl = os.path.join(output_dir, "val_formatted.jsonl")
    
    with open(train_jsonl, 'w', encoding='utf-8') as f:
        for ex in train_examples:
            f.write(json.dumps({'text': ex['text']}) + '\n')
    
    with open(val_jsonl, 'w', encoding='utf-8') as f:
        for ex in val_examples:
            f.write(json.dumps({'text': ex['text']}) + '\n')
    
    print(f"✅ Also saved as JSONL for inspection")
    
    # Print statistics
    avg_length = sum(len(ex['text']) for ex in train_examples) / len(train_examples)
    max_length = max(len(ex['text']) for ex in train_examples)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Average example length: {avg_length:.0f} characters")
    print(f"   Max example length: {max_length} characters")
    print(f"   Estimated tokens per example: {avg_length/4:.0f}")  # Rough estimate
    
    return train_dataset, val_dataset

def preview_formatted_example(examples, num_samples=2):
    """Preview formatted examples"""
    print("\n" + "=" * 70)
    print("Preview Formatted Examples")
    print("=" * 70)
    
    for i in range(min(num_samples, len(examples))):
        ex = examples[i]
        print(f"\n{'='*50}")
        print(f"Example {i+1}:")
        print(f"{'='*50}")
        
        # Show just the first 500 chars to keep it readable
        text = ex['text']
        if len(text) > 1000:
            print(text[:500])
            print("\n... [truncated] ...\n")
            print(text[-500:])
        else:
            print(text)

def main():
    """Main data preparation function"""
    print("\n" + "=" * 70)
    print("FEA TRAINING DATA PREPARATION")
    print("=" * 70)
    print("\nPreparing data with schema prefix for optimal learning...\n")
    
    # Load schema
    schema_str = load_schema("comprehensive_fea_schema.json")
    if schema_str is None:
        return
    
    # Load training examples
    examples = load_training_examples("TrainingExamples_900.jsonl")
    if examples is None:
        return
    
    # Format with schema prefix
    formatted_examples = prepare_dataset(examples, schema_str)
    
    # Split train/val
    train_examples, val_examples = split_dataset(formatted_examples)
    
    # Preview
    preview_formatted_example(train_examples, num_samples=1)
    
    # Save datasets
    train_dataset, val_dataset = save_datasets(train_examples, val_examples)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print("\nPrepared datasets:")
    print("📂 ./prepared_data/train_dataset/")
    print("📂 ./prepared_data/val_dataset/")
    print("\nNext step:")
    print("Run: python 3_finetune_model.py")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Data preparation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
