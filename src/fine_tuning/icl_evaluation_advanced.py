"""
In-Context Learning (ICL) Evaluation Script with Advanced Metrics
==================================================================

Uses sophisticated evaluation metrics:
- Tree Edit Distance (structural similarity)
- Key F1 (structure precision/recall)
- Item F1 (content precision/recall)
- Schema validation
- JSON validity

Tests Qwen2.5-Coder-3B BASE model with different shot counts:
- 0-shot, 1-shot, 3-shot, 10-shot, 30-shot, 50-shot
"""

import json
import torch
import random
import time
import os
import numbers
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
import jsonschema
import zss

# ============================================================================
# EVALUATION METRICS (Tree Edit Distance + F1 Scores)
# ============================================================================

class Tree:
    """Tree structure for tree edit distance calculation"""
    
    def __init__(self, label, is_value=False):
        self.label = label
        self.is_value = is_value
        self.children = []

    @staticmethod
    def get_children(tree):
        return tree.children
    
    @staticmethod
    def get_label(node):
        return node.label

    def size(self):
        size = 1
        for child in self.children:
            size += child.size()
        return size
    
    def get_paths(self, include_values=False):
        if len(self.children) == 0:
            if not self.is_value:
                return set([str(self.label) + "/"])
            elif include_values:
                return set([str(self.label)])
            else:
                return set([""])

        paths = set()
        for child in self.children:
            child_paths = child.get_paths(include_values)
            paths.update([f"{self.label}/{path}" for path in child_paths])

        return paths

def build_tree(data, label, schema):
    """Build tree from JSON data for tree edit distance"""
    tree = Tree(label)

    if isinstance(data, dict):
        properties_schema = schema.get("properties") if schema else None

        for key in sorted(data.keys()):
            child_schema = properties_schema.get(key) if properties_schema else None

            if child_schema and child_schema.get("type") == "string" and "enum" not in child_schema:
                # Ignore non-enum strings
                continue

            sub_tree = build_tree(
                data=data[key],
                label=key,
                schema=child_schema
            )

            tree.children.append(sub_tree)

    elif isinstance(data, list):
        sub_trees = []
        item_schema = schema.get("items") if schema else None

        if item_schema and item_schema.get("type") == "string" and "enum" not in item_schema:
            # Ignore non-enum strings
            return tree

        for i, item in enumerate(data):
            sub_tree = build_tree(
                data=item,
                label=None,
                schema=item_schema
            )

            sort_key = json.dumps(item, sort_keys=True)
            sub_trees.append((sort_key, sub_tree))

        if schema and schema.get("ordered") is False:
            sub_trees.sort(key=lambda x: x[0])

        for i, (sort_key, sub_tree) in enumerate(sub_trees):
            sub_tree.label = f"item_{i}"
            tree.children.append(sub_tree)

    else:
        if schema and schema.get("type") == "string" and "enum" not in schema:
            # Ignore non-enum strings
            return tree
        else:
            tree.children.append(Tree(data, is_value=True))

    return tree

def label_distance(true_label, pred_label):
    """Calculate distance between two node labels"""
    if true_label == pred_label:
        return 0
    
    if not isinstance(true_label, numbers.Number) \
            or not isinstance(pred_label, numbers.Number):
        return 1
    
    denominator = max(abs(true_label), abs(pred_label), 1e-6)
    cost = abs(true_label - pred_label) / denominator
    return min(cost, 1)

def tree_edit_distance(true_tree, pred_tree):
    """Calculate tree edit distance and similarity"""
    distance = zss.simple_distance(
        true_tree,
        pred_tree,
        Tree.get_children,
        Tree.get_label,
        label_distance
    )

    max_distance = true_tree.size() + pred_tree.size()
    if max_distance == 0:
        return 1.0, 0
    
    similarity = 1 - (distance / max_distance)
    return similarity, distance

def calculate_f1(true_set, pred_set):
    """Calculate F1, precision, recall for sets"""
    true_positives = len(true_set & pred_set)
    false_positives = len(pred_set - true_set)
    false_negatives = len(true_set - pred_set)

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if 2 * true_positives + false_positives + false_negatives == 0:
        f1 = 0
    else:
        f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

    return f1, precision, recall

def key_and_item_f1(true_tree, pred_tree):
    """Calculate F1 for keys (structure) and items (content)"""
    true_keys = true_tree.get_paths(include_values=False)
    pred_keys = pred_tree.get_paths(include_values=False)
    key_f1, key_precision, key_recall = calculate_f1(true_keys, pred_keys)

    true_items = true_tree.get_paths(include_values=True)
    pred_items = pred_tree.get_paths(include_values=True)
    item_f1, item_precision, item_recall = calculate_f1(true_items, pred_items)

    return {
        "key_f1": key_f1,
        "key_precision": key_precision,
        "key_recall": key_recall,
        "item_f1": item_f1,
        "item_precision": item_precision,
        "item_recall": item_recall
    }

def calculate_eval_metrics(true_spec_string, pred_spec_string, schema):
    """
    Calculate comprehensive evaluation metrics
    
    Returns dict with:
    - valid_json: bool
    - schema_match: bool
    - key_f1, key_precision, key_recall: float
    - item_f1, item_precision, item_recall: float
    - tree_edit_similarity, tree_edit_distance: float
    """
    scores = {}

    # Parse true spec
    try:
        true_spec = json.loads(true_spec_string)
    except json.JSONDecodeError:
        print(f"Error: true spec isn't valid JSON")
        return {'valid_json': False, 'error': 'true_spec_invalid'}

    # Parse predicted spec
    try:
        pred_spec = json.loads(pred_spec_string)
        scores['valid_json'] = True
    except json.JSONDecodeError:
        scores['valid_json'] = False
        scores['schema_match'] = False
        scores['key_f1'] = 0.0
        scores['key_precision'] = 0.0
        scores['key_recall'] = 0.0
        scores['item_f1'] = 0.0
        scores['item_precision'] = 0.0
        scores['item_recall'] = 0.0
        scores['tree_edit_similarity'] = 0.0
        scores['tree_edit_distance'] = float('inf')
        return scores

    # Schema validation
    try:
        jsonschema.validate(pred_spec, schema)
        scores['schema_match'] = True
    except jsonschema.ValidationError:
        scores['schema_match'] = False

    # Build trees and calculate metrics
    try:
        true_tree = build_tree(true_spec, "root", schema)
        pred_tree = build_tree(pred_spec, "root", schema)

        # F1 scores
        f1_scores = key_and_item_f1(true_tree, pred_tree)
        scores.update(f1_scores)

        # Tree edit distance
        similarity, distance = tree_edit_distance(true_tree, pred_tree)
        scores['tree_edit_similarity'] = similarity
        scores['tree_edit_distance'] = distance
    except Exception as e:
        print(f"Error calculating tree metrics: {e}")
        scores['key_f1'] = 0.0
        scores['key_precision'] = 0.0
        scores['key_recall'] = 0.0
        scores['item_f1'] = 0.0
        scores['item_precision'] = 0.0
        scores['item_recall'] = 0.0
        scores['tree_edit_similarity'] = 0.0
        scores['tree_edit_distance'] = float('inf')

    return scores

# ============================================================================
# ICL EVALUATOR
# ============================================================================

class ICLEvaluator:
    """In-Context Learning Evaluator with Advanced Metrics"""
    
    def __init__(
        self,
        model_path="./base_model",
        adapter_path=None,
        schema_path="comprehensive_fea_schema.json",
        train_data_path="train_data.jsonl",
        test_data_path="test_data.jsonl"
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.schema_path = schema_path
        
        # Load schema
        print("📋 Loading FEA schema...")
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        # Load ICL Pool (from Train Data)
        print(f"📚 Loading ICL pool from {train_data_path}...")
        self.icl_pool = self.load_examples(train_data_path)
        print(f"   Pool size: {len(self.icl_pool)} examples")
        
        # Load Test Set (from Test Data)
        print(f"🧪 Loading Test set from {test_data_path}...")
        self.test_set = self.load_examples(test_data_path)
        print(f"   Test size: {len(self.test_set)} examples\n")
        
        # Load model
        print("🔧 Loading model and tokenizer...")
        self.load_model()
        print("✅ Model ready!\n")
    
    def load_examples(self, data_path: str) -> List[Dict]:
        """Load examples from JSONL"""
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    examples.append(example)
                except json.JSONDecodeError:
                    continue
        return examples
    
    def load_model(self):
        """Load base model (not instruction-tuned)"""
        from transformers import BitsAndBytesConfig
        
        # 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.model.eval()

        # Load adapter if provided
        if hasattr(self, 'adapter_path') and self.adapter_path:
            print(f"🧩 Loading LoRA adapter from: {self.adapter_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model.eval()
    
    def create_prompt(
        self,
        user_input: str,
        num_shots: int = 0,
        include_schema: bool = True
    ) -> str:
        """Create prompt for BASE model (no chat template)"""
        
        prompt_parts = []
        
        # Add schema context
        if include_schema:
            schema_desc = self.get_schema_description()
            prompt_parts.append(schema_desc)
            prompt_parts.append("\n")
        
        # Add few-shot examples
        if num_shots > 0:
            # Stratified sampling: try to get a mix of simple and complex examples
            # We assume "complex" examples have more keys in their JSON
            
            # Sort pool by complexity (approx. JSON length)
            sorted_pool = sorted(self.icl_pool, key=lambda x: len(x['completion']))
            
            if num_shots == 1:
                # For 1-shot, pick a medium complexity example
                mid_idx = len(sorted_pool) // 2
                icl_examples = [sorted_pool[mid_idx]]
            else:
                # For N-shot, pick examples distributed across the complexity range
                indices = [int(i * (len(sorted_pool) - 1) / (num_shots - 1)) for i in range(num_shots)]
                icl_examples = [sorted_pool[i] for i in indices]
                
                # Shuffle so the order in prompt is random
                random.shuffle(icl_examples)
            
            for i, example in enumerate(icl_examples, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Input: {example['prompt']}")
                prompt_parts.append(f"Output: {example['completion']}")
                prompt_parts.append("\n")
        
        # Add the actual query
        prompt_parts.append("Input: " + user_input)
        prompt_parts.append("Output:")
        
        return "\n".join(prompt_parts)
    
    def get_schema_description(self) -> str:
        """Get condensed schema description"""
        schema_str = """You are converting natural language FEA descriptions to JSON.

Output Format (JSON):
{
  "geometry": {
    "type": "cylinder" | "box" | "sphere" | "plate" | "beam" | "shell" | "custom",
    "dimensions": { <geometry-specific dimensions in mm> }
  },
  "material": {
    "type": "steel" | "aluminum" | "titanium" | "concrete" | "composite" | "custom",
    "properties": { "elastic_modulus", "poissons_ratio", "density" }
  },
  "boundary_conditions": [
    {
      "type": "fixed" | "pinned" | "symmetry" | "roller",
      "location": { "description": "where to apply" }
    }
  ],
  "loads": [
    {
      "type": "force" | "pressure" | "torque" | "displacement",
      "magnitude": <value in N or Pa>,
      "direction": { "x": 0, "y": 0, "z": -1 },
      "location": { "description": "where to apply" }
    }
  ]
}

Instructions:
- Generate ONLY valid JSON
- All dimensions in millimeters (mm)
- Forces in Newtons (N), Pressures in Pascals (Pa)
- Include all required fields
"""
        return schema_str
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95
    ) -> Tuple[str, float]:
        """Generate JSON output from prompt"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=4096
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        gen_time = time.time() - start_time
        
        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        generated_text = full_output[len(prompt):].strip()
        
        return generated_text, gen_time
    
    def extract_json(self, text: str) -> str:
        """Extract JSON string from generated text"""
        # Try to find JSON in text
        if "{" in text and "}" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            return text[start:end]
        return text
    
    def evaluate_example(
        self,
        example: Dict,
        num_shots: int
    ) -> Dict:
        """Evaluate a single example with advanced metrics"""
        
        user_input = example['prompt']
        expected_output = example['completion']
        
        # Create prompt
        prompt = self.create_prompt(user_input, num_shots=num_shots)
        
        # Generate
        generated_text, gen_time = self.generate(prompt)
        
        # Extract JSON
        generated_json_str = self.extract_json(generated_text)
        
        # Calculate advanced metrics
        metrics = calculate_eval_metrics(
            true_spec_string=expected_output,
            pred_spec_string=generated_json_str,
            schema=self.schema
        )
        
        # Parse JSONs for display
        try:
            expected_json = json.loads(expected_output)
        except:
            expected_json = None
        
        try:
            generated_json = json.loads(generated_json_str)
        except:
            generated_json = None
        
        return {
            'prompt': user_input,  # FIXED: Keep full prompt (not truncated)
            'generated_text': generated_text[:500],
            'generated_json': generated_json,
            'generated_json_str': generated_json_str,
            'expected_json': expected_json,
            'expected_json_str': expected_output,
            'num_shots': num_shots,
            'generation_time': gen_time,
            **metrics  # Include all evaluation metrics
        }
    
    def evaluate_n_shot(
        self,
        num_shots: int,
        test_samples: List[Dict] = None,
        num_test_samples: int = 5
    ) -> pd.DataFrame:
        """Evaluate model with N-shot prompting"""
        print(f"\n{'='*70}")
        print(f"Evaluating {num_shots}-shot ICL")
        print(f"{'='*70}")
        
        # Use provided test samples, or sample randomly if not provided
        if test_samples is None:
            test_samples = random.sample(
                self.test_set,
                min(num_test_samples, len(self.test_set))
            )
        
        results = []
        
        for example in tqdm(test_samples, desc=f"{num_shots}-shot"):
            result = self.evaluate_example(example, num_shots)
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Calculate aggregate metrics
        metrics = {
            'num_shots': num_shots,
            'total_samples': len(results),
            'valid_json_rate': df['valid_json'].mean(),
            'schema_match_rate': df['schema_match'].mean(),
            'key_f1_mean': df['key_f1'].mean(),
            'key_precision_mean': df['key_precision'].mean(),
            'key_recall_mean': df['key_recall'].mean(),
            'item_f1_mean': df['item_f1'].mean(),
            'item_precision_mean': df['item_precision'].mean(),
            'item_recall_mean': df['item_recall'].mean(),
            'tree_edit_similarity_mean': df['tree_edit_similarity'].mean(),
            'avg_generation_time': df['generation_time'].mean(),
        }
        
        print(f"\n📊 Results for {num_shots}-shot:")
        print(f"   Valid JSON: {metrics['valid_json_rate']*100:.1f}%")
        print(f"   Schema Match: {metrics['schema_match_rate']*100:.1f}%")
        print(f"   Key F1: {metrics['key_f1_mean']:.3f}")
        print(f"   Item F1: {metrics['item_f1_mean']:.3f}")
        print(f"   Tree Edit Similarity: {metrics['tree_edit_similarity_mean']:.3f}")
        print(f"   Avg Time: {metrics['avg_generation_time']:.2f}s")
        
        return df, metrics
    
    def run_full_evaluation(
        self,
        shot_counts: List[int] = [0, 1, 3, 10, 30, 50],
        num_test_samples: int = 5,
        output_dir: str = "./icl_results"
    ):
        """Run full ICL evaluation across all shot counts"""
        print("\n" + "="*70)
        print("STARTING FULL ICL EVALUATION (ADVANCED METRICS)")
        print("="*70)
        print(f"Shot counts to test: {shot_counts}")
        print(f"Test samples per shot: {num_test_samples}")
        print(f"Output directory: {output_dir}\n")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # FIXED: Select test samples ONCE for ALL shot counts
        # This ensures we test the same examples at each shot level
        random.seed(42)  # Reproducible selection
        fixed_test_samples = random.sample(
            self.test_set,
            min(num_test_samples, len(self.test_set))
        )
        
        print(f"✅ Selected {len(fixed_test_samples)} test samples (SAME for all shots)")
        print(f"   Test sample prompts:")
        for i, sample in enumerate(fixed_test_samples, 1):
            print(f"   {i}. {sample['prompt'][:80]}...")
        print()
        
        all_metrics = []
        all_results = {}
        
        for num_shots in shot_counts:
            # Evaluate using the SAME test samples for all shot counts
            df_results, metrics = self.evaluate_n_shot(
                num_shots, 
                test_samples=fixed_test_samples
            )
            
            # Create a more readable version for CSV
            df_readable = df_results.copy()
            
            # Convert JSON objects to single-line strings (better for CSV)
            # Use compact JSON (no indent) to avoid newline issues in CSV
            df_readable['expected_json_str'] = df_readable['expected_json'].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if x else "INVALID"
            )
            df_readable['generated_json_str'] = df_readable['generated_json'].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if x else "INVALID"
            )
            
            # Reorder columns for clarity
            column_order = [
                'num_shots',
                'prompt',
                'expected_json_str',
                'generated_json_str',
                'valid_json',
                'schema_match',
                'key_f1',
                'key_precision',
                'key_recall',
                'item_f1',
                'item_precision',
                'item_recall',
                'tree_edit_similarity',
                'tree_edit_distance',
                'generation_time'
            ]
            df_readable = df_readable[column_order]
            
            # Save CSV with proper quoting
            result_file = os.path.join(output_dir, f"{num_shots}_shot_results.csv")
            df_readable.to_csv(
                result_file, 
                index=False,
                quoting=1,  # QUOTE_ALL - quote all fields
                escapechar='\\'
            )
            print(f"💾 Saved to: {result_file}")
            
            # Also save pretty-printed JSON to separate files for readability
            json_file = os.path.join(output_dir, f"{num_shots}_shot_results_pretty.jsonl")
            with open(json_file, 'w', encoding='utf-8') as f:
                for idx, row in df_results.iterrows():
                    result_entry = {
                        'num_shots': num_shots,
                        'prompt': row['prompt'],
                        'expected_json': row['expected_json'],
                        'generated_json': row['generated_json'],
                        'metrics': {
                            'valid_json': row['valid_json'],
                            'schema_match': row['schema_match'],
                            'key_f1': row['key_f1'],
                            'item_f1': row['item_f1'],
                            'tree_edit_similarity': row['tree_edit_similarity']
                        }
                    }
                    f.write(json.dumps(result_entry, indent=2, ensure_ascii=False) + '\n---\n')
            print(f"📄 Pretty JSON saved to: {json_file}")
            
            all_metrics.append(metrics)
            all_results[num_shots] = df_results
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(all_metrics)
        summary_file = os.path.join(output_dir, "summary_metrics.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print(f"\n📊 Summary Results:")
        print(summary_df.to_string(index=False))
        
        print(f"\n💾 All results saved to: {output_dir}")
        print(f"   - summary_metrics.csv")
        for num_shots in shot_counts:
            print(f"   - {num_shots}_shot_results.csv")
        
        # Create visualization
        self.create_visualization(summary_df, output_dir)
        
        return summary_df, all_results
    
    def create_visualization(self, summary_df: pd.DataFrame, output_dir: str):
        """Create visualization of results"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('ICL Performance with Advanced Metrics', fontsize=16)
            
            # Valid JSON Rate
            axes[0, 0].plot(summary_df['num_shots'], summary_df['valid_json_rate'] * 100, 'o-', linewidth=2)
            axes[0, 0].set_title('Valid JSON Rate')
            axes[0, 0].set_xlabel('Number of Shots')
            axes[0, 0].set_ylabel('Valid JSON (%)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Schema Match Rate
            axes[0, 1].plot(summary_df['num_shots'], summary_df['schema_match_rate'] * 100, 'o-', linewidth=2, color='green')
            axes[0, 1].set_title('Schema Match Rate')
            axes[0, 1].set_xlabel('Number of Shots')
            axes[0, 1].set_ylabel('Schema Match (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Key F1 Score
            axes[0, 2].plot(summary_df['num_shots'], summary_df['key_f1_mean'], 'o-', linewidth=2, color='orange')
            axes[0, 2].set_title('Key F1 Score (Structure)')
            axes[0, 2].set_xlabel('Number of Shots')
            axes[0, 2].set_ylabel('F1 Score')
            axes[0, 2].set_ylim([0, 1])
            axes[0, 2].grid(True, alpha=0.3)
            
            # Item F1 Score
            axes[1, 0].plot(summary_df['num_shots'], summary_df['item_f1_mean'], 'o-', linewidth=2, color='purple')
            axes[1, 0].set_title('Item F1 Score (Content)')
            axes[1, 0].set_xlabel('Number of Shots')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3)
            
            # Tree Edit Similarity
            axes[1, 1].plot(summary_df['num_shots'], summary_df['tree_edit_similarity_mean'], 'o-', linewidth=2, color='red')
            axes[1, 1].set_title('Tree Edit Similarity')
            axes[1, 1].set_xlabel('Number of Shots')
            axes[1, 1].set_ylabel('Similarity')
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].grid(True, alpha=0.3)
            
            # Generation Time
            axes[1, 2].plot(summary_df['num_shots'], summary_df['avg_generation_time'], 'o-', linewidth=2, color='brown')
            axes[1, 2].set_title('Average Generation Time')
            axes[1, 2].set_xlabel('Number of Shots')
            axes[1, 2].set_ylabel('Time (seconds)')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_file = os.path.join(output_dir, 'icl_performance_plot.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"\n📈 Visualization saved to: {plot_file}")
            
        except ImportError:
            print("\n⚠️  matplotlib not installed - skipping visualization")
            print("   Install with: pip install matplotlib")


def main():
    """Main evaluation pipeline"""
    
    print("\n" + "="*70)
    print("IN-CONTEXT LEARNING EVALUATION (ADVANCED METRICS)")
    print("="*70)
    print("\nTesting Qwen2.5-Coder-3B BASE model")
    print("Shot counts: 0, 1, 3, 5, 7")
    print("Test samples: 20 (for better accuracy)")
    print("\nMetrics:")
    print("  ✓ Tree Edit Distance")
    print("  ✓ Key F1 (structure)")
    print("  ✓ Item F1 (content)")
    print("  ✓ Schema validation\n")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ICLEvaluator(
        model_path="./base_model",
        adapter_path=args.adapter,
        schema_path="comprehensive_fea_schema.json",
        train_data_path="train_data.jsonl",
        test_data_path="test_data.jsonl"
    )
    
    output_dir = "./icl_results_finetuned" if args.adapter else "./icl_results_base"
    
    # Run evaluation
    summary_df, all_results = evaluator.run_full_evaluation(
        shot_counts=[0, 1, 3, 5, 7],
        num_test_samples=20,
        output_dir=output_dir
    )
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check ./icl_results/ for detailed results")
    print("2. View summary_metrics.csv for comparison")
    print("3. View icl_performance_plot.png for visualization")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
