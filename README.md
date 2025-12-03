# ICL Evaluation Package

This package contains scripts and data for evaluating In-Context Learning (ICL) performance on Finite Element Analysis (FEA) tasks.

## Contents
- `icl_evaluation_advanced.py`: Main evaluation script using advanced metrics (Tree Edit Distance, F1).
- `analyze_and_compare.py`: Visualization tools.
- `train_data.jsonl`: Training data (used for ICL few-shot examples).
- `test_data.jsonl`: Test data (held-out set for evaluation).
- `comprehensive_fea_schema.json`: JSON schema for validation.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the evaluation script:
```bash
python icl_evaluation_advanced.py
```

This will:
1. Load the model (`Qwen/Qwen2.5-Coder-3B` from Hugging Face).
2. Evaluate 0, 1, 3, 5, 7-shot performance.
3. Save results to `icl_results_base/`.

## Metrics
The script calculates:
- **Valid JSON Rate**: % of outputs that are valid JSON.
- **Schema Match Rate**: % of outputs matching the FEA schema.
- **Key F1**: Structural accuracy (keys).
- **Item F1**: Content accuracy (values).
- **Tree Edit Similarity**: Structural similarity score (0-1).
