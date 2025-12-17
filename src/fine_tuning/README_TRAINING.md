
# FEA Model Training & Evaluation Pack (Transfer)

## contents
1. `finetune_qwen_v2.py`: Script to continue fine-tuning.
2. `icl_evaluation_advanced.py`: Script to evaluate model performance.
3. `checkpoint/`: The existing fine-tuned adapter weights (transferred from local).
4. `train_data.jsonl` & `test_data.jsonl`: Datasets.
5. `download_base_model.py`: Utility to fetch Qwen base model.

## Setup
1. Parse requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Download Base Model:
   ```bash
   python download_base_model.py
   ```
   This will save the 3B model to `./base_model`.

## Usage

### Fine-Tuning
To run fine-tuning (this script is pre-configured to look for `./base_model`):
```bash
python finetune_qwen_v2.py
```
This will save new checkpoints to `./new_ckpt`.

### Evaluation
Modify `icl_evaluation_advanced.py` if needed to point to the correct model adapter path (currently `./checkpoint`).
