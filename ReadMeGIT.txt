steps

1. Download qwen-coder-3b and place it into the folder neamed models in your base directory. Also keep TrainingExamples_900.jsonl and comprehensive_fea_schema.json in your base directory 

    example downloaded from huggingface account

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    import torch
    
    model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
    local_model_dir = "./models/qwen-coder-3b"
    
    # Create directory
    Path(local_model_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading {model_name}...")
    print(f"📂 Saving to: {local_model_dir}")

2. Run 2_prepare_training_data.py (it adds schema from comprehensive_fea_schema.json and adds ome more instructions in pre prompt the add prompts and pair it with corresponding output json)

3. Run python 3_finetune_model.py (LORA and 4 bit quantization so 4GB ram should be enough)

4. Run 4_test_model.py

5 Run 5_inference.py


This is just starter script may have lots of bugs. 

This should do fine tuning. in fine tuning we are pre processing every prompt to json pair first. Our inputs include both schema and english prompt. 

Training example:
System: [Full FEA Schema]
Input: "Create a steel cylinder 20mm radius, 100mm tall"
Output: {"geometry": {"type": "cylinder", ...}}


We currently have 900 examples. 720 are used for training. 


Details of base model Qwen2.5 and why i started with this model.model_name
 1. Good Context length (32,768 tokens) this way we can add full schema, additional instructions and may be few shots later if required.
 2. Base model is trainind on engineering and scientific data.
 3. Small enough to train faster and locally. Also faster for inference time.