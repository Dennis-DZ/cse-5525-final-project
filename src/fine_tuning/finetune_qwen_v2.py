import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Configuration
MODEL_PATH = "./base_model"
DATA_PATH = "train_data.jsonl"
OUTPUT_DIR = "./new_ckpt"

# Hyperparameters (User Requested)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_LENGTH = 768  # Reduced from 1024 since we removed the schema

def format_instruction(example):
    """Format data into instruction-tuning style without system prompt"""
    return f"Input: {example['prompt']}\nOutput: {example['completion']}"

def main():
    print(f" Starting Fine-Tuning (V2 - No Schema)")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Data: {DATA_PATH}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   LR: {LEARNING_RATE}")
    print(f"   Max Length: {MAX_LENGTH}")
    
    # 1. Load Data
    print("\n Loading and formatting data...")
    
    data = []
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    ex = json.loads(line.strip())
                    text = format_instruction(ex)
                    data.append({"text": text})
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f" Error: {DATA_PATH} not found. Please combine data first.")
        return

    dataset = Dataset.from_list(data)
    
    # Split 90/10 (Train/Val) - keeping more for training since we have small data
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    print(f"   Train size: {len(dataset['train'])}")
    print(f"   Val size: {len(dataset['test'])}")

    # 2. Load Model & Tokenizer
    print("\n Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        fp16=True,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none"  # Disable wandb/tensorboard for simplicity
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 6. Train
    trainer.train()

    # 7. Save
    trainer.save_model(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
