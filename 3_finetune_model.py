"""
Step 3: Fine-tune Qwen Coder 3B on RTX 4060
============================================

This script will:
1. Load the prepared training data (with schema prefix)
2. Load Qwen2.5-Coder-3B model with 4-bit quantization
3. Apply LoRA for efficient fine-tuning
4. Train with optimal settings for 8GB VRAM
5. Save the fine-tuned model
6. Provide training metrics

Optimized for: NVIDIA RTX 4060 (8GB VRAM)
Expected training time: 40-60 minutes for 900 examples
"""

import os
import json
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import time

def check_gpu():
    """Check GPU availability and memory"""
    print("=" * 70)
    print("GPU Check")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ No GPU detected! Training will be very slow.")
        print("   Make sure you have:")
        print("   1. NVIDIA GPU drivers installed")
        print("   2. CUDA toolkit installed")
        print("   3. PyTorch with CUDA support")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU detected: {gpu_name}")
    print(f"✅ Total VRAM: {gpu_memory:.1f} GB")
    
    if gpu_memory < 7:
        print(f"⚠️  Warning: Only {gpu_memory:.1f}GB VRAM detected")
        print("   This script is optimized for 8GB. Training may fail.")
    
    return True

def load_prepared_data(data_dir="./prepared_data"):
    """Load the prepared training datasets"""
    print("\n" + "=" * 70)
    print("Loading Prepared Data")
    print("=" * 70)
    
    train_path = os.path.join(data_dir, "train_dataset")
    val_path = os.path.join(data_dir, "val_dataset")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"❌ Prepared data not found in {data_dir}")
        print("   Run: python 2_prepare_training_data.py first")
        return None, None
    
    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path)
    
    print(f"✅ Training examples: {len(train_dataset)}")
    print(f"✅ Validation examples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def load_model_and_tokenizer(model_path="./models/qwen-coder-3b"):
    """Load model with 4-bit quantization for 8GB VRAM"""
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Run: python 1_setup_and_download.py first")
        return None, None
    
    print(f"📂 Loading from: {model_path}")
    print("⚙️  Configuration: 4-bit quantization + LoRA")
    print("⏳ This may take 1-2 minutes...\n")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # Double quantization for extra memory savings
        bnb_4bit_quant_type="nf4",  # Normal float 4-bit
        bnb_4bit_compute_dtype=torch.float16  # Compute in fp16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"  # Important for training
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute across GPU
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    print(f"✅ Model loaded in 4-bit mode")
    
    # Get memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"📊 Current GPU memory used: {memory_allocated:.2f} GB")
    
    return model, tokenizer

def setup_lora(model):
    """Setup LoRA for efficient fine-tuning"""
    print("\n" + "=" * 70)
    print("Configuring LoRA")
    print("=" * 70)
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration optimized for code generation
    # UPDATED: Increased rank for better capacity
    lora_config = LoraConfig(
        r=32,  # Rank - INCREASED from 16 for more capacity
        lora_alpha=64,  # Alpha scaling - INCREASED proportionally
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],  # Target all attention and MLP layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ LoRA applied successfully")
    print(f"📊 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"📊 Total parameters: {total_params:,}")
    
    return model

def tokenize_dataset(dataset, tokenizer, max_length=4096):
    """Tokenize the dataset"""
    print("\n" + "=" * 70)
    print("Tokenizing Dataset")
    print("=" * 70)
    
    def tokenize_function(examples):
        # Tokenize with truncation
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding in collator
        )
        # Add labels (same as input_ids for causal LM)
        result["labels"] = result["input_ids"].copy()
        return result
    
    print(f"⚙️  Max sequence length: {max_length} tokens")
    print("⏳ Tokenizing... (this may take a minute)")
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    print(f"✅ Tokenization complete")
    
    # Check token lengths
    lengths = [len(x) for x in tokenized_dataset["input_ids"]]
    avg_length = sum(lengths) / len(lengths)
    max_found = max(lengths)
    
    print(f"📊 Average sequence length: {avg_length:.0f} tokens")
    print(f"📊 Max sequence length: {max_found} tokens")
    
    if max_found >= max_length:
        print(f"⚠️  Some sequences were truncated (exceeded {max_length} tokens)")
    
    return tokenized_dataset

def train_model(model, tokenizer, train_dataset, val_dataset, output_dir="./finetuned_model"):
    """Train the model"""
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Training arguments optimized for RTX 4060 (8GB VRAM)
    # UPDATED: Better settings for JSON generation
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Training regime - INCREASED for better convergence
        num_train_epochs=10,  # 10 epochs instead of 5 (need lower loss)
        
        # Batch size - tuned for 8GB VRAM
        per_device_train_batch_size=1,  # Small batch
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch = 16
        
        # Optimization - INCREASED learning rate for faster convergence
        learning_rate=5e-4,  # Increased from 2e-4 (faster learning)
        weight_decay=0.01,
        warmup_steps=100,  # More warmup for stability
        lr_scheduler_type="cosine",
        
        # Memory optimization
        fp16=True,  # Mixed precision training
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        
        # Logging and saving
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,  # Keep only 3 checkpoints
        
        # Evaluation
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Other
        report_to="none",  # Disable wandb/tensorboard
        seed=42,
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Print training info
    total_steps = len(train_dataset) * training_args.num_train_epochs // (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )
    
    print(f"\n📊 Training Configuration:")
    print(f"   Total examples: {len(train_dataset)}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   Total training steps: ~{total_steps}")
    print(f"   Learning rate: {training_args.learning_rate}")
    print(f"\n⏳ Estimated training time: 40-60 minutes")
    print(f"\n{'='*70}")
    print("TRAINING STARTED")
    print("="*70)
    print("You can monitor progress below. GPU will be at 100% usage.\n")
    
    # Start training
    start_time = time.time()
    
    try:
        train_result = trainer.train()
        
        # Calculate training time
        training_time = time.time() - start_time
        minutes = int(training_time // 60)
        seconds = int(training_time % 60)
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED!")
        print("="*70)
        print(f"✅ Training time: {minutes}m {seconds}s")
        print(f"✅ Final train loss: {train_result.training_loss:.4f}")
        
        # Get final eval metrics
        eval_results = trainer.evaluate()
        print(f"✅ Final eval loss: {eval_results['eval_loss']:.4f}")
        
        return trainer, train_result
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def save_model(trainer, model, tokenizer, output_dir="./finetuned_model"):
    """Save the fine-tuned model"""
    print("\n" + "=" * 70)
    print("Saving Model")
    print("=" * 70)
    
    # Save the model
    final_model_dir = os.path.join(output_dir, "final_model")
    Path(final_model_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving to: {final_model_dir}")
    
    # Save LoRA adapters
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"✅ Model saved successfully!")
    
    # Print file sizes
    model_files = list(Path(final_model_dir).glob("*.safetensors")) + \
                  list(Path(final_model_dir).glob("*.bin"))
    total_size = sum(f.stat().st_size for f in model_files) / 1e6
    
    print(f"📊 LoRA adapter size: {total_size:.1f} MB")
    print(f"\n💡 To use the model, you need:")
    print(f"   1. Base model: ./models/qwen-coder-3b")
    print(f"   2. LoRA adapters: {final_model_dir}")

def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print("QWEN CODER 3B - FINE-TUNING")
    print("=" * 70)
    print("\nFine-tuning on FEA data with schema prefix...\n")
    
    # Check GPU
    if not check_gpu():
        response = input("\nNo GPU detected. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Load prepared data
    train_dataset, val_dataset = load_prepared_data()
    if train_dataset is None:
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    if model is None:
        return
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Tokenize datasets
    train_tokenized = tokenize_dataset(train_dataset, tokenizer)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer)
    
    # Train
    trainer, train_result = train_model(
        model,
        tokenizer,
        train_tokenized,
        val_tokenized
    )
    
    if trainer is None:
        return
    
    # Save
    save_model(trainer, model, tokenizer)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ FINE-TUNING COMPLETE!")
    print("=" * 70)
    print("\n📂 Your fine-tuned model is ready:")
    print("   Location: ./finetuned_model/final_model/")
    print("\nNext steps:")
    print("   1. Test the model: python 4_test_model.py")
    print("   2. Use in production: python 5_inference.py")
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Training interrupted by user")
        print("   Partial checkpoints may be saved in ./finetuned_model/")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
