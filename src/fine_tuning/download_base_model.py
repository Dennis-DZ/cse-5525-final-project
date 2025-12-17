
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-3B"
save_directory = "./base_model"

print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Saving to {save_directory}...")
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
print("Done!")
