"""
Fixed Inference Script with Proper Generation
==============================================

This uses the correct generation parameters for Qwen models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import json

def generate_fea_json(prompt_text):
    """Generate FEA JSON with corrected parameters"""
    
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "./finetuned_model/final_model",
        trust_remote_code=True,
        padding_side='left'
    )
    
    # Make sure we have pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "./models/qwen-coder-3b",
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        "./finetuned_model/final_model"
    )
    
    print("✅ Model loaded\n")
    
    # Create prompt
    full_prompt = f"""<|im_start|>system
You are an expert FEA engineer. Convert the user's description into valid JSON following this structure:
{{
  "geometry": {{"type": "...", "dimensions": {{...}}}},
  "material": {{"type": "..."}},
  "boundary_conditions": [...],
  "loads": [...]
}}
Generate ONLY valid JSON, nothing else.<|im_end|>
<|im_start|>user
{prompt_text}<|im_end|>
<|im_start|>assistant
"""
    
    print(f"Prompt: {prompt_text}\n")
    print("⏳ Generating...\n")
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with corrected parameters
    with torch.no_grad():
        generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.3,  # Low but not too low
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    print("=" * 70)
    print("RAW OUTPUT:")
    print("=" * 70)
    print(full_response)
    print("=" * 70)
    
    # Extract assistant response
    try:
        # Method 1: Split by assistant marker
        if "<|im_start|>assistant" in full_response:
            parts = full_response.split("<|im_start|>assistant")
            assistant_response = parts[-1]
            
            # Remove end marker if present
            if "<|im_end|>" in assistant_response:
                assistant_response = assistant_response.split("<|im_end|>")[0]
            
            assistant_response = assistant_response.strip()
        else:
            # Method 2: Take everything after the prompt
            assistant_response = full_response.replace(full_prompt, "").strip()
        
        print("\n" + "=" * 70)
        print("EXTRACTED RESPONSE:")
        print("=" * 70)
        print(assistant_response)
        print("=" * 70)
        
        # Try to parse as JSON
        try:
            # Clean up common issues
            cleaned = assistant_response
            
            # Remove any leading/trailing text
            if "{" in cleaned:
                cleaned = cleaned[cleaned.index("{"):]
            if "}" in cleaned:
                cleaned = cleaned[:cleaned.rindex("}") + 1]
            
            parsed = json.loads(cleaned)
            
            print("\n✅ VALID JSON!")
            print("\n" + "=" * 70)
            print("PARSED JSON:")
            print("=" * 70)
            print(json.dumps(parsed, indent=2))
            print("=" * 70)
            
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON Parse Error: {e}")
            print(f"\n❌ The model generated text but not valid JSON")
            print(f"\nTrying to find JSON in output...")
            
            # Last resort: find first { and last }
            if "{" in assistant_response and "}" in assistant_response:
                start = assistant_response.index("{")
                end = assistant_response.rindex("}") + 1
                json_part = assistant_response[start:end]
                
                try:
                    parsed = json.loads(json_part)
                    print("✅ Found valid JSON!")
                    print(json.dumps(parsed, indent=2))
                    return parsed
                except:
                    print("❌ Still invalid")
            
            return None
            
    except Exception as e:
        print(f"\n❌ Error extracting response: {e}")
        return None

if __name__ == "__main__":
    # Test prompts
    test_prompts = [
        "Create a steel cylinder 20mm radius, 100mm tall. Fix base, apply 200N downward.",
        "Design aluminum box 50x30x20mm. Pin corner, 150N rightward.",
        "Model titanium sphere 25mm radius. Symmetry BC, 3 MPa pressure.",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print("\n" + "=" * 70)
        print(f"TEST {i}/{len(test_prompts)}")
        print("=" * 70)
        
        result = generate_fea_json(prompt)
        
        if result:
            print("\n✅ Success!")
        else:
            print("\n❌ Failed")
        
        if i < len(test_prompts):
            input("\nPress Enter for next test...")
