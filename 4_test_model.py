"""
Step 4: Test Fine-tuned Model
==============================

This script will:
1. Load your fine-tuned model
2. Run test prompts
3. Validate JSON outputs
4. Calculate accuracy metrics
5. Show example predictions

Use this to evaluate model quality!
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import jsonschema
from pathlib import Path

def load_finetuned_model(
    base_model_path="./models/qwen-coder-3b",
    lora_path="./finetuned_model/final_model"
):
    """Load the fine-tuned model with LoRA adapters"""
    print("=" * 70)
    print("Loading Fine-tuned Model")
    print("=" * 70)
    
    if not Path(base_model_path).exists():
        print(f"❌ Base model not found: {base_model_path}")
        return None, None
    
    if not Path(lora_path).exists():
        print(f"❌ LoRA adapters not found: {lora_path}")
        return None, None
    
    print(f"📂 Base model: {base_model_path}")
    print(f"📂 LoRA adapters: {lora_path}")
    print("⏳ Loading... (this may take a minute)\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path,
        trust_remote_code=True
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        device_map="auto"
    )
    
    print("✅ Model loaded successfully!\n")
    
    return model, tokenizer

def load_schema(schema_path="comprehensive_fea_schema.json"):
    """Load the FEA schema for validation"""
    if Path(schema_path).exists():
        with open(schema_path, 'r') as f:
            return json.load(f)
    return None

def generate_response(model, tokenizer, user_prompt, schema_str=None, max_new_tokens=1024):
    """Generate JSON response from user prompt"""
    
    # Create system message with schema
    if schema_str:
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
    else:
        system_message = "You are an expert FEA engineer. Convert natural language to structured JSON."
    
    # Format prompt
    prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temp for more deterministic output
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    try:
        # Find the assistant's response after the last <|im_start|>assistant
        assistant_start = full_output.rfind("<|im_start|>assistant")
        if assistant_start != -1:
            response = full_output[assistant_start:].replace("<|im_start|>assistant", "").strip()
        else:
            # Fallback: take everything after the user prompt
            response = full_output.split(user_prompt)[-1].strip()
        
        # Remove any remaining special tokens
        response = response.replace("<|im_end|>", "").strip()
        
        return response
    except:
        return full_output

def validate_json(json_str, schema=None):
    """Validate JSON string"""
    try:
        # Try to parse JSON
        parsed = json.loads(json_str)
        
        # Validate against schema if provided
        if schema:
            jsonschema.validate(instance=parsed, schema=schema)
            return True, parsed, "✅ Valid JSON, schema compliant"
        
        return True, parsed, "✅ Valid JSON (schema not checked)"
        
    except json.JSONDecodeError as e:
        return False, None, f"❌ Invalid JSON: {e}"
    except jsonschema.ValidationError as e:
        return False, None, f"⚠️  Valid JSON but schema error: {e.message}"

def run_test_prompts(model, tokenizer, schema=None):
    """Run a set of test prompts"""
    print("=" * 70)
    print("Running Test Prompts")
    print("=" * 70)
    
    test_prompts = [
        "Create a steel cylinder with 20mm radius and 100mm height. Fix the base and apply 200N downward force at the top center.",
        
        "Design an aluminum box 50mm x 30mm x 20mm. Pin the bottom left corner and apply 150N rightward at the top right corner.",
        
        "Model a hollow titanium sphere with 25mm outer radius and 2mm wall thickness. Apply symmetry BC at bottom and 3 MPa internal pressure.",
        
        "Generate a concrete cone with 30mm base radius, pointed tip, and 80mm height. Fix the base and apply 500N compressive load at apex.",
        
        "Build a CFRP hollow cylinder: outer diameter 40mm, wall thickness 3mm, length 120mm. Pin one end, apply 5 MPa internal pressure.",
    ]
    
    # Get condensed schema string
    schema_str = None
    if schema:
        condensed = {
            "type": "object",
            "required": schema.get("required", []),
            "properties": {k: {"type": v.get("type", "object")} for k, v in schema.get("properties", {}).items()}
        }
        schema_str = json.dumps(condensed, indent=2)
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"{'='*70}")
        print(f"\n📝 Prompt:")
        print(f"   {prompt}\n")
        
        # Generate
        print("⏳ Generating...")
        response = generate_response(model, tokenizer, prompt, schema_str)
        
        # Validate
        is_valid, parsed, message = validate_json(response, schema)
        
        print(f"\n🤖 Generated Response:")
        if parsed:
            print(json.dumps(parsed, indent=2))
        else:
            print(response[:500])  # Show first 500 chars if not valid JSON
        
        print(f"\n{message}")
        
        results.append({
            'prompt': prompt,
            'response': response,
            'valid': is_valid,
            'parsed': parsed
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")
    
    valid_count = sum(1 for r in results if r['valid'])
    print(f"✅ Valid JSON: {valid_count}/{len(results)} ({valid_count/len(results)*100:.1f}%)")
    
    return results

def interactive_mode(model, tokenizer, schema=None):
    """Interactive testing mode"""
    print("\n" + "=" * 70)
    print("Interactive Mode")
    print("=" * 70)
    print("\nType your FEA prompts and see the JSON output!")
    print("Commands: 'quit' to exit, 'test' to run test suite\n")
    
    schema_str = None
    if schema:
        condensed = {
            "type": "object",
            "required": schema.get("required", []),
        }
        schema_str = json.dumps(condensed, indent=2)
    
    while True:
        try:
            prompt = input("\n📝 Your prompt: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if prompt.lower() == 'test':
                run_test_prompts(model, tokenizer, schema)
                continue
            
            # Generate
            print("\n⏳ Generating...")
            response = generate_response(model, tokenizer, prompt, schema_str)
            
            # Validate
            is_valid, parsed, message = validate_json(response, schema)
            
            print(f"\n🤖 Generated JSON:")
            if parsed:
                print(json.dumps(parsed, indent=2))
            else:
                print(response)
            
            print(f"\n{message}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print("QWEN CODER 3B - MODEL TESTING")
    print("=" * 70)
    
    # Load model
    model, tokenizer = load_finetuned_model()
    if model is None:
        return
    
    # Load schema
    schema = load_schema()
    if schema:
        print("✅ Schema loaded for validation\n")
    else:
        print("⚠️  Schema not found, validation will be limited\n")
    
    # Menu
    print("Choose mode:")
    print("1. Run test suite (5 predefined prompts)")
    print("2. Interactive mode (enter your own prompts)")
    print("3. Both")
    
    choice = input("\nYour choice (1/2/3): ").strip()
    
    if choice == '1' or choice == '3':
        run_test_prompts(model, tokenizer, schema)
    
    if choice == '2' or choice == '3':
        interactive_mode(model, tokenizer, schema)
    
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
