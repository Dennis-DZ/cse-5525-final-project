"""
Step 5: Production Inference Script
====================================

This script provides a clean inference interface for production use.
Can be imported as a module or run standalone.

Usage:
    # As module
    from 5_inference import FEAGenerator
    generator = FEAGenerator()
    result = generator.generate("Create a steel cylinder...")
    
    # Standalone
    python 5_inference.py
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import time

class FEAGenerator:
    """FEA JSON Generator using fine-tuned Qwen Coder"""
    
    def __init__(
        self,
        base_model_path="./models/qwen-coder-3b",
        lora_path="./finetuned_model/final_model",
        schema_path="comprehensive_fea_schema.json"
    ):
        """
        Initialize the FEA generator
        
        Args:
            base_model_path: Path to base Qwen model
            lora_path: Path to fine-tuned LoRA adapters
            schema_path: Path to FEA JSON schema
        """
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.schema_path = schema_path
        
        self.model = None
        self.tokenizer = None
        self.schema = None
        self.schema_str = None
        
        # Load everything
        self._load()
    
    def _load(self):
        """Load model, tokenizer, and schema"""
        print("🚀 Initializing FEA Generator...")
        
        # Load schema
        if Path(self.schema_path).exists():
            with open(self.schema_path, 'r') as f:
                self.schema = json.load(f)
            
            # Create condensed schema string
            condensed = {
                "type": "object",
                "required": self.schema.get("required", []),
                "properties": {
                    k: {
                        "type": v.get("type", "object"),
                        "description": v.get("description", "")[:100]
                    }
                    for k, v in self.schema.get("properties", {}).items()
                }
            }
            self.schema_str = json.dumps(condensed, indent=2)
            print("✅ Schema loaded")
        else:
            print("⚠️  Schema not found, continuing without validation")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.lora_path,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(
            base_model,
            self.lora_path,
            device_map="auto"
        )
        print("✅ Model loaded")
        print("✅ Ready to generate!\n")
    
    def generate(
        self,
        prompt,
        max_new_tokens=1024,
        temperature=0.1,
        top_p=0.95,
        return_dict=False
    ):
        """
        Generate FEA JSON from natural language prompt
        
        Args:
            prompt: Natural language description
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            return_dict: If True, return dict with metadata
        
        Returns:
            If return_dict=False: JSON string
            If return_dict=True: Dict with 'json', 'valid', 'time', etc.
        """
        start_time = time.time()
        
        # Create system message
        system_message = f"""You are an expert FEA (Finite Element Analysis) engineer assistant. Your task is to convert natural language descriptions of mechanical engineering problems into structured JSON format.

Follow this JSON schema exactly:

{self.schema_str if self.schema_str else "Generate valid JSON with fields: geometry, material, boundary_conditions, loads"}

Instructions:
- Generate valid JSON matching the schema
- All dimensions in millimeters (mm)
- Forces in Newtons (N)
- Pressures in Pascals (Pa)
- Include all required fields: geometry, material, boundary_conditions, loads
- Use exact material type names from the schema
- Use proper geometry type and dimensions"""
        
        # Format prompt
        formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        try:
            assistant_start = full_output.rfind("<|im_start|>assistant")
            if assistant_start != -1:
                response = full_output[assistant_start:].replace("<|im_start|>assistant", "").strip()
            else:
                response = full_output.split(prompt)[-1].strip()
            response = response.replace("<|im_end|>", "").strip()
        except:
            response = full_output
        
        generation_time = time.time() - start_time
        
        # Validate
        is_valid = False
        parsed = None
        try:
            parsed = json.loads(response)
            is_valid = True
        except:
            pass
        
        if not return_dict:
            return response
        
        return {
            'json': response,
            'parsed': parsed,
            'valid': is_valid,
            'time': generation_time,
            'prompt': prompt
        }
    
    def batch_generate(self, prompts, **kwargs):
        """Generate for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, return_dict=True, **kwargs)
            results.append(result)
        return results

def demo():
    """Demo function"""
    print("=" * 70)
    print("FEA GENERATOR - PRODUCTION DEMO")
    print("=" * 70)
    
    # Initialize generator
    generator = FEAGenerator()
    
    # Example prompts
    examples = [
        "Create a steel cylinder 20mm radius, 100mm tall. Fix bottom, apply 200N downward.",
        "Design aluminum box 50x30x20mm. Pin corners, 150N on top.",
        "Model titanium sphere 25mm radius. Symmetry BC, 3 MPa internal pressure.",
    ]
    
    print("\n📋 Running example generations...\n")
    
    for i, prompt in enumerate(examples, 1):
        print(f"{'='*70}")
        print(f"Example {i}/{len(examples)}")
        print(f"{'='*70}")
        print(f"\n📝 Prompt: {prompt}")
        
        result = generator.generate(prompt, return_dict=True)
        
        print(f"\n⏱️  Generation time: {result['time']:.2f}s")
        print(f"✅ Valid JSON: {result['valid']}")
        
        if result['parsed']:
            print(f"\n🤖 Generated JSON:")
            print(json.dumps(result['parsed'], indent=2))
        else:
            print(f"\n⚠️  Raw output:")
            print(result['json'][:500])
        
        print()
    
    print("=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)

def interactive():
    """Interactive mode"""
    print("=" * 70)
    print("FEA GENERATOR - INTERACTIVE MODE")
    print("=" * 70)
    print("\nType 'quit' to exit\n")
    
    generator = FEAGenerator()
    
    while True:
        try:
            prompt = input("📝 Enter FEA prompt: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            result = generator.generate(prompt, return_dict=True)
            
            print(f"\n⏱️  Generation time: {result['time']:.2f}s")
            
            if result['parsed']:
                print(f"\n🤖 Generated JSON:")
                print(json.dumps(result['parsed'], indent=2))
            else:
                print(f"\n⚠️  Output (may not be valid JSON):")
                print(result['json'])
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            demo()
        elif sys.argv[1] == 'interactive':
            interactive()
        else:
            print("Usage:")
            print("  python 5_inference.py demo        # Run demo")
            print("  python 5_inference.py interactive # Interactive mode")
            print("  python 5_inference.py             # Default (interactive)")
    else:
        interactive()

if __name__ == "__main__":
    main()
