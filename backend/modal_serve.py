"""
Deploy fine-tuned Code Llama as inference server on Modal

Deploy: modal deploy modal_serve_codellama.py
Test: modal run modal_serve_codellama.py::test
"""

import modal
from typing import Dict

app = modal.App("seatac-codellama-serve")

# Reference the same volume
volume = modal.Volume.from_name("seatac-models")

# Inference image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "peft==0.7.1",
        "bitsandbytes==0.41.3",
        "sentencepiece",
        "fastapi",
        "pydantic"
    )
)


@app.cls(
    image=image,
    gpu="T4",  # Cheap GPU for inference ($0.60/hr vs A10G $1.10/hr)
    container_idle_timeout=300,  # Keep warm for 5 minutes
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class CodeLlamaSQL:
    """Code Llama inference class"""
    
    def __enter__(self):
        """Load model on container startup"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        import os
        
        print("🔧 Loading fine-tuned Code Llama...")
        
        hf_token = os.environ.get("HF_TOKEN")
        model_path = "/models/seatac-codellama-final"
        
        # Load base model
        base_model = "codellama/CodeLlama-7b-Instruct-hf"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            token=hf_token,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model in 8-bit
        print("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapters
        print("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()
        
        print("✅ Model loaded and ready!")
    
    @modal.method()
    def generate_sql(self, query: str, max_tokens: int = 512) -> Dict:
        """Generate SQL from natural language query"""
        import torch
        
        # Format prompt (same as training)
        prompt = f"""[INST] Generate SQL for this airport operations query:

Query: {query}

Database Schema:
- flight (call_sign, aircraft_type, operation, flight_number)
- flight_event (call_sign, event_type, event_time, location, operation)
- aircraft_type (aircraft_type, weight_class, wake_category, wingspan_ft)

Generate only the SQL query, no explanation.
[/INST]

"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,  # Low temperature for deterministic SQL
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SQL (remove prompt)
        sql = generated.split("[/INST]")[-1].strip()
        
        return {
            "query": query,
            "sql": sql,
            "model": "seatac-codellama-7b-finetuned"
        }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def test():
    """Test the fine-tuned model"""
    
    print("\n" + "=" * 80)
    print("🧪 Testing Fine-Tuned Code Llama")
    print("=" * 80)
    
    # Initialize model
    model = CodeLlamaSQL()
    
    # Test queries
    test_queries = [
        "Show me taxi-in times by aircraft type",
        "What are the busiest hours?",
        "Compare taxi times by weight class",
        "Display runway utilization",
        "Which aircraft have the longest delays?"
    ]
    
    print("\n📝 Running test queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_queries)}")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"{'-'*80}")
        
        result = model.generate_sql.remote(query)
        
        print(f"SQL Generated:")
        print(result['sql'][:200] + "..." if len(result['sql']) > 200 else result['sql'])
        
    print("\n" + "=" * 80)
    print("✅ Testing Complete!")
    print("=" * 80)


@app.function(
    image=modal.Image.debian_slim().pip_install("fastapi", "pydantic"),
    allow_concurrent_inputs=10
)
@modal.web_endpoint(method="POST")
def generate_sql_api(request: Dict) -> Dict:
    """
    HTTP endpoint for SQL generation
    
    Usage:
        curl -X POST https://your-modal-url/generate_sql_api \\
             -H "Content-Type: application/json" \\
             -d '{"query": "Show taxi-in times"}'
    """
    from datetime import datetime
    
    query = request.get("query", "")
    
    if not query:
        return {
            "error": "No query provided",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Call the model
    model = CodeLlamaSQL()
    result = model.generate_sql.remote(query)
    
    return {
        **result,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "success"
    }