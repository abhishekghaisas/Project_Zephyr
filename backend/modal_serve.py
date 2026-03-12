"""
Deploy Fine-Tuned Gemma on Modal for SeaTac Operations
Run: modal deploy modal_serve.py

This creates a serverless inference API endpoint
"""

import modal
from typing import Dict

app = modal.App("seatac-gemma-api")

# Load the fine-tuned model from volume
volume = modal.Volume.from_name("seatac-models")

# Define image with inference dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.45.0",
        "accelerate==0.35.0",
        "peft==0.7.0",
        "sentencepiece",
        "protobuf"
    )
)

# Model loading (cached in container)
@app.cls(
    image=image,
    gpu="T4",  # $0.60/hour - cost-effective
    volumes={"/models": volume},
    container_idle_timeout=300,  # Keep alive 5 minutes
    allow_concurrent_inputs=10
)
class GemmaInference:
    """Fine-tuned Gemma inference server"""
    
    def __enter__(self):
        """Load model when container starts"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("🔧 Loading fine-tuned model...")
        
        model_path = "/models/seatac-gemma-finetuned"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ Model loaded and ready!")
    
    @modal.method()
    def generate_sql(self, query: str) -> str:
        """Generate SQL from natural language query"""
        
        prompt = f"""<bos><start_of_turn>user
Generate SQL for SeaTac airport operations:

{query}<end_of_turn>
<start_of_turn>model
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SQL (after model turn)
        sql = generated_text.split("<start_of_turn>model")[-1].strip()
        sql = sql.replace("<end_of_turn>", "").replace("<eos>", "").strip()
        
        return sql
    
    @modal.method()
    def analyze_data(self, data: str, query: str) -> str:
        """Analyze query results and provide insights"""
        
        prompt = f"""<bos><start_of_turn>user
Analyze this SeaTac airport operations data:

Original Query: {query}

Data:
{data[:1000]}

Provide 2-3 sentences with insights and recommendations.<end_of_turn>
<start_of_turn>model
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.3)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        analysis = response.split("<start_of_turn>model")[-1].strip()
        analysis = analysis.replace("<end_of_turn>", "").replace("<eos>", "").strip()
        
        return analysis


# Public API endpoints
@app.function(image=modal.Image.debian_slim().pip_install("fastapi", "pydantic"))
@modal.web_endpoint(method="POST")
async def generate_sql_endpoint(query: Dict):
    """Public endpoint for SQL generation"""
    
    gemma = GemmaInference()
    user_query = query.get("query", "")
    
    if not user_query:
        return {"error": "No query provided"}
    
    try:
        sql = gemma.generate_sql.remote(user_query)
        
        return {
            "success": True,
            "query": user_query,
            "sql": sql,
            "model": "gemma-2-2b-seatac-finetuned"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.function(image=modal.Image.debian_slim().pip_install("fastapi"))
@modal.web_endpoint(method="POST")  
async def analyze_endpoint(data: Dict):
    """Public endpoint for data analysis"""
    
    gemma = GemmaInference()
    
    query_text = data.get("query", "")
    data_text = data.get("data", "")
    
    try:
        analysis = gemma.analyze_data.remote(data_text, query_text)
        return {"success": True, "analysis": analysis}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Test function
@app.local_entrypoint()
def test():
    """Test the deployed model"""
    
    gemma = GemmaInference()
    
    test_query = "Show me taxi-in times by aircraft type"
    print(f"\n🧪 Test Query: {test_query}")
    
    sql = gemma.generate_sql.remote(test_query)
    print(f"\n📊 Generated SQL:")
    print(sql)
    
    test_data = '[{"aircraft_type": "B737", "avg_time": 12.5}]'
    analysis = gemma.analyze_data.remote(test_data, test_query)
    print(f"\n💡 Analysis:")
    print(analysis)
    
    print("\n✅ Model is working!")