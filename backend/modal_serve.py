"""
Deploy fine-tuned Code Llama as inference server on Modal

COMPLETE VERSION with:
- Fixed dependencies (scipy, numpy<2)
- Improved prompt for simple queries
- No more over-explaining or numbered lists
- Modal 1.0 API compatibility

Deploy: modal deploy modal_serve.py
Test: modal run modal_serve.py::test
"""

import modal
from typing import Dict

app = modal.App("seatac-codellama-serve")

# Reference the same volume
volume = modal.Volume.from_name("seatac-models")

# Inference image with ALL dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",  # Fix NumPy 2.x compatibility
        "scipy",  # Required by bitsandbytes
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
    gpu="T4",
    scaledown_window=900,  # 15 minutes
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=180
)
class CodeLlamaSQL:
    """Code Llama inference class - optimized for speed"""
    
    @modal.enter()
    def load_model(self):
        """Load model on container startup - optimized"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        import os
        import time
        
        start_time = time.time()
        print("🔧 Loading fine-tuned Code Llama...")
        
        hf_token = os.environ.get("HF_TOKEN")
        model_path = "/models/seatac-codellama-final"
        base_model = "codellama/CodeLlama-7b-Instruct-hf"
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print(f"   Available paths in /models: {os.listdir('/models') if os.path.exists('/models') else 'None'}")
            raise FileNotFoundError(f"Fine-tuned model not found at {model_path}")
        
        print(f"✅ Model found at {model_path}")
        
        # Load tokenizer (fast)
        print("⏳ Loading tokenizer...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            token=hf_token,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"   ✅ Tokenizer loaded in {time.time() - t0:.1f}s")
        
        # Load base model in 4-bit
        print("⏳ Loading base model in 4-bit...")
        t0 = time.time()
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"   ✅ Base model loaded in {time.time() - t0:.1f}s")
        
        # Load LoRA adapters
        print("⏳ Loading LoRA adapters...")
        t0 = time.time()
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()
        print(f"   ✅ LoRA adapters loaded in {time.time() - t0:.1f}s")
        
        total_time = time.time() - start_time
        print(f"✅ Model fully loaded in {total_time:.1f}s and ready!")
        
        self.is_ready = True
    
    @modal.method()
    def health_check(self) -> Dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": hasattr(self, 'model') and hasattr(self, 'tokenizer'),
            "ready": getattr(self, 'is_ready', False)
        }
    
    @modal.method()
    def generate_sql(self, query: str, max_tokens: int = 512) -> Dict:
        """Generate SQL from natural language query"""
        import torch
        import time
        
        start_time = time.time()
        
        try:
            # Check if model is loaded
            if not hasattr(self, 'tokenizer') or not hasattr(self, 'model'):
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            # IMPROVED PROMPT - More explicit about output format
            prompt = f"""[INST] You are a SQL expert. Generate ONE SQL query for the given question. Do not provide explanations, examples, or multiple queries.

QUESTION: {query}

DATABASE SCHEMA:
Tables:
- flight (call_sign, aircraft_type, operation, flight_number)
- flight_event (call_sign, event_type, event_time, location, operation)
- aircraft_type (aircraft_type, weight_class, wake_category, wingspan_ft)

Key Relationships:
- Join on call_sign: flight.call_sign = flight_event.call_sign
- Aircraft type: flight.aircraft_type = aircraft_type.aircraft_type

TAXI TIME FORMULAS (DO NOT USE NOW()):
- Taxi-in = TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)
  * Use for ARRIVALS only
  * Join: landing (Actual_Landing) and inblock (Actual_In_Block)
  
- Taxi-out = TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time)
  * Use for DEPARTURES only
  * Join: offblock (Actual_Off_Block) and takeoff (Actual_Take_Off)

EXAMPLE PATTERNS:

Simple count query:
SELECT COUNT(*) as total_flights
FROM flight
WHERE aircraft_type = 'B738'

Taxi-in by aircraft type:
SELECT f.aircraft_type,
       AVG(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as avg_taxi_in_minutes,
       COUNT(*) as flight_count
FROM flight f
JOIN flight_event landing ON f.call_sign = landing.call_sign 
     AND landing.event_type = 'Actual_Landing' 
     AND landing.operation = 'ARRIVAL'
JOIN flight_event inblock ON f.call_sign = inblock.call_sign
     AND inblock.event_type = 'Actual_In_Block' 
     AND inblock.operation = 'ARRIVAL'
WHERE f.operation = 'ARRIVAL'
  AND TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time) BETWEEN 1 AND 120
GROUP BY f.aircraft_type
ORDER BY avg_taxi_in_minutes DESC

CRITICAL RULES:
1. Return ONLY ONE SQL query
2. NO explanations, NO notes, NO numbered lists
3. Start directly with SELECT
4. NEVER use NOW() - always calculate between event times
5. Filter invalid times: BETWEEN 1 AND 120
6. Match operation field in JOINs

Generate SQL for: {query}
[/INST]

SELECT"""
            
            print(f"📝 Processing query: {query[:50]}...")
            
            # Tokenize
            t0 = time.time()
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            print(f"   Tokenization: {time.time() - t0:.2f}s")
            
            # Generate
            print(f"🤖 Generating SQL...")
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            print(f"   Generation: {time.time() - t0:.2f}s")
            
            # Decode
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract SQL - look for the part after [/INST]
            if "[/INST]" in generated:
                sql = generated.split("[/INST]")[-1].strip()
            else:
                sql = generated.strip()
            
            # The prompt ends with "SELECT" so prepend it if model didn't include it
            if not sql.upper().startswith('SELECT'):
                sql = 'SELECT' + sql
            
            total_time = time.time() - start_time
            print(f"✅ Generated SQL in {total_time:.2f}s total")
            
            return {
                "query": query,
                "sql": sql,
                "model": "seatac-codellama-7b-finetuned",
                "generation_time": round(total_time, 2),
                "error": None
            }
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"❌ Error: {error_msg}")
            traceback.print_exc()
            
            return {
                "query": query,
                "sql": None,
                "model": "seatac-codellama-7b-finetuned",
                "generation_time": round(time.time() - start_time, 2),
                "error": error_msg
            }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/models": volume},
    gpu="T4",
    timeout=300
)
def test():
    """Test the fine-tuned model"""
    
    print("\n" + "=" * 80)
    print("🧪 Testing Fine-Tuned Code Llama")
    print("=" * 80)
    
    # Initialize model
    model = CodeLlamaSQL()
    
    # Health check first
    print("\n🏥 Running health check...")
    health = model.health_check.remote()
    print(f"   Status: {health}")
    
    if not health.get('ready'):
        print("❌ Model not ready!")
        return
    
    # Test queries - mix of simple and complex
    test_queries = [
        "How many A321 flights were there?",
        "Show me taxi-in times by aircraft type",
        "What are the busiest hours?",
        "Compare taxi times by weight class"
    ]
    
    print("\n📝 Running test queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_queries)}")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"{'-'*80}")
        
        result = model.generate_sql.remote(query)
        
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
        else:
            print(f"⏱️  Generation time: {result.get('generation_time', 'N/A')}s")
            print(f"SQL Generated:")
            sql = result['sql']
            print(sql[:300] + "..." if len(sql) > 300 else sql)
    
    print("\n" + "=" * 80)
    print("✅ Testing Complete!")
    print("=" * 80)


@app.function(image=modal.Image.debian_slim().pip_install("fastapi", "pydantic"))
@modal.web_endpoint(method="GET")
def health() -> Dict:
    """Health check endpoint - doesn't load the model"""
    return {
        "status": "online",
        "service": "seatac-codellama-serve",
        "message": "Service is running. Use POST /generate_sql_api to generate SQL."
    }


@app.function(image=modal.Image.debian_slim().pip_install("fastapi", "pydantic"))
@modal.web_endpoint(method="POST")
def generate_sql_api(request: Dict) -> Dict:
    """
    HTTP endpoint for SQL generation
    
    Usage:
        curl -X POST https://your-url/generate_sql_api \
             -H "Content-Type: application/json" \
             -d '{"query": "Show taxi-in times"}'
    """
    from datetime import datetime
    import time
    
    query = request.get("query", "")
    
    if not query:
        return {
            "error": "No query provided",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "error"
        }
    
    try:
        start_time = time.time()
        print(f"📥 Received query: {query}")
        
        # Call the model
        model = CodeLlamaSQL()
        result = model.generate_sql.remote(query)
        
        total_time = time.time() - start_time
        print(f"⏱️  Total request time: {total_time:.2f}s")
        
        return {
            **result,
            "timestamp": datetime.utcnow().isoformat(),
            "total_request_time": round(total_time, 2),
            "status": "success" if not result.get('error') else "error"
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return {
            "error": str(e),
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "error"
        }