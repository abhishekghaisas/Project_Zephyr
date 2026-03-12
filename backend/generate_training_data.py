"""
Complete SeaTac Training Data Generator
Generates 550+ examples using Groq API

Usage:
1. Get Groq API key from: https://console.groq.com/keys
2. Add to .env: GROQ_API_KEY=gsk_...
3. Run: python generate_training_data.py
"""

import json
import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
import time
import re

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# All 11 SeaTac Use Cases
USE_CASES = {
    "1": {
        "name": "Taxi-In Performance by Aircraft Type",
        "purpose": "Identify which aircraft types require additional buffer time",
        "sql": """SELECT 
    f.aircraft_type,
    at.weight_class,
    AVG(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as avg_taxi_in_minutes,
    COUNT(*) as flight_count
FROM flight f
JOIN aircraft_type at ON f.aircraft_type = at.aircraft_type
JOIN flight_event landing ON f.call_sign = landing.call_sign AND landing.event_type = 'Actual_Landing'
JOIN flight_event inblock ON f.call_sign = inblock.call_sign AND inblock.event_type = 'Actual_In_Block'
WHERE f.operation = 'ARRIVAL' AND TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time) BETWEEN 1 AND 120
GROUP BY f.aircraft_type, at.weight_class
ORDER BY avg_taxi_in_minutes DESC"""
    },
    "2": {
        "name": "Taxi-Out Performance by Hour",
        "purpose": "Pinpoint when the surface is most congested",
        "sql": """SELECT 
    HOUR(offblock.event_time) as hour_of_day,
    AVG(TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time)) as avg_taxi_out_minutes,
    COUNT(*) as flight_count
FROM flight f
JOIN flight_event offblock ON f.call_sign = offblock.call_sign AND offblock.event_type = 'Actual_Off_Block'
JOIN flight_event takeoff ON f.call_sign = takeoff.call_sign AND takeoff.event_type = 'Actual_Take_Off'
WHERE f.operation = 'DEPARTURE' AND TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time) BETWEEN 1 AND 120
GROUP BY hour_of_day
ORDER BY hour_of_day"""
    },
    "6": {
        "name": "Weight Class Comparison",
        "purpose": "Assess operational impact of aircraft weight categories",
        "sql": """SELECT 
    at.weight_class,
    AVG(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as avg_taxi_in_minutes,
    COUNT(DISTINCT f.call_sign) as total_flights
FROM aircraft_type at
JOIN flight f ON at.aircraft_type = f.aircraft_type
LEFT JOIN flight_event landing ON f.call_sign = landing.call_sign AND landing.event_type = 'Actual_Landing'
LEFT JOIN flight_event inblock ON f.call_sign = inblock.call_sign AND inblock.event_type = 'Actual_In_Block'
WHERE at.weight_class IN ('L', 'M', 'H')
GROUP BY at.weight_class
ORDER BY FIELD(at.weight_class, 'L', 'M', 'H')"""
    }
}


def generate_variations_batch(use_case_num: str, use_case_info: Dict, num_variations: int = 50) -> List[str]:
    """Generate question variations using Groq"""
    
    prompt = f"""Generate {num_variations} different ways people might ask this airport operations question:

Use Case: {use_case_info['name']}
Purpose: {use_case_info['purpose']}

Generate diverse variations including:
- Formal: "Compare average taxi times by aircraft type"
- Casual: "show taxi times by type"
- Questions: "What are the taxi-in times?"
- Commands: "Display taxi-in performance"
- Natural: "Which planes take longest to taxi?"
- Short: "taxi in by type"

Return EXACTLY {num_variations} variations as a valid JSON array:
["variation 1", "variation 2", ...]

IMPORTANT: Return ONLY the JSON array, nothing else."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            variations = json.loads(json_match.group(0))
            print(f"  ✅ Use Case {use_case_num}: Generated {len(variations)} variations")
            return variations[:num_variations]
        else:
            print(f"  ⚠️  Could not parse JSON")
            return []
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return []


def generate_all_training_data() -> List[Dict]:
    """Generate complete training dataset"""
    
    all_examples = []
    
    print("=" * 70)
    print("🤖 Generating Training Data with Groq")
    print("=" * 70)
    print(f"Target: {len(USE_CASES) * 50} examples\n")
    
    for use_case_num, use_case_info in USE_CASES.items():
        print(f"Use Case {use_case_num}: {use_case_info['name']}")
        
        variations = generate_variations_batch(use_case_num, use_case_info, 50)
        
        for variation in variations:
            all_examples.append({
                "text_input": variation.strip(),
                "output": use_case_info['sql'].strip(),
                "use_case": use_case_num,
                "use_case_name": use_case_info['name'],
                "purpose": use_case_info['purpose']
            })
        
        time.sleep(1)  # Rate limiting
    
    return all_examples


def save_training_data(examples: List[Dict]):
    """Save in multiple formats"""
    
    # Format 1: Standard JSON
    with open('seatac_training_data.json', 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"\n💾 Saved: seatac_training_data.json ({len(examples)} examples)")
    
    # Format 2: Gemini JSONL
    import jsonlines
    gemini_format = []
    for ex in examples:
        gemini_format.append({
            "contents": [
                {"role": "user", "parts": [{"text": f"Generate SQL: {ex['text_input']}"}]},
                {"role": "model", "parts": [{"text": ex['output']}]}
            ]
        })
    
    with jsonlines.open('seatac_gemini_training.jsonl', 'w') as writer:
        writer.write_all(gemini_format)
    print(f"💾 Saved: seatac_gemini_training.jsonl")
    
    # Format 3: Llama/Gemma JSON
    llama_format = []
    for ex in examples:
        llama_format.append({
            "instruction": ex['text_input'],
            "input": "Database: flight, flight_event, aircraft_type",
            "output": ex['output']
        })
    
    with open('seatac_llama_training.json', 'w') as f:
        json.dump(llama_format, f, indent=2)
    print(f"💾 Saved: seatac_llama_training.json")


if __name__ == "__main__":
    print("SeaTac Training Data Generator")
    print("=" * 70)
    
    if not os.getenv('GROQ_API_KEY'):
        print("❌ GROQ_API_KEY not found in .env")
        print("Get your key from: https://console.groq.com/keys")
        exit(1)
    
    training_data = generate_all_training_data()
    
    print(f"\n✅ Generated {len(training_data)} training examples")
    
    save_training_data(training_data)
    
    print("\n✅ Training data generation complete!")
    print("Files created:")
    print("  - seatac_training_data.json")
    print("  - seatac_gemini_training.jsonl")
    print("  - seatac_llama_training.json")