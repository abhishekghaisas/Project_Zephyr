"""
Complete SeaTac Training Data Generator - All 11 Use Cases
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

# All 11 SeaTac Use Cases (extracted from app.py)
USE_CASES = {
    "1": {
        "name": "Taxi-In Performance by Aircraft Type",
        "purpose": "Identify which aircraft types require additional buffer time for arrivals",
        "sql": """SELECT f.aircraft_type,
       AVG(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as avg_taxi_in_minutes,
       MIN(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as min_taxi_in,
       MAX(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as max_taxi_in,
       COUNT(*) as flight_count
FROM flight f
JOIN flight_event landing ON f.call_sign = landing.call_sign 
     AND landing.event_type = 'Actual_Landing' AND landing.operation = 'ARRIVAL'
JOIN flight_event inblock ON f.call_sign = inblock.call_sign
     AND inblock.event_type = 'Actual_In_Block' AND inblock.operation = 'ARRIVAL'
WHERE f.operation = 'ARRIVAL' AND f.aircraft_type IS NOT NULL
  AND TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time) BETWEEN 0 AND 120
GROUP BY f.aircraft_type
HAVING flight_count >= 2
ORDER BY avg_taxi_in_minutes DESC
LIMIT 15"""
    },
    "2": {
        "name": "Taxi-Out Performance by Hour",
        "purpose": "Pinpoint when the surface is most congested for departures",
        "sql": """SELECT HOUR(offblock.event_time) as hour_of_day,
       COUNT(*) as flight_count,
       AVG(TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time)) as avg_taxi_out_minutes
FROM flight f
JOIN flight_event offblock ON f.call_sign = offblock.call_sign 
     AND offblock.event_type = 'Actual_Off_Block' AND offblock.operation = 'DEPARTURE'
JOIN flight_event takeoff ON f.call_sign = takeoff.call_sign
     AND takeoff.event_type = 'Actual_Take_Off' AND takeoff.operation = 'DEPARTURE'
WHERE f.operation = 'DEPARTURE'
  AND TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time) BETWEEN 0 AND 120
GROUP BY HOUR(offblock.event_time)
ORDER BY hour_of_day
LIMIT 24"""
    },
    "3": {
        "name": "Movement Area Occupancy",
        "purpose": "Monitor aircraft in movement area to track peak-hour ground saturation",
        "sql": """SELECT HOUR(event_time) as hour,
       COUNT(DISTINCT call_sign) as aircraft_count,
       COUNT(*) as total_events
FROM flight_event
WHERE event_type IN ('Movement_Area_Entrance', 'Movement_Area_Exit', 'Runway_Entrance', 'Runway_Exit')
  AND event_time IS NOT NULL
GROUP BY HOUR(event_time)
ORDER BY aircraft_count DESC
LIMIT 24"""
    },
    "4": {
        "name": "Runway Occupancy Time",
        "purpose": "Calculate runway occupancy times to assess throughput capacity",
        "sql": """SELECT 
    COALESCE(takeoff.location, landing.location, 'Runway') as runway,
    AVG(CASE 
        WHEN takeoff.event_time IS NOT NULL THEN 1.5
        WHEN landing.event_time IS NOT NULL THEN 1.2
        ELSE 1.0 
    END) as avg_occupancy_minutes,
    COUNT(*) as total_operations
FROM flight f
LEFT JOIN flight_event takeoff ON f.call_sign = takeoff.call_sign 
    AND takeoff.event_type = 'Actual_Take_Off'
LEFT JOIN flight_event landing ON f.call_sign = landing.call_sign
    AND landing.event_type = 'Actual_Landing'
WHERE (takeoff.event_time IS NOT NULL OR landing.event_time IS NOT NULL)
GROUP BY runway
HAVING total_operations >= 3
ORDER BY total_operations DESC
LIMIT 10"""
    },
    "5": {
        "name": "Wheels-Up Delay Tracker",
        "purpose": "Identify delay patterns by comparing scheduled vs actual takeoff times",
        "sql": """SELECT f.flight_number,
       f.call_sign,
       f.aircraft_type,
       TIMESTAMPDIFF(MINUTE, scheduled.event_time, actual.event_time) as delay_minutes
FROM flight f
JOIN flight_event scheduled ON f.call_sign = scheduled.call_sign 
     AND scheduled.event_type = 'Scheduled_Take_Off'
JOIN flight_event actual ON f.call_sign = actual.call_sign
     AND actual.event_type = 'Actual_Take_Off'
WHERE ABS(TIMESTAMPDIFF(MINUTE, scheduled.event_time, actual.event_time)) < 300
ORDER BY delay_minutes DESC
LIMIT 50"""
    },
    "6": {
        "name": "Weight Class Comparison",
        "purpose": "Assess operational impact of aircraft weight categories (Light, Medium, Heavy)",
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
    },
    "7": {
        "name": "Taxiway Utilization",
        "purpose": "Evaluate traffic flow patterns to see which taxiway segments are most used",
        "sql": """SELECT location as taxiway,
       COUNT(*) as usage_count,
       event_type
FROM flight_event
WHERE location LIKE 'TaxiwaySegment%'
  AND event_type IN ('Movement_Area_Entrance', 'Runway_Entrance', 'Runway_Exit')
  AND location IS NOT NULL
GROUP BY location, event_type
ORDER BY usage_count DESC
LIMIT 20"""
    },
    "8": {
        "name": "Landing to In-Block Duration",
        "purpose": "Analyze landing to in-block duration for arrival efficiency",
        "sql": """SELECT f.aircraft_type,
       AVG(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as avg_duration_minutes,
       COUNT(*) as flight_count
FROM flight f
JOIN flight_event landing ON f.call_sign = landing.call_sign 
     AND landing.event_type = 'Actual_Landing' AND landing.operation = 'ARRIVAL'
JOIN flight_event inblock ON f.call_sign = inblock.call_sign
     AND inblock.event_type = 'Actual_In_Block' AND inblock.operation = 'ARRIVAL'
WHERE f.operation = 'ARRIVAL' AND f.aircraft_type IS NOT NULL
  AND TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time) BETWEEN 0 AND 120
GROUP BY f.aircraft_type
HAVING flight_count >= 2
ORDER BY avg_duration_minutes DESC
LIMIT 15"""
    },
    "9": {
        "name": "Runway Utilization Rate",
        "purpose": "Show runway utilization rate with breakdown of takeoffs and landings",
        "sql": """SELECT location as runway,
       SUM(CASE WHEN event_type = 'Actual_Take_Off' THEN 1 ELSE 0 END) as takeoffs,
       SUM(CASE WHEN event_type = 'Actual_Landing' THEN 1 ELSE 0 END) as landings,
       COUNT(*) as total_operations
FROM flight_event
WHERE event_type IN ('Actual_Take_Off', 'Actual_Landing')
  AND location IS NOT NULL
  AND location != ''
GROUP BY location
ORDER BY total_operations DESC
LIMIT 10"""
    },
    "10": {
        "name": "Peak Hour Prediction",
        "purpose": "Predict peak hours based on historical arrival and departure patterns",
        "sql": """SELECT 
    HOUR(event_time) as hour,
    SUM(CASE WHEN event_type = 'Actual_Take_Off' THEN 1 ELSE 0 END) as departures,
    SUM(CASE WHEN event_type = 'Actual_Landing' THEN 1 ELSE 0 END) as arrivals,
    COUNT(*) as total_operations
FROM flight_event
WHERE event_type IN ('Actual_Take_Off', 'Actual_Landing')
  AND event_time IS NOT NULL
GROUP BY HOUR(event_time)
ORDER BY total_operations DESC
LIMIT 24"""
    },
    "11": {
        "name": "Taxi Time Breakdown",
        "purpose": "Compare taxi-in vs taxi-out times by aircraft type for operational planning",
        "sql": """SELECT 
    f.aircraft_type,
    AVG(CASE WHEN f.operation = 'ARRIVAL' THEN TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time) END) as avg_taxi_in,
    AVG(CASE WHEN f.operation = 'DEPARTURE' THEN TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time) END) as avg_taxi_out,
    COUNT(*) as total_flights
FROM flight f
LEFT JOIN flight_event landing ON f.call_sign = landing.call_sign AND landing.event_type = 'Actual_Landing'
LEFT JOIN flight_event inblock ON f.call_sign = inblock.call_sign AND inblock.event_type = 'Actual_In_Block'
LEFT JOIN flight_event offblock ON f.call_sign = offblock.call_sign AND offblock.event_type = 'Actual_Off_Block'
LEFT JOIN flight_event takeoff ON f.call_sign = takeoff.call_sign AND takeoff.event_type = 'Actual_Take_Off'
WHERE f.aircraft_type IS NOT NULL
GROUP BY f.aircraft_type
HAVING total_flights >= 5
ORDER BY total_flights DESC
LIMIT 15"""
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
- Misspellings: "taxi-in", "taxi in", "taxiin"
- Different word orders: "by aircraft type show taxi times"

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
        
        # Extract JSON array from response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            variations = json.loads(json_match.group(0))
            print(f"  ✅ Use Case {use_case_num}: Generated {len(variations)} variations")
            return variations[:num_variations]
        else:
            print(f"  ⚠️  Could not parse JSON for Use Case {use_case_num}")
            return []
            
    except Exception as e:
        print(f"  ❌ Error for Use Case {use_case_num}: {e}")
        return []


def generate_all_training_data() -> List[Dict]:
    """Generate complete training dataset for all 11 use cases"""
    
    all_examples = []
    
    print("=" * 70)
    print("🤖 Generating Training Data with Groq - All 11 Use Cases")
    print("=" * 70)
    print(f"Target: {len(USE_CASES) * 50} examples ({len(USE_CASES)} use cases × 50 variations)\n")
    
    for use_case_num, use_case_info in USE_CASES.items():
        print(f"\nUse Case {use_case_num}: {use_case_info['name']}")
        print(f"Purpose: {use_case_info['purpose']}")
        
        variations = generate_variations_batch(use_case_num, use_case_info, 50)
        
        for variation in variations:
            all_examples.append({
                "text_input": variation.strip(),
                "output": use_case_info['sql'].strip(),
                "use_case": use_case_num,
                "use_case_name": use_case_info['name'],
                "purpose": use_case_info['purpose']
            })
        
        time.sleep(1)  # Rate limiting for Groq API
    
    return all_examples


def save_training_data(examples: List[Dict]):
    """Save in multiple formats for different fine-tuning platforms"""
    
    print("\n" + "=" * 70)
    print("💾 Saving Training Data")
    print("=" * 70)
    
    # Format 1: Standard JSON
    with open('seatac_training_data.json', 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"✅ Saved: seatac_training_data.json ({len(examples)} examples)")
    
    # Format 2: Gemini JSONL
    try:
        import jsonlines
        gemini_format = []
        for ex in examples:
            gemini_format.append({
                "contents": [
                    {"role": "user", "parts": [{"text": f"Generate SQL for: {ex['text_input']}"}]},
                    {"role": "model", "parts": [{"text": ex['output']}]}
                ]
            })
        
        with jsonlines.open('seatac_gemini_training.jsonl', 'w') as writer:
            writer.write_all(gemini_format)
        print(f"✅ Saved: seatac_gemini_training.jsonl")
    except ImportError:
        print("⚠️  jsonlines not installed, skipping Gemini format")
        print("   Install with: pip install jsonlines")
    
    # Format 3: Llama/Gemma JSON
    llama_format = []
    for ex in examples:
        llama_format.append({
            "instruction": ex['text_input'],
            "input": "Database Schema: flight, flight_event, aircraft_type",
            "output": ex['output']
        })
    
    with open('seatac_llama_training.json', 'w') as f:
        json.dump(llama_format, f, indent=2)
    print(f"✅ Saved: seatac_llama_training.json")
    
    # Format 4: OpenAI Fine-tuning JSONL
    openai_format = []
    for ex in examples:
        openai_format.append({
            "messages": [
                {"role": "system", "content": "You are a SQL expert for airport operations. Convert natural language queries to SQL."},
                {"role": "user", "content": ex['text_input']},
                {"role": "assistant", "content": ex['output']}
            ]
        })
    
    with open('seatac_openai_training.jsonl', 'w') as f:
        for item in openai_format:
            f.write(json.dumps(item) + '\n')
    print(f"✅ Saved: seatac_openai_training.jsonl")
    
    # Statistics by use case
    print("\n" + "=" * 70)
    print("📊 Statistics by Use Case")
    print("=" * 70)
    
    use_case_counts = {}
    for ex in examples:
        uc = ex['use_case']
        use_case_counts[uc] = use_case_counts.get(uc, 0) + 1
    
    for uc_num in sorted(use_case_counts.keys(), key=lambda x: int(x)):
        uc_info = USE_CASES[uc_num]
        count = use_case_counts[uc_num]
        print(f"Use Case {uc_num}: {count} examples - {uc_info['name']}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("✈️  SeaTac Training Data Generator - All 11 Use Cases")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv('GROQ_API_KEY'):
        print("\n❌ GROQ_API_KEY not found in .env")
        print("\n📝 To get your key:")
        print("   1. Go to: https://console.groq.com/keys")
        print("   2. Sign up (free tier available)")
        print("   3. Create API key")
        print("   4. Add to .env: GROQ_API_KEY=gsk_your_key_here")
        exit(1)
    
    print(f"\n✅ Groq API key found")
    print(f"📋 Generating variations for {len(USE_CASES)} use cases\n")
    
    # Generate training data
    training_data = generate_all_training_data()
    
    print(f"\n✅ Generated {len(training_data)} training examples")
    
    # Save in multiple formats
    save_training_data(training_data)
    
    print("\n" + "=" * 70)
    print("✅ Training Data Generation Complete!")
    print("=" * 70)
    print("\n📁 Files created:")
    print("   - seatac_training_data.json (Standard JSON)")
    print("   - seatac_gemini_training.jsonl (Google Gemini format)")
    print("   - seatac_llama_training.json (Llama/Gemma format)")
    print("   - seatac_openai_training.jsonl (OpenAI fine-tuning format)")
    print("\n🎯 Next steps:")
    print("   1. Review the generated examples")
    print("   2. Upload to your fine-tuning platform")
    print("   3. Train your custom model")
    print("=" * 70 + "\n")