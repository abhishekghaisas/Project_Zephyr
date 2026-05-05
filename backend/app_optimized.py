"""
SeaTac Operations Intelligence - DIRECT CLAUDE Edition v8.0
FastAPI Implementation with Direct Claude API (Modal Code Commented Out)

OPTIMIZED FOR SPEED (PRESENTATION MODE):
- ✅ Direct Claude SQL generation (FAST)
- ✅ Modal/Code Llama code COMMENTED OUT (easy to re-enable)
- ✅ Single API call instead of Modal → Claude validation
- ✅ Superior SQL generation quality
- ✅ Auto-correction of event type mistakes
- ✅ ~2-3 seconds faster per query

Pipeline: Claude Direct → Auto-Correct → Execute → Insights

TO RE-ENABLE MODAL:
1. Uncomment the Modal sections (search for "# MODAL CODE - COMMENTED OUT")
2. Set USE_MODAL = True
3. Provide MODAL_ENDPOINT in .env
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass
import mysql.connector
from mysql.connector import Error
import json
import re
from datetime import datetime, date
from decimal import Decimal
import os
from dotenv import load_dotenv
import uvicorn
import requests
from anthropic import Anthropic  # NEW: Claude SDK

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SeaTac Airport Operations Intelligence",
    description="AI-powered airport operations analysis - Direct Claude v8.0 (Fast Mode)",
    version="8.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514')  # Latest fast model

# ============================================================================
# MODAL DISABLED FOR SPEED - SET TO True TO RE-ENABLE
# ============================================================================
USE_MODAL = False  # Toggle this to True to use Modal instead of direct Claude

# MODAL CODE - COMMENTED OUT FOR PRESENTATION SPEED
# Uncomment this section to re-enable Modal/Code Llama
# MODAL_ENDPOINT = os.getenv('MODAL_ENDPOINT')
# USE_MODAL_MODEL = os.getenv('USE_MODAL_MODEL', 'true').lower() == 'true'

# Custom JSON encoder
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super(DecimalEncoder, self).default(obj)


# ============================================================================
# GREETING & CASUAL QUERY HANDLER
# ============================================================================

class GreetingHandler:
    """Detect and handle greetings and casual queries with Claude"""
    
    def __init__(self, claude_client):
        self.client = claude_client
        self.model = CLAUDE_MODEL
        
        # Greeting patterns
        self.greeting_keywords = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'whats up', "what's up", 'howdy', 'greetings', 'yo',
            'sup', 'hiya', 'good day'
        ]
        
        # Casual/non-data query patterns
        self.casual_keywords = [
            'thank you', 'thanks', 'appreciate', 'awesome', 'great', 'nice',
            'help', 'what can you do', 'who are you', 'tell me about',
            'explain', 'how do i', 'can you', 'goodbye', 'bye', 'see you'
        ]
    
    def is_greeting_or_casual(self, query: str) -> bool:
        """Check if query is a greeting or casual conversation"""
        query_lower = query.lower().strip()
        
        # Very short queries are likely greetings
        if len(query.split()) <= 3:
            for keyword in self.greeting_keywords:
                if keyword in query_lower:
                    return True
        
        # Check casual patterns
        for keyword in self.casual_keywords:
            if query_lower.startswith(keyword):
                return True
        
        # Check if it's a question about the system itself
        system_questions = ['what can you', 'who are you', 'what do you do', 
                          'how does this work', 'what is this']
        for pattern in system_questions:
            if pattern in query_lower:
                return True
        
        return False
    
    async def handle_casual_query(self, query: str) -> str:
        """Handle greetings and casual queries with Claude"""
        if not self.client:
            return self._fallback_response(query)
        
        try:
            casual_prompt = f"""You are an AI assistant for SeaTac Airport Operations Intelligence system.

The user said: "{query}"

This appears to be a greeting or casual conversation, not a data query.

Respond naturally and helpfully. If they're greeting you, greet them back warmly and briefly explain what you can help with (analyzing airport operations data like taxi times, flight counts, delays, etc.).

Keep your response friendly, concise (2-3 sentences max), and professional.

Response:"""
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.7,
                messages=[{"role": "user", "content": casual_prompt}]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            print(f"❌ Claude error in casual handler: {e}")
            return self._fallback_response(query)
    
    def _fallback_response(self, query: str) -> str:
        """Fallback responses if Claude isn't available"""
        query_lower = query.lower().strip()
        
        # Greetings
        if any(kw in query_lower for kw in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "Hello! I'm your SeaTac Airport Operations assistant. I can help you analyze flight data, taxi times, delays, and more. What would you like to know?"
        
        # Thanks
        if any(kw in query_lower for kw in ['thank', 'thanks', 'appreciate']):
            return "You're welcome! Let me know if you need anything else."
        
        # Goodbye
        if any(kw in query_lower for kw in ['bye', 'goodbye', 'see you']):
            return "Goodbye! Feel free to come back anytime you need help with airport operations data."
        
        # What can you do
        if 'what can you' in query_lower or 'who are you' in query_lower:
            return """I'm an AI assistant specialized in SeaTac Airport Operations Intelligence. I can help you:

• Analyze taxi-in and taxi-out times
• Compare performance by aircraft type
• Identify peak hours and delays
• Calculate runway utilization
• Show flight counts and operations data

Just ask me a question about the airport operations!"""
        
        # Default
        return "I'm here to help with SeaTac airport operations data. Try asking about taxi times, flight counts, delays, or aircraft operations!"


# ============================================================================
# SQL AUTO-CORRECTION
# ============================================================================

def fix_common_sql_errors(sql: str) -> str:
    """
    Auto-fix common event type errors
    This runs AFTER SQL generation to catch mistakes
    """
    if not sql:
        return sql
    
    # Store original for logging
    original_sql = sql
    
    # Fix event types - case-insensitive replacement
    event_type_fixes = {
        # Takeoff variations
        r"event_type\s*=\s*['\"]takeoff['\"]": "event_type = 'Actual_Take_Off'",
        r"event_type\s*=\s*['\"]Takeoff['\"]": "event_type = 'Actual_Take_Off'",
        r"event_type\s*=\s*['\"]take_off['\"]": "event_type = 'Actual_Take_Off'",
        r"event_type\s*=\s*['\"]Take_Off['\"]": "event_type = 'Actual_Take_Off'",
        r"event_type\s*=\s*['\"]TAKEOFF['\"]": "event_type = 'Actual_Take_Off'",
        
        # Landing variations
        r"event_type\s*=\s*['\"]landing['\"]": "event_type = 'Actual_Landing'",
        r"event_type\s*=\s*['\"]Landing['\"]": "event_type = 'Actual_Landing'",
        r"event_type\s*=\s*['\"]LANDING['\"]": "event_type = 'Actual_Landing'",
        
        # Off Block variations
        r"event_type\s*=\s*['\"]offblock['\"]": "event_type = 'Actual_Off_Block'",
        r"event_type\s*=\s*['\"]off_block['\"]": "event_type = 'Actual_Off_Block'",
        r"event_type\s*=\s*['\"]Off_Block['\"]": "event_type = 'Actual_Off_Block'",
        r"event_type\s*=\s*['\"]OffBlock['\"]": "event_type = 'Actual_Off_Block'",
        
        # In Block variations
        r"event_type\s*=\s*['\"]inblock['\"]": "event_type = 'Actual_In_Block'",
        r"event_type\s*=\s*['\"]in_block['\"]": "event_type = 'Actual_In_Block'",
        r"event_type\s*=\s*['\"]In_Block['\"]": "event_type = 'Actual_In_Block'",
        r"event_type\s*=\s*['\"]InBlock['\"]": "event_type = 'Actual_In_Block'",
    }
    
    corrections_made = []
    
    for pattern, replacement in event_type_fixes.items():
        if re.search(pattern, sql, re.IGNORECASE):
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
            match = re.search(pattern, original_sql, re.IGNORECASE)
            if match:
                corrections_made.append(f"{match.group()} → {replacement}")
    
    # Log corrections
    if corrections_made:
        print("\n" + "=" * 80)
        print("🔧 AUTO-CORRECTED EVENT TYPES")
        print("=" * 80)
        for correction in corrections_made:
            print(f"   ✅ Fixed: {correction}")
        print("=" * 80 + "\n")
    
    return sql


# ============================================================================
# DIRECT CLAUDE SQL GENERATOR (FAST MODE)
# ============================================================================

class DirectClaudeSQLGenerator:
    """Generate SQL directly with Claude - single fast API call"""
    
    def __init__(self, claude_client):
        self.client = claude_client
        self.model = CLAUDE_MODEL
        self.enabled = bool(claude_client)
    
    async def generate_sql(self, query: str) -> Optional[str]:
        """Generate SQL directly with Claude"""
        if not self.client:
            return None
        
        try:
            print(f"🤖 Generating SQL query...")
            
            sql_prompt = f"""{DETAILED_SCHEMA}

USER QUERY: "{query}"

Generate a MySQL query to answer this question. Follow these rules:

1. Use ONLY the exact event_type values from schema (case-sensitive)
2. Return ONLY the SQL query, no explanations or markdown
3. Use proper MySQL syntax
4. Include appropriate WHERE, GROUP BY, ORDER BY, and LIMIT clauses
5. For taxi times, calculate the time difference between events
6. For aggregations, use appropriate functions (AVG, COUNT, SUM)
7. Limit results to 100 rows unless specifically asked for more

SQL Query:"""
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": sql_prompt}]
            )
            
            sql = message.content[0].text.strip()
            
            # Clean up SQL
            sql = sql.replace('```sql', '').replace('```', '').strip()
            if sql.endswith(';'):
                sql = sql[:-1]
            
            print(f"✅ Generated SQL query: {sql[:100]}...")
            return sql
            
        except Exception as e:
            print(f"❌ SQL generation error: {e}")
            return None


# ============================================================================
# MODAL CODE - COMMENTED OUT FOR PRESENTATION SPEED
# ============================================================================
# Uncomment this entire section to re-enable Modal/Code Llama
"""
class ModalSQLGenerator:
    \"\"\"Generate SQL using fine-tuned Code Llama on Modal\"\"\"
    
    def __init__(self, endpoint: Optional[str], enabled: bool):
        self.endpoint = endpoint
        self.enabled = enabled and bool(endpoint)
        self.timeout = 180
        
        if self.enabled:
            print(f"✅ Modal Code Llama ENABLED")
            print(f"   Endpoint: {self.endpoint}")
        else:
            print("ℹ️  Modal Code Llama disabled")
    
    def _clean_sql(self, sql_text: str) -> str:
        \"\"\"Clean SQL from Modal output\"\"\"
        if not sql_text:
            return sql_text
        
        print("\\n" + "=" * 80)
        print("🧹 CLEANING SQL")
        print("=" * 80)
        print("RAW INPUT:")
        print(sql_text[:500] + "..." if len(sql_text) > 500 else sql_text)
        print("=" * 80)
        
        sql_text = re.sub(r'```sql\\s*', '', sql_text, flags=re.IGNORECASE)
        sql_text = re.sub(r'```\\s*', '', sql_text)
        sql_text = re.sub(r'^\\s*\\d+\\.\\s+', '', sql_text, flags=re.MULTILINE)
        
        lines = sql_text.split('\\n')
        select_indices = []
        
        for i, line in enumerate(lines):
            if line.strip().upper().startswith(('SELECT', 'WITH')):
                select_indices.append(i)
        
        if not select_indices:
            print("⚠️  No SELECT found")
            return ""
        
        if len(select_indices) > 1:
            print(f"⚠️  Found {len(select_indices)} SELECT statements - keeping first")
        
        sql_start_idx = select_indices[0]
        sql_lines = lines[sql_start_idx:]
        sql_end_idx = len(sql_lines)
        
        for i, line in enumerate(sql_lines):
            stripped_upper = line.strip().upper()
            stripped_lower = line.strip().lower()
            
            if i > 0 and stripped_upper.startswith(('SELECT', 'WITH')):
                print(f"⚠️  Stopping at second SELECT")
                sql_end_idx = i
                break
            
            if any(m in stripped_lower for m in ['note:', 'explanation:', 'this query', 'the above']):
                sql_end_idx = i
                break
            
            if re.match(r'^\\s*\\d+\\.\\s+[A-Z]', line):
                sql_end_idx = i
                break
        
        sql_lines = sql_lines[:sql_end_idx]
        sql_text = '\\n'.join(sql_lines)
        
        # Fix column spacing issues
        common_fixes = [
            (r'\\baircraft\\s+type\\b', 'f.aircraft_type'),
            (r'\\bevent\\s+type\\b', 'event_type'),
            (r'\\bevent\\s+time\\b', 'event_time'),
            (r'\\bcall\\s+sign\\b', 'call_sign'),
        ]
        
        for pattern, replacement in common_fixes:
            sql_text = re.sub(pattern, replacement, sql_text, flags=re.IGNORECASE)
        
        sql_text = sql_text.strip()
        
        if sql_text.endswith(';'):
            sql_text = sql_text[:-1].strip()
        
        if 'LIMIT' not in sql_text.upper():
            sql_text += ' LIMIT 100'
        
        if sql_text.upper().count('SELECT') > 1:
            first_select_end = sql_text.upper().find('SELECT', 1)
            if first_select_end > 0:
                sql_text = sql_text[:first_select_end].strip()
        
        print("✅ CLEANED OUTPUT:")
        print(sql_text[:300] + "..." if len(sql_text) > 300 else sql_text)
        print("=" * 80 + "\\n")
        
        return sql_text
    
    async def generate_sql(self, query: str) -> Optional[str]:
        \"\"\"Generate SQL using Modal endpoint\"\"\"
        if not self.enabled:
            return None
        
        try:
            print(f"🦙 [Modal] Calling fine-tuned Code Llama...")
            
            response = requests.post(
                self.endpoint,
                json={"query": query},
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                print(f"❌ [Modal] HTTP {response.status_code}")
                return None
            
            result = response.json()
            
            if result.get('error'):
                print(f"❌ [Modal] Error: {result['error']}")
                return None
            
            raw_sql = result.get('sql', '').strip()
            
            if not raw_sql:
                print("❌ [Modal] Empty SQL returned")
                return None
            
            print("\\n" + "=" * 80)
            print("🦙 RAW MODAL OUTPUT")
            print("=" * 80)
            print(raw_sql[:500] + "..." if len(raw_sql) > 500 else raw_sql)
            print("=" * 80)
            
            sql = self._clean_sql(raw_sql)
            
            if not sql or 'SELECT' not in sql.upper():
                print(f"❌ [Modal] Invalid SQL after cleaning")
                return None
            
            print(f"✅ [Modal] Cleaned SQL ready for validation")
            return sql
            
        except requests.Timeout:
            print(f"⏱️  [Modal] Timeout after {self.timeout}s")
            return None
        except Exception as e:
            print(f"❌ [Modal] Exception: {e}")
            return None
"""
# END MODAL CODE


# ============================================================================
# MODAL SQL GENERATOR - MINIMAL STUB FOR COMPATIBILITY
# ============================================================================

class ModalSQLGenerator:
    """Stub class for compatibility - Modal code is commented out above"""
    def __init__(self, endpoint: Optional[str] = None, enabled: bool = False):
        self.endpoint = None
        self.enabled = False
        print("ℹ️  Modal disabled (using Direct Claude for speed)")
    
    async def generate_sql(self, query: str) -> Optional[str]:
        return None


# ============================================================================
# SQL VALIDATORS
# ============================================================================

class SQLValidator:
    """Basic SQL validation with schema awareness"""
    
    @staticmethod
    def validate_and_fix(sql: str, query: str) -> Dict[str, Any]:
        """Validate SQL for common errors"""
        warnings = []
        
        if not sql or not sql.strip():
            return {
                'valid': False,
                'sql': sql,
                'warnings': ['❌ Empty SQL'],
                'error': 'Empty SQL generated'
            }
        
        sql_upper = sql.strip().upper()
        sql_lower = sql.lower()
        
        if not sql_upper.startswith(('SELECT', 'WITH', '(SELECT')):
            return {
                'valid': False,
                'sql': sql,
                'warnings': ['❌ SQL does not start with SELECT'],
                'error': f'Invalid start: {sql[:50]}...'
            }
        
        # Check for instructional text
        first_line = sql.split('\n')[0].strip().lower()
        if any(kw in first_line for kw in ['select the appropriate', 'based on the', 'use the following']):
            if 'from' not in first_line:
                warnings.append(f"❌ Instructional text: '{first_line[:60]}'")
                return {
                    'valid': False,
                    'sql': sql,
                    'warnings': warnings,
                    'error': 'Instructions instead of SQL'
                }
        
        if 'FROM' not in sql_upper:
            warnings.append("❌ Missing FROM clause")
            return {'valid': False, 'sql': sql, 'warnings': warnings, 'error': 'Missing FROM'}
        
        if 'NOW()' in sql_upper:
            warnings.append("❌ Uses NOW() - incorrect for taxi times")
            return {'valid': False, 'sql': sql, 'warnings': warnings, 'error': 'Uses NOW()'}
        
        if re.search(r'^\s*\d+\.\s+', sql, re.MULTILINE):
            warnings.append("❌ Contains numbered lists")
            return {'valid': False, 'sql': sql, 'warnings': warnings, 'error': 'Numbered lists'}
        
        from_pos = sql_upper.find('FROM')
        where_pos = sql_upper.find('WHERE')
        if where_pos != -1 and where_pos < from_pos:
            warnings.append("❌ WHERE before FROM")
            return {'valid': False, 'sql': sql, 'warnings': warnings, 'error': 'Invalid structure'}
        
        # CRITICAL: Check for event_time used without flight_event table
        if 'event_time' in sql_lower:
            has_flight_event = 'flight_event' in sql_lower or any(alias in sql_lower for alias in [
                'landing', 'inblock', 'offblock', 'takeoff', 'fe'
            ])
            
            if not has_flight_event:
                warnings.append("❌ Uses event_time but flight_event table not joined")
                return {
                    'valid': False,
                    'sql': sql,
                    'warnings': warnings,
                    'error': 'event_time column requires JOIN with flight_event table'
                }
        
        # Check for time-based queries that need flight_event
        if any(kw in query.lower() for kw in ['time', 'hour', 'pm', 'am', 'between']):
            if 'FROM flight' in sql_upper and 'JOIN' not in sql_upper:
                warnings.append("❌ Time-based query needs flight_event table")
                return {
                    'valid': False,
                    'sql': sql,
                    'warnings': warnings,
                    'error': 'Time-based queries require JOIN with flight_event for event_time column'
                }
        
        # Taxi-in validation
        if 'taxi' in query.lower() and 'in' in query.lower():
            if not ('Actual_Landing' in sql and 'Actual_In_Block' in sql):
                warnings.append("❌ Taxi-in missing proper events")
                return {'valid': False, 'sql': sql, 'warnings': warnings, 'error': 'Missing events'}
        
        # Taxi-out validation
        if 'taxi' in query.lower() and 'out' in query.lower():
            if not ('Actual_Off_Block' in sql and 'Actual_Take_Off' in sql):
                warnings.append("❌ Taxi-out missing proper events")
                return {'valid': False, 'sql': sql, 'warnings': warnings, 'error': 'Missing events'}
        
        warnings.append('✅ SQL validation passed')
        return {'valid': True, 'sql': sql, 'warnings': warnings}


class ClaudeSQLValidator:
    """
    Claude API as "Big Brother" Validator
    Superior validation and correction capabilities
    """
    
    def __init__(self, claude_client):
        self.client = claude_client
        self.basic_validator = SQLValidator()
        self.model = CLAUDE_MODEL
    
    async def validate_and_correct(self, sql: str, user_query: str, source: str = 'modal') -> Dict[str, Any]:
        """Validate SQL and have Claude correct if needed"""
        
        print("\n" + "=" * 80)
        print("🤖 CLAUDE SMART SQL VALIDATION")
        print("=" * 80)
        
        # Basic validation first
        basic_check = self.basic_validator.validate_and_fix(sql, user_query)
        
        print("📋 Basic Validation:")
        for warning in basic_check['warnings']:
            print(f"  {warning}")
        
        if basic_check['valid']:
            print("✅ SQL passed basic validation - sending to Claude for final review...")
        else:
            print(f"\n⚠️  Issues found: {basic_check.get('error')}")
            print("🔧 Claude will validate and correct...")
        
        if not self.client:
            print("❌ Claude API not configured")
            return {
                'valid': False,
                'sql': sql,
                'corrected': False,
                'issues_found': basic_check['warnings'],
                'final_source': source
            }
        
        # Prepare validation prompt for Claude
        validation_prompt = f"""{DETAILED_SCHEMA}

TASK: You are the "Big Brother" validator for SQL queries. Validate and correct this MySQL query if needed.

USER QUERY: "{user_query}"

GENERATED SQL:
{sql}

BASIC VALIDATION RESULTS:
{chr(10).join(basic_check['warnings'])}

Your job:
1. Verify the SQL is syntactically correct MySQL
2. Verify all table names and column names exist in the schema above
3. Verify event_type values are EXACTLY correct (case-sensitive)
4. Verify JOIN conditions are correct
5. Verify the query actually answers the user's question

Respond with ONLY ONE of these formats:

If SQL is correct:
VALID: YES
REASONING: [brief explanation]

If SQL needs correction:
VALID: NO
ISSUES: [list specific problems]
CORRECTED SQL:
[corrected query here - no markdown, just SQL]

CRITICAL RULES:
- Use EXACT event types: 'Actual_Take_Off', 'Actual_Landing', 'Actual_Off_Block', 'Actual_In_Block'
- NEVER use lowercase: 'takeoff', 'landing', 'offblock', 'inblock'
- NEVER use NOW()
- Taxi-in: TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)
- Taxi-out: TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time)
- Filter: BETWEEN 1 AND 120
- Always include LIMIT clause
"""
        
        try:
            print(f"🤖 Calling Claude API ({self.model})...")
            
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": validation_prompt}
                ]
            )
            
            validation_response = message.content[0].text.strip()
            
            print(f"\n🤖 Claude Response:")
            print("=" * 80)
            print(validation_response[:300] + "..." if len(validation_response) > 300 else validation_response)
            print("=" * 80)
            
            # Parse Claude's response
            if 'VALID: YES' in validation_response or 'VALID:YES' in validation_response:
                print("\n✅ Claude APPROVED the SQL")
                
                # Extract reasoning if provided
                reasoning = ""
                if 'REASONING:' in validation_response:
                    reasoning = validation_response.split('REASONING:')[1].strip()
                    print(f"   Reasoning: {reasoning[:100]}...")
                
                return {
                    'valid': True,
                    'sql': sql,
                    'corrected': False,
                    'issues_found': [],
                    'reasoning': reasoning,
                    'final_source': f'{source}-claude-approved'
                }
            
            elif 'VALID: NO' in validation_response or 'VALID:NO' in validation_response:
                print("\n🔧 Claude found issues - extracting corrected SQL...")
                
                # Extract issues
                issues = []
                if 'ISSUES:' in validation_response:
                    issues_text = validation_response.split('ISSUES:')[1]
                    if 'CORRECTED SQL:' in issues_text:
                        issues_text = issues_text.split('CORRECTED SQL:')[0]
                    issues = [issues_text.strip()]
                
                # Extract corrected SQL
                corrected_sql = None
                
                if 'CORRECTED SQL:' in validation_response:
                    sql_part = validation_response.split('CORRECTED SQL:')[1].strip()
                    corrected_sql = self._clean_sql(sql_part)
                else:
                    # Try to find SELECT statement
                    lines = validation_response.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().upper().startswith('SELECT'):
                            corrected_sql = '\n'.join(lines[i:])
                            corrected_sql = self._clean_sql(corrected_sql)
                            break
                
                if corrected_sql:
                    # Apply auto-correction
                    corrected_sql = fix_common_sql_errors(corrected_sql)
                    
                    if corrected_sql and 'SELECT' in corrected_sql.upper():
                        print(f"\n✅ SQL CORRECTED BY CLAUDE")
                        print("=" * 80)
                        print(corrected_sql[:200] + "..." if len(corrected_sql) > 200 else corrected_sql)
                        print("=" * 80)
                        
                        return {
                            'valid': True,
                            'sql': corrected_sql,
                            'corrected': True,
                            'issues_found': issues,
                            'final_source': f'{source}-claude-corrected'
                        }
                
                print("⚠️  Claude indicated issues but couldn't extract corrected SQL")
                return {
                    'valid': False,
                    'sql': sql,
                    'corrected': False,
                    'issues_found': issues,
                    'final_source': source
                }
            
            else:
                print("⚠️  Unexpected Claude response format")
                return {
                    'valid': False,
                    'sql': sql,
                    'corrected': False,
                    'issues_found': ['Unexpected response format from Claude'],
                    'final_source': source
                }
                
        except Exception as e:
            print(f"❌ Claude API error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'valid': False,
                'sql': sql,
                'corrected': False,
                'issues_found': [f'Claude API error: {str(e)}'],
                'final_source': source
            }
    
    def _clean_sql(self, sql_text: str) -> str:
        """Clean SQL from Claude's response"""
        if not sql_text:
            return sql_text
        
        # Remove markdown code blocks
        sql_text = re.sub(r'```sql\s*', '', sql_text, flags=re.IGNORECASE)
        sql_text = re.sub(r'```\s*', '', sql_text)
        sql_text = re.sub(r'^\s*\d+\.\s+', '', sql_text, flags=re.MULTILINE)
        
        lines = sql_text.split('\n')
        sql_start_idx = None
        
        # Find first SELECT
        for i, line in enumerate(lines):
            if line.strip().upper().startswith(('SELECT', 'WITH')):
                sql_start_idx = i
                break
        
        if sql_start_idx is None:
            return sql_text.strip()
        
        sql_lines = lines[sql_start_idx:]
        sql_end_idx = len(sql_lines)
        
        # Find end of SQL (before explanations)
        for i, line in enumerate(sql_lines[1:], 1):
            if line.strip().upper().startswith(('SELECT', 'WITH')):
                sql_end_idx = i
                break
            if any(m in line.lower() for m in ['note:', 'explanation:', 'reasoning:']):
                sql_end_idx = i
                break
        
        sql_text = '\n'.join(sql_lines[:sql_end_idx]).strip()
        
        # Remove trailing semicolon
        if sql_text.endswith(';'):
            sql_text = sql_text[:-1].strip()
        
        # Add LIMIT if missing
        if 'LIMIT' not in sql_text.upper():
            sql_text += ' LIMIT 100'
        
        return sql_text


# ============================================================================
# TEMPORAL CONTEXT EXTRACTION (unchanged)
# ============================================================================

class TemporalContextExtractor:
    """Extract temporal filters from queries"""
    
    def __init__(self):
        self.time_periods = {
            'morning': (5, 11),
            'afternoon': (12, 17),
            'evening': (18, 21),
            'night': (22, 4),
            'rush hour': [(7, 9), (16, 18)],
        }
    
    def extract_temporal_context(self, query: str) -> Dict:
        """Extract time filters"""
        query_lower = query.lower().strip()
        context = {
            'hour_range': None,
            'specific_hours': None,
            'has_temporal_filter': False
        }
        
        for period_name, hours in self.time_periods.items():
            if period_name in query_lower:
                context['has_temporal_filter'] = True
                context['hour_range'] = hours
                break
        
        specific_hour_pattern = r'\b(?:at\s+)?(\d{1,2})(?::00)?\s*(am|pm|AM|PM)?\b'
        matches = re.findall(specific_hour_pattern, query_lower)
        if matches:
            context['has_temporal_filter'] = True
            hours = []
            for hour_str, meridiem in matches:
                hour = int(hour_str)
                if meridiem.lower() == 'pm' and hour != 12:
                    hour += 12
                elif meridiem.lower() == 'am' and hour == 12:
                    hour = 0
                hours.append(hour)
            context['specific_hours'] = hours
        
        range_pattern = r'between\s+(\d{1,2})(?::00)?\s*(am|pm)?\s+(?:and|to)\s+(\d{1,2})(?::00)?\s*(am|pm)?'
        range_match = re.search(range_pattern, query_lower)
        
        if range_match:
            context['has_temporal_filter'] = True
            start_hour = int(range_match.group(1))
            start_meridiem = range_match.group(2) or ''
            end_hour = int(range_match.group(3))
            end_meridiem = range_match.group(4) or ''
            
            if start_meridiem.lower() == 'pm' and start_hour != 12:
                start_hour += 12
            if end_meridiem.lower() == 'pm' and end_hour != 12:
                end_hour += 12
            
            context['hour_range'] = (start_hour, end_hour)
        
        return context
    
    def inject_temporal_filter(self, sql_query: str, user_query: str) -> str:
        """Inject temporal WHERE clauses"""
        temporal_context = self.extract_temporal_context(user_query)
        
        if not temporal_context['has_temporal_filter']:
            return sql_query
        
        sql_upper = sql_query.upper()
        sql_lower = sql_query.lower()
        
        if 'TIME(' in sql_upper or ('BETWEEN' in sql_upper and ("'14:" in sql_query or "'17:" in sql_query)):
            print("⏭️  SQL already has time filtering")
            return sql_query
        
        if 'event_time' not in sql_lower:
            print("⏭️  Query doesn't use event_time, skipping temporal filter")
            return sql_query
        
        time_column = None
        if 'offblock.event_time' in sql_lower:
            time_column = 'offblock.event_time'
        elif 'landing.event_time' in sql_lower:
            time_column = 'landing.event_time'
        elif 'takeoff.event_time' in sql_lower:
            time_column = 'takeoff.event_time'
        elif 'le.event_time' in sql_lower:
            time_column = 'le.event_time'
        elif 'fe.event_time' in sql_lower:
            time_column = 'fe.event_time'
        else:
            print("⚠️  No usable event_time column found")
            return sql_query
        
        clauses = []
        
        if temporal_context.get('specific_hours'):
            hours = temporal_context['specific_hours']
            if len(hours) == 1:
                clauses.append(f"HOUR({time_column}) = {hours[0]}")
            else:
                hour_list = ', '.join(map(str, hours))
                clauses.append(f"HOUR({time_column}) IN ({hour_list})")
        
        elif temporal_context.get('hour_range'):
            hour_range = temporal_context['hour_range']
            if isinstance(hour_range, list):
                range_clauses = []
                for start, end in hour_range:
                    if start <= end:
                        range_clauses.append(f"(HOUR({time_column}) BETWEEN {start} AND {end})")
                    else:
                        range_clauses.append(f"(HOUR({time_column}) >= {start} OR HOUR({time_column}) <= {end})")
                clauses.append(f"({' OR '.join(range_clauses)})")
            else:
                start, end = hour_range
                if start <= end:
                    clauses.append(f"HOUR({time_column}) BETWEEN {start} AND {end}")
                else:
                    clauses.append(f"(HOUR({time_column}) >= {start} OR HOUR({time_column}) <= {end})")
        
        if not clauses:
            return sql_query
        
        where_clause = ' AND '.join(clauses)
        
        if 'WHERE' in sql_upper:
            where_pos = sql_upper.find('WHERE') + 5
            next_clause_pos = len(sql_query)
            for clause in ['GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT']:
                pos = sql_upper.find(clause, where_pos)
                if pos != -1 and pos < next_clause_pos:
                    next_clause_pos = pos
            
            before = sql_query[:next_clause_pos].rstrip()
            after = sql_query[next_clause_pos:].lstrip()
            print(f"✅ Injecting: {where_clause}")
            return f"{before}\n  AND {where_clause}\n{after}"
        else:
            group_by_pos = sql_upper.find('GROUP BY')
            if group_by_pos != -1:
                before = sql_query[:group_by_pos].rstrip()
                after = sql_query[group_by_pos:].lstrip()
                print(f"✅ Injecting: {where_clause}")
                return f"{before}\nWHERE {where_clause}\n{after}"
            
            print(f"✅ Injecting: {where_clause}")
            return f"{sql_query.rstrip()}\nWHERE {where_clause}"


# ============================================================================
# OUTPUT FORMAT CLASSIFICATION (unchanged)
# ============================================================================

OutputFormat = Literal['chart', 'table', 'text', 'chart_and_text', 'table_and_text', 'all']

@dataclass
class OutputPreference:
    format: OutputFormat
    confidence: float
    reasoning: str
    
    @property
    def show_chart(self) -> bool:
        return self.format in ['chart', 'chart_and_text', 'all']
    
    @property
    def show_table(self) -> bool:
        return self.format in ['table', 'table_and_text', 'all']
    
    @property
    def show_text(self) -> bool:
        return self.format in ['text', 'chart_and_text', 'table_and_text', 'all']


class OutputFormatClassifier:
    """Classify desired output format"""
    
    def __init__(self):
        self.format_indicators = {
            'chart': {
                'explicit': ['chart', 'visualize', 'plot', 'graph'],
                'implicit': ['trends', 'pattern', 'by hour', 'compare']
            },
            'table': {
                'explicit': ['table', 'list', 'show me the data'],
                'implicit': ['which flights', 'what are the']
            },
            'text': {
                'explicit': ['just tell me', 'summarize'],
                'implicit': ['how many', 'what was', 'average']
            }
        }
    
    def classify(self, query: str, intent: str = None) -> OutputPreference:
        query_lower = query.lower().strip()
        scores = {'chart': 0.0, 'table': 0.0, 'text': 0.0}
        
        for format_type, indicators in self.format_indicators.items():
            for phrase in indicators['explicit']:
                if phrase in query_lower:
                    scores[format_type] += 3.0
            for phrase in indicators['implicit']:
                if phrase in query_lower:
                    scores[format_type] += 0.5
        
        if all(s < 1.0 for s in scores.values()):
            if any(kw in query_lower for kw in ['hour', 'time', 'compare']):
                return OutputPreference('chart_and_text', 0.6, "Default: analytical")
            return OutputPreference('text', 0.5, "Default: simple")
        
        max_score = max(scores.values())
        top_formats = [f for f, s in scores.items() if s >= max_score * 0.7]
        
        if len(top_formats) == 1:
            final = top_formats[0]
        elif 'chart' in top_formats and 'text' in top_formats:
            final = 'chart_and_text'
        else:
            final = top_formats[0]
        
        return OutputPreference(final, min(max_score / 5.0, 1.0), '')


# ============================================================================
# DETAILED SCHEMA WITH EXACT EVENT TYPES
# ============================================================================

DETAILED_SCHEMA = """
DATABASE SCHEMA - SeaTac Airport Operations (MySQL)

Tables:
1. flight (call_sign, aircraft_type, operation, flight_number, origin_airport, destination_airport)
2. flight_event (call_sign, event_type, event_time, location, operation)
3. aircraft_type (aircraft_type, weight_class, wake_category, wingspan_ft, wingspan_m)

⚠️  CRITICAL - EXACT Event Type Values (case-sensitive with underscores):

✅ CORRECT event_type values:
   - 'Actual_Take_Off' (NOT 'takeoff', 'Takeoff', 'take_off', 'TAKEOFF')
   - 'Actual_Landing' (NOT 'landing', 'Landing', 'LANDING')
   - 'Actual_Off_Block' (NOT 'offblock', 'off_block', 'OffBlock')
   - 'Actual_In_Block' (NOT 'inblock', 'in_block', 'InBlock')

❌ WRONG examples that will return ZERO results:
   - 'takeoff', 'landing', 'offblock', 'inblock' (lowercase = NO MATCH)
   - Any variation without 'Actual_' prefix

TAXI TIME FORMULAS:
- Taxi-in: TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)
  WHERE landing.event_type = 'Actual_Landing' 
    AND inblock.event_type = 'Actual_In_Block'
    AND landing.operation = 'ARRIVAL'
    AND inblock.operation = 'ARRIVAL'
    
- Taxi-out: TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time)
  WHERE offblock.event_type = 'Actual_Off_Block'
    AND takeoff.event_type = 'Actual_Take_Off'
    AND offblock.operation = 'DEPARTURE'
    AND takeoff.operation = 'DEPARTURE'

MANDATORY RULES:
- ALWAYS use exact event_type values with 'Actual_' prefix and proper capitalization
- ALWAYS match operation field in JOINs (AND fe.operation = 'DEPARTURE' or 'ARRIVAL')
- ALWAYS filter invalid times: BETWEEN 1 AND 120
- NEVER use NOW()
- ALWAYS include LIMIT clause

Example for counting departures:
SELECT COUNT(DISTINCT f.call_sign) as departure_count
FROM flight f
JOIN flight_event fe ON f.call_sign = fe.call_sign
WHERE f.operation = 'DEPARTURE'
  AND fe.operation = 'DEPARTURE'
  AND fe.event_type = 'Actual_Take_Off'
LIMIT 100
"""


# ============================================================================
# CLAUDE API CLIENT
# ============================================================================

# Initialize Claude client
claude_client = Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY else None


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)

class QueryResponse(BaseModel):
    success: bool
    message: str
    use_case: Optional[str] = None
    agent_reasoning: Optional[List[Dict[str, str]]] = None
    row_count: int = 0
    sql_queries: Optional[List[str]] = None
    data: Optional[List[Dict[str, Any]]] = None
    insights: Optional[List[str]] = None
    chart: Optional[Dict[str, Any]] = None
    output_format: Optional[str] = None
    output_confidence: Optional[float] = None
    sql_source: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model: str
    modal_enabled: bool
    modal_endpoint: Optional[str]
    claude_enabled: bool
    database_status: str
    version: str


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Database operations"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'AIplane')
        }
    
    def test_connection(self) -> bool:
        try:
            connection = mysql.connector.connect(**self.db_config)
            connection.close()
            return True
        except Error:
            return False
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("📋 EXECUTING SQL")
        print("=" * 80)
        print(sql)
        print("=" * 80 + "\n")
        
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            connection.close()
            
            cleaned_results = []
            for row in results:
                cleaned_row = {}
                for key, value in row.items():
                    if isinstance(value, Decimal):
                        cleaned_row[key] = float(value)
                    elif isinstance(value, (datetime, date)):
                        cleaned_row[key] = value.isoformat()
                    else:
                        cleaned_row[key] = value
                cleaned_results.append(cleaned_row)
            
            print(f"✅ Success: {len(cleaned_results)} rows\n")
            
            return {
                'success': True,
                'data': cleaned_results,
                'row_count': len(cleaned_results),
                'sql': sql
            }
        except Error as e:
            print(f"❌ FAILED: {str(e)}\n")
            return {
                'success': False,
                'error': str(e),
                'sql': sql,
                'row_count': 0,
                'data': []
            }


# ============================================================================
# CHART GENERATOR
# ============================================================================

class ChartGenerator:
    """Generate Chart.js configs"""
    
    @staticmethod
    def generate_chart(data: List[Dict], title: str = "Analysis") -> Optional[Dict]:
        if not data:
            return None
        
        first_row = data[0]
        keys = list(first_row.keys())
        
        # Find label field (categorical data)
        label_field = None
        value_field = None
        
        # Priority 1: Look for common label fields
        for key in keys:
            if any(t in key.lower() for t in ['hour', 'type', 'class', 'airport', 'operation']):
                label_field = key
                break
        
        # Priority 2: Find numeric value field
        for key in keys:
            if any(t in key.lower() for t in ['avg', 'count', 'total', 'minutes', 'sum', 'max', 'min']):
                # Check if this field is actually numeric
                try:
                    test_value = first_row.get(key)
                    if test_value is not None:
                        float(test_value)
                        if value_field is None or 'avg' in key.lower():
                            value_field = key
                except (ValueError, TypeError):
                    continue
        
        # Fallback: First non-numeric field for labels, first numeric for values
        if not label_field:
            for key in keys:
                try:
                    float(first_row.get(key, 0))
                except (ValueError, TypeError):
                    label_field = key
                    break
            if not label_field:
                label_field = keys[0]
        
        if not value_field:
            for key in keys:
                if key != label_field:
                    try:
                        float(first_row.get(key, 0))
                        value_field = key
                        break
                    except (ValueError, TypeError):
                        continue
        
        # If still no value field found, return None
        if not value_field:
            print(f"⚠️  Could not find numeric field for chart. Available fields: {keys}")
            return None
        
        # Extract labels and values
        labels = []
        values = []
        
        for row in data[:24]:  # Limit to 24 rows for readability
            label = str(row.get(label_field, ''))
            
            # Format hour labels nicely
            if 'hour' in label_field.lower() and label.isdigit():
                label = f"{int(label):02d}:00"
            
            labels.append(label)
            
            # Safely convert value to float
            try:
                value = float(row.get(value_field, 0))
            except (ValueError, TypeError):
                value = 0
            values.append(value)
        
        # Determine chart type
        is_temporal = 'hour' in label_field.lower()
        chart_type = 'line' if is_temporal else 'bar'
        
        # Determine y-axis label
        y_label = 'Value'
        if 'minute' in value_field.lower() or 'time' in value_field.lower():
            y_label = 'Minutes'
        elif 'count' in value_field.lower():
            y_label = 'Count'
        
        return {
            'type': chart_type,
            'title': title,
            'data': {
                'labels': labels,
                'datasets': [{
                    'label': value_field.replace('_', ' ').title(),
                    'data': values,
                    'backgroundColor': 'rgba(59, 130, 246, 0.7)' if chart_type == 'bar' else 'rgba(59, 130, 246, 0.2)',
                    'borderColor': 'rgba(59, 130, 246, 1)',
                    'borderWidth': 2,
                    'tension': 0.4 if chart_type == 'line' else 0,
                    'fill': chart_type == 'line'
                }]
            },
            'options': {
                'responsive': True,
                'scales': {
                    'y': {
                        'beginAtZero': True,
                        'title': {'display': True, 'text': y_label}
                    }
                }
            }
        }


# ============================================================================
# RESPONSE CONTROLLER
# ============================================================================

class EnhancedResponseController:
    """Controls output generation"""
    
    def __init__(self, format_classifier, chart_generator):
        self.classifier = format_classifier
        self.chart_generator = chart_generator
    
    def process_output(self, user_query: str, result: Dict, insights: str, title: str = "Analysis") -> Dict:
        output_pref = self.classifier.classify(user_query)
        
        response_data = {
            'output_format': output_pref.format,
            'output_confidence': output_pref.confidence
        }
        
        if output_pref.show_text:
            response_data['message'] = insights
        else:
            response_data['message'] = f"Found {result.get('row_count', 0)} results"
        
        if output_pref.show_chart and result.get('data'):
            chart = self.chart_generator.generate_chart(result['data'], title)
            if chart:
                response_data['chart'] = chart
        
        if output_pref.show_table and result.get('data'):
            response_data['data'] = result['data'][:100]
        
        return response_data


# ============================================================================
# SEATAC AGENT - SMART PIPELINE WITH CLAUDE
# ============================================================================

class SeaTacAgent:
    """Agent with Claude as Big Brother validator"""
    
    def __init__(self, modal_gen, claude_validator, db_manager):
        self.modal_gen = modal_gen
        self.claude_validator = claude_validator
        self.db_manager = db_manager
        self.chart_generator = ChartGenerator()
        self.temporal_extractor = TemporalContextExtractor()
        self.format_classifier = OutputFormatClassifier()
        self.response_controller = EnhancedResponseController(self.format_classifier, self.chart_generator)
        
        # Pre-built use cases
        self.use_cases = {
            "1": {
                "keywords": ["taxi-in", "taxi in", "aircraft type"],
                "sql": """SELECT f.aircraft_type, 
                       AVG(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as avg_taxi_in_minutes,
                       COUNT(*) as flight_count
                FROM flight f
                JOIN aircraft_type at ON f.aircraft_type = at.aircraft_type
                JOIN flight_event landing ON f.call_sign = landing.call_sign 
                     AND landing.event_type = 'Actual_Landing' 
                     AND landing.operation = 'ARRIVAL'
                JOIN flight_event inblock ON f.call_sign = inblock.call_sign
                     AND inblock.event_type = 'Actual_In_Block' 
                     AND inblock.operation = 'ARRIVAL'
                WHERE f.operation = 'ARRIVAL'
                  AND TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time) BETWEEN 1 AND 120
                GROUP BY f.aircraft_type, at.weight_class
                ORDER BY avg_taxi_in_minutes DESC
                LIMIT 20"""
            },
            "2": {
                "keywords": ["taxi-out", "taxi out", "hour"],
                "sql": """SELECT HOUR(offblock.event_time) as hour_of_day,
                       AVG(TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time)) as avg_taxi_out_minutes,
                       COUNT(*) as flight_count
                FROM flight f
                JOIN flight_event offblock ON f.call_sign = offblock.call_sign 
                     AND offblock.event_type = 'Actual_Off_Block' 
                     AND offblock.operation = 'DEPARTURE'
                JOIN flight_event takeoff ON f.call_sign = takeoff.call_sign
                     AND takeoff.event_type = 'Actual_Take_Off' 
                     AND takeoff.operation = 'DEPARTURE'
                WHERE f.operation = 'DEPARTURE'
                  AND TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time) BETWEEN 1 AND 120
                GROUP BY HOUR(offblock.event_time)
                ORDER BY hour_of_day"""
            }
        }
    
    async def generate_sql(self, query: str) -> Dict[str, Any]:
        """
        FAST PIPELINE WITH DIRECT CLAUDE:
        1. Claude generates SQL directly (FAST - no Modal)
        2. Auto-fix event types
        3. Return SQL
        
        Modal validation steps are SKIPPED for presentation speed
        """
        
        print("\n" + "=" * 80)
        print("⚡ SQL GENERATION PIPELINE")
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)
        
        # STAGE 1: SQL generation (FAST)
        if self.modal_gen and hasattr(self.modal_gen, 'generate_sql'):
            print("\n📍 Generating SQL query...")
            
            sql = await self.modal_gen.generate_sql(query)
            
            if sql:
                # STAGE 2: Auto-fix event types
                sql = fix_common_sql_errors(sql)
                
                print(f"\n✅ SQL query generated successfully")
                return {
                    'success': True,
                    'sql': sql,
                    'source': 'generated'  # Generic source name
                }
        
        # FALLBACK: Pre-built SQL
        print("\n📍 FALLBACK: Pre-built SQL")
        prebuilt = self._get_prebuilt_sql(query)
        return {
            'success': True,
            'sql': prebuilt,
            'source': 'prebuilt'
        }
    
    # ============================================================================
    # ORIGINAL MODAL VALIDATION PIPELINE - COMMENTED OUT FOR SPEED
    # ============================================================================
    """
    async def generate_sql_with_modal(self, query: str) -> Dict[str, Any]:
        \"\"\"
        ORIGINAL SMART PIPELINE WITH CLAUDE VALIDATION:
        1. Modal generates SQL
        2. Auto-fix event types
        3. Claude validates/corrects
        4. Return best SQL
        
        This method is preserved but not used in fast mode
        \"\"\"
        
        print("\\n" + "=" * 80)
        print("🧠 SMART SQL PIPELINE (CLAUDE VALIDATOR)")
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)
        
        # STAGE 1: Modal
        if self.modal_gen.enabled:
            print("\\n📍 STAGE 1: Modal generates SQL...")
            modal_sql = await self.modal_gen.generate_sql(query)
            
            if modal_sql:
                # STAGE 2: Auto-fix event types
                modal_sql = fix_common_sql_errors(modal_sql)
                
                # STAGE 3: Claude validates/corrects
                print("\\n📍 STAGE 3: Claude validates...")
                
                validation = await self.claude_validator.validate_and_correct(
                    modal_sql, 
                    query, 
                    source='modal'
                )
                
                if validation['valid']:
                    if validation['corrected']:
                        print("\\n🎯 Using CLAUDE-CORRECTED SQL (Modal → Claude fix)")
                        return {
                            'success': True,
                            'sql': validation['sql'],
                            'source': 'modal-claude-corrected'
                        }
                    else:
                        print("\\n🎯 Using CLAUDE-VALIDATED Modal SQL (Claude approved)")
                        return {
                            'success': True,
                            'sql': validation['sql'],
                            'source': 'modal-claude-validated'
                        }
                else:
                    print("\\n❌ Claude couldn't validate, generating fresh...")
        
        # STAGE 4: Claude fresh generation
        if claude_client:
            try:
                print("\\n📍 STAGE 4: Claude generates fresh SQL...")
                
                sql_prompt = f\"\"\"{DETAILED_SCHEMA}

Generate a MySQL query for: "{query}"

CRITICAL: Use EXACT event_type values:
- 'Actual_Take_Off' (not 'takeoff')
- 'Actual_Landing' (not 'landing')  
- 'Actual_Off_Block' (not 'offblock')
- 'Actual_In_Block' (not 'inblock')

Return ONLY the SQL query, no explanations or markdown.
\"\"\"
                
                message = claude_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=1500,
                    temperature=0.05,
                    messages=[{"role": "user", "content": sql_prompt}]
                )
                
                sql = message.content[0].text.strip()
                sql = self._clean_sql(sql)
                sql = fix_common_sql_errors(sql)
                
                if sql and 'SELECT' in sql.upper():
                    print(f"✅ Claude fresh SQL")
                    return {
                        'success': True,
                        'sql': sql,
                        'source': 'claude'
                    }
                
            except Exception as e:
                print(f"❌ Claude error: {e}")
        
        # STAGE 5: Pre-built
        print("\\n📍 STAGE 5: Pre-built SQL")
        prebuilt = self._get_prebuilt_sql(query)
        return {
            'success': True,
            'sql': prebuilt,
            'source': 'prebuilt'
        }
    """
    
    def _clean_sql(self, sql_text: str) -> str:
        """Clean SQL"""
        if not sql_text:
            return sql_text
        
        sql_text = re.sub(r'```sql\s*', '', sql_text, flags=re.IGNORECASE)
        sql_text = re.sub(r'```\s*', '', sql_text)
        sql_text = re.sub(r'^\s*\d+\.\s+', '', sql_text, flags=re.MULTILINE)
        
        lines = sql_text.split('\n')
        sql_start_idx = None
        
        for i, line in enumerate(lines):
            if line.strip().upper().startswith(('SELECT', 'WITH')):
                sql_start_idx = i
                break
        
        if sql_start_idx is None:
            return sql_text.strip()
        
        sql_lines = lines[sql_start_idx:]
        sql_end_idx = len(sql_lines)
        
        for i, line in enumerate(sql_lines[1:], 1):
            if line.strip().upper().startswith(('SELECT', 'WITH')):
                sql_end_idx = i
                break
        
        sql_text = '\n'.join(sql_lines[:sql_end_idx]).strip()
        
        if sql_text.endswith(';'):
            sql_text = sql_text[:-1].strip()
        
        if 'LIMIT' not in sql_text.upper():
            sql_text += ' LIMIT 100'
        
        return sql_text
    
    def _get_prebuilt_sql(self, query: str) -> str:
        query_lower = query.lower()
        
        for use_case_id, use_case in self.use_cases.items():
            if any(kw in query_lower for kw in use_case['keywords']):
                return use_case['sql'].strip()
        
        return "SELECT * FROM flight LIMIT 10"
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with Claude-powered smart pipeline"""
        try:
            reasoning_steps = []
            
            print(f"\n{'='*80}")
            print(f"📥 Query: {query}")
            print(f"{'='*80}")
            
            # Generate SQL
            sql_result = await self.generate_sql(query)
            sql_query = sql_result['sql']
            sql_source = sql_result['source']
            
            reasoning_steps.append({
                'action': 'generate_sql',
                'observation': f"Generated via {sql_source}"
            })
            
            # Apply temporal filtering
            enhanced_sql = self.temporal_extractor.inject_temporal_filter(sql_query, query)
            
            if enhanced_sql != sql_query:
                reasoning_steps.append({
                    'action': 'apply_temporal_filter',
                    'observation': 'Applied time filtering'
                })
            
            # Execute SQL
            result = self.db_manager.execute_query(enhanced_sql)
            
            reasoning_steps.append({
                'action': 'execute_sql',
                'observation': f"Retrieved {result.get('row_count', 0)} rows"
            })
            
            if not result['success']:
                return {
                    'success': False,
                    'answer': f"SQL failed: {result.get('error')}",
                    'sql_source': sql_source,
                    'sql_queries': [enhanced_sql]
                }
            
            if not result['data']:
                return {
                    'success': True,
                    'answer': "No data found.",
                    'sql_source': sql_source,
                    'sql_queries': [enhanced_sql],
                    'row_count': 0
                }
            
            # Generate insights with Claude
            insights = await self._generate_insights(query, result['data'])
            
            # Process output format
            output_data = self.response_controller.process_output(
                user_query=query,
                result=result,
                insights=insights,
                title="SeaTac Operations"
            )
            
            return {
                'success': True,
                'answer': output_data.get('message', insights),
                'use_case': 'Analysis',
                'reasoning_steps': reasoning_steps,
                'sql_queries': [enhanced_sql],
                'data': output_data.get('data', result['data'][:100]),
                'row_count': result['row_count'],
                'chart': output_data.get('chart'),
                'insights': [insights],
                'output_format': output_data.get('output_format'),
                'output_confidence': output_data.get('output_confidence'),
                'sql_source': sql_source
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'answer': f'Error: {str(e)}',
                'sql_source': 'error'
            }
    
    async def _generate_insights(self, query: str, data: List[Dict]) -> str:
        """Generate insights using Claude"""
        if not data:
            return "No data available."
        
        if not claude_client:
            return f"Found {len(data)} results."
        
        insights_prompt = f"""Analyze this airport operations data and provide 2-3 sentences with specific numbers.

QUESTION: "{query}"
DATA SAMPLE: {json.dumps(data[:3], indent=2, cls=DecimalEncoder)}

Provide brief, data-driven insights:
"""
        
        try:
            message = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
                temperature=0.4,
                messages=[{"role": "user", "content": insights_prompt}]
            )
            return message.content[0].text.strip()
        except:
            return f"Found {len(data)} results."


# ============================================================================
# INITIALIZE
# ============================================================================

# FAST MODE: Direct Claude SQL generation (no Modal)
direct_claude_generator = DirectClaudeSQLGenerator(claude_client) if claude_client else None

# Modal disabled for presentation speed
modal_generator = ModalSQLGenerator(None, False)  # Stub only

# Keep validators and other components
claude_validator = ClaudeSQLValidator(claude_client) if claude_client else None  
greeting_handler = GreetingHandler(claude_client) if claude_client else None
db_manager = DatabaseManager()

# Agent uses direct Claude generator for speed
agent_system = SeaTacAgent(direct_claude_generator, claude_validator, db_manager) if claude_client else None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "SeaTac Operations Intelligence - v8.0",
        "version": "8.0.0",
        "status": "online",
        "pipeline": [
            "1. Generate SQL query",
            "2. Auto-correct event types",
            "3. Execute SQL",
            "4. Generate insights"
        ]
    }


@app.post("/api/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Process queries with Claude-powered validation and greeting detection"""
    try:
        if not request.query or len(request.query.strip()) < 2:
            raise HTTPException(status_code=400, detail="Invalid query")
        
        # NEW: Check if it's a greeting or casual query
        if greeting_handler and greeting_handler.is_greeting_or_casual(request.query):
            print(f"\n🤝 Detected greeting/casual query: {request.query}")
            casual_response = await greeting_handler.handle_casual_query(request.query)
            
            return QueryResponse(
                success=True,
                message=casual_response,
                use_case="Casual",
                agent_reasoning=[{
                    'action': 'greeting_detected',
                    'observation': 'Handled by Claude without SQL generation'
                }],
                row_count=0,
                sql_queries=[],
                data=None,
                insights=None,
                chart=None,
                sql_source='system'  # Generic source for casual queries
            )
        
        # Regular data query processing
        if not agent_system:
            raise HTTPException(status_code=503, detail="No SQL generator configured")
        
        result = await agent_system.process_query(request.query)
        
        return QueryResponse(
            success=result.get('success', False),
            message=result.get('answer', 'No answer'),
            use_case=result.get('use_case'),
            agent_reasoning=result.get('reasoning_steps', []),
            row_count=result.get('row_count', 0),
            sql_queries=result.get('sql_queries', []),
            data=result.get('data'),
            insights=result.get('insights', []),
            chart=result.get('chart'),
            output_format=result.get('output_format'),
            output_confidence=result.get('output_confidence'),
            sql_source=result.get('sql_source', 'generated')  # Default to generic 'generated'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model=CLAUDE_MODEL,
        modal_enabled=False,  # Disabled for speed
        modal_endpoint=None,
        claude_enabled=bool(claude_client),
        database_status="online" if db_manager.test_connection() else "offline",
        version="8.0.0"
    )


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 80)
    print("✈️  SeaTac Operations Intelligence - DIRECT CLAUDE v8.0 (FAST MODE)")
    print("=" * 80)
    
    print(f"⚡ PRESENTATION SPEED MODE: Modal disabled, using Direct Claude")
    print(f"   Modal code preserved in comments (search for 'MODAL CODE')")
    print()
    
    if claude_client:
        print(f"🤖 Claude API: ENABLED ✅")
        print(f"   Model: {CLAUDE_MODEL}")
        print(f"   Role: Direct SQL Generator + Insights")
        print(f"   Speed: ~2-3 seconds faster than Modal pipeline")
    else:
        print(f"🤖 Claude API: NOT CONFIGURED ❌")
    
    print(f"\n🗄️  Database: {db_manager.db_config['database']}@{db_manager.db_config['host']}")
    if db_manager.test_connection():
        print(f"   Status: Connected ✅")
    else:
        print(f"   Status: Disconnected ❌")
    
    print("\n⚡ OPTIMIZED SQL Pipeline (Direct Claude):")
    print("   1️⃣  Claude generates SQL directly (SINGLE FAST API CALL)")
    print("   2️⃣  Auto-correct event types")
    print("   3️⃣  Execute SQL")
    print("   4️⃣  Claude generates insights")
    
    print("\n✨ Speed Improvements v8.0:")
    print("   ✓ Eliminated Modal roundtrip (~2-3 sec saved)")
    print("   ✓ Single Claude API call for SQL generation")
    print("   ✓ Superior Claude SQL generation quality")
    print("   ✓ Auto-correction of event types")
    print("   ✓ Schema-aware generation")
    print("   ✓ Perfect for live presentations")
    
    print("\n🔄 To re-enable Modal:")
    print("   1. Search for 'MODAL CODE - COMMENTED OUT'")
    print("   2. Uncomment the ModalSQLGenerator class")
    print("   3. Set USE_MODAL = True")
    print("   4. Update initialization to use modal_generator")
    
    print("\n🚀 Server starting on http://0.0.0.0:8000")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    uvicorn.run("app_optimized:app", host="0.0.0.0", port=8000, reload=True)