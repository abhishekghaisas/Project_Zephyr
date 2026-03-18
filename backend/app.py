"""
SeaTac Operations Intelligence - Modal Edition v7.0
FastAPI Implementation with Fine-Tuned Code Llama

Complete version with:
- Modal fine-tuned Code Llama integration
- Robust SQL cleaning and validation
- 3-tier SQL generation (Modal → OpenRouter → Pre-built)
- Temporal filtering and output format classification
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

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SeaTac Airport Operations Intelligence",
    description="AI-powered airport operations analysis - Modal Edition v7.0",
    version="7.0.0"
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
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'google/gemini-2.0-flash-exp:free')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Modal Configuration
MODAL_ENDPOINT = os.getenv('MODAL_ENDPOINT')
USE_MODAL_MODEL = os.getenv('USE_MODAL_MODEL', 'true').lower() == 'true'

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
# MODAL SQL GENERATOR (YOUR FINE-TUNED MODEL)
# ============================================================================

class ModalSQLGenerator:
    """Generate SQL using fine-tuned Code Llama on Modal"""
    
    def __init__(self, endpoint: Optional[str], enabled: bool):
        self.endpoint = endpoint
        self.enabled = enabled and bool(endpoint)
        self.timeout = 180  # 3 minutes for cold starts
        
        if self.enabled:
            print(f"✅ Modal Code Llama ENABLED")
            print(f"   Endpoint: {self.endpoint}")
            print(f"   Your fine-tuned model will be used first!")
        else:
            if enabled and not endpoint:
                print("⚠️  Modal enabled but MODAL_ENDPOINT not set")
                print("   Will use OpenRouter fallback")
            else:
                print("ℹ️  Modal Code Llama disabled (USE_MODAL_MODEL=false)")
    
    def _clean_sql(self, sql_text: str) -> str:
        """
        Aggressively clean SQL from LLM output.
        Handles numbered lists, markdown, explanations, multiple queries.
        """
        if not sql_text:
            return sql_text
    
        print("\n" + "=" * 80)
        print("🧹 CLEANING SQL")
        print("=" * 80)
        print("RAW INPUT:")
        print(sql_text[:500] + "..." if len(sql_text) > 500 else sql_text)
        print("=" * 80)
    
        # Step 1: Remove markdown code blocks
        sql_text = re.sub(r'```sql\s*', '', sql_text, flags=re.IGNORECASE)
        sql_text = re.sub(r'```\s*', '', sql_text)
    
        # Step 2: Remove numbered list markers at start
        sql_text = re.sub(r'^\s*\d+\.\s+', '', sql_text, flags=re.MULTILINE)
    
        # Step 3: Find ALL SELECT statements
        lines = sql_text.split('\n')
        select_indices = []
    
        for i, line in enumerate(lines):
            stripped = line.strip().upper()
            if stripped.startswith(('SELECT', 'WITH')):
                select_indices.append(i)
    
        if not select_indices:
            print("⚠️  No SELECT found, returning empty")
            return ""
    
        # If multiple SELECT statements, only use the FIRST complete one
        if len(select_indices) > 1:
            print(f"⚠️  Found {len(select_indices)} SELECT statements - keeping only the first")
    
        # Start from first SELECT
        sql_start_idx = select_indices[0]
        sql_lines = lines[sql_start_idx:]
    
        # Step 4: Find where FIRST query ends
        sql_end_idx = len(sql_lines)
    
        for i, line in enumerate(sql_lines):
            stripped = line.strip()
            stripped_upper = stripped.upper()
            stripped_lower = stripped.lower()
        
            # Stop at next SELECT statement (query #2)
            if i > 0 and stripped_upper.startswith(('SELECT', 'WITH')):
                print(f"⚠️  Stopping at second SELECT on line {i}")
                sql_end_idx = i
                break
        
            # Stop at explanation markers
            if any(marker in stripped_lower for marker in [
                'note:',
                'explanation:',
                'this query',
                'the above',
                'assumptions:',
                'this sql',
                'this will',
                'you can',
                'to use this',
                'replace',
                'modify',
                'example',
            ]):
                sql_end_idx = i
                break
        
            # Stop at numbered list items
            if re.match(r'^\s*\d+\.\s+[A-Z]', line):
                sql_end_idx = i
                break
    
        # Extract just the FIRST SQL query
        sql_lines = sql_lines[:sql_end_idx]
        sql_text = '\n'.join(sql_lines)
    
        # Step 5: Remove trailing semicolons and whitespace
        sql_text = sql_text.strip()
    
        if sql_text.endswith(';'):
            sql_text = sql_text[:-1].strip()
        
        # Step 6: Fix common column name spacing issues
    # "aircraft type" → "f.aircraft_type"
    # This happens when model doesn't include table prefix
    
    # Pattern: word1 word2 that should be word1_word2
        common_fixes = [
        (r'\baircraft\s+type\b', 'f.aircraft_type'),
        (r'\bevent\s+type\b', 'event_type'),
        (r'\bevent\s+time\b', 'event_time'),
        (r'\bcall\s+sign\b', 'call_sign'),
        (r'\bflight\s+count\b', 'flight_count'),
        (r'\bweight\s+class\b', 'weight_class'),
    ]
    
        for pattern, replacement in common_fixes:
            sql_text = re.sub(pattern, replacement, sql_text, flags=re.IGNORECASE)
    
        # Step 7: Clean trailing comments
        lines = sql_text.split('\n')
        cleaned_lines = []
    
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('--') and len(stripped) > 2:
                comment_text = stripped[2:].strip().lower()
                if any(word in comment_text for word in ['note', 'explanation', 'this', 'you can', 'replace']):
                    break
            cleaned_lines.append(line)
    
        sql_text = '\n'.join(cleaned_lines).strip()
    
        # Step 8: Ensure LIMIT clause
        if 'LIMIT' not in sql_text.upper():
            sql_text += ' LIMIT 100'
    
        # Step 9: Final validation
        if not sql_text.strip().upper().startswith(('SELECT', 'WITH', '(SELECT')):
            print("❌ Cleaned SQL doesn't start with SELECT!")
            print("CLEANED OUTPUT:")
            print(sql_text[:300])
            return ""
    
        # Step 10: Double-check no duplicate SELECT (should be impossible now)
        if sql_text.upper().count('SELECT') > 1:
            # Keep only up to first occurrence after the initial SELECT
            first_select_end = sql_text.upper().find('SELECT', 1)
            if first_select_end > 0:
                sql_text = sql_text[:first_select_end].strip()
                print(f"⚠️  Removed duplicate SELECT statements")
    
        print("✅ CLEANED OUTPUT:")
        print(sql_text[:300] + "..." if len(sql_text) > 300 else sql_text)
        print("=" * 80 + "\n")
    
        return sql_text
    
    async def generate_sql(self, query: str) -> Optional[str]:
        """Generate SQL using Modal endpoint"""
        if not self.enabled:
            return None
        
        try:
            print(f"🦙 [Modal] Calling fine-tuned Code Llama...")
            print(f"   Query: {query[:80]}...")
            
            response = requests.post(
                self.endpoint,
                json={"query": query},
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                print(f"❌ [Modal] HTTP {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return None
            
            result = response.json()
            
            if result.get('error'):
                print(f"❌ [Modal] Error: {result['error']}")
                return None
            
            raw_sql = result.get('sql', '').strip()
            
            if not raw_sql:
                print("❌ [Modal] Empty SQL returned")
                return None
            
            print("\n" + "=" * 80)
            print("🦙 RAW MODAL OUTPUT")
            print("=" * 80)
            print(raw_sql)
            print("=" * 80)
            
            # Clean the SQL
            sql = self._clean_sql(raw_sql)
            
            # Validate SQL
            if not sql or 'SELECT' not in sql.upper():
                print(f"❌ [Modal] Invalid SQL after cleaning")
                return None
            
            print(f"✅ [Modal] Generated SQL successfully")
            print(f"   Raw length: {len(raw_sql)} chars")
            print(f"   Cleaned length: {len(sql)} chars")
            
            return sql
            
        except requests.Timeout:
            print(f"⏱️  [Modal] Timeout after {self.timeout}s")
            print("   Endpoint may be cold-starting (first request takes longer)")
            return None
        except requests.ConnectionError as e:
            print(f"❌ [Modal] Connection failed")
            print(f"   Is Modal inference server deployed?")
            return None
        except Exception as e:
            print(f"❌ [Modal] Exception: {type(e).__name__}: {e}")
            return None


# ============================================================================
# SQL VALIDATOR
# ============================================================================

class SQLValidator:
    """Validate and fix common SQL generation errors"""
    
    @staticmethod
    def validate_and_fix(sql: str, query: str) -> Dict[str, Any]:
        """Validate and fix common SQL generation errors"""
    
    @staticmethod
    def validate_and_fix(sql: str, query: str) -> Dict[str, Any]:
        """Validate SQL and attempt to fix common errors"""
        warnings = []
        
        # Pre-check: SQL must not be empty
        if not sql or not sql.strip():
            return {
                'valid': False,
                'sql': sql,
                'fixed': False,
                'warnings': ['❌ Empty SQL generated'],
                'error': 'SQL generation produced empty output'
            }
        
        # Pre-check: Must start with SELECT/WITH
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith(('SELECT', 'WITH', '(SELECT')):
            return {
                'valid': False,
                'sql': sql,
                'fixed': False,
                'warnings': ['❌ SQL does not start with SELECT'],
                'error': f'Invalid SQL start: {sql[:50]}...'
            }
        
        # Check 0: Detect instructional text instead of SQL
        # Look for common instruction phrases in the first line
        first_line = sql.split('\n')[0].strip().lower()
        instruction_keywords = [
            'select the appropriate',
            'choose the',
            'based on the',
            'use the following',
            'according to',
            'refer to',
            'consider the',
            'determine the',
            'identify the',
            'find the',
        ]
        
        for keyword in instruction_keywords:
            if keyword in first_line and 'from' not in first_line.lower():
                warnings.append(f"❌ SQL contains instructional text: '{first_line[:60]}'")
                return {
                    'valid': False,
                    'sql': sql,
                    'fixed': False,
                    'warnings': warnings,
                    'error': 'Model generated instructions instead of SQL query'
                }
        
        # Check 1: Must have FROM clause
        if 'FROM' not in sql_upper:
            warnings.append("❌ SQL missing FROM clause")
            return {
                'valid': False,
                'sql': sql,
                'fixed': False,
                'warnings': warnings,
                'error': 'SQL must include FROM clause'
            }
        
        # Check 2: Detect NOW() usage
        if 'NOW()' in sql_upper:
            warnings.append("❌ CRITICAL: SQL uses NOW() - incorrect for taxi time calculations!")
            return {
                'valid': False,
                'sql': sql,
                'fixed': False,
                'warnings': warnings,
                'error': 'SQL uses NOW() which is incorrect for taxi time calculations'
            }
        
        # Check 3: Detect numbered lists
        if re.search(r'^\s*\d+\.\s+', sql, re.MULTILINE):
            warnings.append("❌ SQL contains numbered list markers")
            return {
                'valid': False,
                'sql': sql,
                'fixed': False,
                'warnings': warnings,
                'error': 'SQL contains numbered lists or formatting artifacts'
            }
        
        # Check 4: WHERE before FROM (syntax error)
        from_pos = sql_upper.find('FROM')
        where_pos = sql_upper.find('WHERE')
        if where_pos != -1 and where_pos < from_pos:
            warnings.append("❌ WHERE clause appears before FROM clause")
            return {
                'valid': False,
                'sql': sql,
                'fixed': False,
                'warnings': warnings,
                'error': 'Invalid SQL structure: WHERE before FROM'
            }
        
        # Check 5: Taxi-in validation
        if 'taxi' in query.lower() and 'in' in query.lower():
            has_landing = 'Actual_Landing' in sql
            has_inblock = 'Actual_In_Block' in sql
            
            if not (has_landing and has_inblock):
                warnings.append("❌ Taxi-in query missing proper events")
                return {
                    'valid': False,
                    'sql': sql,
                    'fixed': False,
                    'warnings': warnings,
                    'error': 'Taxi-in calculation must use Actual_Landing and Actual_In_Block events'
                }
        
        # Check 6: Taxi-out validation
        if 'taxi' in query.lower() and 'out' in query.lower():
            has_offblock = 'Actual_Off_Block' in sql
            has_takeoff = 'Actual_Take_Off' in sql
            
            if not (has_offblock and has_takeoff):
                warnings.append("❌ Taxi-out query missing proper events")
                return {
                    'valid': False,
                    'sql': sql,
                    'fixed': False,
                    'warnings': warnings,
                    'error': 'Taxi-out calculation must use Actual_Off_Block and Actual_Take_Off events'
                }
        
        # Check 7: Dangerous operations
        dangerous = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        for op in dangerous:
            if op in sql_upper.split():
                warnings.append(f"❌ Dangerous operation detected: {op}")
                return {
                    'valid': False,
                    'sql': sql,
                    'fixed': False,
                    'warnings': warnings,
                    'error': f'Dangerous SQL operation not allowed: {op}'
                }
        
        # Check 8: Basic structure validation
        # SQL should have reasonable structure
        has_select = 'SELECT' in sql_upper
        has_from = 'FROM' in sql_upper
        
        if not (has_select and has_from):
            warnings.append("❌ SQL missing basic structure (SELECT...FROM)")
            return {
                'valid': False,
                'sql': sql,
                'fixed': False,
                'warnings': warnings,
                'error': 'SQL must have SELECT...FROM structure'
            }
        
        # All checks passed
        warnings.append('✅ SQL validation passed')
        return {
            'valid': True,
            'sql': sql,
            'fixed': False,
            'warnings': warnings
        }


# ============================================================================
# ENHANCEMENT 1: Temporal Context Extraction
# ============================================================================

class TemporalContextExtractor:
    """Extract temporal filters from natural language queries"""
    
    def __init__(self):
        self.time_periods = {
            'morning': (5, 11),
            'early morning': (5, 8),
            'late morning': (9, 11),
            'afternoon': (12, 17),
            'early afternoon': (12, 14),
            'late afternoon': (15, 17),
            'evening': (18, 21),
            'night': (22, 4),
            'midnight': (0, 2),
            'noon': (11, 13),
            'rush hour': [(7, 9), (16, 18)],
            'peak hours': [(7, 9), (16, 18)],
            'business hours': (9, 17),
            'overnight': (22, 5)
        }
    
    def extract_temporal_context(self, query: str) -> Dict:
        """Extract all temporal filters from query"""
        query_lower = query.lower().strip()
        context = {
            'hour_range': None,
            'specific_hours': None,
            'has_temporal_filter': False
        }
        
        # Extract named periods
        for period_name, hours in self.time_periods.items():
            if period_name in query_lower:
                context['has_temporal_filter'] = True
                context['hour_range'] = hours
                break
        
        # Extract specific hours
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
        
        # Extract ranges
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
        """Inject temporal WHERE clauses into existing SQL"""
        temporal_context = self.extract_temporal_context(user_query)
    
        if not temporal_context['has_temporal_filter']:
            return sql_query
    
        # Check if SQL already has time filtering (avoid duplicate)
        sql_upper = sql_query.upper()
        sql_lower = sql_query.lower()
    
        if 'TIME(' in sql_upper or 'BETWEEN' in sql_upper and ("'14:" in sql_query or "'17:" in sql_query):
            print("⏭️  SQL already has time filtering, skipping temporal injection")
            return sql_query
    
        # CRITICAL: Check if event_time column exists in this query
        # Don't inject temporal filters on queries that don't use flight_event table
        if 'event_time' not in sql_lower:
            print("⏭️  Query doesn't use event_time column, skipping temporal injection")
            return sql_query
    
        # Detect which time column/alias to use based on what's in the query
        time_column = None
    
        # Check for common alias patterns
        if 'offblock.event_time' in sql_lower:
            time_column = 'offblock.event_time'
        elif 'landing.event_time' in sql_lower:
            time_column = 'landing.event_time'
        elif 'takeoff.event_time' in sql_lower:
            time_column = 'takeoff.event_time'
        elif 'inblock.event_time' in sql_lower:
            time_column = 'inblock.event_time'
        elif 'le.event_time' in sql_lower:
            time_column = 'le.event_time'
        elif 'ib.event_time' in sql_lower:
            time_column = 'ib.event_time'
        elif 'fe.event_time' in sql_lower:
            time_column = 'fe.event_time'
        elif 'flight_event' in sql_lower and 'event_time' in sql_lower:
        # Query uses flight_event table but no specific alias
        # Try to find what alias is used for flight_event
        # Look for patterns like "JOIN flight_event X ON" or "FROM flight_event X"
            import re
            alias_match = re.search(r'flight_event\s+(\w+)\s+ON', sql_lower)
            if alias_match:
                alias = alias_match.group(1)
                time_column = f'{alias}.event_time'
            else:
                time_column = 'event_time'
        else:
            print("⚠️  No usable event_time column found in query, skipping temporal filter")
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
    
        # Find WHERE clause position
        if 'WHERE' in sql_upper:
            where_pos = sql_upper.find('WHERE') + 5
            next_clause_pos = len(sql_query)
            for clause in ['GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT']:
                pos = sql_upper.find(clause, where_pos)
                if pos != -1 and pos < next_clause_pos:
                    next_clause_pos = pos
        
            before = sql_query[:next_clause_pos].rstrip()
            after = sql_query[next_clause_pos:]
        
            print(f"✅ Injecting temporal filter: {where_clause}")
            return f"{before}\n     AND {where_clause}{after}"
        else:
            group_by_pos = sql_upper.find('GROUP BY')
            if group_by_pos != -1:
                before = sql_query[:group_by_pos].rstrip()
                after = sql_query[group_by_pos:]
                print(f"✅ Injecting temporal filter: {where_clause}")
                return f"{before}\nWHERE {where_clause}\n{after}"
        
            print(f"✅ Injecting temporal filter: {where_clause}")
            return f"{sql_query.rstrip()}\nWHERE {where_clause}"


# ============================================================================
# ENHANCEMENT 2: Output Format Classification
# ============================================================================

OutputFormat = Literal['chart', 'table', 'text', 'chart_and_text', 'table_and_text', 'all']

@dataclass
class OutputPreference:
    """User's preferred output format"""
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
    """Classify user's desired output format"""
    
    def __init__(self):
        self.format_indicators = {
            'chart': {
                'explicit': ['show me a chart', 'create a chart', 'visualize', 'plot', 'graph'],
                'implicit': ['trends', 'pattern', 'over time', 'by hour', 'compare']
            },
            'table': {
                'explicit': ['show me a table', 'list', 'show me the data', 'table view'],
                'implicit': ['which flights', 'what are the', 'individual', 'details']
            },
            'text': {
                'explicit': ['just tell me', 'summarize', 'summary', 'brief', 'no chart'],
                'implicit': ['how many', 'what was', 'average', 'total']
            }
        }
    
    def classify(self, query: str, intent: str = None) -> OutputPreference:
        """Classify desired output format"""
        query_lower = query.lower().strip()
        
        scores = {'chart': 0.0, 'table': 0.0, 'text': 0.0}
        reasons = []
        
        for format_type, indicators in self.format_indicators.items():
            for phrase in indicators['explicit']:
                if phrase in query_lower:
                    scores[format_type] += 3.0
                    reasons.append(f"'{phrase}' → {format_type}")
            
            for phrase in indicators['implicit']:
                if phrase in query_lower:
                    scores[format_type] += 0.5
        
        if all(s < 1.0 for s in scores.values()):
            if any(kw in query_lower for kw in ['hour', 'time', 'compare', 'trend']):
                return OutputPreference('chart_and_text', 0.6, "Default: analytical query")
            return OutputPreference('text', 0.5, "Default: simple query")
        
        max_score = max(scores.values())
        top_formats = [f for f, s in scores.items() if s >= max_score * 0.7]
        confidence = min(max_score / 5.0, 1.0)
        
        if len(top_formats) == 1:
            final = top_formats[0]
        elif 'chart' in top_formats and 'text' in top_formats:
            final = 'chart_and_text'
        elif 'table' in top_formats and 'text' in top_formats:
            final = 'table_and_text'
        else:
            final = top_formats[0]
        
        return OutputPreference(final, confidence, ' | '.join(reasons[:3]))


# ============================================================================
# DETAILED SCHEMA
# ============================================================================

DETAILED_SCHEMA = """
DATABASE SCHEMA - SeaTac Airport Operations (MySQL)

=== TABLE 1: flight ===
- call_sign (VARCHAR) - Flight identifier, USE FOR JOINS - Examples: 'QTA1', 'ZYX966'
- aircraft_type (VARCHAR) - Aircraft model - Examples: 'B738', 'A321'
- flight_number (VARCHAR) - Flight number
- operation (ENUM) - 'ARRIVAL' or 'DEPARTURE'

=== TABLE 2: flight_event ===
- call_sign (VARCHAR) - Links to flight.call_sign - USE FOR JOINS
- operation (ENUM) - 'ARRIVAL' or 'DEPARTURE' - MUST MATCH
- event_type (VARCHAR) - 'Actual_Off_Block', 'Actual_Take_Off', 'Actual_Landing', 'Actual_In_Block'
- event_time (DATETIME) - When event occurred
- location (VARCHAR) - Gate, runway, or taxiway

=== TABLE 3: aircraft_type ===
- aircraft_type (VARCHAR, PRIMARY KEY)
- weight_class (VARCHAR) - 'L', 'M', 'H'
- wingspan_ft, wingspan_m (DECIMAL)

=== TAXI TIME CALCULATIONS ===
Taxi-out: TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time)
Taxi-in: TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)

ALWAYS filter: BETWEEN 1 AND 120
NEVER use NOW() - always calculate between actual event times
"""


# ============================================================================
# OpenRouter LLM Wrapper
# ============================================================================

class OpenRouterLLM:
    """Wrapper for OpenRouter API"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
    
    async def ainvoke(self, messages, temperature: float = 0.1):
        """Call OpenRouter API"""
        if isinstance(messages, list):
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    formatted_messages.append(msg)
                else:
                    formatted_messages.append({"role": "user", "content": str(msg)})
        else:
            formatted_messages = [{"role": "user", "content": str(messages)}]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "SeaTac Operations Intelligence"
        }
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.status_code}")
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        class Response:
            def __init__(self, text):
                self.content = text
        
        return Response(content)


# Initialize Modal and OpenRouter
modal_generator = ModalSQLGenerator(MODAL_ENDPOINT, USE_MODAL_MODEL)
llm = OpenRouterLLM(OPENROUTER_API_KEY, OPENROUTER_MODEL) if OPENROUTER_API_KEY else None


# ============================================================================
# Pydantic Models
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
    openrouter_enabled: bool
    database_status: str
    version: str


# ============================================================================
# Database Manager
# ============================================================================

class DatabaseManager:
    """Database connection and query execution"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'aiplane')
        }
    
    def test_connection(self) -> bool:
        try:
            connection = mysql.connector.connect(**self.db_config)
            connection.close()
            return True
        except Error:
            return False
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        # LOG THE FULL SQL BEFORE EXECUTION
        print("\n" + "=" * 80)
        print("📋 EXECUTING SQL QUERY")
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
            
            # Convert Decimal, datetime, date
            cleaned_results = []
            for row in results:
                cleaned_row = {}
                for key, value in row.items():
                    if isinstance(value, Decimal):
                        cleaned_row[key] = float(value)
                    elif isinstance(value, datetime):
                        cleaned_row[key] = value.isoformat()
                    elif isinstance(value, date):
                        cleaned_row[key] = value.isoformat()
                    elif value is None:
                        cleaned_row[key] = None
                    else:
                        try:
                            json.dumps(value)
                            cleaned_row[key] = value
                        except (TypeError, ValueError):
                            cleaned_row[key] = str(value)
                cleaned_results.append(cleaned_row)
            
            print(f"✅ Query executed successfully: {len(cleaned_results)} rows returned\n")
            
            return {
                'success': True,
                'data': cleaned_results,
                'row_count': len(cleaned_results),
                'sql': sql
            }
        except Error as e:
            print(f"❌ SQL EXECUTION FAILED")
            print(f"Error: {str(e)}\n")
            return {
                'success': False,
                'error': str(e),
                'sql': sql,
                'row_count': 0,
                'data': []
            }


# ============================================================================
# Chart Generator
# ============================================================================

class ChartGenerator:
    """Generate Chart.js configurations"""
    
    @staticmethod
    def generate_chart(data: List[Dict], title: str = "Analysis Results") -> Optional[Dict]:
        if not data:
            return None
        
        first_row = data[0]
        keys = list(first_row.keys())
        
        # Detect label and value fields
        label_field = None
        value_field = None
        
        for key in keys:
            if any(term in key.lower() for term in ['hour', 'type', 'class', 'runway']):
                label_field = key
                break
        
        for key in keys:
            if any(term in key.lower() for term in ['avg', 'count', 'total', 'minutes']):
                if value_field is None and 'avg' in key.lower():
                    value_field = key
        
        if not label_field:
            label_field = keys[0]
        if not value_field:
            value_field = keys[1] if len(keys) > 1 else keys[0]
        
        # Format labels
        labels = []
        for row in data[:24]:
            label = str(row.get(label_field, ''))
            if 'hour' in label_field.lower() and label.isdigit():
                label = f"{int(label):02d}:00"
            labels.append(label)
        
        values = [float(row.get(value_field, 0)) for row in data[:24]]
        
        is_temporal = 'hour' in label_field.lower()
        chart_type = 'line' if is_temporal else 'bar'
        
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
                        'title': {'display': True, 'text': 'Minutes'}
                    }
                }
            }
        }


# ============================================================================
# Enhanced Response Controller
# ============================================================================

class EnhancedResponseController:
    """Controls output generation"""
    
    def __init__(self, format_classifier: OutputFormatClassifier, chart_generator: ChartGenerator):
        self.classifier = format_classifier
        self.chart_generator = chart_generator
    
    def process_output(self, user_query: str, result: Dict, insights: str, 
                      title: str = "Analysis Results") -> Dict:
        """Process and format output"""
        output_pref = self.classifier.classify(user_query)
        
        print(f"[Output] Format: {output_pref.format} (confidence: {output_pref.confidence:.2f})")
        
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
# Enhanced SeaTac Agent (Modal-First)
# ============================================================================

class SeaTacAgent:
    """Enhanced agent with Modal Code Llama integration"""
    
    def __init__(self, modal_gen: ModalSQLGenerator, openrouter_llm, db_manager: DatabaseManager):
        self.modal_gen = modal_gen
        self.llm = openrouter_llm
        self.db_manager = db_manager
        self.chart_generator = ChartGenerator()
        self.temporal_extractor = TemporalContextExtractor()
        self.format_classifier = OutputFormatClassifier()
        self.response_controller = EnhancedResponseController(self.format_classifier, self.chart_generator)
        self.sql_validator = SQLValidator()
        
        # Pre-built use cases (fallback)
        self.use_cases = {
            "1": {
                "name": "Taxi-In Performance by Aircraft Type",
                "keywords": ["taxi-in", "taxi in", "aircraft type"],
                "sql": """
                    SELECT f.aircraft_type, 
                           at.weight_class,
                           AVG(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as avg_taxi_in_minutes,
                           MIN(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as min_taxi_in,
                           MAX(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as max_taxi_in,
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
                    HAVING flight_count >= 2
                    ORDER BY avg_taxi_in_minutes DESC
                    LIMIT 20
                """
            },
            "2": {
                "name": "Taxi-Out Performance by Hour",
                "keywords": ["taxi-out", "taxi out", "hour", "hourly"],
                "sql": """
                    SELECT HOUR(offblock.event_time) as hour_of_day,
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
                    ORDER BY hour_of_day
                """
            }
        }
    
    def _clean_sql(self, sql_text: str) -> str:
        """
        Aggressively clean SQL from LLM output.
        Handles numbered lists, markdown, explanations, multiple queries.
        """
        if not sql_text:
            return sql_text
    
        print("\n" + "=" * 80)
        print("🧹 CLEANING SQL")
        print("=" * 80)
        print("RAW INPUT:")
        print(sql_text[:500] + "..." if len(sql_text) > 500 else sql_text)
        print("=" * 80)
    
        # Step 1: Remove markdown code blocks
        sql_text = re.sub(r'```sql\s*', '', sql_text, flags=re.IGNORECASE)
        sql_text = re.sub(r'```\s*', '', sql_text)
    
        # Step 2: Remove numbered list markers at start
        sql_text = re.sub(r'^\s*\d+\.\s+', '', sql_text, flags=re.MULTILINE)
    
        # Step 3: Find ALL SELECT statements
        lines = sql_text.split('\n')
        select_indices = []
    
        for i, line in enumerate(lines):
            stripped = line.strip().upper()
            if stripped.startswith(('SELECT', 'WITH')):
                select_indices.append(i)
    
        if not select_indices:
            print("⚠️  No SELECT found, returning empty")
            return ""
    
        # If multiple SELECT statements, only use the FIRST complete one
        if len(select_indices) > 1:
            print(f"⚠️  Found {len(select_indices)} SELECT statements - keeping only the first")
    
        # Start from first SELECT
        sql_start_idx = select_indices[0]
        sql_lines = lines[sql_start_idx:]
    
        # Step 4: Find where FIRST query ends
        sql_end_idx = len(sql_lines)
    
        for i, line in enumerate(sql_lines):
            stripped = line.strip()
            stripped_upper = stripped.upper()
            stripped_lower = stripped.lower()
        
            # Stop at next SELECT statement (query #2)
            if i > 0 and stripped_upper.startswith(('SELECT', 'WITH')):
                print(f"⚠️  Stopping at second SELECT on line {i}")
                sql_end_idx = i
                break
        
            # Stop at explanation markers
            if any(marker in stripped_lower for marker in [
                'note:',
                'explanation:',
                'this query',
                'the above',
                'assumptions:',
                'this sql',
                'this will',
                'you can',
                'to use this',
                'replace',
                'modify',
                'example',
            ]):
                sql_end_idx = i
                break
        
            # Stop at numbered list items
            if re.match(r'^\s*\d+\.\s+[A-Z]', line):
                sql_end_idx = i
                break
    
        # Extract just the FIRST SQL query
        sql_lines = sql_lines[:sql_end_idx]
        sql_text = '\n'.join(sql_lines)

        # Step 5: Fix common column name spacing issues
    # "aircraft type" → "f.aircraft_type"
    # This happens when model doesn't include table prefix
    
    # Pattern: word1 word2 that should be word1_word2
        common_fixes = [
        (r'\baircraft\s+type\b', 'f.aircraft_type'),
        (r'\bevent\s+type\b', 'event_type'),
        (r'\bevent\s+time\b', 'event_time'),
        (r'\bcall\s+sign\b', 'call_sign'),
        (r'\bflight\s+count\b', 'flight_count'),
        (r'\bweight\s+class\b', 'weight_class'),
    ]
    
        for pattern, replacement in common_fixes:
            sql_text = re.sub(pattern, replacement, sql_text, flags=re.IGNORECASE)
    
        # Step 6: Remove trailing semicolons and whitespace
        sql_text = sql_text.strip()
    
        if sql_text.endswith(';'):
            sql_text = sql_text[:-1].strip()
    
        # Step 7: Clean trailing comments
        lines = sql_text.split('\n')
        cleaned_lines = []
    
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('--') and len(stripped) > 2:
                comment_text = stripped[2:].strip().lower()
                if any(word in comment_text for word in ['note', 'explanation', 'this', 'you can', 'replace']):
                    break
            cleaned_lines.append(line)
    
        sql_text = '\n'.join(cleaned_lines).strip()
    
        # Step 8: Ensure LIMIT clause
        if 'LIMIT' not in sql_text.upper():
            sql_text += ' LIMIT 100'
    
        # Step 9: Final validation
        if not sql_text.strip().upper().startswith(('SELECT', 'WITH', '(SELECT')):
            print("❌ Cleaned SQL doesn't start with SELECT!")
            print("CLEANED OUTPUT:")
            print(sql_text[:300])
            return ""
    
        # Step 10: Double-check no duplicate SELECT (should be impossible now)
        if sql_text.upper().count('SELECT') > 1:
            # Keep only up to first occurrence after the initial SELECT
            first_select_end = sql_text.upper().find('SELECT', 1)
            if first_select_end > 0:
                sql_text = sql_text[:first_select_end].strip()
                print(f"⚠️  Removed duplicate SELECT statements")
    
        print("✅ CLEANED OUTPUT:")
        print(sql_text[:300] + "..." if len(sql_text) > 300 else sql_text)
        print("=" * 80 + "\n")
    
        return sql_text
    
    async def generate_sql(self, query: str) -> Dict[str, Any]:
        """Generate SQL with 3-tier hierarchy + validation"""
        
        print("\n" + "=" * 80)
        print("🔍 SQL GENERATION PROCESS")
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80 + "\n")
        
        # TIER 1: Try Modal
        if self.modal_gen.enabled:
            print("📍 TIER 1: Trying Modal fine-tuned Code Llama...")
            modal_sql = await self.modal_gen.generate_sql(query)
            if modal_sql:
                # VALIDATE
                validation = self.sql_validator.validate_and_fix(modal_sql, query)
                
                print("\n" + "=" * 80)
                print("🔍 SQL VALIDATION")
                print("=" * 80)
                for warning in validation['warnings']:
                    print(warning)
                print("=" * 80 + "\n")
                
                if validation['valid']:
                    print(f"✅ [TIER 1] Using Modal fine-tuned Code Llama\n")
                    return {
                        'success': True,
                        'sql': validation['sql'],
                        'source': 'modal'
                    }
                else:
                    print(f"❌ [TIER 1] Modal SQL failed validation: {validation.get('error')}")
                    print(f"⚠️  Falling back to TIER 2...\n")
        
        # TIER 2: Try OpenRouter
        if self.llm:
            try:
                print("📍 TIER 2: Trying OpenRouter...")
                
                sql_prompt = f"""{DETAILED_SCHEMA}

Generate SQL for: "{query}"

CRITICAL RULES:
- NEVER use NOW() - always calculate between actual event times
- For taxi-in: TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)
- For taxi-out: TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time)
- Always filter: BETWEEN 1 AND 120 minutes

Return ONLY the SQL query, no explanations.
"""
                
                response = await self.llm.ainvoke(
                    [{"role": "user", "content": sql_prompt}],
                    temperature=0.05
                )
                
                sql = self._clean_sql(response.content)
                
                # VALIDATE
                validation = self.sql_validator.validate_and_fix(sql, query)
                
                print("\n" + "=" * 80)
                print("🔍 SQL VALIDATION")
                print("=" * 80)
                for warning in validation['warnings']:
                    print(warning)
                print("=" * 80 + "\n")
                
                if validation['valid'] and 'SELECT' in sql.upper():
                    print(f"✅ [TIER 2] Using OpenRouter\n")
                    return {
                        'success': True,
                        'sql': validation['sql'],
                        'source': 'openrouter'
                    }
                else:
                    print(f"❌ [TIER 2] OpenRouter SQL failed validation")
                
            except Exception as e:
                print(f"❌ [TIER 2] OpenRouter error: {e}\n")
        
        # TIER 3: Pre-built SQL
        print("📍 TIER 3: Using pre-built SQL (fallback)")
        prebuilt = self._get_prebuilt_sql(query)
        print(f"✅ [TIER 3] Using pre-built SQL\n")
        return {
            'success': True,
            'sql': prebuilt,
            'source': 'prebuilt'
        }
    
    def _get_prebuilt_sql(self, query: str) -> str:
        """Get pre-built SQL based on query keywords"""
        query_lower = query.lower()
        
        for use_case_id, use_case in self.use_cases.items():
            if any(kw in query_lower for kw in use_case['keywords']):
                return use_case['sql'].strip()
        
        return "SELECT * FROM flight LIMIT 10"
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with all enhancements"""
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
                'input': query,
                'observation': f"Generated via {sql_source}"
            })
            
            # Apply temporal filtering
            enhanced_sql = self.temporal_extractor.inject_temporal_filter(sql_query, query)
            
            if enhanced_sql != sql_query:
                reasoning_steps.append({
                    'action': 'apply_temporal_filter',
                    'input': query,
                    'observation': 'Applied time-based filtering'
                })
            
            # Execute SQL
            result = self.db_manager.execute_query(enhanced_sql)
            
            reasoning_steps.append({
                'action': 'execute_sql',
                'input': enhanced_sql[:100] + '...',
                'observation': f"Retrieved {result.get('row_count', 0)} rows"
            })
            
            if not result['success']:
                return {
                    'success': False,
                    'answer': f"SQL execution failed: {result.get('error')}",
                    'reasoning_steps': reasoning_steps,
                    'sql_queries': [enhanced_sql],
                    'sql_source': sql_source
                }
            
            if not result['data']:
                return {
                    'success': True,
                    'answer': "No data found for your query.",
                    'reasoning_steps': reasoning_steps,
                    'sql_queries': [enhanced_sql],
                    'row_count': 0,
                    'sql_source': sql_source
                }
            
            # Generate insights
            insights = await self._generate_insights(query, result['data'])
            
            reasoning_steps.append({
                'action': 'generate_insights',
                'input': f"{len(result['data'])} rows",
                'observation': insights[:100] + '...'
            })
            
            # Process output format
            output_data = self.response_controller.process_output(
                user_query=query,
                result=result,
                insights=insights,
                title="SeaTac Operations Analysis"
            )
            
            return {
                'success': True,
                'answer': output_data.get('message', insights),
                'use_case': 'Dynamic Analysis',
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
                'reasoning_steps': reasoning_steps if 'reasoning_steps' in locals() else []
            }
    
    async def _generate_insights(self, query: str, data: List[Dict]) -> str:
        """Generate insights from results"""
        if not data:
            return "No data available."
        
        # Calculate statistics
        numeric_fields = {}
        for key in data[0].keys():
            if any(term in key.lower() for term in ['avg', 'min', 'max', 'count']):
                try:
                    values = [float(row.get(key, 0)) for row in data if row.get(key) is not None]
                    if values:
                        numeric_fields[key] = {
                            'avg': sum(values) / len(values),
                            'min': min(values),
                            'max': max(values)
                        }
                except:
                    pass
        
        if not self.llm:
            return f"Found {len(data)} results."
        
        insights_prompt = f"""Analyze this airport operations data and provide 2-3 sentences with specific numbers.

QUESTION: "{query}"
RECORDS: {len(data)}

SAMPLE DATA:
{json.dumps(data[:3], indent=2, cls=DecimalEncoder)}

STATISTICS:
{json.dumps(numeric_fields, indent=2, cls=DecimalEncoder)}

Provide clear insights with specific numbers:
"""
        
        try:
            response = await self.llm.ainvoke(
                [{"role": "user", "content": insights_prompt}],
                temperature=0.4
            )
            return response.content.strip()
        except:
            return f"Found {len(data)} results with various taxi time patterns."


# ============================================================================
# Initialize
# ============================================================================

db_manager = DatabaseManager()
agent_system = SeaTacAgent(modal_generator, llm, db_manager) if (modal_generator or llm) else None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "SeaTac Operations Intelligence - Modal Edition v7.0",
        "version": "7.0.0",
        "modal_enabled": modal_generator.enabled,
        "openrouter_model": OPENROUTER_MODEL,
        "sql_hierarchy": [
            "1. Modal fine-tuned Code Llama (your custom model)",
            "2. OpenRouter (general purpose fallback)",
            "3. Pre-built SQL (guaranteed fallback)"
        ]
    }


@app.post("/api/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Process queries - uses Modal fine-tuned model first"""
    try:
        if not request.query or len(request.query.strip()) < 2:
            raise HTTPException(status_code=400, detail="Please provide a valid query")
        
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
            sql_source=result.get('sql_source')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model=OPENROUTER_MODEL,
        modal_enabled=modal_generator.enabled,
        modal_endpoint=MODAL_ENDPOINT if modal_generator.enabled else None,
        openrouter_enabled=bool(llm),
        database_status="online" if db_manager.test_connection() else "offline",
        version="7.0.0"
    )


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 80)
    print("✈️  SeaTac Operations Intelligence - MODAL EDITION v7.0")
    print("=" * 80)
    print(f"Version: 7.0.0")
    
    if modal_generator.enabled:
        print(f"🦙 Modal Code Llama: ENABLED ✅")
        print(f"   Endpoint: {MODAL_ENDPOINT}")
        print(f"   Status: Your fine-tuned model (PRIMARY)")
    else:
        print(f"🦙 Modal Code Llama: DISABLED")
    
    if llm:
        print(f"🌐 OpenRouter: Configured ✅ (FALLBACK)")
        print(f"   Model: {OPENROUTER_MODEL}")
    else:
        print(f"🌐 OpenRouter: Not configured")
    
    print(f"🗄️  Database: {db_manager.db_config['database']}@{db_manager.db_config['host']}")
    if db_manager.test_connection():
        print(f"   Status: Connected ✅")
    else:
        print(f"   Status: Disconnected ❌")
    
    print("\n📋 SQL Generation Hierarchy:")
    print("   1️⃣  Modal fine-tuned Code Llama (your custom model)")
    print("   2️⃣  OpenRouter (general purpose)")
    print("   3️⃣  Pre-built SQL (guaranteed fallback)")
    
    print("\n✨ Features v7.0:")
    print("   ✓ Modal fine-tuned Code Llama integration")
    print("   ✓ Robust SQL cleaning and validation")
    print("   ✓ Temporal filtering (afternoon, morning, time ranges)")
    print("   ✓ Output format classification (chart/table/text)")
    print("   ✓ Smart 3-tier SQL generation")
    
    print("\n🚀 Server starting on http://0.0.0.0:8000")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)