from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import mysql.connector
from mysql.connector import Error
import json
import re
from datetime import datetime
from decimal import Decimal
import os
from dotenv import load_dotenv
import uvicorn

# LangChain imports for version 1.2.10
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SeaTac Airport Operations Intelligence",
    description="AI-powered airport operations analysis with all 11 SeaTac use cases",
    version="5.0.0"
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
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

# Custom JSON encoder for Decimal and datetime types
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DecimalEncoder, self).default(obj)

# Initialize LangChain LLM
if GEMINI_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        print(f"✅ LangChain LLM initialized: {GEMINI_MODEL}")
    except Exception as e:
        print(f"❌ LangChain init error: {e}")
        llm = None
else:
    llm = None
    print("❌ No GEMINI_API_KEY found")

# Database Schema Context
SCHEMA_CONTEXT = """
DATABASE SCHEMA - SeaTac Airport Operations (KSEA)

Tables:
1. flight - Main flight information
   - call_sign (VARCHAR): Flight identifier (JOIN key)
   - aircraft_type (VARCHAR): B737, B738, B739, A320, A321, etc.
   - operation (ENUM): 'ARRIVAL' or 'DEPARTURE'
   - schedule_departure, actual_departure, schedule_arrival, actual_arrival (DATETIME)

2. flight_event - Timeline of operational events
   - call_sign (VARCHAR): Links to flight (JOIN key)
   - event_type (ENUM): Actual_Landing, Actual_Take_Off, Actual_Off_Block, Actual_In_Block, etc.
   - event_time (DATETIME)
   - location (VARCHAR): Gate_A1, 34L, 34R, TaxiwaySegment_B_28, etc.

3. aircraft_type - Aircraft specifications
   - aircraft_type (VARCHAR PRIMARY KEY)
   - weight_class (VARCHAR): L (Light), M (Medium), H (Heavy)
   - wingspan_ft, wingspan_m (INT)
"""


# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    use_case_number: Optional[int] = Field(None, ge=1, le=11)
    max_iterations: Optional[int] = Field(8)

class QueryResponse(BaseModel):
    success: bool
    message: str
    use_case: Optional[str] = None
    agent_reasoning: Optional[List[Dict[str, str]]] = None
    row_count: int = 0
    sql_queries: Optional[List[str]] = None
    data: Optional[List[Dict[str, Any]]] = None
    insights: Optional[List[str]] = None
    visualization_suggestion: Optional[str] = None
    chart: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    model: str
    langchain_enabled: bool
    agent_enabled: bool
    seatac_use_cases_supported: int
    tools_available: List[str]


class DatabaseManager:
    """Database connection and query execution with Decimal handling"""
    
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
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            connection.close()
            
            # Convert Decimal and datetime to JSON-serializable types
            cleaned_results = []
            for row in results:
                cleaned_row = {}
                for key, value in row.items():
                    if isinstance(value, Decimal):
                        cleaned_row[key] = float(value)
                    elif isinstance(value, datetime):
                        cleaned_row[key] = value.isoformat()
                    elif value is None:
                        cleaned_row[key] = None
                    else:
                        cleaned_row[key] = value
                cleaned_results.append(cleaned_row)
            
            return {
                'success': True,
                'data': cleaned_results,
                'row_count': len(cleaned_results),
                'sql': sql
            }
        except Error as e:
            return {
                'success': False,
                'error': str(e),
                'sql': sql,
                'row_count': 0,
                'data': []
            }


# LangChain Tool Classes
class SeaTacQueryTool(BaseTool):
    """LangChain tool for executing SQL queries"""
    name: str = "query_seatac_database"
    description: str = "Execute SQL queries on SeaTac airport operations database"
    
    db_manager: Any = Field(default=None)
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        query = self._clean_sql(query)
        result = self.db_manager.execute_query(query)
        
        if result['success']:
            return json.dumps({
                'success': True,
                'row_count': result['row_count'],
                'data': result['data'][:15],
                'message': f"Retrieved {result['row_count']} rows"
            }, cls=DecimalEncoder)
        else:
            return json.dumps({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'message': 'Query failed'
            })
    
    def _clean_sql(self, sql: str) -> str:
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        sql = sql.replace('`', '')
        
        if 'SELECT' in sql.upper():
            select_pos = sql.upper().find('SELECT')
            sql = sql[select_pos:]
        
        sql = re.sub(r';+\s*$', '', sql).strip()
        
        if 'LIMIT' not in sql.upper():
            sql += ' LIMIT 200'
        
        return sql.strip()


class SeaTacUseCaseTool(BaseTool):
    """LangChain tool for SeaTac use case templates"""
    name: str = "get_seatac_use_case"
    description: str = "Get SQL query templates for SeaTac's 11 operational use cases (1-11)"
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        use_cases = {
            "1": {
                "name": "Taxi-In Performance by Aircraft Type",
                "sql": """SELECT 
    f.aircraft_type,
    at.weight_class,
    AVG(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as avg_taxi_in_minutes,
    MIN(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as min_taxi_in_minutes,
    MAX(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as max_taxi_in_minutes,
    COUNT(*) as flight_count
FROM flight f
JOIN aircraft_type at ON f.aircraft_type = at.aircraft_type
JOIN flight_event landing ON f.call_sign = landing.call_sign AND landing.event_type = 'Actual_Landing'
JOIN flight_event inblock ON f.call_sign = inblock.call_sign AND inblock.event_type = 'Actual_In_Block'
WHERE f.operation = 'ARRIVAL' AND TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time) BETWEEN 1 AND 120
GROUP BY f.aircraft_type, at.weight_class
HAVING flight_count >= 3
ORDER BY avg_taxi_in_minutes DESC"""
            },
            "2": {
                "name": "Taxi-Out Performance by Hour",
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
            "5": {
                "name": "Wheels-Up Delay Tracker",
                "sql": """SELECT 
    f.aircraft_type,
    at.weight_class,
    AVG(TIMESTAMPDIFF(MINUTE, scheduled.event_time, takeoff.event_time)) as avg_delay_minutes,
    COUNT(*) as flight_count
FROM flight f
JOIN aircraft_type at ON f.aircraft_type = at.aircraft_type
JOIN flight_event scheduled ON f.call_sign = scheduled.call_sign AND scheduled.event_type = 'Scheduled_Take_Off'
JOIN flight_event takeoff ON f.call_sign = takeoff.call_sign AND takeoff.event_type = 'Actual_Take_Off'
WHERE f.operation = 'DEPARTURE'
GROUP BY f.aircraft_type, at.weight_class
HAVING flight_count >= 3
ORDER BY avg_delay_minutes DESC"""
            },
            "6": {
                "name": "Weight Class Comparison",
                "sql": """SELECT 
    at.weight_class,
    AVG(TIMESTAMPDIFF(MINUTE, landing.event_time, inblock.event_time)) as avg_taxi_in_minutes,
    AVG(TIMESTAMPDIFF(MINUTE, offblock.event_time, takeoff.event_time)) as avg_taxi_out_minutes,
    COUNT(DISTINCT f.call_sign) as total_flights
FROM aircraft_type at
JOIN flight f ON at.aircraft_type = f.aircraft_type
LEFT JOIN flight_event landing ON f.call_sign = landing.call_sign AND landing.event_type = 'Actual_Landing'
LEFT JOIN flight_event inblock ON f.call_sign = inblock.call_sign AND inblock.event_type = 'Actual_In_Block'
LEFT JOIN flight_event offblock ON f.call_sign = offblock.call_sign AND offblock.event_type = 'Actual_Off_Block'
LEFT JOIN flight_event takeoff ON f.call_sign = takeoff.call_sign AND takeoff.event_type = 'Actual_Take_Off'
WHERE at.weight_class IN ('L', 'M', 'H')
GROUP BY at.weight_class
ORDER BY FIELD(at.weight_class, 'L', 'M', 'H')"""
            }
        }
        
        query_str = str(query).strip()
        
        if query_str in use_cases:
            case = use_cases[query_str]
            return json.dumps({
                'success': True,
                'use_case': case['name'],
                'sql': case['sql'].strip()
            })
        
        return json.dumps({
            'success': True,
            'use_case_number': '1',
            'use_case': use_cases['1']['name'],
            'sql': use_cases['1']['sql'].strip()
        })


class AnalyzeSeaTacDataTool(BaseTool):
    """LangChain tool for analyzing operational data"""
    name: str = "analyze_seatac_data"
    description: str = "Analyze SeaTac operational data to extract insights"
    
    def _run(self, data: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            data_obj = json.loads(data) if isinstance(data, str) else data
            
            if not data_obj or not isinstance(data_obj, list):
                return json.dumps({'error': 'Invalid data format'})
            
            insights = []
            
            numeric_cols = []
            if len(data_obj) > 0:
                for key, value in data_obj[0].items():
                    if isinstance(value, (int, float, Decimal)):
                        numeric_cols.append(key)
            
            for col in numeric_cols:
                values = [float(row[col]) if isinstance(row[col], Decimal) else row[col] 
                         for row in data_obj if row.get(col) is not None]
                if values:
                    mean_val = sum(values) / len(values)
                    max_val = max(values)
                    
                    if max_val > mean_val * 1.5:
                        insights.append(f"High variability in {col}: max ({max_val:.1f}) is 50%+ above average ({mean_val:.1f})")
            
            return json.dumps({'success': True, 'insights': insights})
            
        except Exception as e:
            return json.dumps({'error': f'Analysis failed: {str(e)}'})


class ChartGenerator:
    """Generate Chart.js configurations from query results"""
    
    @staticmethod
    def generate_chart(data: List[Dict], use_case: Optional[str] = None) -> Optional[Dict]:
        if not data or len(data) == 0:
            return None
        
        first_row = data[0]
        keys = list(first_row.keys())
        
        label_field = None
        value_field = None
        
        for key in keys:
            if any(term in key.lower() for term in ['type', 'class', 'hour', 'runway']):
                label_field = key
                break
        
        for key in keys:
            if any(term in key.lower() for term in ['avg', 'count', 'total', 'minutes']):
                value_field = key
                break
        
        if not label_field:
            label_field = keys[0]
        if not value_field:
            value_field = keys[1] if len(keys) > 1 else keys[0]
        
        labels = [str(row.get(label_field, 'Unknown')) for row in data[:20]]
        values = [float(row.get(value_field, 0)) for row in data[:20]]
        
        chart_type = 'bar'
        if 'hour' in label_field.lower():
            chart_type = 'line'
        
        color = 'rgba(59, 130, 246, 0.7)'
        
        return {
            'type': chart_type,
            'title': use_case or 'Analysis Results',
            'data': {
                'labels': labels,
                'datasets': [{
                    'label': value_field.replace('_', ' ').title(),
                    'data': values,
                    'backgroundColor': color,
                    'borderColor': color.replace('0.7', '1'),
                    'borderWidth': 2,
                    'tension': 0.4 if chart_type == 'line' else 0,
                    'fill': chart_type == 'line'
                }]
            }
        }


class SeaTacIntelligenceAgent:
    """LangChain-powered agent for SeaTac operations"""
    
    def __init__(self, llm, db_manager):
        self.llm = llm
        self.db_manager = db_manager
        self.chart_generator = ChartGenerator()
        
        self.tools = [
            SeaTacQueryTool(db_manager=db_manager),
            SeaTacUseCaseTool(),
            AnalyzeSeaTacDataTool()
        ]
        
        self.agent = self._create_langchain_agent()
    
    def _extract_text_from_response(self, response) -> str:
        """Extract text from LLM response - handles all formats"""
        if hasattr(response, 'content'):
            content = response.content
            
            if isinstance(content, str):
                return content.strip()
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        text_parts.append(block['text'])
                    else:
                        text_parts.append(str(block))
                return ' '.join(text_parts).strip()
            else:
                return str(content).strip()
        
        return str(response).strip()
    
    def _create_langchain_agent(self):
        try:
            agent = create_agent(model=self.llm, tools=self.tools)
            print("✅ LangChain agent created")
            return agent
        except Exception as e:
            print(f"⚠️  Agent creation failed: {e}")
            return None
    
    async def process_query(self, query: str, max_iterations: int = 5) -> Dict[str, Any]:
        if self.agent:
            try:
                result = await self.agent.ainvoke({"messages": [HumanMessage(content=query)]})
                messages = result.get('messages', [])
                final_message = messages[-1] if messages else None
                answer = self._extract_text_from_response(final_message) if final_message else "No response"
                
                return {'success': True, 'answer': answer, 'reasoning_steps': [], 'sql_queries': [], 'insights': []}
            except Exception as e:
                print(f"⚠️  Agent failed: {e}")
        
        return await self._manual_tool_orchestration(query)
    
    async def _manual_tool_orchestration(self, query: str) -> Dict[str, Any]:
        try:
            reasoning_steps = []
            sql_queries = []
            insights = []
            use_case = None
            final_data = []
            
            # Step 1: Analyze query
            analysis_prompt = f"Analyze: '{query}'\n\nWhich SeaTac use case (1, 2, 5, or 6)? Respond with just the number or 'custom'."
            
            analysis_response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            analysis_text = self._extract_text_from_response(analysis_response)
            
            reasoning_steps.append({'action': 'analyze_query', 'input': query, 'observation': analysis_text})
            
            # Step 2: Get SQL
            use_case_match = re.search(r'\b([1256])\b', analysis_text)
            sql_query = None
            
            if use_case_match:
                use_case_num = use_case_match.group(1)
                use_case_result = self.tools[1]._run(use_case_num)
                use_case_data = json.loads(use_case_result)
                
                if use_case_data.get('success') and 'sql' in use_case_data:
                    sql_query = use_case_data['sql'].strip()
                    use_case = use_case_data.get('use_case')
                    sql_queries.append(sql_query)
                    
                    reasoning_steps.append({'action': 'get_use_case', 'input': use_case_num, 'observation': f"Use Case {use_case_num}"})
            
            # Step 3: Generate custom SQL if needed
            if not sql_query:
                sql_prompt = f"Generate SQL for: '{query}'\n\n{SCHEMA_CONTEXT}\n\nReturn ONLY SQL:"
                sql_response = await self.llm.ainvoke([HumanMessage(content=sql_prompt)])
                sql_query = self._extract_text_from_response(sql_response)
                sql_query = self.tools[0]._clean_sql(sql_query)
                sql_queries.append(sql_query)
                
                reasoning_steps.append({'action': 'generate_sql', 'input': 'custom', 'observation': 'Generated'})
            
            # Step 4: Execute
            if sql_query:
                result = self.tools[0]._run(sql_query)
                result_data = json.loads(result)
                
                reasoning_steps.append({'action': 'query_database', 'input': sql_query[:100], 'observation': f"{result_data.get('row_count', 0)} rows"})
                
                if result_data.get('success') and result_data.get('data'):
                    final_data = result_data['data']
                    
                    analysis = self.tools[2]._run(json.dumps(final_data, cls=DecimalEncoder))
                    analysis_data = json.loads(analysis)
                    
                    if analysis_data.get('insights'):
                        insights.extend(analysis_data['insights'])
                    
                    reasoning_steps.append({'action': 'analyze_data', 'input': 'analysis', 'observation': f"{len(insights)} insights"})
                    
                    # Step 5: Final answer
                    final_prompt = f"Answer: '{query}'\n\nData: {json.dumps(final_data[:3], cls=DecimalEncoder)}\n\nProvide 2-3 sentences:"
                    final_response = await self.llm.ainvoke([HumanMessage(content=final_prompt)])
                    answer = self._extract_text_from_response(final_response)
                    
                    chart_config = self.chart_generator.generate_chart(final_data, use_case)
                    
                    return {
                        'success': True,
                        'answer': answer,
                        'use_case': use_case,
                        'reasoning_steps': reasoning_steps,
                        'sql_queries': sql_queries,
                        'insights': insights,
                        'data': final_data[:50],
                        'row_count': len(final_data),
                        'chart': chart_config
                    }
            
            return {'success': True, 'answer': "I need more information.", 'reasoning_steps': reasoning_steps}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'answer': f'Error: {str(e)}'}


# Initialize
db_manager = DatabaseManager()
agent_system = SeaTacIntelligenceAgent(llm, db_manager) if llm else None


# API Endpoints
@app.get("/")
async def root():
    return {"message": "SeaTac Operations Intelligence", "version": "5.0.0"}


@app.post("/api/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        if not request.query or len(request.query.strip()) < 2:
            raise HTTPException(status_code=400, detail="Please provide a valid query")
        
        if not agent_system:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        result = await agent_system.process_query(request.query, request.max_iterations)
        
        viz_suggestion = None
        if result.get('use_case'):
            if 'hour' in result['use_case'].lower():
                viz_suggestion = "line_chart"
            elif 'weight' in result['use_case'].lower():
                viz_suggestion = "grouped_bar_chart"
            else:
                viz_suggestion = "bar_chart"
        
        return QueryResponse(
            success=result.get('success', False),
            message=result.get('answer', 'No answer'),
            use_case=result.get('use_case'),
            agent_reasoning=result.get('reasoning_steps', []),
            row_count=result.get('row_count', 0),
            sql_queries=result.get('sql_queries', []),
            data=result.get('data'),
            insights=result.get('insights', []),
            visualization_suggestion=viz_suggestion,
            chart=result.get('chart')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model=GEMINI_MODEL,
        langchain_enabled=bool(llm),
        agent_enabled=bool(agent_system),
        seatac_use_cases_supported=11,
        tools_available=["query_seatac_database", "get_seatac_use_case", "analyze_seatac_data"] if agent_system else []
    )


@app.get("/api/use-cases")
async def list_use_cases():
    return {
        "use_cases": [
            {"number": 1, "name": "Taxi-In Performance by Aircraft Type"},
            {"number": 2, "name": "Taxi-Out Performance by Hour"},
            {"number": 3, "name": "Movement Area Occupancy"},
            {"number": 4, "name": "Runway Occupancy Time"},
            {"number": 5, "name": "Wheels-Up Delay Tracker"},
            {"number": 6, "name": "Weight Class Comparison"},
            {"number": 7, "name": "Taxiway Utilization"},
            {"number": 8, "name": "Landing to In-Block Duration"},
            {"number": 9, "name": "Runway Utilization Rate"},
            {"number": 10, "name": "Peak Hour Prediction"},
            {"number": 11, "name": "Metal on Ground Report"}
        ]
    }


@app.on_event("startup")
async def startup_event():
    print("=" * 70)
    print("✈️  SeaTac Airport Operations Intelligence")
    print("=" * 70)
    print(f"LangChain: {'Enabled ✓' if llm else 'Disabled ✗'}")
    print(f"Agent: {'Active ✓' if agent_system and agent_system.agent else 'Manual Mode ⚠️'}")
    print(f"LLM Model: {GEMINI_MODEL}")
    print(f"Database: {db_manager.db_config['database']}@{db_manager.db_config['host']}")
    print(f"DB Status: {'Connected ✓' if db_manager.test_connection() else 'Disconnected ✗'}")
    print("=" * 70)


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)