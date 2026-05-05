// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface QueryResponse {
  success: boolean;
  message: string;
  tableData?: any[];
  chart?: any;
  row_count?: number;
  output_format?: string;
  output_confidence?: number;
  sql_source?: string;
  use_case?: string;
  insights?: string[];
  sql_queries?: string[];
}

export async function queryAPI(query: string): Promise<QueryResponse> {
  const url = `${API_BASE_URL}/api/query`;
  
  console.log('🚀 Sending query to:', url);
  console.log('📝 Query:', query);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });

    console.log('📡 Response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ Response error:', errorText);
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    
    console.log('✅ Response data:', data);
    console.log('📊 Data fields:', {
      hasSuccess: !!data.success,
      hasMessage: !!data.message,
      hasData: !!data.data,
      dataLength: data.data?.length || 0,
      hasChart: !!data.chart,
      sqlSource: data.sql_source
    });

    // Backend returns success: false on errors
    if (data.success === false) {
      return {
        success: false,
        message: data.message || 'Query failed'
      };
    }

    // Return with tableData mapped from backend's data field
    return {
      success: data.success,
      message: data.message,
      tableData: data.data, // ✅ Map backend's "data" to "tableData"
      chart: data.chart,
      row_count: data.row_count,
      output_format: data.output_format,
      output_confidence: data.output_confidence,
      sql_source: data.sql_source,
      use_case: data.use_case,
      insights: data.insights,
      sql_queries: data.sql_queries
    };

  } catch (error) {
    console.error('❌ Query error:', error);

    if (error instanceof TypeError && error.message.includes('fetch')) {
      return {
        success: false,
        message: `Cannot connect to backend at ${API_BASE_URL}. Please ensure backend is running.`
      };
    }

    return {
      success: false,
      message: error instanceof Error ? error.message : 'An unexpected error occurred'
    };
  }
}

// Health check function
export async function checkBackendHealth(): Promise<{
  healthy: boolean;
  message: string;
  details?: any;
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
    });

    if (!response.ok) {
      return {
        healthy: false,
        message: `Backend returned status ${response.status}`,
      };
    }

    const data = await response.json();
    return {
      healthy: data.status === 'healthy',
      message: data.status === 'healthy' ? 'Backend is healthy' : 'Backend is unhealthy',
      details: data,
    };
  } catch (error) {
    return {
      healthy: false,
      message: 'Cannot connect to backend at ' + API_BASE_URL,
    };
  }
}
