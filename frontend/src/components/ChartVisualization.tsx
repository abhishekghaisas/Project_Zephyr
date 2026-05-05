import { useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';
import { Button } from './ui/button';
import { BarChart3, Table as TableIcon } from 'lucide-react';
import { DataTableWithExport } from './DataTableWithExport';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface ChartConfig {
  type: string;
  title: string;
  data: {
    labels: string[];
    datasets: Array<{
      label: string;
      data: number[];
      backgroundColor: string | string[];
      borderColor: string | string[];
      borderWidth: number;
      tension?: number;
      fill?: boolean;
    }>;
  };
  options?: any;
}

interface ChartVisualizationProps {
  chartConfig: ChartConfig | null;
  tableData?: any[];
}

const ChartVisualization: React.FC<ChartVisualizationProps> = ({ chartConfig, tableData }) => {
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart');

  console.log('📊 ChartViz received:', { 
    hasChart: !!chartConfig, 
    hasTableData: !!tableData, 
    tableDataLength: tableData?.length 
  });

  if (!chartConfig) {
    return null;
  }

  const defaultOptions: ChartOptions<any> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          font: {
            size: 12,
            family: "'Inter', sans-serif"
          },
          padding: 15,
          usePointStyle: true
        }
      },
      title: {
        display: true,
        text: chartConfig.title,
        font: {
          size: 18,
          family: "'Inter', sans-serif",
          weight: 'bold'
        },
        padding: {
          top: 10,
          bottom: 20
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        titleFont: {
          size: 14,
          family: "'Inter', sans-serif"
        },
        bodyFont: {
          size: 13,
          family: "'Inter', sans-serif"
        },
        cornerRadius: 8
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        },
        ticks: {
          font: {
            size: 11,
            family: "'Inter', sans-serif"
          }
        },
        ...(chartConfig.options?.scales?.x || {})
      },
      y: {
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        ticks: {
          font: {
            size: 11,
            family: "'Inter', sans-serif"
          }
        },
        ...(chartConfig.options?.scales?.y || {})
      }
    },
    ...chartConfig.options
  };

  if (chartConfig.type === 'horizontalBar') {
    const horizontalOptions: ChartOptions<'bar'> = {
      ...defaultOptions,
      indexAxis: 'y' as const,
      scales: {
        x: {
          grid: {
            color: 'rgba(0, 0, 0, 0.05)'
          },
          ticks: {
            font: {
              size: 11,
              family: "'Inter', sans-serif"
            }
          },
          ...(chartConfig.options?.scales?.x || {})
        },
        y: {
          grid: {
            display: false
          },
          ticks: {
            font: {
              size: 10,
              family: "'Inter', sans-serif"
            }
          },
          ...(chartConfig.options?.scales?.y || {})
        }
      }
    };

    return (
      <div className="mt-4 bg-white rounded-lg border border-gray-200 shadow-sm">
        {tableData && tableData.length > 0 && (
          <div className="p-3 border-b border-gray-200 flex items-center gap-2">
            <Button
              variant={viewMode === 'chart' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('chart')}
              className="h-8 px-3 text-xs"
            >
              <BarChart3 className="h-3 w-3 mr-1.5" />
              Chart
            </Button>
            <Button
              variant={viewMode === 'table' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('table')}
              className="h-8 px-3 text-xs"
            >
              <TableIcon className="h-3 w-3 mr-1.5" />
              Table
            </Button>
          </div>
        )}
        
        {viewMode === 'chart' ? (
          <div className="p-4">
            <div style={{ height: '500px' }}>
              <Bar data={chartConfig.data} options={horizontalOptions} />
            </div>
          </div>
        ) : (
          tableData && <DataTableWithExport data={tableData} />
        )}
      </div>
    );
  }

  const ChartComponent = chartConfig.type === 'line' ? Line : Bar;

  return (
    <div className="mt-4 bg-white rounded-lg border border-gray-200 shadow-sm">
      {tableData && tableData.length > 0 && (
        <div className="p-3 border-b border-gray-200 flex items-center gap-2">
          <Button
            variant={viewMode === 'chart' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('chart')}
            className="h-8 px-3 text-xs"
          >
            <BarChart3 className="h-3 w-3 mr-1.5" />
            Chart
          </Button>
          <Button
            variant={viewMode === 'table' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('table')}
            className="h-8 px-3 text-xs"
          >
            <TableIcon className="h-3 w-3 mr-1.5" />
            Table
          </Button>
        </div>
      )}
      
      {viewMode === 'chart' ? (
        <div className="p-4">
          <div style={{ height: '400px' }}>
            <ChartComponent data={chartConfig.data} options={defaultOptions} />
          </div>
        </div>
      ) : (
        tableData && <DataTableWithExport data={tableData} />
      )}
    </div>
  );
};

export default ChartVisualization;
