import React, { useRef, useEffect, useState } from 'react';
import { Chart as ChartJS, registerables } from 'chart.js';
import { Button } from './ui/button';
import { Download, Image as ImageIcon, FileText, Check, BarChart3, LineChart, PieChart } from 'lucide-react';

ChartJS.register(...registerables);

interface ChartVisualizationProps {
  chartConfig: {
    type: string;
    title?: string;
    data: {
      labels: string[];
      datasets: Array<{
        label: string;
        data: number[];
        backgroundColor?: string | string[];
        borderColor?: string | string[];
        borderWidth?: number;
        tension?: number;
        fill?: boolean;
      }>;
    };
    options?: any;
  };
  insightsText?: string;
}

export default function ChartVisualization({ chartConfig, insightsText }: ChartVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<ChartJS | null>(null);
  const [copied, setCopied] = useState(false);
  const [chartType, setChartType] = useState<'bar' | 'line' | 'pie'>(
    chartConfig.type as 'bar' | 'line' | 'pie'
  );

  useEffect(() => {
    if (!canvasRef.current) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Prepare data based on chart type
    let chartData = { ...chartConfig.data };
    let chartOptions = { ...chartConfig.options };

    // For pie charts, use multiple colors
    if (chartType === 'pie') {
      const colors = [
        'rgba(59, 130, 246, 0.7)',   // Blue
        'rgba(16, 185, 129, 0.7)',   // Green
        'rgba(245, 158, 11, 0.7)',   // Amber
        'rgba(239, 68, 68, 0.7)',    // Red
        'rgba(139, 92, 246, 0.7)',   // Purple
        'rgba(236, 72, 153, 0.7)',   // Pink
        'rgba(20, 184, 166, 0.7)',   // Teal
        'rgba(251, 146, 60, 0.7)',   // Orange
        'rgba(14, 165, 233, 0.7)',   // Sky
        'rgba(168, 85, 247, 0.7)',   // Violet
      ];

      chartData = {
        ...chartConfig.data,
        datasets: chartConfig.data.datasets.map(dataset => ({
          ...dataset,
          backgroundColor: colors.slice(0, chartConfig.data.labels.length),
          borderColor: colors.slice(0, chartConfig.data.labels.length).map(c => c.replace('0.7', '1')),
          borderWidth: 2
        }))
      };

      chartOptions = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            display: true,
            position: 'right' as const,
            labels: {
              generateLabels: function(chart: any) {
                const data = chart.data;
                const dataset = data.datasets[0];
                const total = dataset.data.reduce((a: number, b: number) => a + b, 0);
                
                return data.labels.map((label: string, i: number) => {
                  const value = dataset.data[i];
                  const percentage = ((value / total) * 100).toFixed(1);
                  return {
                    text: `${label}: ${value} (${percentage}%)`,
                    fillStyle: dataset.backgroundColor[i],
                    hidden: false,
                    index: i
                  };
                });
              },
              font: { size: 11 },
              padding: 8
            }
          },
          title: {
            display: !!chartConfig.title,
            text: chartConfig.title || '',
            font: {
              size: 16,
              weight: 'bold' as const,
            },
          },
        }
      };
    }

    chartRef.current = new ChartJS(ctx, {
      type: chartType,
      data: chartData,
      options: chartType === 'pie' ? chartOptions : {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            display: true,
            position: 'top' as const,
          },
          title: {
            display: !!chartConfig.title,
            text: chartConfig.title || '',
            font: {
              size: 16,
              weight: 'bold' as const,
            },
          },
        },
        ...chartConfig.options,
      },
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, [chartConfig, chartType]);

  const downloadChartWithInsights = async () => {
    if (!canvasRef.current) return;

    const padding = 80;
    const chartScale = 2;
    const fontSize = 24;
    const lineHeight = 36;
    const titleFontSize = 32;
    
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;
    
    tempCtx.font = `${fontSize}px Inter, system-ui, sans-serif`;
    const maxTextWidth = (canvasRef.current.width * chartScale) - (padding * 3);
    
    const words = insightsText?.split(' ') || [];
    const lines: string[] = [];
    let currentLine = '';
    
    for (const word of words) {
      const testLine = currentLine + word + ' ';
      const metrics = tempCtx.measureText(testLine);
      
      if (metrics.width > maxTextWidth && currentLine.length > 0) {
        lines.push(currentLine.trim());
        currentLine = word + ' ';
      } else {
        currentLine = testLine;
      }
    }
    if (currentLine.trim()) {
      lines.push(currentLine.trim());
    }
    
    const textBlockHeight = insightsText 
      ? (lines.length * lineHeight) + titleFontSize + 100
      : 0;
    
    const exportCanvas = document.createElement('canvas');
    const chartWidth = canvasRef.current.width * chartScale;
    const chartHeight = canvasRef.current.height * chartScale;
    
    exportCanvas.width = chartWidth + (padding * 2);
    exportCanvas.height = chartHeight + textBlockHeight + (padding * 3);
    
    const ctx = exportCanvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 3;
    ctx.strokeRect(2, 2, exportCanvas.width - 4, exportCanvas.height - 4);

    ctx.drawImage(
      canvasRef.current, 
      padding, 
      padding, 
      chartWidth, 
      chartHeight
    );

    if (insightsText && lines.length > 0) {
      const textStartY = chartHeight + (padding * 2);
      
      ctx.fillStyle = '#f3f4f6';
      const textBoxPadding = 40;
      ctx.fillRect(
        padding, 
        textStartY - textBoxPadding, 
        exportCanvas.width - (padding * 2), 
        textBlockHeight + textBoxPadding
      );
      
      ctx.strokeStyle = '#d1d5db';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        padding, 
        textStartY - textBoxPadding, 
        exportCanvas.width - (padding * 2), 
        textBlockHeight + textBoxPadding
      );
      
      ctx.fillStyle = '#111827';
      ctx.font = `bold ${titleFontSize}px Inter, system-ui, sans-serif`;
      ctx.fillText('📊 Analysis:', padding + 40, textStartY + 20);
      
      ctx.strokeStyle = '#d1d5db';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding + 40, textStartY + 40);
      ctx.lineTo(exportCanvas.width - padding - 40, textStartY + 40);
      ctx.stroke();
      
      ctx.font = `${fontSize}px Inter, system-ui, sans-serif`;
      ctx.fillStyle = '#1f2937';
      
      let y = textStartY + 75;
      for (const line of lines) {
        ctx.fillText(line, padding + 40, y);
        y += lineHeight;
      }
    }
    
    const footerY = exportCanvas.height - 35;
    ctx.fillStyle = '#6b7280';
    ctx.font = 'bold 16px Inter, system-ui, sans-serif';
    ctx.fillText('AIport Intelligence', padding, footerY);
    
    ctx.font = '16px Inter, system-ui, sans-serif';
    const dateText = new Date().toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
    ctx.fillText(dateText, exportCanvas.width - padding - ctx.measureText(dateText).width, footerY);

    const link = document.createElement('a');
    link.download = `aiport-analysis-${Date.now()}.png`;
    link.href = exportCanvas.toDataURL('image/png', 1.0);
    link.click();
  };

  const downloadChartOnly = () => {
    if (!canvasRef.current) return;

    const link = document.createElement('a');
    link.download = `aiport-chart-${Date.now()}.png`;
    link.href = canvasRef.current.toDataURL('image/png');
    link.click();
  };

  const downloadHighResPNG = () => {
    if (!chartRef.current || !canvasRef.current) return;

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = canvasRef.current.width * 2;
    exportCanvas.height = canvasRef.current.height * 2;
    
    const exportCtx = exportCanvas.getContext('2d');
    if (!exportCtx) return;

    exportCtx.scale(2, 2);

    const tempChart = new ChartJS(exportCtx, {
      type: chartType,
      data: chartConfig.data,
      options: {
        responsive: false,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            display: true,
            position: 'top' as const,
          },
          title: {
            display: !!chartConfig.title,
            text: chartConfig.title || '',
            font: {
              size: 16,
              weight: 'bold' as const,
            },
          },
        },
        ...chartConfig.options,
      },
    });

    setTimeout(() => {
      const link = document.createElement('a');
      link.download = `aiport-chart-highres-${Date.now()}.png`;
      link.href = exportCanvas.toDataURL('image/png');
      link.click();
      
      tempChart.destroy();
    }, 100);
  };

  const copyImageAndText = async () => {
    if (!canvasRef.current) return;

    try {
      const blob = await new Promise<Blob>((resolve) => {
        canvasRef.current?.toBlob((blob) => {
          if (blob) resolve(blob);
        }, 'image/png');
      });

      const clipboardItems: ClipboardItem[] = [];

      if (insightsText) {
        clipboardItems.push(
          new ClipboardItem({
            'image/png': blob,
            'text/plain': new Blob([insightsText], { type: 'text/plain' }),
          })
        );
      } else {
        clipboardItems.push(
          new ClipboardItem({ 'image/png': blob })
        );
      }

      await navigator.clipboard.write(clipboardItems);

      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
      if (insightsText) {
        await navigator.clipboard.writeText(insightsText);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
    }
  };

  const copyTextOnly = async () => {
    if (!insightsText) return;

    try {
      await navigator.clipboard.writeText(insightsText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  return (
    <div className="space-y-3">
      {/* Chart Type Selector */}
      <div className="flex items-center gap-2">
        <span className="text-sm text-gray-600">Chart Type:</span>
        <div className="flex items-center gap-1 bg-gray-100 rounded-md p-1">
          <Button
            variant={chartType === 'bar' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setChartType('bar')}
            className="gap-2 h-8"
          >
            <BarChart3 className="h-4 w-4" />
            Bar
          </Button>
          <Button
            variant={chartType === 'line' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setChartType('line')}
            className="gap-2 h-8"
          >
            <LineChart className="h-4 w-4" />
            Line
          </Button>
          <Button
            variant={chartType === 'pie' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setChartType('pie')}
            className="gap-2 h-8"
          >
            <PieChart className="h-4 w-4" />
            Pie
          </Button>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="relative bg-white rounded-lg p-4">
        <canvas ref={canvasRef} />
      </div>

      {/* Export Controls */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Download Options */}
        <div className="flex items-center gap-2">
          {insightsText && (
            <Button
              variant="default"
              size="sm"
              onClick={downloadChartWithInsights}
              className="gap-2"
            >
              <Download className="h-4 w-4" />
              Download with Insights
            </Button>
          )}
          
          <Button
            variant="outline"
            size="sm"
            onClick={downloadChartOnly}
            className="gap-2"
          >
            <Download className="h-4 w-4" />
            Chart Only
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={downloadHighResPNG}
            className="gap-2"
          >
            <ImageIcon className="h-4 w-4" />
            High-Res
          </Button>
        </div>

        {/* Divider */}
        <div className="h-6 w-px bg-gray-300" />

        {/* Copy Options */}
        <div className="flex items-center gap-2">
          {insightsText && (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={copyImageAndText}
                className="gap-2"
              >
                {copied ? (
                  <>
                    <Check className="h-4 w-4" />
                    Copied!
                  </>
                ) : (
                  <>
                    <ImageIcon className="h-4 w-4" />
                    Copy Image + Text
                  </>
                )}
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={copyTextOnly}
                className="gap-2"
              >
                <FileText className="h-4 w-4" />
                Copy Text
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Helper text */}
      {insightsText && (
        <p className="text-xs text-gray-500">
          💡 Tip: Switch chart types to find the best visualization. "Download with Insights" includes analysis text.
        </p>
      )}
    </div>
  );
}
