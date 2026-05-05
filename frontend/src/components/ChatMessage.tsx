import { useState } from 'react';
import { Button } from './ui/button';
import { Avatar, AvatarFallback } from './ui/avatar';
import { Copy, Check, Download, Search } from 'lucide-react';
import { cn } from './ui/utils';
import { DataTableWithExport } from './DataTableWithExport';

import ChartVisualization from './ChartVisualization';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date | string;
  chart?: any;
  data?: any[];
  metadata?: any;
}

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState<{
    key: string;
    direction: 'asc' | 'desc';
  } | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;

  console.log('🎨 ChatMessage rendering:', {
    id: message.id,
    role: message.role,
    content: message.content?.slice(0, 50),
    hasChart: !!message.chart,
    hasData: !!message.data,
    dataLength: message.data?.length
  });

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatTime = (date: Date | string) => {
    try {
      const dateObj = typeof date === 'string' ? new Date(date) : date;
      return dateObj.toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit',
        hour12: true 
      });
    } catch {
      return '';
    }
  };

  // Table functions
  const handleSort = (key: string) => {
    let direction: 'asc' | 'desc' = 'asc';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const exportToCSV = () => {
    if (!message.data || message.data.length === 0) return;

    const headers = Object.keys(message.data[0]);
    const csvContent = [
      headers.join(','),
      ...message.data.map(row => 
        headers.map(header => {
          const value = row[header];
          return typeof value === 'string' && value.includes(',') 
            ? `"${value}"` 
            : value;
        }).join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aiport-data-${Date.now()}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  // Filter and sort data
  const getProcessedData = () => {
    if (!message.data) return [];

    let filtered = message.data;

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(row =>
        Object.values(row).some(value =>
          String(value).toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }

    // Sort
    if (sortConfig) {
      filtered = [...filtered].sort((a, b) => {
        const aVal = a[sortConfig.key];
        const bVal = b[sortConfig.key];

        if (aVal === null || aVal === undefined) return 1;
        if (bVal === null || bVal === undefined) return -1;

        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
        }

        const aStr = String(aVal);
        const bStr = String(bVal);
        return sortConfig.direction === 'asc'
          ? aStr.localeCompare(bStr)
          : bStr.localeCompare(aStr);
      });
    }

    return filtered;
  };

  const processedData = getProcessedData();
  const totalPages = Math.ceil(processedData.length / rowsPerPage);
  const paginatedData = processedData.slice(
    (currentPage - 1) * rowsPerPage,
    currentPage * rowsPerPage
  );

  return (
    <div className={cn(
      "group flex gap-4 p-6 hover:bg-gray-50/50 transition-colors",
      message.role === 'assistant' && "bg-gray-50/30"
    )}>
      {/* Avatar */}
      <div className="flex-shrink-0">
        <Avatar className="h-8 w-8">
          <AvatarFallback className={cn(
            "text-sm font-medium",
            message.role === 'user' 
              ? "bg-blue-100 text-blue-700" 
              : "bg-green-100 text-green-700"
          )}>
            {message.role === 'user' ? 'U' : 'AI'}
          </AvatarFallback>
        </Avatar>
      </div>

      {/* Message Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm font-medium text-gray-900">
            {message.role === 'user' ? 'You' : 'AIport Intelligence'}
          </span>
          <span className="text-xs text-gray-500">
            {formatTime(message.timestamp)}
          </span>
          {message.metadata?.sqlSource && (
            <span className="text-xs px-2 py-0.5 bg-purple-100 text-purple-700 rounded font-mono">
              {message.metadata.sqlSource}
            </span>
          )}
        </div>
        
        {/* Text Content - ALWAYS RENDER */}
        <div className="prose prose-sm max-w-none text-gray-800 leading-relaxed mb-4">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>

        {/* View Toggle - Show if assistant has both chart and data */}
        {message.role === 'assistant' && message.chart && message.data && message.data.length > 0 && (
          <div className="flex items-center gap-2 mb-4">
            <Button
              variant={viewMode === 'chart' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('chart')}
              className="gap-2"
            >
              📊 Chart View
            </Button>
            <Button
              variant={viewMode === 'table' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('table')}
              className="gap-2"
            >
              📋 Table View
            </Button>
          </div>
        )}

        {/* Chart - Show in chart mode */}
        {message.role === 'assistant' && message.chart && viewMode === 'chart' && (
          <div className="mt-4 bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
            <ChartVisualization chartConfig={message.chart}
            insightsText={message.content} />
          </div>
        )}

        {/* Table - Show in table mode */}
        {message.role === 'assistant' && message.data && message.data.length > 0 && viewMode === 'table' && (
          <div className="mt-4 bg-white rounded-lg border border-gray-200 shadow-sm">
            {/* Table Header with Controls */}
            <div className="p-4 border-b border-gray-200 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <h3 className="text-sm font-semibold text-gray-900">
                  {message.chart?.title || 'Query Results'}
                </h3>
                <span className="text-xs text-gray-500">
                  {processedData.length} rows {searchTerm && `(filtered from ${message.data.length})`}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search..."
                    value={searchTerm}
                    onChange={(e) => {
                      setSearchTerm(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="pl-8 pr-3 py-1 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={exportToCSV}
                  className="gap-2"
                >
                  <Download className="h-4 w-4" />
                  Export CSV
                </Button>
              </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    {message.data.length > 0 && Object.keys(message.data[0]).map(key => (
                      <th
                        key={key}
                        onClick={() => handleSort(key)}
                        className="px-4 py-3 text-left font-medium text-gray-700 cursor-pointer hover:bg-gray-100 transition-colors"
                      >
                        <div className="flex items-center gap-2">
                          <span>{key}</span>
                          {sortConfig?.key === key && (
                            <span className="text-blue-600">
                              {sortConfig.direction === 'asc' ? '↑' : '↓'}
                            </span>
                          )}
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {paginatedData.map((row, idx) => (
                    <tr 
                      key={idx} 
                      className="border-b border-gray-100 hover:bg-gray-50 transition-colors"
                    >
                      {Object.values(row).map((value: any, i) => (
                        <td key={i} className="px-4 py-3 text-gray-900">
                          {value != null ? String(value) : '-'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="p-4 border-t border-gray-200 flex items-center justify-between">
                <div className="text-xs text-gray-500">
                  Page {currentPage} of {totalPages}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Debug Info */}
        {message.role === 'assistant' && (
          <div className="mt-2 text-xs text-gray-500 font-mono">
            {message.data?.length || 0} rows
            {message.chart && ' | has chart'}
            {message.data && message.data.length > 0 && ' | has data'}
            {message.chart && message.data && message.data.length > 0 && ' | toggle available'}
          </div>
        )}

        {/* Copy Button */}
        {message.role === 'assistant' && (
          <div className="flex items-center gap-2 mt-3 opacity-0 group-hover:opacity-100 transition-opacity">
            <Button
              variant="ghost"
              size="sm"
              onClick={copyToClipboard}
              className="h-7 px-2 text-xs text-gray-500 hover:text-gray-700"
            >
              {copied ? (
                <>
                  <Check className="h-3 w-3 mr-1" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-3 w-3 mr-1" />
                  Copy
                </>
              )}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
