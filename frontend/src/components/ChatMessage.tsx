import { useState } from 'react';
import { Button } from './ui/button';
import { Avatar, AvatarFallback } from './ui/avatar';
import { Copy, Check, BarChart3, Table as TableIcon } from 'lucide-react';
import { cn } from './ui/utils';
import ChartVisualization from './ChartVisualization';
import { DataTableWithExport } from './DataTableWithExport';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date | string;
  chart?: any;
  data?: any[];
}

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart');

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatTime = (timestamp: Date | string) => {
    const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
    return date.toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit',
      hour12: true 
    });
  };

  console.log('🎨 ChatMessage rendering:', message);

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
            {message.role === 'user' ? 'You' : 'Gate Assistant'}
          </span>
          <span className="text-xs text-gray-500">
            {formatTime(message.timestamp)}
          </span>
        </div>
        
        {/* Markdown content */}
        <div className="prose prose-sm max-w-none text-gray-800 leading-relaxed">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Chart/Table Toggle for responses with both */}
        {message.role === 'assistant' && message.chart && message.data && message.data.length > 0 && (
          <div className="mt-4">
            <div className="flex items-center gap-2 mb-3">
              <Button
                variant={viewMode === 'chart' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('chart')}
                className="h-8 px-3 text-xs"
              >
                <BarChart3 className="h-3 w-3 mr-1.5" />
                Chart View
              </Button>
              <Button
                variant={viewMode === 'table' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('table')}
                className="h-8 px-3 text-xs"
              >
                <TableIcon className="h-3 w-3 mr-1.5" />
                Table View
              </Button>
            </div>

            {viewMode === 'chart' ? (
              <ChartVisualization chartConfig={message.chart} insightsText={message.content} />
            ) : (
              <DataTable 
                data={message.data} 
                title="SeaTac Operations Analysis"
                pageSize={15}
              />
            )}
          </div>
        )}

        {/* Chart Only (no table data) */}
        {message.role === 'assistant' && message.chart && (!message.data || message.data.length === 0) && (
          <div className="mt-4">
            <ChartVisualization chartConfig={message.chart} insightsText={message.content} />
          </div>
        )}

        {/* Table Only (no chart) */}
        {message.role === 'assistant' && !message.chart && message.data && message.data.length > 0 && (
          <div className="mt-4">
            <DataTable 
              data={message.data} 
              title="Query Results"
              pageSize={15}
            />
          </div>
        )}

        {/* Action Buttons */}
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
