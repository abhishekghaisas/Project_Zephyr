import { useState } from 'react';
import { Button } from './ui/button';
import { Avatar, AvatarFallback } from './ui/avatar';
import { Copy, Check } from 'lucide-react';
import { cn } from './ui/utils';
import ChartVisualization from './ChartVisualization';
import { DataTableWithExport } from './DataTableWithExport';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  chart?: any; // Chart configuration from backend
  data?: any[]; // Raw data from backend
}

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit',
      hour12: true 
    });
  };

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
        
        {/* <div className="prose prose-sm max-w-none">
          <div className="text-gray-800 whitespace-pre-wrap leading-relaxed">
            {message.content}
          </div>
        </div> */}
        {/* support markdown rendering */}
        <div className="prose prose-sm max-w-none text-gray-800 leading-relaxed">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Chart Visualization with Table Toggle */}
        {message.role === 'assistant' && message.chart && (
          <ChartVisualization chartConfig={message.chart} tableData={message.data} />
        )}

        {/* Table Data Only (when no chart) */}
        {message.role === 'assistant' && !message.chart && message.data && message.data.length > 0 && (
          <DataTableWithExport data={message.data} />
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
