import { useState } from 'react';
import { Button } from './ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Copy, Check, Download } from 'lucide-react';
import { toast } from 'sonner';

interface DataTableWithExportProps {
  data: any[];
}

export function DataTableWithExport({ data }: DataTableWithExportProps) {
  const [copied, setCopied] = useState(false);

  if (!data || data.length === 0) {
    return null;
  }

  const columns = Object.keys(data[0]);

  // Convert data to CSV format
  const convertToCSV = () => {
    const headers = columns.join(',');
    const rows = data.map(row => 
      columns.map(col => {
        const value = row[col];
        // Escape quotes and wrap in quotes if contains comma
        const stringValue = value === null || value === undefined ? '' : String(value);
        return stringValue.includes(',') ? `"${stringValue.replace(/"/g, '""')}"` : stringValue;
      }).join(',')
    );
    return [headers, ...rows].join('\n');
  };

  // Download as CSV file
  const downloadCSV = () => {
    const csv = convertToCSV();
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aiport-data-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    toast.success('CSV downloaded!');
  };

  // Copy to clipboard in tab-separated format (for pasting into Excel/Sheets)
  const copyToClipboard = async () => {
    // Use tab-separated values for better Excel/Sheets compatibility
    const headers = columns.join('\t');
    const rows = data.map(row => 
      columns.map(col => row[col] === null || row[col] === undefined ? '' : String(row[col])).join('\t')
    );
    const tsvData = [headers, ...rows].join('\n');
    
    await navigator.clipboard.writeText(tsvData);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast.success('Table copied! Paste into Excel or Google Sheets');
  };

  return (
    <div className="mt-4 bg-white rounded-lg border border-gray-200 shadow-sm">
      <div className="p-4 border-b border-gray-200 flex items-center justify-between">
        <div className="text-sm font-medium text-gray-700">
          {data.length} {data.length === 1 ? 'row' : 'rows'}
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={copyToClipboard}
            className="h-8 px-3 text-xs"
          >
            {copied ? (
              <>
                <Check className="h-3 w-3 mr-1.5" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-3 w-3 mr-1.5" />
                Copy to Excel
              </>
            )}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={downloadCSV}
            className="h-8 px-3 text-xs"
          >
            <Download className="h-3 w-3 mr-1.5" />
            Download CSV
          </Button>
        </div>
      </div>
      
      <div className="max-h-96 overflow-auto">
        <Table>
          <TableHeader>
            <TableRow>
              {columns.map((col) => (
                <TableHead key={col} className="font-semibold">
                  {col.replace(/_/g, ' ').toUpperCase()}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.map((row, idx) => (
              <TableRow key={idx}>
                {columns.map((col) => (
                  <TableCell key={col}>
                    {row[col] === null || row[col] === undefined ? '-' : String(row[col])}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
