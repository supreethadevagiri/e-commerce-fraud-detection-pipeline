import { useState } from 'react';
import { 
  Cloud, Database, Table, RefreshCw, 
  Clock, FileSpreadsheet 
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Table as UITable, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';

interface SnowflakePreviewProps {
  data: {
    tables: {
      name: string;
      schema: string;
      database: string;
      rowCount: number;
      lastUpdated: string;
      sizeMB: number;
    }[];
    totalRows: number;
    lastExport: string;
    exportStatus: string;
  };
}

// Mock table data
const mockTableData = {
  'RAW_TRANSACTIONS': [
    { transaction_id: 'TXN-ABC123', customer_id: 'CUST-001', amount: 150.00, category: 'Electronics', is_fraudulent: false },
    { transaction_id: 'TXN-DEF456', customer_id: 'CUST-002', amount: 2500.00, category: 'Jewelry', is_fraudulent: true },
    { transaction_id: 'TXN-GHI789', customer_id: 'CUST-003', amount: 45.50, category: 'Food', is_fraudulent: false },
    { transaction_id: 'TXN-JKL012', customer_id: 'CUST-004', amount: 899.99, category: 'Electronics', is_fraudulent: false },
    { transaction_id: 'TXN-MNO345', customer_id: 'CUST-005', amount: 15.00, category: 'Books', is_fraudulent: false },
  ],
  'CLEANED_TRANSACTIONS': [
    { transaction_id: 'TXN-ABC123', customer_id: 'CUST-001', amount: 150.00, category: 'Electronics', is_fraudulent: false, is_duplicate: false },
    { transaction_id: 'TXN-DEF456', customer_id: 'CUST-002', amount: 2500.00, category: 'Jewelry', is_fraudulent: true, is_duplicate: false },
    { transaction_id: 'TXN-GHI789', customer_id: 'CUST-003', amount: 45.50, category: 'Food', is_fraudulent: false, is_duplicate: false },
    { transaction_id: 'TXN-JKL012', customer_id: 'CUST-004', amount: 899.99, category: 'Electronics', is_fraudulent: false, is_duplicate: false },
    { transaction_id: 'TXN-MNO345', customer_id: 'CUST-005', amount: 15.00, category: 'Books', is_fraudulent: false, is_duplicate: false },
  ],
  'HOURLY_CATEGORY_SUMMARY': [
    { summary_hour: '2024-01-15 10:00', category: 'Electronics', transaction_count: 145, total_amount: 45230.50, fraud_count: 8 },
    { summary_hour: '2024-01-15 10:00', category: 'Clothing', transaction_count: 89, total_amount: 12340.75, fraud_count: 3 },
    { summary_hour: '2024-01-15 10:00', category: 'Food', transaction_count: 234, total_amount: 5670.25, fraud_count: 1 },
    { summary_hour: '2024-01-15 11:00', category: 'Electronics', transaction_count: 156, total_amount: 48900.00, fraud_count: 9 },
    { summary_hour: '2024-01-15 11:00', category: 'Clothing', transaction_count: 92, total_amount: 13100.50, fraud_count: 4 },
  ],
  'FRAUD_DETECTION_FEATURES': [
    { transaction_id: 'TXN-ABC123', risk_score: 23.5, hour_of_day: 14, amount_zscore: 0.5, is_fraudulent: false },
    { transaction_id: 'TXN-DEF456', risk_score: 87.3, hour_of_day: 3, amount_zscore: 3.2, is_fraudulent: true },
    { transaction_id: 'TXN-GHI789', risk_score: 15.2, hour_of_day: 12, amount_zscore: -0.8, is_fraudulent: false },
    { transaction_id: 'TXN-JKL012', risk_score: 45.8, hour_of_day: 18, amount_zscore: 1.2, is_fraudulent: false },
    { transaction_id: 'TXN-MNO345', risk_score: 12.1, hour_of_day: 9, amount_zscore: -1.2, is_fraudulent: false },
  ]
};

export default function SnowflakePreview({ data }: SnowflakePreviewProps) {
  const [selectedTable, setSelectedTable] = useState('RAW_TRANSACTIONS');
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, string> = {
      success: 'bg-green-500/20 text-green-400 border-green-500/30',
      failed: 'bg-red-500/20 text-red-400 border-red-500/30',
      in_progress: 'bg-blue-500/20 text-blue-400 border-blue-500/30'
    };
    return variants[status] || 'bg-slate-500/20 text-slate-400 border-slate-500/30';
  };

  const tableSizeData = data.tables.map(t => ({
    name: t.name,
    size: t.sizeMB,
    rows: t.rowCount
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-500/20 rounded-lg">
                <Cloud className="w-6 h-6 text-indigo-400" />
              </div>
              <div>
                <CardTitle className="text-white">Snowflake Data Warehouse</CardTitle>
                <CardDescription className="text-slate-400">
                  Exported data and analytics tables
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge 
                variant="outline" 
                className={getStatusBadge(data.exportStatus)}
              >
                {data.exportStatus === 'in_progress' ? 'Exporting...' : 
                 data.exportStatus === 'success' ? 'Export Complete' : 'Export Failed'}
              </Badge>
              <Button 
                variant="outline" 
                size="icon"
                onClick={handleRefresh}
                disabled={isRefreshing}
              >
                <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <Database className="w-4 h-4" />
                <span className="text-sm">Total Tables</span>
              </div>
              <p className="text-2xl font-bold text-white">{data.tables.length}</p>
              <p className="text-sm text-slate-500 mt-1">Across 4 schemas</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <FileSpreadsheet className="w-4 h-4" />
                <span className="text-sm">Total Rows</span>
              </div>
              <p className="text-2xl font-bold text-indigo-400">{data.totalRows.toLocaleString()}</p>
              <p className="text-sm text-slate-500 mt-1">All tables combined</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm">Last Export</span>
              </div>
              <p className="text-lg font-semibold text-white">
                {new Date(data.lastExport).toLocaleTimeString()}
              </p>
              <p className="text-sm text-slate-500 mt-1">
                {new Date(data.lastExport).toLocaleDateString()}
              </p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <Table className="w-4 h-4" />
                <span className="text-sm">Total Size</span>
              </div>
              <p className="text-2xl font-bold text-white">
                {data.tables.reduce((acc, t) => acc + t.sizeMB, 0).toFixed(1)} MB
              </p>
              <p className="text-sm text-slate-500 mt-1">Storage used</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tables and Data */}
      <Tabs defaultValue="tables" className="space-y-4">
        <TabsList className="bg-slate-900 border border-slate-800">
          <TabsTrigger value="tables">Tables</TabsTrigger>
          <TabsTrigger value="data">Data Preview</TabsTrigger>
          <TabsTrigger value="storage">Storage Stats</TabsTrigger>
        </TabsList>

        <TabsContent value="tables">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg">Database Tables</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {data.tables.map((table) => (
                  <div 
                    key={table.name} 
                    className="flex items-center justify-between p-4 bg-slate-950 rounded-lg border border-slate-800 hover:border-slate-700 cursor-pointer transition-colors"
                    onClick={() => setSelectedTable(table.name)}
                  >
                    <div className="flex items-center gap-4">
                      <div className="p-2 bg-indigo-500/20 rounded-lg">
                        <Table className="w-5 h-5 text-indigo-400" />
                      </div>
                      <div>
                        <p className="font-medium text-white">{table.name}</p>
                        <p className="text-sm text-slate-400">
                          {table.database}.{table.schema}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-6 text-sm">
                      <div className="text-right">
                        <p className="text-white">{table.rowCount.toLocaleString()}</p>
                        <p className="text-slate-400">rows</p>
                      </div>
                      <div className="text-right">
                        <p className="text-white">{table.sizeMB} MB</p>
                        <p className="text-slate-400">size</p>
                      </div>
                      <div className="text-right">
                        <p className="text-slate-300">
                          {new Date(table.lastUpdated).toLocaleTimeString()}
                        </p>
                        <p className="text-slate-400">updated</p>
                      </div>
                      <Button variant="ghost" size="sm">
                        View
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="data">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-white text-lg">Data Preview</CardTitle>
                <select 
                  className="bg-slate-950 border border-slate-800 rounded px-3 py-1 text-white text-sm"
                  value={selectedTable}
                  onChange={(e) => setSelectedTable(e.target.value)}
                >
                  {data.tables.map(t => (
                    <option key={t.name} value={t.name}>{t.name}</option>
                  ))}
                </select>
              </div>
            </CardHeader>
            <CardContent>
              <div className="rounded-md border border-slate-800">
                <UITable>
                  <TableHeader>
                    <TableRow className="bg-slate-950">
                      {mockTableData[selectedTable as keyof typeof mockTableData] && 
                        Object.keys(mockTableData[selectedTable as keyof typeof mockTableData][0]).map((key) => (
                          <TableHead key={key} className="text-slate-400 capitalize">
                            {key.replace(/_/g, ' ')}
                          </TableHead>
                        ))
                      }
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {mockTableData[selectedTable as keyof typeof mockTableData]?.map((row, index) => (
                      <TableRow key={index} className="border-slate-800">
                        {Object.entries(row).map(([key, value]: [string, any]) => (
                          <TableCell key={key} className="text-slate-300">
                            {typeof value === 'boolean' ? (
                              <Badge 
                                variant="outline" 
                                className={value 
                                  ? 'bg-green-500/20 text-green-400 border-green-500/30' 
                                  : 'bg-slate-500/20 text-slate-400 border-slate-500/30'
                                }
                              >
                                {value ? 'Yes' : 'No'}
                              </Badge>
                            ) : (
                              value
                            )}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </UITable>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="storage">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg">Storage Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={tableSizeData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" angle={-45} textAnchor="end" height={100} />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    formatter={(value: number) => [`${value} MB`, 'Size']}
                  />
                  <Bar dataKey="size" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
