import { useState } from 'react';
import { 
  Database, Play, Pause, RefreshCw, 
  TrendingUp, AlertTriangle, Clock, CheckCircle 
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';

interface DataGeneratorPanelProps {
  data: {
    recordsGenerated: number;
    recordsPerHour: number;
    fraudCount: number;
    missingValues: number;
    duplicates: number;
    outliers: number;
    startTime: string;
    isRunning: boolean;
    status: string;
  };
}

// Mock historical data for charts
const generateHistoryData = () => {
  const data = [];
  for (let i = 6; i >= 0; i--) {
    data.push({
      time: `${i}h ago`,
      records: 1000 + Math.floor(Math.random() * 200),
      fraud: 50 + Math.floor(Math.random() * 20)
    });
  }
  return data;
};

const qualityData = [
  { name: 'Valid', value: 92, color: '#22c55e' },
  { name: 'Missing', value: 5, color: '#eab308' },
  { name: 'Duplicates', value: 2, color: '#f97316' },
  { name: 'Outliers', value: 1, color: '#ef4444' }
];

export default function DataGeneratorPanel({ data }: DataGeneratorPanelProps) {
  const [isRunning, setIsRunning] = useState(data.isRunning);
  const [historyData] = useState(generateHistoryData());

  const duration = Math.floor((Date.now() - new Date(data.startTime).getTime()) / (1000 * 60 * 60));
  const fraudRate = ((data.fraudCount / data.recordsGenerated) * 100).toFixed(2);

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/20 rounded-lg">
                <Database className="w-6 h-6 text-blue-400" />
              </div>
              <div>
                <CardTitle className="text-white">Data Generator</CardTitle>
                <CardDescription className="text-slate-400">
                  Synthetic transaction data generation
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge 
                variant="outline" 
                className={isRunning 
                  ? 'bg-green-500/20 text-green-400 border-green-500/30' 
                  : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                }
              >
                {isRunning ? 'Running' : 'Paused'}
              </Badge>
              <Button 
                variant={isRunning ? "destructive" : "default"}
                size="sm"
                onClick={() => setIsRunning(!isRunning)}
              >
                {isRunning ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                {isRunning ? 'Stop' : 'Start'}
              </Button>
              <Button variant="outline" size="icon">
                <RefreshCw className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <TrendingUp className="w-4 h-4" />
                <span className="text-sm">Records Generated</span>
              </div>
              <p className="text-2xl font-bold text-white">{data.recordsGenerated.toLocaleString()}</p>
              <p className="text-sm text-green-400 mt-1">+{data.recordsPerHour}/hour</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <AlertTriangle className="w-4 h-4" />
                <span className="text-sm">Fraud Cases</span>
              </div>
              <p className="text-2xl font-bold text-red-400">{data.fraudCount.toLocaleString()}</p>
              <p className="text-sm text-slate-500 mt-1">{fraudRate}% rate</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm">Duration</span>
              </div>
              <p className="text-2xl font-bold text-white">{duration}h</p>
              <p className="text-sm text-slate-500 mt-1">Since start</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <CheckCircle className="w-4 h-4" />
                <span className="text-sm">Data Quality</span>
              </div>
              <p className="text-2xl font-bold text-green-400">92%</p>
              <p className="text-sm text-slate-500 mt-1">Valid records</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Charts */}
      <Tabs defaultValue="throughput" className="space-y-4">
        <TabsList className="bg-slate-900 border border-slate-800">
          <TabsTrigger value="throughput">Throughput</TabsTrigger>
          <TabsTrigger value="quality">Data Quality</TabsTrigger>
          <TabsTrigger value="distribution">Distribution</TabsTrigger>
        </TabsList>

        <TabsContent value="throughput">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg">Generation Rate (Last 7 Hours)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={historyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="records" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    dot={{ fill: '#3b82f6' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="fraud" 
                    stroke="#ef4444" 
                    strokeWidth={2}
                    dot={{ fill: '#ef4444' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="quality">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg">Data Quality Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-8">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={qualityData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {qualityData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                
                <div className="space-y-4">
                  {qualityData.map((item) => (
                    <div key={item.name} className="flex items-center justify-between p-3 bg-slate-950 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: item.color }}
                        />
                        <span className="text-slate-300">{item.name}</span>
                      </div>
                      <span className="font-semibold text-white">{item.value}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="distribution">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg">Data Issues Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                  { name: 'Missing Values', count: data.missingValues, color: '#eab308' },
                  { name: 'Duplicates', count: data.duplicates, color: '#f97316' },
                  { name: 'Outliers', count: data.outliers, color: '#ef4444' }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                    {
                      [0, 1, 2].map((_, index) => (
                        <Cell key={`cell-${index}`} fill={['#eab308', '#f97316', '#ef4444'][index]} />
                      ))
                    }
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Configuration */}
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white text-lg">Generator Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="text-sm text-slate-400">Records per Hour</label>
              <p className="text-lg font-semibold text-white">1,000</p>
            </div>
            <div>
              <label className="text-sm text-slate-400">Duration</label>
              <p className="text-lg font-semibold text-white">6 hours</p>
            </div>
            <div>
              <label className="text-sm text-slate-400">Fraud Rate</label>
              <p className="text-lg font-semibold text-white">5%</p>
            </div>
            <div>
              <label className="text-sm text-slate-400">Missing Values</label>
              <p className="text-lg font-semibold text-white">5%</p>
            </div>
            <div>
              <label className="text-sm text-slate-400">Duplicates</label>
              <p className="text-lg font-semibold text-white">2%</p>
            </div>
            <div>
              <label className="text-sm text-slate-400">Outliers</label>
              <p className="text-lg font-semibold text-white">1%</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
