/**
 * Updated App.tsx for Real Data Integration
 */

import { useState } from 'react';
import { 
  Activity, Database, Server, Shield, 
  TrendingUp, AlertTriangle, CheckCircle, 
  Zap, BarChart3, RefreshCw,
  Pause, Settings, Bell
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

import PipelineFlow from './sections/PipelineFlow';
import DataGeneratorPanel from './sections/DataGeneratorPanel';
import AirflowDashboard from './sections/AirflowDashboard';
import KafkaMonitor from './sections/KafkaMonitor';
import MLPredictions from './sections/MLPredictions';
import SnowflakePreview from './sections/SnowflakePreview';
import FraudAnalytics from './sections/FraudAnalytics';
import SystemMetrics from './sections/SystemMetrics';

import { 
  useDataGenerator, 
  useAirflowDAG, 
  useKafkaStream, 
  useMLAPI, 
  useSnowflake,
  useFraudAnalytics 
} from './hooks/useRealData';

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  
  const generator = useDataGenerator();
  const airflow = useAirflowDAG();
  const kafka = useKafkaStream();
  const ml = useMLAPI();
  const snowflake = useSnowflake();
  const fraud = useFraudAnalytics();

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
      case 'success':
      case 'online':
      case 'active':
        return 'bg-green-500';
      case 'warning':
      case 'paused':
        return 'bg-yellow-500';
      case 'error':
      case 'failed':
      case 'offline':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Type-safe data wrappers
  const generatorData = {
    ...generator,
    startTime: generator.startTime || ''
  };

  const airflowData = {
    ...airflow,
    lastRun: airflow.lastRun || '',
    nextRun: airflow.nextRun || ''
  };

  const snowflakeData = {
    ...snowflake,
    lastExport: snowflake.lastExport || ''
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-2 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg">
                <Shield className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                  Fraud Detection Pipeline
                </h1>
                <p className="text-sm text-slate-400">Real-time E-Commerce Transaction Monitoring</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-sm text-slate-400">Auto Refresh</span>
                <Button
                  variant={isAutoRefresh ? "default" : "outline"}
                  size="sm"
                  onClick={() => setIsAutoRefresh(!isAutoRefresh)}
                  className={isAutoRefresh ? "bg-green-600 hover:bg-green-700" : ""}
                >
                  {isAutoRefresh ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Pause className="w-4 h-4" />}
                </Button>
              </div>
              
              <Button variant="outline" size="icon">
                <Bell className="w-4 h-4" />
              </Button>
              
              <Button variant="outline" size="icon">
                <Settings className="w-4 h-4" />
              </Button>
            </div>
          </div>
          
          {/* Status Bar */}
          <div className="flex items-center gap-6 mt-4 text-sm">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${getStatusColor(generator.status)} animate-pulse`} />
              <span className="text-slate-400">Generator:</span>
              <span className="font-medium capitalize">{generator.status}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${getStatusColor(airflow.status)} animate-pulse`} />
              <span className="text-slate-400">Airflow:</span>
              <span className="font-medium capitalize">{airflow.status}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${getStatusColor(kafka.isConnected ? 'online' : 'offline')} animate-pulse`} />
              <span className="text-slate-400">Kafka:</span>
              <span className="font-medium">{kafka.isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${getStatusColor(ml.apiStatus)} animate-pulse`} />
              <span className="text-slate-400">ML API:</span>
              <span className="font-medium capitalize">{ml.apiStatus}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid grid-cols-7 gap-2 bg-slate-900 p-1">
            <TabsTrigger value="overview" className="data-[state=active]:bg-slate-800">
              <Activity className="w-4 h-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="generator" className="data-[state=active]:bg-slate-800">
              <Database className="w-4 h-4 mr-2" />
              Generator
            </TabsTrigger>
            <TabsTrigger value="airflow" className="data-[state=active]:bg-slate-800">
              <Server className="w-4 h-4 mr-2" />
              Airflow
            </TabsTrigger>
            <TabsTrigger value="kafka" className="data-[state=active]:bg-slate-800">
              <Zap className="w-4 h-4 mr-2" />
              Kafka
            </TabsTrigger>
            <TabsTrigger value="ml" className="data-[state=active]:bg-slate-800">
              <TrendingUp className="w-4 h-4 mr-2" />
              ML API
            </TabsTrigger>
            <TabsTrigger value="snowflake" className="data-[state=active]:bg-slate-800">
              <Database className="w-4 h-4 mr-2" />
              Snowflake
            </TabsTrigger>
            <TabsTrigger value="analytics" className="data-[state=active]:bg-slate-800">
              <BarChart3 className="w-4 h-4 mr-2" />
              Analytics
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <PipelineFlow 
              generator={generator}
              airflow={airflow}
              kafka={kafka}
              ml={ml}
              snowflake={snowflake}
            />
            
            <div className="grid grid-cols-4 gap-4">
              <Card className="bg-slate-900 border-slate-800">
                <CardHeader className="pb-2">
                  <CardDescription className="text-slate-400">Total Transactions</CardDescription>
                  <CardTitle className="text-3xl text-white">{fraud.totalTransactions.toLocaleString()}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center text-sm text-green-400">
                    <TrendingUp className="w-4 h-4 mr-1" />
                    +{generator.recordsPerHour.toLocaleString()}/hour
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-slate-900 border-slate-800">
                <CardHeader className="pb-2">
                  <CardDescription className="text-slate-400">Fraud Detected</CardDescription>
                  <CardTitle className="text-3xl text-red-400">{fraud.fraudCount.toLocaleString()}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center text-sm text-red-400">
                    <AlertTriangle className="w-4 h-4 mr-1" />
                    {fraud.fraudRate.toFixed(2)}% rate
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-slate-900 border-slate-800">
                <CardHeader className="pb-2">
                  <CardDescription className="text-slate-400">ML Predictions</CardDescription>
                  <CardTitle className="text-3xl text-blue-400">{ml.totalPredictions.toLocaleString()}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center text-sm text-blue-400">
                    <CheckCircle className="w-4 h-4 mr-1" />
                    {(ml.averageConfidence * 100).toFixed(1)}% avg confidence
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-slate-900 border-slate-800">
                <CardHeader className="pb-2">
                  <CardDescription className="text-slate-400">Kafka Messages</CardDescription>
                  <CardTitle className="text-3xl text-purple-400">{kafka.totalMessages.toLocaleString()}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center text-sm text-purple-400">
                    <Zap className="w-4 h-4 mr-1" />
                    {kafka.messagesPerSecond}/sec
                  </div>
                </CardContent>
              </Card>
            </div>
            
            <SystemMetrics />
          </TabsContent>

          {/* Generator Tab */}
          <TabsContent value="generator">
            <DataGeneratorPanel data={generatorData as any} />
          </TabsContent>

          {/* Airflow Tab */}
          <TabsContent value="airflow">
            <AirflowDashboard data={airflowData as any} />
          </TabsContent>

          {/* Kafka Tab */}
          <TabsContent value="kafka">
            <KafkaMonitor data={kafka as any} />
          </TabsContent>

          {/* ML Tab */}
          <TabsContent value="ml">
            <MLPredictions data={ml as any} />
          </TabsContent>

          {/* Snowflake Tab */}
          <TabsContent value="snowflake">
            <SnowflakePreview data={snowflakeData as any} />
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics">
            <FraudAnalytics data={fraud as any} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default App;