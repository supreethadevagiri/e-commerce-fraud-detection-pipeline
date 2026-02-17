import { useState, useEffect } from 'react';
import { 
  Cpu, HardDrive, Wifi, Activity, 
  Server, Database 
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

// Generate mock system metrics data
const generateMetricsData = (points: number = 20) => {
  const data = [];
  for (let i = points; i >= 0; i--) {
    data.push({
      time: `${i * 5}s ago`,
      cpu: Math.floor(Math.random() * 30) + 40,
      memory: Math.floor(Math.random() * 20) + 50,
      network: Math.floor(Math.random() * 50) + 20
    });
  }
  return data;
};

export default function SystemMetrics() {
  const [metrics, setMetrics] = useState(generateMetricsData());
  const [currentMetrics, setCurrentMetrics] = useState({
    cpu: 45,
    memory: 62,
    disk: 78,
    networkIn: 12.5,
    networkOut: 8.3,
    uptime: 86400 * 3 // 3 days in seconds
  });

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => {
        const newData = [...prev.slice(1)];
        newData.push({
          time: 'now',
          cpu: Math.floor(Math.random() * 30) + 40,
          memory: Math.floor(Math.random() * 20) + 50,
          network: Math.floor(Math.random() * 50) + 20
        });
        return newData;
      });

      setCurrentMetrics(prev => ({
        ...prev,
        cpu: Math.floor(Math.random() * 30) + 40,
        memory: Math.floor(Math.random() * 20) + 50,
        networkIn: Math.round((Math.random() * 10 + 5) * 10) / 10,
        networkOut: Math.round((Math.random() * 8 + 3) * 10) / 10
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}d ${hours}h ${minutes}m`;
  };

  return (
    <div className="space-y-6">
      {/* Current Metrics */}
      <div className="grid grid-cols-6 gap-4">
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-400 flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              CPU Usage
            </CardDescription>
            <CardTitle className="text-2xl text-white">{currentMetrics.cpu}%</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={currentMetrics.cpu} className="h-2" />
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-400 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Memory
            </CardDescription>
            <CardTitle className="text-2xl text-white">{currentMetrics.memory}%</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={currentMetrics.memory} className="h-2" />
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-400 flex items-center gap-2">
              <HardDrive className="w-4 h-4" />
              Disk
            </CardDescription>
            <CardTitle className="text-2xl text-white">{currentMetrics.disk}%</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={currentMetrics.disk} className="h-2" />
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-400 flex items-center gap-2">
              <Wifi className="w-4 h-4" />
              Network In
            </CardDescription>
            <CardTitle className="text-2xl text-blue-400">{currentMetrics.networkIn} MB/s</CardTitle>
          </CardHeader>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-400 flex items-center gap-2">
              <Wifi className="w-4 h-4" />
              Network Out
            </CardDescription>
            <CardTitle className="text-2xl text-green-400">{currentMetrics.networkOut} MB/s</CardTitle>
          </CardHeader>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-400 flex items-center gap-2">
              <Server className="w-4 h-4" />
              Uptime
            </CardDescription>
            <CardTitle className="text-2xl text-white">{formatUptime(currentMetrics.uptime)}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader>
            <CardTitle className="text-white text-lg flex items-center gap-2">
              <Cpu className="w-5 h-5" />
              CPU Usage History
            </CardTitle>
            <CardDescription className="text-slate-400">
              CPU utilization over the last 2 minutes
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={metrics}>
                <defs>
                  <linearGradient id="colorCpu" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  formatter={(value: number) => [`${value}%`, 'CPU']}
                />
                <Area 
                  type="monotone" 
                  dataKey="cpu" 
                  stroke="#3b82f6" 
                  fillOpacity={1} 
                  fill="url(#colorCpu)" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader>
            <CardTitle className="text-white text-lg flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Memory Usage History
            </CardTitle>
            <CardDescription className="text-slate-400">
              Memory utilization over the last 2 minutes
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={metrics}>
                <defs>
                  <linearGradient id="colorMemory" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  formatter={(value: number) => [`${value}%`, 'Memory']}
                />
                <Area 
                  type="monotone" 
                  dataKey="memory" 
                  stroke="#8b5cf6" 
                  fillOpacity={1} 
                  fill="url(#colorMemory)" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Pipeline Component Status */}
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white text-lg flex items-center gap-2">
            <Database className="w-5 h-5" />
            Pipeline Component Health
          </CardTitle>
          <CardDescription className="text-slate-400">
            Status and resource usage of pipeline components
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-5 gap-4">
            {[
              { name: 'Data Generator', status: 'healthy', cpu: 12, memory: 8 },
              { name: 'Kafka Broker', status: 'healthy', cpu: 25, memory: 32 },
              { name: 'Airflow Scheduler', status: 'healthy', cpu: 8, memory: 15 },
              { name: 'ML API Server', status: 'healthy', cpu: 35, memory: 45 },
              { name: 'Snowflake Connector', status: 'healthy', cpu: 5, memory: 10 }
            ].map((component) => (
              <div key={component.name} className="p-4 bg-slate-950 rounded-lg border border-slate-800">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium text-white">{component.name}</span>
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                </div>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">CPU</span>
                      <span className="text-slate-300">{component.cpu}%</span>
                    </div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-blue-500 rounded-full transition-all"
                        style={{ width: `${component.cpu}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">Memory</span>
                      <span className="text-slate-300">{component.memory}%</span>
                    </div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-purple-500 rounded-full transition-all"
                        style={{ width: `${component.memory}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
