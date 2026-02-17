import { useState, useEffect } from 'react';
import { 
  Zap, Play, Pause, TrendingUp, 
  Layers, Activity, AlertCircle 
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

interface KafkaMonitorProps {
  data: {
    messagesPerSecond: number;
    totalMessages: number;
    topics: string[];
    partitions: number;
    lag: number;
    isConnected: boolean;
    messages: any[];
  };
}

// Generate mock throughput data
const generateThroughputData = () => {
  const data = [];
  for (let i = 60; i >= 0; i--) {
    data.push({
      time: `${i}s ago`,
      throughput: Math.floor(Math.random() * 30) + 30,
      lag: Math.floor(Math.random() * 20) + 5
    });
  }
  return data;
};

export default function KafkaMonitor({ data }: KafkaMonitorProps) {
  const [isStreaming, setIsStreaming] = useState(true);
  const [throughputData, setThroughputData] = useState(generateThroughputData());
  const [messages, setMessages] = useState(data.messages);

  // Simulate real-time updates
  useEffect(() => {
    if (!isStreaming) return;

    const interval = setInterval(() => {
      setThroughputData(prev => {
        const newData = [...prev.slice(1)];
        newData.push({
          time: 'now',
          throughput: Math.floor(Math.random() * 30) + 30,
          lag: Math.floor(Math.random() * 20) + 5
        });
        return newData;
      });

      // Add new message
      const newMessage = {
        key: `CUST-${Math.floor(Math.random() * 100000)}`,
        value: {
          transaction_id: `TXN-${Math.random().toString(36).substr(2, 16).toUpperCase()}`,
          customer_id: `CUST-${Math.floor(Math.random() * 100000)}`,
          amount: Math.round(Math.random() * 500 * 100) / 100,
          category: ['Electronics', 'Clothing', 'Home & Garden', 'Sports'][Math.floor(Math.random() * 4)],
          is_fraudulent: Math.random() < 0.05
        },
        timestamp: new Date().toISOString(),
        topic: 'ecommerce-transactions',
        partition: Math.floor(Math.random() * 6),
        offset: Math.floor(Math.random() * 100000)
      };

      setMessages(prev => [newMessage, ...prev].slice(0, 50));
    }, 1000);

    return () => clearInterval(interval);
  }, [isStreaming]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-500/20 rounded-lg">
                <Zap className="w-6 h-6 text-purple-400" />
              </div>
              <div>
                <CardTitle className="text-white">Kafka Stream Monitor</CardTitle>
                <CardDescription className="text-slate-400">
                  Real-time message streaming and monitoring
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge 
                variant="outline" 
                className={data.isConnected 
                  ? 'bg-green-500/20 text-green-400 border-green-500/30' 
                  : 'bg-red-500/20 text-red-400 border-red-500/30'
                }
              >
                {data.isConnected ? 'Connected' : 'Disconnected'}
              </Badge>
              <Button 
                variant={isStreaming ? "destructive" : "default"}
                size="sm"
                onClick={() => setIsStreaming(!isStreaming)}
              >
                {isStreaming ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                {isStreaming ? 'Pause' : 'Resume'}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-5 gap-4">
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <TrendingUp className="w-4 h-4" />
                <span className="text-sm">Messages/sec</span>
              </div>
              <p className="text-2xl font-bold text-white">{data.messagesPerSecond}</p>
              <p className="text-sm text-green-400 mt-1">+12% from avg</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <Layers className="w-4 h-4" />
                <span className="text-sm">Total Messages</span>
              </div>
              <p className="text-2xl font-bold text-purple-400">{data.totalMessages.toLocaleString()}</p>
              <p className="text-sm text-slate-500 mt-1">Since start</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <Activity className="w-4 h-4" />
                <span className="text-sm">Partitions</span>
              </div>
              <p className="text-2xl font-bold text-white">{data.partitions}</p>
              <p className="text-sm text-slate-500 mt-1">Across topics</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">Consumer Lag</span>
              </div>
              <p className="text-2xl font-bold text-yellow-400">{data.lag}</p>
              <p className="text-sm text-slate-500 mt-1">Messages behind</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <Layers className="w-4 h-4" />
                <span className="text-sm">Topics</span>
              </div>
              <p className="text-2xl font-bold text-white">{data.topics.length}</p>
              <p className="text-sm text-slate-500 mt-1">Active topics</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Charts and Messages */}
      <Tabs defaultValue="throughput" className="space-y-4">
        <TabsList className="bg-slate-900 border border-slate-800">
          <TabsTrigger value="throughput">Throughput</TabsTrigger>
          <TabsTrigger value="messages">Live Messages</TabsTrigger>
          <TabsTrigger value="topics">Topics</TabsTrigger>
        </TabsList>

        <TabsContent value="throughput">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg">Real-time Throughput (Last 60s)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={throughputData}>
                  <defs>
                    <linearGradient id="colorThroughput" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="throughput" 
                    stroke="#8b5cf6" 
                    fillOpacity={1} 
                    fill="url(#colorThroughput)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="messages">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg">Live Messages</CardTitle>
              <CardDescription className="text-slate-400">
                Real-time stream of Kafka messages
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px] w-full rounded-md border border-slate-800">
                <div className="space-y-2 p-4">
                  {messages.map((msg, index) => (
                    <div 
                      key={index} 
                      className="p-3 bg-slate-950 rounded-lg border border-slate-800 font-mono text-sm"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs bg-purple-500/20 text-purple-400 border-purple-500/30">
                            {msg.topic}
                          </Badge>
                          <span className="text-slate-500">Partition: {msg.partition}</span>
                          <span className="text-slate-500">Offset: {msg.offset}</span>
                        </div>
                        <span className="text-slate-500 text-xs">
                          {new Date(msg.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-slate-500">Key: </span>
                          <span className="text-slate-300">{msg.key}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Transaction: </span>
                          <span className="text-slate-300">{msg.value.transaction_id}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Amount: </span>
                          <span className="text-green-400">${msg.value.amount}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Category: </span>
                          <span className="text-blue-400">{msg.value.category}</span>
                        </div>
                        {msg.value.is_fraudulent && (
                          <div className="col-span-2">
                            <Badge variant="outline" className="bg-red-500/20 text-red-400 border-red-500/30">
                              FRAUD ALERT
                            </Badge>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="topics">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg">Kafka Topics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {data.topics.map((topic, index) => (
                  <div key={topic} className="flex items-center justify-between p-4 bg-slate-950 rounded-lg border border-slate-800">
                    <div className="flex items-center gap-4">
                      <div className="p-2 bg-purple-500/20 rounded-lg">
                        <Layers className="w-5 h-5 text-purple-400" />
                      </div>
                      <div>
                        <p className="font-medium text-white">{topic}</p>
                        <p className="text-sm text-slate-400">
                          {index === 0 ? 'Main transaction stream' : 
                           index === 1 ? 'Fraud detection alerts' : 
                           'ML prediction results'}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <div className="text-right">
                        <p className="text-white">{data.partitions} partitions</p>
                        <p className="text-slate-400">Replication: 3</p>
                      </div>
                      <Badge variant="outline" className="bg-green-500/20 text-green-400 border-green-500/30">
                        Active
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
