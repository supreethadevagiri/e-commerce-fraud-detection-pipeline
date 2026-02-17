import { useState, useEffect } from 'react';
import { 
  TrendingUp, AlertTriangle, CheckCircle, 
  Clock, Brain 
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';

interface MLPredictionsProps {
  data: {
    totalPredictions: number;
    fraudDetected: number;
    averageConfidence: number;
    modelVersion: string;
    apiStatus: string;
    responseTime: number;
    predictions: any[];
  };
}

const confidenceDistribution = [
  { range: '90-100%', count: 3421, color: '#22c55e' },
  { range: '80-90%', count: 2893, color: '#84cc16' },
  { range: '70-80%', count: 1876, color: '#eab308' },
  { range: '60-70%', count: 543, color: '#f97316' },
  { range: '<60%', count: 234, color: '#ef4444' }
];

const fraudByHour = Array.from({ length: 24 }, (_, i) => ({
  hour: `${i}:00`,
  fraud: Math.floor(Math.random() * 50) + 20,
  legitimate: Math.floor(Math.random() * 200) + 100
}));

export default function MLPredictions({ data }: MLPredictionsProps) {
  const [predictions, setPredictions] = useState(data.predictions);
  const isLive = true;

  // Simulate real-time predictions
  useEffect(() => {
    if (!isLive) return;

    const interval = setInterval(() => {
      const isFraud = Math.random() < 0.05;
      const confidence = 0.7 + Math.random() * 0.28;
      
      const newPrediction = {
        transactionId: `TXN-${Math.random().toString(36).substr(2, 16).toUpperCase()}`,
        prediction: isFraud ? 'fraud' : 'legitimate',
        confidence: confidence,
        riskScore: isFraud ? Math.floor(70 + Math.random() * 30) : Math.floor(Math.random() * 40),
        features: {
          transaction_amount: Math.round(Math.random() * 500 * 100) / 100,
          transaction_hour: Math.floor(Math.random() * 24),
          merchant_risk_score: Math.round(Math.random() * 100),
          customer_age_months: Math.floor(Math.random() * 60),
          num_transactions_24h: Math.floor(Math.random() * 20)
        },
        timestamp: new Date().toISOString()
      };

      setPredictions(prev => [newPrediction, ...prev].slice(0, 30));
    }, 2000);

    return () => clearInterval(interval);
  }, [isLive]);

  const fraudRate = ((data.fraudDetected / data.totalPredictions) * 100).toFixed(2);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-orange-500/20 rounded-lg">
                <Brain className="w-6 h-6 text-orange-400" />
              </div>
              <div>
                <CardTitle className="text-white">ML Fraud Detection API</CardTitle>
                <CardDescription className="text-slate-400">
                  Real-time fraud prediction and risk scoring
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge 
                variant="outline" 
                className={data.apiStatus === 'online' 
                  ? 'bg-green-500/20 text-green-400 border-green-500/30' 
                  : 'bg-red-500/20 text-red-400 border-red-500/30'
                }
              >
                {data.apiStatus === 'online' ? 'API Online' : 'API Offline'}
              </Badge>
              <Badge variant="outline" className="bg-blue-500/20 text-blue-400 border-blue-500/30">
                v{data.modelVersion}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <TrendingUp className="w-4 h-4" />
                <span className="text-sm">Total Predictions</span>
              </div>
              <p className="text-2xl font-bold text-white">{data.totalPredictions.toLocaleString()}</p>
              <p className="text-sm text-green-400 mt-1">+45/min</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <AlertTriangle className="w-4 h-4" />
                <span className="text-sm">Fraud Detected</span>
              </div>
              <p className="text-2xl font-bold text-red-400">{data.fraudDetected.toLocaleString()}</p>
              <p className="text-sm text-slate-500 mt-1">{fraudRate}% of total</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <CheckCircle className="w-4 h-4" />
                <span className="text-sm">Avg Confidence</span>
              </div>
              <p className="text-2xl font-bold text-green-400">{(data.averageConfidence * 100).toFixed(1)}%</p>
              <p className="text-sm text-slate-500 mt-1">Model accuracy</p>
            </div>
            
            <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm">Response Time</span>
              </div>
              <p className="text-2xl font-bold text-blue-400">{data.responseTime}ms</p>
              <p className="text-sm text-slate-500 mt-1">Average latency</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Charts and Predictions */}
      <Tabs defaultValue="predictions" className="space-y-4">
        <TabsList className="bg-slate-900 border border-slate-800">
          <TabsTrigger value="predictions">Live Predictions</TabsTrigger>
          <TabsTrigger value="confidence">Confidence Distribution</TabsTrigger>
          <TabsTrigger value="patterns">Fraud Patterns</TabsTrigger>
        </TabsList>

        <TabsContent value="predictions">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-white text-lg">Real-time Predictions</CardTitle>
                  <CardDescription className="text-slate-400">
                    Live fraud detection results
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  <span className="text-sm text-slate-400">Live</span>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px] w-full rounded-md border border-slate-800">
                <div className="space-y-2 p-4">
                  {predictions.map((pred, index) => (
                    <div 
                      key={index} 
                      className={`p-4 rounded-lg border ${
                        pred.prediction === 'fraud' 
                          ? 'bg-red-950/30 border-red-800' 
                          : 'bg-green-950/30 border-green-800'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          {pred.prediction === 'fraud' ? (
                            <AlertTriangle className="w-5 h-5 text-red-400" />
                          ) : (
                            <CheckCircle className="w-5 h-5 text-green-400" />
                          )}
                          <span className="font-medium text-white">{pred.transactionId}</span>
                          <Badge 
                            variant="outline" 
                            className={pred.prediction === 'fraud' 
                              ? 'bg-red-500/20 text-red-400 border-red-500/30' 
                              : 'bg-green-500/20 text-green-400 border-green-500/30'
                            }
                          >
                            {pred.prediction.toUpperCase()}
                          </Badge>
                        </div>
                        <span className="text-sm text-slate-500">
                          {new Date(pred.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-4 mb-3">
                        <div>
                          <span className="text-sm text-slate-400">Confidence</span>
                          <div className="flex items-center gap-2">
                            <Progress value={pred.confidence * 100} className="w-24 h-2" />
                            <span className="text-sm text-white">{(pred.confidence * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                        <div>
                          <span className="text-sm text-slate-400">Risk Score</span>
                          <p className={`font-semibold ${
                            pred.riskScore > 70 ? 'text-red-400' : 
                            pred.riskScore > 40 ? 'text-yellow-400' : 'text-green-400'
                          }`}>
                            {pred.riskScore}/100
                          </p>
                        </div>
                        <div>
                          <span className="text-sm text-slate-400">Amount</span>
                          <p className="text-white">${pred.features.transaction_amount}</p>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-4 gap-2 text-xs">
                        <div className="bg-slate-950 p-2 rounded">
                          <span className="text-slate-500">Hour: </span>
                          <span className="text-slate-300">{pred.features.transaction_hour}:00</span>
                        </div>
                        <div className="bg-slate-950 p-2 rounded">
                          <span className="text-slate-500">Merchant Risk: </span>
                          <span className="text-slate-300">{pred.features.merchant_risk_score}</span>
                        </div>
                        <div className="bg-slate-950 p-2 rounded">
                          <span className="text-slate-500">Customer Age: </span>
                          <span className="text-slate-300">{pred.features.customer_age_months}mo</span>
                        </div>
                        <div className="bg-slate-950 p-2 rounded">
                          <span className="text-slate-500">Txns 24h: </span>
                          <span className="text-slate-300">{pred.features.num_transactions_24h}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="confidence">
          <div className="grid grid-cols-2 gap-6">
            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white text-lg">Confidence Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={confidenceDistribution} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis type="number" stroke="#94a3b8" />
                    <YAxis dataKey="range" type="category" stroke="#94a3b8" width={80} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    />
                    <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                      {confidenceDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white text-lg">Model Performance</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-slate-400">Precision</span>
                    <span className="text-white">94.2%</span>
                  </div>
                  <Progress value={94.2} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-slate-400">Recall</span>
                    <span className="text-white">91.8%</span>
                  </div>
                  <Progress value={91.8} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-slate-400">F1 Score</span>
                    <span className="text-white">93.0%</span>
                  </div>
                  <Progress value={93.0} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-slate-400">AUC-ROC</span>
                    <span className="text-white">97.5%</span>
                  </div>
                  <Progress value={97.5} className="h-2" />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="patterns">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg">Fraud Patterns by Hour</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={fraudByHour}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="hour" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  />
                  <Bar dataKey="legitimate" stackId="a" fill="#22c55e" name="Legitimate" />
                  <Bar dataKey="fraud" stackId="a" fill="#ef4444" name="Fraud" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
