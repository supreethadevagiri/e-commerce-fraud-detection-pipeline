import { 
  AlertTriangle, TrendingUp, MapPin, ShoppingBag, 
  Clock, Shield, Activity 
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area
} from 'recharts';

interface FraudAnalyticsProps {
  data: {
    totalTransactions: number;
    fraudCount: number;
    fraudRate: number;
    fraudByCategory: Record<string, number>;
    fraudByCountry: Record<string, number>;
    fraudByHour: number[];
    riskDistribution: number[];
    topRiskyMerchants: { merchantId: string; riskScore: number }[];
  };
}

const COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899'];

export default function FraudAnalytics({ data }: FraudAnalyticsProps) {
  const categoryData = Object.entries(data.fraudByCategory).map(([name, value]) => ({ name, value }));
  const countryData = Object.entries(data.fraudByCountry).map(([name, value]) => ({ name, value }));
  const hourlyData = data.fraudByHour.map((count, hour) => ({ hour: `${hour}:00`, count }));
  
  const riskData = [
    { name: 'Low (0-25)', value: data.riskDistribution[0], color: '#22c55e' },
    { name: 'Medium (25-50)', value: data.riskDistribution[1], color: '#eab308' },
    { name: 'High (50-75)', value: data.riskDistribution[2], color: '#f97316' },
    { name: 'Critical (75-100)', value: data.riskDistribution[3], color: '#ef4444' }
  ];

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-400 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Total Transactions
            </CardDescription>
            <CardTitle className="text-3xl text-white">
              {data.totalTransactions.toLocaleString()}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-400 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              Fraud Cases
            </CardDescription>
            <CardTitle className="text-3xl text-red-400">
              {data.fraudCount.toLocaleString()}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-400 flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Fraud Rate
            </CardDescription>
            <CardTitle className="text-3xl text-orange-400">
              {data.fraudRate.toFixed(2)}%
            </CardTitle>
          </CardHeader>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-400 flex items-center gap-2">
              <Shield className="w-4 h-4" />
              Prevention Rate
            </CardDescription>
            <CardTitle className="text-3xl text-green-400">
              94.2%
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Analytics Tabs */}
      <Tabs defaultValue="category" className="space-y-4">
        <TabsList className="bg-slate-900 border border-slate-800">
          <TabsTrigger value="category">By Category</TabsTrigger>
          <TabsTrigger value="country">By Country</TabsTrigger>
          <TabsTrigger value="hourly">By Hour</TabsTrigger>
          <TabsTrigger value="risk">Risk Distribution</TabsTrigger>
          <TabsTrigger value="merchants">Risky Merchants</TabsTrigger>
        </TabsList>

        <TabsContent value="category">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg flex items-center gap-2">
                <ShoppingBag className="w-5 h-5" />
                Fraud by Category
              </CardTitle>
              <CardDescription className="text-slate-400">
                Distribution of fraudulent transactions across product categories
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-8">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={categoryData}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      dataKey="value"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {categoryData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    />
                  </PieChart>
                </ResponsiveContainer>

                <div className="space-y-2">
                  {categoryData.map((item, index) => (
                    <div key={item.name} className="flex items-center justify-between p-3 bg-slate-950 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: COLORS[index % COLORS.length] }}
                        />
                        <span className="text-slate-300">{item.name}</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-white font-semibold">{item.value}</span>
                        <span className="text-slate-500 text-sm">
                          {((item.value / data.fraudCount) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="country">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg flex items-center gap-2">
                <MapPin className="w-5 h-5" />
                Fraud by Country
              </CardTitle>
              <CardDescription className="text-slate-400">
                Geographic distribution of fraudulent transactions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={countryData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" stroke="#94a3b8" />
                  <YAxis dataKey="name" type="category" stroke="#94a3b8" width={100} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  />
                  <Bar dataKey="value" fill="#ef4444" radius={[0, 4, 4, 0]}>
                    {countryData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="hourly">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg flex items-center gap-2">
                <Clock className="w-5 h-5" />
                Fraud by Hour of Day
              </CardTitle>
              <CardDescription className="text-slate-400">
                Temporal patterns of fraudulent activity
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={hourlyData}>
                  <defs>
                    <linearGradient id="colorFraud" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="hour" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="count" 
                    stroke="#ef4444" 
                    fillOpacity={1} 
                    fill="url(#colorFraud)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="risk">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg flex items-center gap-2">
                <Shield className="w-5 h-5" />
                Risk Score Distribution
              </CardTitle>
              <CardDescription className="text-slate-400">
                Distribution of transactions by risk score ranges
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-8">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={riskData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      dataKey="value"
                      label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                    >
                      {riskData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    />
                  </PieChart>
                </ResponsiveContainer>

                <div className="space-y-3">
                  {riskData.map((item) => (
                    <div key={item.name} className="p-4 bg-slate-950 rounded-lg border border-slate-800">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: item.color }}
                          />
                          <span className="text-slate-300">{item.name}</span>
                        </div>
                        <span className="text-white font-semibold">{item.value}%</span>
                      </div>
                      <Progress 
                        value={item.value} 
                        className="h-2"
                        style={{ 
                          backgroundColor: '#1e293b',
                          ['--progress-background' as string]: item.color 
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="merchants">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white text-lg flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                Highest Risk Merchants
              </CardTitle>
              <CardDescription className="text-slate-400">
                Merchants with the highest average risk scores
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {data.topRiskyMerchants.map((merchant, index) => (
                  <div 
                    key={merchant.merchantId} 
                    className="flex items-center justify-between p-4 bg-slate-950 rounded-lg border border-slate-800"
                  >
                    <div className="flex items-center gap-4">
                      <span className="text-slate-500 font-mono">#{index + 1}</span>
                      <div>
                        <p className="font-medium text-white">{merchant.merchantId}</p>
                        <p className="text-sm text-slate-400">Merchant ID</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <p className={`text-lg font-semibold ${
                          merchant.riskScore > 90 ? 'text-red-400' : 
                          merchant.riskScore > 80 ? 'text-orange-400' : 'text-yellow-400'
                        }`}>
                          {merchant.riskScore.toFixed(1)}
                        </p>
                        <p className="text-sm text-slate-400">Risk Score</p>
                      </div>
                      <Badge 
                        variant="outline" 
                        className={merchant.riskScore > 90 
                          ? 'bg-red-500/20 text-red-400 border-red-500/30' 
                          : merchant.riskScore > 80 
                            ? 'bg-orange-500/20 text-orange-400 border-orange-500/30'
                            : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                        }
                      >
                        {merchant.riskScore > 90 ? 'Critical' : merchant.riskScore > 80 ? 'High' : 'Medium'}
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
