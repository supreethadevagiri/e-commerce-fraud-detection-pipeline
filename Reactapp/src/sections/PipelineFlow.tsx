import { 
  Database, Server, Zap, TrendingUp, 
  Cloud, ArrowRight, CheckCircle, AlertCircle, Clock 
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface PipelineFlowProps {
  generator: { status: string; recordsGenerated: number; recordsPerHour: number };
  airflow: { status: string; tasks: { status: string }[] };
  kafka: { isConnected: boolean; messagesPerSecond: number; totalMessages: number };
  ml: { apiStatus: string; totalPredictions: number; fraudDetected: number };
  snowflake: { tables: { name: string; rowCount: number }[]; exportStatus: string };
}

export default function PipelineFlow({ generator, airflow, kafka, ml, snowflake }: PipelineFlowProps) {
  const stages = [
    {
      id: 'generator',
      name: 'Data Generator',
      icon: Database,
      status: generator.status,
      metrics: `${generator.recordsGenerated.toLocaleString()} records`,
      throughput: `${generator.recordsPerHour}/hr`,
      color: 'from-blue-500 to-cyan-500'
    },
    {
      id: 'kafka',
      name: 'Kafka Stream',
      icon: Zap,
      status: kafka.isConnected ? 'running' : 'error',
      metrics: `${kafka.totalMessages.toLocaleString()} messages`,
      throughput: `${kafka.messagesPerSecond}/sec`,
      color: 'from-purple-500 to-pink-500'
    },
    {
      id: 'airflow',
      name: 'Airflow DAG',
      icon: Server,
      status: airflow.status,
      metrics: `${airflow.tasks.filter(t => t.status === 'success').length}/${airflow.tasks.length} tasks`,
      throughput: 'Every 6h',
      color: 'from-green-500 to-emerald-500'
    },
    {
      id: 'ml',
      name: 'ML API',
      icon: TrendingUp,
      status: ml.apiStatus,
      metrics: `${ml.totalPredictions.toLocaleString()} predictions`,
      throughput: `${ml.fraudDetected} frauds`,
      color: 'from-orange-500 to-amber-500'
    },
    {
      id: 'snowflake',
      name: 'Snowflake',
      icon: Cloud,
      status: snowflake.exportStatus,
      metrics: `${snowflake.tables.reduce((acc, t) => acc + t.rowCount, 0).toLocaleString()} rows`,
      throughput: `${snowflake.tables.length} tables`,
      color: 'from-indigo-500 to-violet-500'
    }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
      case 'success':
      case 'online':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'warning':
      case 'paused':
        return <Clock className="w-5 h-5 text-yellow-400" />;
      case 'error':
      case 'failed':
      case 'offline':
        return <AlertCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Clock className="w-5 h-5 text-slate-400" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, string> = {
      running: 'bg-green-500/20 text-green-400 border-green-500/30',
      success: 'bg-green-500/20 text-green-400 border-green-500/30',
      online: 'bg-green-500/20 text-green-400 border-green-500/30',
      error: 'bg-red-500/20 text-red-400 border-red-500/30',
      failed: 'bg-red-500/20 text-red-400 border-red-500/30',
      offline: 'bg-red-500/20 text-red-400 border-red-500/30',
      warning: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      paused: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      in_progress: 'bg-blue-500/20 text-blue-400 border-blue-500/30'
    };
    return variants[status] || 'bg-slate-500/20 text-slate-400 border-slate-500/30';
  };

  return (
    <Card className="bg-slate-900 border-slate-800">
      <CardContent className="p-6">
        <h2 className="text-xl font-semibold mb-6 text-white">Pipeline Flow</h2>
        
        <div className="flex items-center justify-between gap-4">
          {stages.map((stage, index) => (
            <div key={stage.id} className="flex items-center gap-4 flex-1">
              {/* Stage Card */}
              <div className="flex-1 relative">
                <div className={`p-4 rounded-xl bg-gradient-to-br ${stage.color} bg-opacity-10 border border-slate-700 hover:border-slate-600 transition-all`}>
                  <div className="flex items-start justify-between mb-3">
                    <div className={`p-2 rounded-lg bg-gradient-to-br ${stage.color}`}>
                      <stage.icon className="w-5 h-5 text-white" />
                    </div>
                    {getStatusIcon(stage.status)}
                  </div>
                  
                  <h3 className="font-semibold text-white mb-1">{stage.name}</h3>
                  <p className="text-sm text-slate-400 mb-2">{stage.metrics}</p>
                  
                  <div className="flex items-center justify-between">
                    <Badge variant="outline" className={`text-xs ${getStatusBadge(stage.status)}`}>
                      {stage.status}
                    </Badge>
                    <span className="text-xs text-slate-500">{stage.throughput}</span>
                  </div>
                </div>
              </div>
              
              {/* Arrow */}
              {index < stages.length - 1 && (
                <div className="flex-shrink-0">
                  <ArrowRight className="w-6 h-6 text-slate-600" />
                </div>
              )}
            </div>
          ))}
        </div>
        
        {/* Data Flow Animation */}
        <div className="mt-6 p-4 bg-slate-950 rounded-lg border border-slate-800">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">Data Flow:</span>
            <div className="flex items-center gap-4">
              <span className="text-blue-400">Generator → Kafka</span>
              <span className="text-purple-400">→ Airflow</span>
              <span className="text-orange-400">→ ML API</span>
              <span className="text-indigo-400">→ Snowflake</span>
            </div>
            <span className="text-green-400 flex items-center gap-2">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              Active
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
