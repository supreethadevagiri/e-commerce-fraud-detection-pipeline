// Types for Fraud Detection Pipeline Dashboard

export interface Transaction {
  transaction_id: string;
  customer_id: string;
  timestamp: string;
  amount: number;
  category: string;
  payment_method: string;
  device_type: string;
  country: string;
  merchant_id: string;
  is_fraudulent: boolean;
  fraud_type?: string;
  risk_score?: number;
}

export interface PipelineStatus {
  status: 'running' | 'stopped' | 'error' | 'warning';
  message: string;
  lastUpdated: string;
}

export interface DataGeneratorMetrics {
  recordsGenerated: number;
  recordsPerHour: number;
  fraudCount: number;
  missingValues: number;
  duplicates: number;
  outliers: number;
  startTime: string;
  isRunning: boolean;
}

export interface AirflowTask {
  id: string;
  name: string;
  status: 'success' | 'running' | 'failed' | 'pending' | 'skipped';
  startTime?: string;
  endTime?: string;
  duration?: number;
  logs?: string[];
}

export interface AirflowDAG {
  dagId: string;
  status: 'running' | 'success' | 'failed' | 'paused';
  lastRun: string;
  nextRun: string;
  totalRuns: number;
  tasks: AirflowTask[];
  schedule: string;
}

export interface KafkaMessage {
  key: string;
  value: Transaction;
  timestamp: string;
  topic: string;
  partition: number;
  offset: number;
}

export interface KafkaMetrics {
  messagesPerSecond: number;
  totalMessages: number;
  topics: string[];
  partitions: number;
  lag: number;
  isConnected: boolean;
}

export interface MLPrediction {
  transactionId: string;
  prediction: 'fraud' | 'legitimate';
  confidence: number;
  riskScore: number;
  features: Record<string, number>;
  timestamp: string;
}

export interface MLMetrics {
  totalPredictions: number;
  fraudDetected: number;
  averageConfidence: number;
  modelVersion: string;
  apiStatus: 'online' | 'offline';
  responseTime: number;
}

export interface SnowflakeTable {
  name: string;
  schema: string;
  database: string;
  rowCount: number;
  lastUpdated: string;
  sizeMB: number;
}

export interface SnowflakeMetrics {
  tables: SnowflakeTable[];
  totalRows: number;
  lastExport: string;
  exportStatus: 'success' | 'failed' | 'in_progress';
}

export interface FraudAnalytics {
  totalTransactions: number;
  fraudCount: number;
  fraudRate: number;
  fraudByCategory: Record<string, number>;
  fraudByCountry: Record<string, number>;
  fraudByHour: number[];
  riskDistribution: number[];
  topRiskyMerchants: { merchantId: string; riskScore: number }[];
}

export interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkIn: number;
  networkOut: number;
  uptime: number;
}

export interface PipelineStage {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'inactive' | 'error';
  throughput: number;
  latency: number;
  icon: string;
}
