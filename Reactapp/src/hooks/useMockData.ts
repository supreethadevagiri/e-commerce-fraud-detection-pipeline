import { useState, useCallback } from 'react';
import type { 
  DataGeneratorMetrics, 
  AirflowDAG, 
  KafkaMetrics, 
  MLMetrics, 
  SnowflakeMetrics,
  FraudAnalytics 
} from '@/types';

// Mock Data Generator Hook
export function useDataGenerator() {
  const [data, setData] = useState<DataGeneratorMetrics & { status: string }>({
    recordsGenerated: 15420,
    recordsPerHour: 1000,
    fraudCount: 771,
    missingValues: 771,
    duplicates: 308,
    outliers: 154,
    startTime: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
    isRunning: true,
    status: 'running'
  });

  const refresh = useCallback(() => {
    setData(prev => ({
      ...prev,
      recordsGenerated: prev.recordsGenerated + Math.floor(Math.random() * 50),
      fraudCount: prev.fraudCount + Math.floor(Math.random() * 3),
      missingValues: prev.missingValues + Math.floor(Math.random() * 2),
      duplicates: prev.duplicates + Math.floor(Math.random() * 2),
      outliers: prev.outliers + Math.floor(Math.random() * 1),
    }));
  }, []);

  return { ...data, refresh };
}

// Mock Airflow DAG Hook
export function useAirflowDAG() {
  const [data, setData] = useState<AirflowDAG & { status: string }>({
    dagId: 'ecommerce_batch_pipeline',
    status: 'running',
    lastRun: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    nextRun: new Date(Date.now() + 30 * 60 * 1000).toISOString(),
    totalRuns: 147,
    schedule: '0 */6 * * *',
    tasks: [
      { id: '1', name: 'ingest_data', status: 'success', startTime: '2024-01-15T10:00:00Z', endTime: '2024-01-15T10:02:15Z', duration: 135 },
      { id: '2', name: 'validate_raw_data', status: 'success', startTime: '2024-01-15T10:02:16Z', endTime: '2024-01-15T10:03:42Z', duration: 86 },
      { id: '3', name: 'clean_data', status: 'success', startTime: '2024-01-15T10:03:43Z', endTime: '2024-01-15T10:05:21Z', duration: 98 },
      { id: '4', name: 'validate_cleaned_data', status: 'success', startTime: '2024-01-15T10:05:22Z', endTime: '2024-01-15T10:06:15Z', duration: 53 },
      { id: '5', name: 'aggregate_data', status: 'success', startTime: '2024-01-15T10:06:16Z', endTime: '2024-01-15T10:08:45Z', duration: 149 },
      { id: '6', name: 'engineer_features', status: 'success', startTime: '2024-01-15T10:08:46Z', endTime: '2024-01-15T10:12:33Z', duration: 227 },
      { id: '7', name: 'export_to_snowflake', status: 'running', startTime: '2024-01-15T10:12:34Z', duration: 45 },
    ]
  });

  const refresh = useCallback(() => {
    setData(prev => ({
      ...prev,
      tasks: prev.tasks.map(task => 
        task.status === 'running' 
          ? { ...task, duration: (task.duration || 0) + 1 }
          : task
      )
    }));
  }, []);

  return { ...data, refresh };
}

// Mock Kafka Stream Hook
export function useKafkaStream() {
  const [data, setData] = useState<KafkaMetrics & { messages: any[] }>({
    messagesPerSecond: 45,
    totalMessages: 89234,
    topics: ['ecommerce-transactions', 'fraud-alerts', 'ml-predictions'],
    partitions: 6,
    lag: 12,
    isConnected: true,
    messages: []
  });

  const refresh = useCallback(() => {
    setData(prev => {
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
        offset: prev.totalMessages + 1
      };
      
      return {
        ...prev,
        messagesPerSecond: Math.floor(Math.random() * 20) + 35,
        totalMessages: prev.totalMessages + Math.floor(Math.random() * 10),
        lag: Math.max(0, prev.lag + Math.floor(Math.random() * 5) - 2),
        messages: [newMessage, ...prev.messages].slice(0, 50)
      };
    });
  }, []);

  return { ...data, refresh };
}

// Mock ML API Hook
export function useMLAPI() {
  const [data, setData] = useState<MLMetrics & { predictions: any[] }>({
    totalPredictions: 89234,
    fraudDetected: 4462,
    averageConfidence: 0.87,
    modelVersion: 'v2.1.0',
    apiStatus: 'online',
    responseTime: 45,
    predictions: []
  });

  const refresh = useCallback(() => {
    setData(prev => {
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
      
      return {
        ...prev,
        totalPredictions: prev.totalPredictions + 1,
        fraudDetected: isFraud ? prev.fraudDetected + 1 : prev.fraudDetected,
        averageConfidence: prev.averageConfidence + (Math.random() - 0.5) * 0.01,
        responseTime: Math.floor(Math.random() * 30) + 30,
        predictions: [newPrediction, ...prev.predictions].slice(0, 30)
      };
    });
  }, []);

  return { ...data, refresh };
}

// Mock Snowflake Hook
export function useSnowflake() {
  const [data, setData] = useState<SnowflakeMetrics>({
    tables: [
      { name: 'RAW_TRANSACTIONS', schema: 'RAW_DATA', database: 'ECOMMERCE_DW', rowCount: 89234, lastUpdated: '2024-01-15T10:12:00Z', sizeMB: 45.2 },
      { name: 'CLEANED_TRANSACTIONS', schema: 'CLEANED_DATA', database: 'ECOMMERCE_DW', rowCount: 87652, lastUpdated: '2024-01-15T10:15:00Z', sizeMB: 52.8 },
      { name: 'HOURLY_CATEGORY_SUMMARY', schema: 'ANALYTICS', database: 'ECOMMERCE_DW', rowCount: 1847, lastUpdated: '2024-01-15T10:18:00Z', sizeMB: 2.1 },
      { name: 'HOURLY_OVERALL_SUMMARY', schema: 'ANALYTICS', database: 'ECOMMERCE_DW', rowCount: 231, lastUpdated: '2024-01-15T10:18:00Z', sizeMB: 0.8 },
      { name: 'DAILY_CATEGORY_SUMMARY', schema: 'ANALYTICS', database: 'ECOMMERCE_DW', rowCount: 96, lastUpdated: '2024-01-15T10:18:00Z', sizeMB: 0.5 },
      { name: 'FRAUD_DETECTION_FEATURES', schema: 'ML_FEATURES', database: 'ECOMMERCE_DW', rowCount: 87652, lastUpdated: '2024-01-15T10:22:00Z', sizeMB: 128.4 },
    ],
    totalRows: 266612,
    lastExport: '2024-01-15T10:22:00Z',
    exportStatus: 'success'
  });

  const refresh = useCallback(() => {
    setData(prev => ({
      ...prev,
      tables: prev.tables.map(table => ({
        ...table,
        rowCount: table.rowCount + Math.floor(Math.random() * 10)
      })),
      totalRows: prev.totalRows + Math.floor(Math.random() * 50)
    }));
  }, []);

  return { ...data, refresh };
}

// Mock Fraud Analytics Hook
export function useFraudAnalytics() {
  const [data, setData] = useState<FraudAnalytics>({
    totalTransactions: 89234,
    fraudCount: 4462,
    fraudRate: 5.0,
    fraudByCategory: {
      'Electronics': 1245,
      'Clothing': 892,
      'Jewelry & Watches': 756,
      'Health & Beauty': 534,
      'Automotive': 423,
      'Home & Garden': 312,
      'Sports & Outdoors': 198,
      'Others': 102
    },
    fraudByCountry: {
      'Nigeria': 892,
      'Romania': 654,
      'Russia': 543,
      'Unknown': 432,
      'USA': 387,
      'UK': 298,
      'Others': 1256
    },
    fraudByHour: [45, 38, 42, 56, 78, 92, 87, 76, 65, 54, 48, 52, 58, 62, 71, 83, 95, 102, 98, 87, 76, 68, 59, 48],
    riskDistribution: [45, 32, 18, 5],
    topRiskyMerchants: [
      { merchantId: 'MERCH_042', riskScore: 94.2 },
      { merchantId: 'MERCH_087', riskScore: 91.8 },
      { merchantId: 'MERCH_015', riskScore: 89.5 },
      { merchantId: 'MERCH_063', riskScore: 87.3 },
      { merchantId: 'MERCH_029', riskScore: 85.1 }
    ]
  });

  const refresh = useCallback(() => {
    setData(prev => ({
      ...prev,
      totalTransactions: prev.totalTransactions + Math.floor(Math.random() * 10),
      fraudCount: prev.fraudCount + Math.floor(Math.random() * 2),
      fraudRate: prev.fraudRate + (Math.random() - 0.5) * 0.1
    }));
  }, []);

  return { ...data, refresh };
}
