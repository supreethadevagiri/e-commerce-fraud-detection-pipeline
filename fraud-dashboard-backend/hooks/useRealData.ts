/**
 * Updated React Hooks for Real Data Integration
 * 
 * These hooks replace the mock data hooks and connect to the backend API.
 * Copy these files to your React app's src/hooks/ folder.
 */

// ============================================
// FILE: src/hooks/useRealData.ts
// ============================================

import { useState, useEffect, useCallback, useRef } from 'react';

// API Base URL - change this to your backend URL
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:3001';

// Generic fetch helper
async function fetchApi(endpoint: string) {
  const response = await fetch(`${API_URL}${endpoint}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}

// ============================================
// Data Generator Hook
// ============================================

export function useDataGenerator() {
  const [data, setData] = useState({
    recordsGenerated: 0,
    recordsPerHour: 1000,
    fraudCount: 0,
    missingValues: 0,
    duplicates: 0,
    outliers: 0,
    startTime: null,
    isRunning: false,
    status: 'stopped'
  });

  const refresh = useCallback(async () => {
    try {
      const result = await fetchApi('/api/generator');
      setData(result);
    } catch (error) {
      console.error('Failed to fetch generator data:', error);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [refresh]);

  return { ...data, refresh };
}

// ============================================
// Airflow DAG Hook
// ============================================

export function useAirflowDAG() {
  const [data, setData] = useState({
    dagId: 'ecommerce_batch_pipeline',
    status: 'unknown',
    lastRun: null,
    nextRun: null,
    totalRuns: 0,
    schedule: '0 */6 * * *',
    tasks: []
  });

  const refresh = useCallback(async () => {
    try {
      const result = await fetchApi('/api/airflow');
      setData(result);
    } catch (error) {
      console.error('Failed to fetch Airflow data:', error);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 30000);
    return () => clearInterval(interval);
  }, [refresh]);

  return { ...data, refresh };
}

// ============================================
// Kafka Stream Hook (with WebSocket)
// ============================================

export function useKafkaStream() {
  const [data, setData] = useState({
    messagesPerSecond: 0,
    totalMessages: 0,
    topics: ['ecommerce-transactions', 'fraud-alerts', 'ml-predictions'],
    partitions: 6,
    lag: 0,
    isConnected: false,
    messages: []
  });
  
  const wsRef = useRef<WebSocket | null>(null);

  const refresh = useCallback(async () => {
    try {
      const result = await fetchApi('/api/kafka');
      setData(prev => ({ ...prev, ...result }));
    } catch (error) {
      console.error('Failed to fetch Kafka data:', error);
    }
  }, []);

  useEffect(() => {
    // Initial fetch
    refresh();

    // Setup WebSocket for real-time messages
    const ws = new WebSocket(`${WS_URL}/ws/kafka`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Kafka WebSocket connected');
      setData(prev => ({ ...prev, isConnected: true }));
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === 'kafka-message') {
        setData(prev => ({
          ...prev,
          messages: [message.data, ...prev.messages].slice(0, 50)
        }));
      } else if (message.type === 'initial') {
        setData(prev => ({ ...prev, ...message.data }));
      }
    };

    ws.onclose = () => {
      console.log('Kafka WebSocket disconnected');
      setData(prev => ({ ...prev, isConnected: false }));
    };

    ws.onerror = (error) => {
      console.error('Kafka WebSocket error:', error);
      setData(prev => ({ ...prev, isConnected: false }));
    };

    // Poll for status updates
    const interval = setInterval(refresh, 30000);

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, [refresh]);

  return { ...data, refresh };
}

// ============================================
// ML API Hook
// ============================================

export function useMLAPI() {
  const [data, setData] = useState({
    totalPredictions: 0,
    fraudDetected: 0,
    averageConfidence: 0.87,
    modelVersion: 'v2.1.0',
    apiStatus: 'offline',
    responseTime: 0,
    predictions: []
  });

  const refresh = useCallback(async () => {
    try {
      const result = await fetchApi('/api/ml');
      setData(result);
    } catch (error) {
      console.error('Failed to fetch ML data:', error);
    }
  }, []);

  // Function to make a prediction
  const predict = useCallback(async (features: any) => {
    try {
      const response = await fetch(`${API_URL}/api/ml/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
      });
      return await response.json();
    } catch (error) {
      console.error('Prediction failed:', error);
      throw error;
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 30000);
    return () => clearInterval(interval);
  }, [refresh]);

  return { ...data, refresh, predict };
}

// ============================================
// Snowflake Hook
// ============================================

export function useSnowflake() {
  const [data, setData] = useState({
    tables: [],
    totalRows: 0,
    lastExport: null,
    exportStatus: 'unknown'
  });

  const refresh = useCallback(async () => {
    try {
      const result = await fetchApi('/api/snowflake');
      setData(result);
    } catch (error) {
      console.error('Failed to fetch Snowflake data:', error);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 300000); // Refresh every 5 minutes
    return () => clearInterval(interval);
  }, [refresh]);

  return { ...data, refresh };
}

// ============================================
// Fraud Analytics Hook
// ============================================

export function useFraudAnalytics() {
  const [data, setData] = useState({
    totalTransactions: 0,
    fraudCount: 0,
    fraudRate: 0,
    fraudByCategory: {},
    fraudByCountry: {},
    fraudByHour: new Array(24).fill(0),
    riskDistribution: [45, 32, 18, 5],
    topRiskyMerchants: []
  });

  const refresh = useCallback(async () => {
    try {
      const result = await fetchApi('/api/fraud');
      setData(result);
    } catch (error) {
      console.error('Failed to fetch fraud analytics:', error);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 30000);
    return () => clearInterval(interval);
  }, [refresh]);

  return { ...data, refresh };
}

// ============================================
// System Metrics Hook
// ============================================

export function useSystemMetrics() {
  const [data, setData] = useState({
    cpuUsage: 0,
    memoryUsage: 0,
    diskUsage: 0,
    networkIn: 0,
    networkOut: 0,
    uptime: 0
  });

  const refresh = useCallback(async () => {
    try {
      const result = await fetchApi('/api/system');
      setData(result);
    } catch (error) {
      console.error('Failed to fetch system metrics:', error);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, [refresh]);

  return { ...data, refresh };
}

// ============================================
// Combined Hook for All Data
// ============================================

export function useAllData() {
  const generator = useDataGenerator();
  const airflow = useAirflowDAG();
  const kafka = useKafkaStream();
  const ml = useMLAPI();
  const snowflake = useSnowflake();
  const fraud = useFraudAnalytics();

  const refreshAll = useCallback(async () => {
    await Promise.all([
      generator.refresh(),
      airflow.refresh(),
      kafka.refresh(),
      ml.refresh(),
      snowflake.refresh(),
      fraud.refresh()
    ]);
  }, [generator, airflow, kafka, ml, snowflake, fraud]);

  return {
    generator,
    airflow,
    kafka,
    ml,
    snowflake,
    fraud,
    refreshAll
  };
}
