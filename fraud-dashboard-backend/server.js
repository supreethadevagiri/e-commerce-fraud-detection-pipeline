/**
 * Fraud Detection Pipeline Dashboard - Backend API
 */

const express = require('express');
const cors = require('cors');
const WebSocket = require('ws');
const http = require('http');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const glob = require('glob');
const cron = require('node-cron');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server, path: '/ws/kafka' });

app.use(cors());
app.use(express.json());

const CONFIG = {
  AIRFLOW_DATA_PATH: process.env.AIRFLOW_DATA_PATH || '/home/user/airflow/data',
  AIRFLOW_URL: process.env.AIRFLOW_URL || 'http://0.0.0.0:8085',
  AIRFLOW_USER: process.env.AIRFLOW_USER || 'admin',
  AIRFLOW_PASS: process.env.AIRFLOW_PASS || 'admin',
  KAFKA_BROKERS: (process.env.KAFKA_BROKERS || 'localhost:9092').split(','),
  KAFKA_TOPIC: process.env.KAFKA_TOPIC || 'ecommerce-transactions',
  ML_API_URL: process.env.ML_API_URL || 'http://localhost:4500',
  SNOWFLAKE_ACCOUNT: process.env.SNOWFLAKE_ACCOUNT || 'SFEDU02-FEB92475',
  SNOWFLAKE_USER: process.env.SNOWFLAKE_USER || 'BADGER',
  SNOWFLAKE_DATABASE: process.env.SNOWFLAKE_DATABASE || 'ECOMMERCE_DW',
  SNOWFLAKE_WAREHOUSE: process.env.SNOWFLAKE_WAREHOUSE || 'ECOMMERCE_LOAD_WH',
  SNOWFLAKE_ROLE: process.env.SNOWFLAKE_ROLE || 'TRAINING_ROLE',
  SNOWFLAKE_KEY_PATH: process.env.SNOWFLAKE_KEY_PATH || '/home/user/snowflake_key.p8',
};

const dataStore = {
  generator: { recordsGenerated: 0, recordsPerHour: 1000, fraudCount: 0, missingValues: 0, duplicates: 0, outliers: 0, startTime: '', isRunning: false, status: 'stopped' },
  airflow: { dagId: 'ecommerce_batch_pipeline', status: 'unknown', lastRun: '', nextRun: '', totalRuns: 0, schedule: '0 */6 * * *', tasks: [] },
  kafka: { messagesPerSecond: 0, totalMessages: 0, topics: ['ecommerce-transactions', 'fraud-alerts', 'ml-predictions'], partitions: 6, lag: 0, isConnected: false, recentMessages: [] },
  ml: { totalPredictions: 0, fraudDetected: 0, averageConfidence: 0.87, modelVersion: 'v1.0.0', apiStatus: 'offline', responseTime: 0, recentPredictions: [] },
  snowflake: { tables: [], totalRows: 0, lastExport: '', exportStatus: 'unknown' },
  fraud: { totalTransactions: 0, fraudCount: 0, fraudRate: 0, fraudByCategory: {}, fraudByCountry: {}, fraudByHour: new Array(24).fill(0), riskDistribution: [45, 32, 18, 5], topRiskyMerchants: [] }
};

async function analyzeCSVFiles() {
  const rawPath = path.join(CONFIG.AIRFLOW_DATA_PATH, 'raw');
  let totalRecords = 0, fraudCount = 0, missingValues = 0, duplicates = 0, outliers = 0;
  
  try {
    const csvFiles = glob.sync('*.csv', { cwd: rawPath });
    
    for (const file of csvFiles) {
      const filePath = path.join(rawPath, file);
      const stats = fs.statSync(filePath);
      
      if (Date.now() - stats.mtime.getTime() < 3600000) {
        dataStore.generator.isRunning = true;
        dataStore.generator.status = 'running';
        if (!dataStore.generator.startTime) dataStore.generator.startTime = stats.mtime.toISOString();
      }
      
      await new Promise((resolve, reject) => {
        const seenIds = new Set();
        fs.createReadStream(filePath).pipe(csv()).on('data', (row) => {
          totalRecords++;
          if (row.is_fraudulent === 'true' || row.is_fraudulent === '1' || row.is_fraudulent === true) fraudCount++;
          if (!row.payment_method || !row.device_type || !row.category) missingValues++;
          if (seenIds.has(row.transaction_id)) duplicates++; else seenIds.add(row.transaction_id);
          const amount = parseFloat(row.amount);
          if (amount > 10000 || amount < 1) outliers++;
        }).on('end', resolve).on('error', reject);
      });
    }
    
    dataStore.generator.recordsGenerated = totalRecords;
    dataStore.generator.fraudCount = fraudCount;
    dataStore.generator.missingValues = missingValues;
    dataStore.generator.duplicates = duplicates;
    dataStore.generator.outliers = outliers;
    dataStore.fraud.totalTransactions = totalRecords;
    dataStore.fraud.fraudCount = fraudCount;
    dataStore.fraud.fraudRate = totalRecords > 0 ? (fraudCount / totalRecords) * 100 : 0;
  } catch (error) {
    console.error('Error analyzing CSV files:', error.message);
  }
}

async function fetchAirflowStatus() {
  console.log('Airflow skipped - using mock data');
  dataStore.airflow = {
    dagId: 'ecommerce_batch_pipeline', status: 'running',
    lastRun: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    nextRun: new Date(Date.now() + 30 * 60 * 1000).toISOString(),
    totalRuns: 147, schedule: '0 */6 * * *',
    tasks: [
      { id: '1', name: 'ingest_data', status: 'success', startTime: '2024-01-15T10:00:00Z', endTime: '2024-01-15T10:02:15Z', duration: 135 },
      { id: '2', name: 'validate_raw_data', status: 'success', startTime: '2024-01-15T10:02:16Z', endTime: '2024-01-15T10:03:42Z', duration: 86 },
      { id: '3', name: 'clean_data', status: 'success', startTime: '2024-01-15T10:03:43Z', endTime: '2024-01-15T10:05:21Z', duration: 98 },
      { id: '4', name: 'validate_cleaned_data', status: 'success', startTime: '2024-01-15T10:05:22Z', endTime: '2024-01-15T10:06:15Z', duration: 53 },
      { id: '5', name: 'aggregate_data', status: 'success', startTime: '2024-01-15T10:06:16Z', endTime: '2024-01-15T10:08:45Z', duration: 149 },
      { id: '6', name: 'engineer_features', status: 'success', startTime: '2024-01-15T10:08:46Z', endTime: '2024-01-15T10:12:33Z', duration: 227 },
      { id: '7', name: 'export_to_snowflake', status: 'running', startTime: '2024-01-15T10:12:34Z', duration: 45 }
    ]
  };
}

async function checkMLAPI() {
  try {
    const startTime = Date.now();
    const response = await axios.get(`${CONFIG.ML_API_URL}/health`, { timeout: 5000 });
    const responseTime = Date.now() - startTime;
    
    dataStore.ml.apiStatus = response.data.status === 'healthy' ? 'online' : 'offline';
    dataStore.ml.responseTime = responseTime;
    if (response.data.model_loaded) dataStore.ml.modelVersion = 'v1.0.0';
  } catch (error) {
    console.error('ML API unreachable:', error.message);
    dataStore.ml.apiStatus = 'offline';
  }
}

async function querySnowflake() {
  try {
    const snowflake = require('snowflake-sdk');
    const connection = snowflake.createConnection({
      account: CONFIG.SNOWFLAKE_ACCOUNT, username: CONFIG.SNOWFLAKE_USER,
      authenticator: 'SNOWFLAKE_JWT', privateKeyPath: CONFIG.SNOWFLAKE_KEY_PATH,
      database: CONFIG.SNOWFLAKE_DATABASE, warehouse: CONFIG.SNOWFLAKE_WAREHOUSE, role: CONFIG.SNOWFLAKE_ROLE
    });
    await new Promise((resolve, reject) => connection.connect((err) => err ? reject(err) : resolve()));
    
    const tables = [
      { name: 'RAW_TRANSACTIONS', schema: 'RAW_DATA' }, { name: 'CLEANED_TRANSACTIONS', schema: 'CLEANED_DATA' },
      { name: 'HOURLY_CATEGORY_SUMMARY', schema: 'ANALYTICS' }, { name: 'HOURLY_OVERALL_SUMMARY', schema: 'ANALYTICS' },
      { name: 'DAILY_CATEGORY_SUMMARY', schema: 'ANALYTICS' }, { name: 'FRAUD_DETECTION_FEATURES', schema: 'ML_FEATURES' }
    ];
    
    dataStore.snowflake.tables = []; let totalRows = 0;
    for (const table of tables) {
      try {
        const rows = await new Promise((resolve, reject) => {
          connection.execute({ sqlText: `SELECT COUNT(*) as count FROM ${CONFIG.SNOWFLAKE_DATABASE}.${table.schema}.${table.name}`,
            complete: (err, stmt, rows) => err ? reject(err) : resolve(rows[0].COUNT) });
        });
        dataStore.snowflake.tables.push({ name: table.name, schema: table.schema, database: CONFIG.SNOWFLAKE_DATABASE, rowCount: rows, lastUpdated: new Date().toISOString(), sizeMB: 0 });
        totalRows += rows;
      } catch (e) { console.error(`Error querying ${table.name}:`, e.message); }
    }
    dataStore.snowflake.totalRows = totalRows;
    dataStore.snowflake.lastExport = new Date().toISOString();
    dataStore.snowflake.exportStatus = 'success';
    connection.destroy();
  } catch (error) {
    console.error('Snowflake query error:', error.message);
    dataStore.snowflake.exportStatus = 'error';
  }
}

let kafkaConsumer = null;
async function setupKafkaConsumer() {
  try {
    const { Kafka } = require('kafkajs');
    const kafka = new Kafka({ clientId: 'fraud-dashboard', brokers: CONFIG.KAFKA_BROKERS });
    kafkaConsumer = kafka.consumer({ groupId: 'dashboard-consumer' });
    await kafkaConsumer.connect();
    await kafkaConsumer.subscribe({ topic: CONFIG.KAFKA_TOPIC, fromBeginning: false });
    dataStore.kafka.isConnected = true;
    
    await kafkaConsumer.run({
      eachMessage: async ({ topic, partition, message }) => {
        const value = JSON.parse(message.value.toString());
        dataStore.kafka.totalMessages++;
        dataStore.kafka.recentMessages.unshift({ key: message.key?.toString() || '', value, timestamp: new Date().toISOString(), topic, partition, offset: message.offset });
        if (dataStore.kafka.recentMessages.length > 100) dataStore.kafka.recentMessages.pop();
        wss.clients.forEach(client => { if (client.readyState === WebSocket.OPEN) client.send(JSON.stringify({ type: 'kafka-message', data: dataStore.kafka.recentMessages[0] })); });
      }
    });
    console.log('Kafka consumer connected');
  } catch (error) {
    console.error('Kafka setup error:', error.message);
    dataStore.kafka.isConnected = false;
  }
}

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  ws.send(JSON.stringify({ type: 'initial', data: dataStore.kafka }));
  ws.on('close', () => console.log('WebSocket client disconnected'));
});

app.get('/api/health', (req, res) => res.json({ status: 'ok', timestamp: new Date().toISOString() }));
app.get('/api/generator', (req, res) => res.json(dataStore.generator));
app.get('/api/airflow', (req, res) => res.json(dataStore.airflow));
app.get('/api/kafka', (req, res) => res.json(dataStore.kafka));
app.get('/api/ml', (req, res) => res.json(dataStore.ml));

app.post('/api/ml/predict', async (req, res) => {
  try {
    const response = await axios.post(`${CONFIG.ML_API_URL}/predict`, req.body, {
      timeout: 10000,
      headers: { 'Content-Type': 'application/json' }
    });
    
    dataStore.ml.totalPredictions++;
    if (response.data.is_fraud) dataStore.ml.fraudDetected++;
    
    dataStore.ml.recentPredictions.unshift({
      transactionId: req.body.transaction_id || `TXN-${Date.now()}`,
      prediction: response.data.is_fraud ? 'fraud' : 'legitimate',
      confidence: response.data.fraud_probability,
      riskScore: Math.round(response.data.fraud_probability * 100),
      riskLevel: response.data.risk_level,
      features: req.body,
      timestamp: response.data.timestamp || new Date().toISOString()
    });
    if (dataStore.ml.recentPredictions.length > 50) dataStore.ml.recentPredictions.pop();
    
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/snowflake', (req, res) => res.json(dataStore.snowflake));
app.get('/api/fraud', (req, res) => res.json(dataStore.fraud));
app.get('/api/system', (req, res) => {
  const os = require('os');
  const totalMem = os.totalmem(), freeMem = os.freemem();
  res.json({ cpuUsage: os.loadavg()[0] * 10, memoryUsage: Math.round(((totalMem - freeMem) / totalMem) * 100), diskUsage: 78, networkIn: 12.5, networkOut: 8.3, uptime: os.uptime() });
});
app.get('/api/all', (req, res) => res.json({ generator: dataStore.generator, airflow: dataStore.airflow, kafka: dataStore.kafka, ml: dataStore.ml, snowflake: dataStore.snowflake, fraud: dataStore.fraud }));
app.post('/api/refresh', async (req, res) => { await Promise.all([analyzeCSVFiles(), fetchAirflowStatus(), checkMLAPI(), querySnowflake()]); res.json({ status: 'refreshed' }); });

cron.schedule('*/30 * * * * *', async () => { await analyzeCSVFiles(); await fetchAirflowStatus(); await checkMLAPI(); });
cron.schedule('*/5 * * * *', async () => await querySnowflake());
let lastMessageCount = 0;
cron.schedule('* * * * * *', () => { const currentCount = dataStore.kafka.totalMessages; dataStore.kafka.messagesPerSecond = currentCount - lastMessageCount; lastMessageCount = currentCount; });

const PORT = process.env.PORT || 3001;
server.listen(PORT, async () => {
  console.log(`Server running on: http://localhost:${PORT}`);
  console.log(`WebSocket: ws://localhost:${PORT}/ws/kafka`);
  await analyzeCSVFiles(); await fetchAirflowStatus(); await checkMLAPI();
  setupKafkaConsumer().catch(console.error);
});

process.on('SIGINT', async () => {
  console.log('\nShutting down...');
  if (kafkaConsumer) await kafkaConsumer.disconnect();
  server.close(() => { console.log('Server closed'); process.exit(0); });
});