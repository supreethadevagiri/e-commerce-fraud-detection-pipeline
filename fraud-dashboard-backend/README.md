# Fraud Detection Pipeline Dashboard - Real Data Integration

This folder contains everything you need to connect your React dashboard to real data sources.

## 📁 Files Included

```
fraud-dashboard-backend/
├── server.js              # Backend API server
├── package.json           # Node.js dependencies
├── .env.example           # Configuration template
├── README.md              # This file
├── hooks/
│   └── useRealData.ts     # Updated React hooks for real data
└── App.tsx.real           # Updated App.tsx using real hooks
```

## 🚀 Setup Instructions

### Step 1: Install Backend Dependencies

```bash
cd fraud-dashboard-backend
npm install
```

### Step 2: Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual values
nano .env
```

**Required Configuration:**

```env
# Path to your Airflow data folder
AIRFLOW_DATA_PATH=/home/YOUR_USERNAME/airflow/data

# Airflow API credentials
AIRFLOW_URL=http://localhost:8080
AIRFLOW_USER=admin
AIRFLOW_PASS=your_password

# Kafka configuration
KAFKA_BROKERS=localhost:9092
KAFKA_TOPIC=ecommerce-transactions

# ML API (your Flask app)
ML_API_URL=http://localhost:4500

# Snowflake (optional - only if you want to query Snowflake)
SNOWFLAKE_ACCOUNT=SFEDU02-FEB92475
SNOWFLAKE_USER=BADGER
SNOWFLAKE_DATABASE=ECOMMERCE_DW
SNOWFLAKE_WAREHOUSE=ECOMMERCE_LOAD_WH
SNOWFLAKE_ROLE=TRAINING_ROLE
SNOWFLAKE_KEY_PATH=/home/YOUR_USERNAME/snowflake_key.p8
```

### Step 3: Start the Backend Server

```bash
npm start
```

The server will start on `http://localhost:3001`

You should see:
```
╔════════════════════════════════════════════════════════════╗
║     Fraud Detection Pipeline Dashboard - Backend API       ║
╠════════════════════════════════════════════════════════════╣
║  Server running on: http://localhost:3001                  ║
║  WebSocket: ws://localhost:3001/ws/kafka                   ║
╚════════════════════════════════════════════════════════════╝
```

### Step 4: Update Your React App

1. **Copy the new hooks:**
   ```bash
   cp hooks/useRealData.ts /path/to/your/react-app/src/hooks/
   ```

2. **Update App.tsx:**
   ```bash
   cp App.tsx.real /path/to/your/react-app/src/App.tsx
   ```

3. **Add environment variable for API URL:**
   
   Create a `.env` file in your React app root:
   ```env
   VITE_API_URL=http://localhost:3001
   VITE_WS_URL=ws://localhost:3001
   ```

4. **Rebuild and run your React app:**
   ```bash
   cd /path/to/your/react-app
   npm run build
   npm run dev
   ```

## 🔌 API Endpoints

The backend provides these endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `GET /api/generator` | Data generator stats from CSV files |
| `GET /api/airflow` | DAG status from Airflow API |
| `GET /api/kafka` | Kafka consumer stats |
| `GET /api/ml` | ML API status |
| `POST /api/ml/predict` | Make a prediction via ML API |
| `GET /api/snowflake` | Snowflake table stats |
| `GET /api/fraud` | Fraud analytics |
| `GET /api/system` | System metrics (CPU, memory) |
| `GET /api/all` | All data in one request |
| `POST /api/refresh` | Manually refresh all data |
| `WS /ws/kafka` | WebSocket for real-time Kafka messages |

## 📊 Data Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Generator │────▶│   CSV Files     │────▶│  Backend API    │
│  (Python)       │     │  (~/airflow/    │     │  (server.js)    │
│                 │     │   data/raw/)    │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│   Kafka Broker  │◀───▶│  Kafka Consumer │◀─────────────┘
│  (localhost:    │     │  (in server.js) │
│   9092)         │     │                 │
└────────┬────────┘     └─────────────────┘
         │
         │ WebSocket
         ▼
┌─────────────────┐
│  React Frontend │
│  (Dashboard UI) │
└─────────────────┘
```

## 🔧 Troubleshooting

### Backend can't find CSV files
- Check `AIRFLOW_DATA_PATH` in `.env`
- Make sure the path ends without a trailing slash
- Verify the folder has read permissions

### Airflow API connection failed
- Ensure Airflow is running: `airflow webserver`
- Check your Airflow username/password
- Verify Airflow REST API is enabled

### Kafka connection failed
- Ensure Kafka is running: `docker ps` (if using Docker)
- Check `KAFKA_BROKERS` matches your setup
- The consumer will retry automatically

### ML API connection failed
- Ensure your Flask API is running on port 4500
- Check `ML_API_URL` in `.env`
- Verify the `/health` endpoint exists

### Snowflake connection failed
- Ensure your RSA key file exists at `SNOWFLAKE_KEY_PATH`
- Verify the key has correct permissions (chmod 600)
- Snowflake queries are optional - the dashboard works without them

## 📝 Notes

- The backend polls data sources every 30 seconds
- Kafka messages are pushed via WebSocket in real-time
- System metrics are fetched every 5 seconds
- Snowflake queries run every 5 minutes (to avoid costs)
- All data is cached in memory - restart the server to clear

## 🛠️ Customization

### Add a new data source

1. Create a fetch function in `server.js`:
```javascript
async function fetchMyDataSource() {
  // Your logic here
  dataStore.myData = result;
}
```

2. Add an API endpoint:
```javascript
app.get('/api/mydata', (req, res) => {
  res.json(dataStore.myData);
});
```

3. Add a React hook in `useRealData.ts`:
```typescript
export function useMyData() {
  const [data, setData] = useState({});
  
  useEffect(() => {
    fetchApi('/api/mydata').then(setData);
  }, []);
  
  return data;
}
```

## 📞 Support

If you have issues:
1. Check the backend logs for errors
2. Verify all services are running (Airflow, Kafka, ML API)
3. Test individual API endpoints with curl:
   ```bash
   curl http://localhost:3001/api/health
   curl http://localhost:3001/api/generator
   ```
