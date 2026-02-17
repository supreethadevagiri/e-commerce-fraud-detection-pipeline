# E-Commerce Customer Purchase Analytics with Real-Time Fraud Detection
## Data Engineering Final Project - Architecture Design Document

---

## 1. ARCHITECTURE DIAGRAM (Text-Based)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                           │
│  │  Web App     │  │ Mobile App   │  │  POS System  │                           │
│  │  Events      │  │  Events      │  │   Events     │                           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                           │
└─────────┼─────────────────┼─────────────────┼───────────────────────────────────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  COMPONENT 1: DATA INGESTION (Apache Kafka / Google Pub/Sub)                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Kafka Cluster (3 brokers)                                               │    │
│  │  ├── Topic: purchase-events (3 partitions, replication=3)                │    │
│  │  ├── Topic: user-behavior (2 partitions, replication=3)                  │    │
│  │  └── Topic: fraud-alerts (2 partitions, replication=3)                   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────┬───────────────────────────────┘
                         │                        │
         ┌───────────────┘                        └───────────────┐
         │                                                       │
         ▼                                                       ▼
┌─────────────────────────────┐                    ┌─────────────────────────────┐
│ COMPONENT 3: STREAM         │                    │ COMPONENT 2: BATCH PROCESS  │
│ PROCESSING (Kafka Consumer) │                    │ (Airflow + Snowflake)       │
│ ┌─────────────────────────┐ │                    │ ┌─────────────────────────┐ │
│ │ Fraud Detection Service │ │                    │ │ Apache Airflow          │ │
│ │ (Python + Kafka Client) │ │                    │ │ ├─ DAG: daily_etl        │ │
│ │                         │ │                    │ │ ├─ DAG: hourly_agg       │ │
│ │ Real-time ML Inference  │─┼──►┌─────────────┐ │ │ └─ DAG: weekly_reports   │ │
│ │ - Check patterns        │ │   │  COMPONENT 4│ │ └──────────┬──────────────┘ │
│ │ - Call Vertex AI API    │─┼──►│  ML/Vertex  │ │            │                │
│ │ - Route to alerts       │ │   │    AI       │ │            ▼                │
│ └─────────────────────────┘ │   └──────┬──────┘ │   ┌─────────────────────┐   │
│         Consumer Group: 2   │          │        │   │    Snowflake        │   │
│         (2 workers)         │          │        │   │  ├─ RAW schema      │   │
│                             │          │        │   │  ├─ STAGING schema  │   │
│                             │          │        │   │  ├─ ANALYTICS schema│   │
│                             │          │        │   │  └─ ML_FEATURES     │   │
└─────────────────────────────┘          │        │   └─────────────────────┘   │
                                         │        └─────────────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  COMPONENT 5:       │
                              │  MONITORING &       │
                              │  STORAGE            │
                              │  ├─ CloudWatch/GCP  │
                              │  │   Monitoring      │
                              │  ├─ Grafana         │
                              │  │   Dashboards      │
                              │  └─ GCS/S3 (Data    │
                              │      Lake Backup)   │
                              └─────────────────────┘
```

---

## 2. DATA FLOW DESCRIPTION

### Flow 1: Real-Time Fraud Detection (Hot Path)
```
Purchase Event → Kafka → Stream Processor → Vertex AI → Fraud Alert/Approval
     (50ms)      (10ms)      (100ms)         (200ms)      (10ms)
     
TOTAL LATENCY: ~370ms (sub-second fraud detection)
```

### Flow 2: Batch Analytics (Cold Path)
```
Raw Events → Kafka → Airflow DAG → Snowflake → Analytics Dashboard
 (hourly)    (buffer)  (schedule)   (transform)   (query)
 
FREQUENCY: Every hour for recent data, daily for full refresh
```

### Flow 3: ML Model Training
```
Historical Data → Snowflake → Feature Store → Vertex AI → Model Registry
   (batch)        (query)      (engineer)     (train)      (deploy)
   
FREQUENCY: Weekly retraining with new fraud patterns
```

---

## 3. LITTLE'S LAW CALCULATIONS (L = λ × W)

### Given Parameters:
- **Expected Load (λ)**: 1,000 events/hour
- **Peak Load**: 3,000 events/hour (3x safety factor)
- **Event Types**: Purchase events, user behavior, fraud alerts

### Step-by-Step Calculations:

#### Step 1: Convert Load to Events per Second
```
λ_normal = 1,000 events/hour ÷ 3,600 seconds = 0.28 events/second
λ_peak   = 3,000 events/hour ÷ 3,600 seconds = 0.83 events/second
```

#### Step 2: Define Processing Times (W)
| Component | Processing Time | Notes |
|-----------|-----------------|-------|
| Kafka Ingestion | 10 ms | Network latency + broker write |
| Stream Processing | 100 ms | Parse + validate + enrich |
| ML Inference (Vertex AI) | 200 ms | API call + prediction |
| Alert Routing | 10 ms | Route to notification service |
| **Total W (end-to-end)** | **320 ms** | **0.32 seconds** |

#### Step 3: Calculate Queue Size (L) using Little's Law

**Formula: L = λ × W**

**Normal Load:**
```
L_normal = 0.28 events/sec × 0.32 sec = 0.09 events in system
```

**Peak Load:**
```
L_peak = 0.83 events/sec × 0.32 sec = 0.27 events in system
```

**With 5x Safety Buffer (recommended for production):**
```
L_buffer = 0.27 × 5 = 1.35 → Round up to 2 events
```

#### Step 4: Optimal Kafka Queue Size
```
Recommended Queue Depth = 1,000 events (per partition)

Calculation:
- At peak: 0.83 events/sec incoming
- If consumer fails for 20 minutes (1,200 sec)
- Queue needed: 0.83 × 1,200 = 996 events
- With buffer: 1,000 events
```

#### Step 5: Worker/Consumer Sizing

**Kafka Consumer Group Configuration:**
```
Partitions: 3 (for purchase-events topic)
Consumers: 2 (in same consumer group)

Distribution:
- Consumer 1: Handles partitions 0, 1
- Consumer 2: Handles partition 2

Benefits:
- Parallel processing
- Fault tolerance (if 1 consumer fails, other takes over)
- Can scale to 3 consumers max (1 per partition)
```

**Processing Capacity per Worker:**
```
Single Worker Capacity:
- Processing time per event: 320 ms
- Events per second per worker: 1 ÷ 0.32 = 3.125 events/sec
- Events per hour per worker: 3.125 × 3,600 = 11,250 events/hour

Required Workers for Peak Load (3,000 events/hour):
Workers needed = 3,000 ÷ 11,250 = 0.27 → 1 worker minimum

Recommended: 2 workers (for redundancy + headroom)
```

#### Step 6: Airflow Worker Sizing (Batch Processing)
```
Daily Batch Load:
- 1,000 events/hour × 24 hours = 24,000 events/day
- Batch processing time per event: 50 ms
- Total batch time: 24,000 × 0.05 = 1,200 seconds = 20 minutes

Airflow Configuration:
- 1 scheduler (manages DAG runs)
- 2 workers (execute tasks in parallel)
- Each worker can handle 4 concurrent tasks

Parallel Task Execution:
- Task 1: Extract from Kafka (5 min)
- Task 2: Transform in Snowflake (10 min) - depends on Task 1
- Task 3: Load to Analytics (5 min) - depends on Task 2
- Task 4: Generate reports (3 min) - depends on Task 3

Total DAG Runtime: ~23 minutes
```

---

## 4. TOOL JUSTIFICATION

### Component 1: Apache Kafka (Data Ingestion)

**Why Kafka?**
| Criteria | Kafka Advantage |
|----------|-----------------|
| Throughput | Handles 100K+ events/sec (we need 1K/hour) |
| Durability | Messages persisted to disk with replication |
| Scalability | Add brokers/partitions as load grows |
| Ecosystem | Native integration with Spark, Flink, Python |
| Cost | Open source, free to use |

**Alternative Considered**: Google Pub/Sub
- Pub/Sub is serverless (less management)
- Kafka chosen for: learning value, on-premise option, no cloud vendor lock-in

**Configuration:**
```yaml
Cluster: 3 brokers (minimum for production)
Topics:
  purchase-events:
    partitions: 3
    replication: 3
    retention: 7 days
  user-behavior:
    partitions: 2
    replication: 3
    retention: 3 days
```

---

### Component 2: Apache Airflow (Batch Orchestration)

**Why Airflow?**
| Criteria | Airflow Advantage |
|----------|-------------------|
| Scheduling | Cron-like scheduling with dependencies |
| Monitoring | Built-in UI for DAG runs, logs, retries |
| Integration | 300+ operators (Snowflake, Kafka, GCS, etc.) |
| Python | Write workflows in Python (student-friendly) |
| Community | Large community, extensive documentation |

**Alternative Considered**: Prefect, Dagster
- Airflow chosen for: industry standard, best learning resource availability

**Key DAGs:**
```python
# 1. hourly_etl.py - Runs every hour
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

# 2. daily_fraud_report.py - Runs daily at 6 AM
# 3. weekly_model_retrain.py - Runs Sundays at 2 AM
```

---

### Component 3: Snowflake (Data Warehouse)

**Why Snowflake?**
| Criteria | Snowflake Advantage |
|----------|---------------------|
| Separation | Compute separate from storage (cost-efficient) |
| Scaling | Auto-scale compute (XS to XXL warehouses) |
| SQL | Standard SQL interface |
| Features | Built-in ML (Snowpark), time travel, cloning |
| Student | Free trial credits, academic program |

**Database Schema Design:**
```sql
-- RAW Layer (landing zone)
CREATE SCHEMA raw;
CREATE TABLE raw.purchase_events (
    event_id VARCHAR,
    user_id VARCHAR,
    amount DECIMAL(10,2),
    timestamp TIMESTAMP,
    raw_json VARIANT  -- Store full event as JSON
);

-- STAGING Layer (cleaned)
CREATE SCHEMA staging;
CREATE TABLE staging.purchases AS
SELECT 
    event_id,
    user_id,
    amount,
    timestamp,
    DATE(timestamp) as purchase_date
FROM raw.purchase_events
WHERE amount > 0;

-- ANALYTICS Layer (aggregated)
CREATE SCHEMA analytics;
CREATE TABLE analytics.daily_sales AS
SELECT 
    purchase_date,
    COUNT(*) as total_orders,
    SUM(amount) as total_revenue,
    AVG(amount) as avg_order_value
FROM staging.purchases
GROUP BY purchase_date;

-- ML_FEATURES Layer (for model training)
CREATE SCHEMA ml_features;
CREATE TABLE ml_features.user_behavior (
    user_id VARCHAR,
    total_purchases INT,
    avg_purchase_amount DECIMAL(10,2),
    days_since_last_purchase INT,
    is_fraud_label BOOLEAN
);
```

---

### Component 4: Vertex AI (ML Platform)

**Why Vertex AI?**
| Criteria | Vertex AI Advantage |
|----------|---------------------|
| Unified | End-to-end ML platform (train → deploy → monitor) |
| AutoML | Can train models without deep ML expertise |
| Serving | Auto-scaling prediction endpoints |
| Integration | Native GCP integration with Pub/Sub, BigQuery |
| Cost | Pay-per-prediction (cost-effective for low volume) |

**Fraud Detection Model Architecture:**
```
Input Features (8 total):
├── Transaction amount
├── Time of day (hour)
├── Day of week
├── User's historical avg purchase
├── User's purchase frequency
├── Days since last purchase
├── Number of items in cart
└── Payment method type

Model Type: Gradient Boosted Trees (XGBoost)
Training Data: Historical labeled transactions (fraud/not fraud)
Prediction Output: Fraud probability (0.0 to 1.0)
Threshold: > 0.7 = Flag as potential fraud
```

**API Integration:**
```python
from google.cloud import aiplatform

def predict_fraud(transaction_features):
    endpoint = aiplatform.Endpoint("fraud-detection-endpoint")
    response = endpoint.predict(instances=[transaction_features])
    fraud_probability = response.predictions[0]
    return fraud_probability > 0.7  # Returns True/False
```

---

### Component 5: Monitoring & Storage

**Monitoring Stack:**
| Tool | Purpose |
|------|---------|
| Prometheus + Grafana | Metrics visualization |
| Kafka Manager | Kafka cluster monitoring |
| Airflow UI | DAG run tracking |
| Snowflake Account Usage | Query performance, storage |
| CloudWatch/GCP Monitoring | Infrastructure metrics |

**Key Metrics to Track:**
```yaml
System Metrics:
  - Kafka consumer lag (should be < 1000)
  - Airflow DAG success rate (target: > 99%)
  - Snowflake query duration (p95 < 30s)
  - Vertex AI prediction latency (p95 < 300ms)

Business Metrics:
  - Events processed per minute
  - Fraud detection rate
  - False positive rate (target: < 5%)
  - End-to-end latency (target: < 500ms)
```

**Storage Layers:**
```
Hot Storage (Active Querying):
  - Kafka: Last 7 days
  - Snowflake: Last 90 days

Warm Storage (Occasional Access):
  - GCS/S3: 90 days to 1 year
  - Compressed Parquet format

Cold Storage (Compliance):
  - GCS Coldline/S3 Glacier: 1+ years
  - For audit and regulatory requirements
```

---

## 5. SCALABILITY CONSIDERATIONS

### Horizontal Scaling Strategy

**Kafka Scaling:**
```
Current: 3 brokers, 3 partitions
Scale Trigger: Consumer lag > 10,000

Scale Actions:
1. Add partitions: 3 → 6 → 12
2. Add brokers: 3 → 5 → 7
3. Increase consumer instances: 2 → 4 → 6

Max Capacity: 100,000 events/hour with 12 partitions
```

**Snowflake Scaling:**
```
Current: X-Small warehouse (1 credit/hour)
Scale Trigger: Query queue time > 30 seconds

Scale Actions:
1. Resize warehouse: XS → S → M → L
2. Enable auto-scaling: 1-3 clusters
3. Use warehouse per workload (ETL vs Analytics)

Cost Impact:
- XS: $2/hour → $48/day
- S: $4/hour → $96/day
- M: $8/hour → $192/day
```

**Vertex AI Scaling:**
```
Current: 1 prediction node
Scale Trigger: Prediction latency > 500ms (p95)

Scale Actions:
1. Increase min nodes: 1 → 2 → 4
2. Enable auto-scaling: 1-10 nodes
3. Use batch prediction for non-real-time needs

Cost Impact:
- 1 node: ~$0.40/hour
- 4 nodes: ~$1.60/hour
```

### Load Projections

| Timeframe | Expected Events | Scale Action |
|-----------|-----------------|--------------|
| Month 1-3 | 1,000/hour | Current setup |
| Month 4-6 | 5,000/hour | Add Kafka partitions (3→6) |
| Month 7-12 | 15,000/hour | Add brokers, Snowflake S |
| Year 2+ | 50,000/hour | Full horizontal scaling |

---

## 6. FAULT TOLERANCE STRATEGIES

### Kafka Fault Tolerance
```yaml
Replication Factor: 3
  - Each message stored on 3 brokers
  - Can tolerate 2 broker failures

Min ISR (In-Sync Replicas): 2
  - Requires 2 replicas to acknowledge write
  - Balances durability vs availability

Consumer Group:
  - 2 consumers in group
  - If 1 fails, other takes all partitions
  - Automatic rebalancing on failure

Retention: 7 days
  - Replay capability if consumer fails
  - Buffer for recovery scenarios
```

### Airflow Fault Tolerance
```python
# DAG Configuration for Fault Tolerance
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,  # Don't block on previous failure
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),  # Kill stuck tasks
    'sla': timedelta(hours=1),  # Alert if running too long
}
```

### Snowflake Fault Tolerance
```sql
-- Time Travel (built-in recovery)
SELECT * FROM purchases AT (OFFSET => -60*60);  -- 1 hour ago

-- Fail-safe (7 days after time travel expires)
-- Contact Snowflake support for recovery

-- Zero-copy cloning for testing
CREATE CLONE purchases_backup CLONE purchases;

-- Multi-cluster warehouses for HA
ALTER WAREHOUSE etl_warehouse SET 
    MIN_CLUSTER_COUNT = 1 
    MAX_CLUSTER_COUNT = 3;
```

### Vertex AI Fault Tolerance
```yaml
Model Deployment:
  - Deployed to 2 regions (us-central1, us-east1)
  - Traffic split: 50/50
  - Automatic failover if region down

Prediction Endpoint:
  - Min nodes: 1 (always available)
  - Max nodes: 5 (handle spikes)
  - Health checks every 30 seconds
```

### Disaster Recovery Plan
```
RPO (Recovery Point Objective): 1 hour
- Kafka retention covers this
- Hourly Airflow backups to GCS

RTO (Recovery Time Objective): 30 minutes
- Automated failover for Kafka
- Manual intervention for Snowflake
- Runbook documented

Backup Strategy:
- Kafka: Replicated across 3 brokers (no separate backup needed)
- Snowflake: Time travel + periodic clones
- Airflow DAGs: Git repository (infrastructure as code)
- ML Models: Versioned in Vertex AI Model Registry
```

---

## 7. COST ESTIMATES (Monthly)

| Component | Configuration | Monthly Cost |
|-----------|--------------|--------------|
| Kafka (self-hosted) | 3 brokers (n1-standard-2) | $200 |
| OR Kafka (managed - Confluent) | Basic cluster | $500 |
| Airflow (Cloud Composer) | Small environment | $300 |
| Snowflake | X-Small, 4 hrs/day avg | $300 |
| Vertex AI | 1 prediction node | $300 |
| Storage (GCS) | 500GB standard | $25 |
| Monitoring | CloudWatch + Grafana | $50 |
| **Total** | | **$1,175 - $1,475** |

**Student Project Optimization:**
- Use Kafka Docker locally: $0
- Use Airflow Docker locally: $0
- Snowflake free trial: $0 (first $400)
- Vertex AI free tier: $0 (first 6 months)
- **Total for learning: $0-100/month**

---

## 8. IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1-2)
- [ ] Set up Kafka cluster (Docker for local)
- [ ] Create topics with partitions
- [ ] Build simple producer (Python)
- [ ] Verify message flow

### Phase 2: Stream Processing (Week 3-4)
- [ ] Build Kafka consumer
- [ ] Integrate with Vertex AI
- [ ] Implement fraud detection logic
- [ ] Add alert routing

### Phase 3: Batch Pipeline (Week 5-6)
- [ ] Set up Airflow
- [ ] Create ETL DAGs
- [ ] Configure Snowflake connection
- [ ] Build dimensional model

### Phase 4: ML Pipeline (Week 7-8)
- [ ] Create feature engineering pipeline
- [ ] Train fraud detection model
- [ ] Deploy to Vertex AI endpoint
- [ ] Set up model monitoring

### Phase 5: Monitoring & Polish (Week 9-10)
- [ ] Set up Grafana dashboards
- [ ] Configure alerts
- [ ] Document runbooks
- [ ] Performance testing

---

## 9. SUMMARY: KEY NUMBERS

| Metric | Value |
|--------|-------|
| Expected Load | 1,000 events/hour (0.28 events/sec) |
| Peak Load | 3,000 events/hour (0.83 events/sec) |
| End-to-End Latency | 320ms (fraud detection) |
| Queue Size (Little's Law) | 0.27 events at peak |
| Recommended Queue Depth | 1,000 events |
| Kafka Consumers | 2 (in consumer group) |
| Kafka Partitions | 3 (for purchase-events) |
| Airflow Workers | 2 |
| Snowflake Warehouse | X-Small |
| Vertex AI Nodes | 1 |
| Monthly Cost (production) | $1,175 - $1,475 |
| Monthly Cost (student) | $0 - $100 |

---

*Document Version: 1.0*
*Created for: Data Engineering Final Project*
*Domain: E-commerce Customer Purchase Analytics with Real-Time Fraud Detection*
