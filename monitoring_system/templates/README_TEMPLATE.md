# E-Commerce Fraud Detection Pipeline - Monitoring System

## Overview

This monitoring system provides comprehensive visibility into the E-Commerce Fraud Detection Pipeline, tracking:
- **Data Volume**: Records processed per hour
- **Processing Time**: Batch duration and stream latency
- **Error Rates**: Failed records and exceptions
- **ML Performance**: Prediction latency and confidence scores

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
├─────────────────────────────────────────────────────────────────┤
│  Airflow Logs    │  Kafka Logs    │  ML API Logs                │
│  (Text/JSON)     │  (JSON)        │  (JSON)                     │
└────────┬─────────┴────────┬───────┴────────┬────────────────────┘
         │                  │                │
         └──────────────────┼────────────────┘
                            │
                    ┌───────▼────────┐
                    │  Log Parsers   │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │   Metrics      │
                    │  Aggregator    │
                    └───────┬────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
  ┌──────▼──────┐  ┌───────▼───────┐  ┌───────▼───────┐
  │ CLI Report  │  │ JSON Export   │  │ Dashboard     │
  │ (Text)      │  │ (API)         │  │ (Visual)      │
  └─────────────┘  └───────────────┘  └───────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd monitoring_system
pip install -r requirements.txt
```

### 2. Run Monitoring Script

```bash
# Generate text report
python scripts/pipeline_monitor.py --hours 24

# Generate JSON report
python scripts/pipeline_monitor.py --output json --hours 6
```

### 3. View Dashboard

```bash
# Simple CLI dashboard
python scripts/pipeline_dashboard.py
```

## File Structure

```
monitoring_system/
├── configs/                    # Logging configurations
│   ├── airflow_logging.conf   # Airflow logging config
│   ├── kafka_logging.yaml     # Kafka logging config
│   ├── ml_logging.conf        # ML service logging config
│   └── unified_logging.json   # Centralized logging config
├── scripts/                    # Python scripts
│   ├── pipeline_monitor.py    # Main monitoring script
│   └── pipeline_dashboard.py  # CLI dashboard
├── logs/                       # Example log files
│   ├── example_airflow_logs.jsonl
│   ├── example_kafka_logs.jsonl
│   └── example_ml_logs.jsonl
├── templates/                  # Documentation templates
│   └── README_TEMPLATE.md     # This file
└── requirements.txt           # Python dependencies
```

## Metrics Tracked

| Category | Metric | Description | Unit |
|----------|--------|-------------|------|
| **Data Volume** | Records Processed | Total records ingested | count |
| | Records Cleaned | Records after cleaning | count |
| | Duplicates Removed | Duplicate records found | count |
| | Outliers Flagged | Outlier records detected | count |
| | Records Exported | Records sent to Snowflake | count |
| **Processing Time** | DAG Duration | Total pipeline execution time | seconds |
| | Task Duration | Individual task execution time | seconds |
| **Error Rates** | Error Count | Number of errors | count |
| | Error Rate | Percentage of failed operations | % |
| **ML Performance** | Prediction Latency | Time to make prediction | milliseconds |
| | Confidence Score | Model confidence | 0-1 |

## Configuration

### Log Paths

Update log paths in `scripts/pipeline_monitor.py`:

```python
AIRFLOW_LOGS_DIR = os.path.expanduser('~/airflow/logs/dag_id=ecommerce_batch_pipeline')
LOG_PATHS = {
    'airflow': os.path.expanduser('~/airflow/logs/airflow_metrics.log'),
    'kafka': '/var/log/kafka/consumer_stats.log',
    'ml': '/var/log/ml/metrics.log'
}
```

### Adding Custom Metrics

1. Edit `scripts/pipeline_monitor.py`
2. Add metric extraction pattern in `extract_metric()` method
3. Run the script to verify

## Usage Examples

### Generate Report for Last 24 Hours

```bash
python scripts/pipeline_monitor.py --hours 24
```

### Export to JSON

```bash
python scripts/pipeline_monitor.py --output json --hours 6 > report.json
```

### View Real-time Dashboard

```bash
python scripts/pipeline_dashboard.py
```

## Troubleshooting

### Issue: "Log file not found"

**Solution**: Verify log paths in configuration
```bash
ls -la ~/airflow/logs/dag_id=ecommerce_batch_pipeline/
```

### Issue: "No metrics extracted"

**Solution**: Check log format matches expected patterns
```bash
head -20 ~/airflow/logs/dag_id=ecommerce_batch_pipeline/run_id=*/task_id=*/attempt=*.log
```

### Issue: "Permission denied"

**Solution**: Fix log directory permissions
```bash
chmod -R 755 ~/airflow/logs/
```

## Integration with Pipeline Components

### Airflow DAG

Add to your DAG file:
```python
import logging

logger = logging.getLogger('airflow.processor')
logger.info(f"records_processed={len(df)} duration={duration}")
```

### Kafka Consumer

Add to your consumer:
```python
import logging
import json

logger = logging.getLogger('kafka.consumer')
logger.info(json.dumps({
    "consumer_lag": lag,
    "latency_ms": latency,
    "messages_per_second": throughput
}))
```

### ML API

Add to your API:
```python
import logging
import json
import time

start = time.time()
# ... prediction code ...
latency = (time.time() - start) * 1000

logger = logging.getLogger('ml.predictor')
logger.info(json.dumps({
    "latency_ms": latency,
    "confidence": confidence,
    "model_version": "v1.0.0"
}))
```

## License

This project is part of the MSc Applied Data Science and Analytics curriculum.
