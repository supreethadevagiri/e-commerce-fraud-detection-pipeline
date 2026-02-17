# Quick Start Guide

Get up and running with the Data Pipeline Monitoring System in 5 minutes.

---

## Prerequisites

- Python 3.8+
- pip package manager
- sudo access (for creating log directories)

---

## 1. Installation (2 minutes)

```bash
# Navigate to the monitoring system directory
cd monitoring_system

# Run the setup script
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Set up log directories
- Copy example log files

---

## 2. Activate Environment (10 seconds)

```bash
source venv/bin/activate
```

---

## 3. Test the Monitoring Script (1 minute)

### Generate a quick report

```bash
# All components, last 24 hours
python scripts/pipeline_monitor.py --component all --hours 24

# Specific component
python scripts/pipeline_monitor.py --component kafka --metric latency

# JSON output for programmatic use
python scripts/pipeline_monitor.py --output json --hours 6 > report.json
```

### Expected Output

```
================================================================================
PIPELINE METRICS REPORT
Generated: 2024-01-15T10:30:00
Time Window: Last 24 hours
================================================================================

----------------------------------------
DATA VOLUME METRICS
----------------------------------------
  total_records: 90000
  avg_per_hour: 3750.0
  unique_hours: 24

----------------------------------------
LATENCY METRICS
----------------------------------------
  avg_latency_ms: 45.23
  min_latency_ms: 41.8
  max_latency_ms: 48.5
  p50_latency_ms: 45.1
  p95_latency_ms: 47.8
  p99_latency_ms: 48.3
  sample_count: 30

----------------------------------------
ERROR METRICS
----------------------------------------
  error_count: 0
  total_operations: 30
  error_rate_percent: 0.0
  time_window_hours: 24
```

---

## 4. Launch the Dashboard (2 minutes)

```bash
jupyter notebook notebooks/pipeline_dashboard.ipynb
```

### Dashboard Features

1. **Summary Cards**: Real-time KPIs at a glance
2. **Data Volume Chart**: Records processed per hour
3. **Latency Distribution**: Box plots by component
4. **Error Rate Timeline**: Track failures over time
5. **Component Breakdown**: Pie chart of activity

### Using the Dashboard

1. Select time range from dropdown
2. Click "Refresh" to update
3. Enable "Auto-refresh" for live monitoring
4. Filter components using multi-select

---

## 5. Run Monitoring Server (Optional)

For continuous monitoring:

```bash
# Start server with 60-second intervals
python scripts/pipeline_monitor.py --server --interval 60

# Output is written to /var/log/pipeline/monitoring_output.log
```

Press `Ctrl+C` to stop the server.

---

## Common Tasks

### View Logs

```bash
# Airflow logs
tail -f /var/log/airflow/airflow_metrics.log

# Kafka logs
tail -f /var/log/kafka/consumer_stats.log

# ML logs
tail -f /var/log/ml/metrics.log
```

### Export Metrics

```bash
# Export to CSV
python -c "
from scripts.pipeline_monitor import *
import pandas as pd

collectors = {
    'airflow': AirflowMetricsCollector(),
    'kafka': KafkaMetricsCollector(),
    'ml': MLMetricsCollector()
}

all_metrics = []
for c in collectors.values():
    all_metrics.extend(c.collect_metrics(24))

df = pd.DataFrame([m.to_dict() for m in all_metrics])
df.to_csv('metrics_export.csv', index=False)
print('Exported to metrics_export.csv')
"
```

### Custom Log Paths

```bash
# Use custom log file path
python scripts/pipeline_monitor.py \
    --component airflow \
    --log-path /custom/path/to/airflow.log \
    --hours 12
```

---

## Troubleshooting

### Issue: "Log file not found"

**Solution**: Create log directories and copy example logs
```bash
mkdir -p logs/airflow logs/kafka logs/ml
# Copy example logs (see setup.sh)
```

### Issue: "Permission denied"

**Solution**: Fix permissions
```bash
sudo chown -R $USER:$USER /var/log/pipeline
sudo chmod 755 /var/log/pipeline
```

### Issue: "Module not found"

**Solution**: Ensure virtual environment is activated
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## Next Steps

1. **Configure Logging**: Update your pipeline components to use the logging configs in `configs/`
2. **Customize Dashboard**: Modify `notebooks/pipeline_dashboard.ipynb` for your needs
3. **Set Up Alerts**: Extend the monitoring script to send alerts
4. **Integrate with Your Pipeline**: Add the logging configuration to your Airflow, Kafka, and ML components

---

## File Structure

```
monitoring_system/
├── configs/              # Logging configurations
│   ├── airflow_logging.conf
│   ├── kafka_logging.yaml
│   ├── ml_logging.conf
│   └── unified_logging.json
├── scripts/              # Python scripts
│   └── pipeline_monitor.py
├── notebooks/            # Jupyter notebooks
│   └── pipeline_dashboard.ipynb
├── logs/                 # Example log files
│   ├── example_airflow_logs.jsonl
│   ├── example_kafka_logs.jsonl
│   └── example_ml_logs.jsonl
├── templates/            # Documentation templates
│   ├── README_TEMPLATE.md
│   ├── ARCHITECTURE_TEMPLATE.md
│   └── CONTRIBUTION_TEMPLATE.md
├── requirements.txt      # Python dependencies
├── setup.sh             # Setup script
└── QUICKSTART.md        # This file
```

---

## Support

- **Documentation**: See `templates/README_TEMPLATE.md`
- **Architecture**: See `templates/ARCHITECTURE_TEMPLATE.md`
- **Contributing**: See `templates/CONTRIBUTION_TEMPLATE.md`

---

**You're all set!** 🎉

Start monitoring your data pipeline with:
```bash
python scripts/pipeline_monitor.py --component all --hours 24
```
