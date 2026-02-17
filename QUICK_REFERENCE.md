# Quick Reference Guide

## 1-Minute Quick Start

```bash
# 1. Install dependencies
pip install faker kafka-python

# 2. Generate sample data
python ecommerce_fraud_data_generator.py --sample 100

# 3. Run full generator (1000 records/hour for 6 hours)
python ecommerce_fraud_data_generator.py --output transactions.jsonl
```

## Common Commands

### Generate Data
```bash
# Basic (1000 records/hour for 6 hours)
python ecommerce_fraud_data_generator.py

# Custom rate and duration
python ecommerce_fraud_data_generator.py --rate 2000 --hours 12

# Custom output file
python ecommerce_fraud_data_generator.py --output /path/to/output.jsonl

# Generate samples only
python ecommerce_fraud_data_generator.py --sample 1000
```

### With Kafka
```bash
# Start Kafka
docker-compose -f docker-compose-kafka.yml up -d

# Run with Kafka
python ecommerce_fraud_data_generator.py --kafka

# Custom Kafka settings
python ecommerce_fraud_data_generator.py \
  --kafka \
  --kafka-servers localhost:9092 \
  --kafka-topic ecommerce-transactions
```

### Adjust Data Quality Rates
```bash
# More missing values (10%)
python ecommerce_fraud_data_generator.py --missing-rate 0.10

# More duplicates (5%)
python ecommerce_fraud_data_generator.py --duplicate-rate 0.05

# More fraud cases (10%)
python ecommerce_fraud_data_generator.py --fraud-rate 0.10

# Combine options
python ecommerce_fraud_data_generator.py \
  --rate 5000 \
  --hours 2 \
  --missing-rate 0.10 \
  --duplicate-rate 0.05 \
  --fraud-rate 0.10 \
  --output high-volume.jsonl
```

## Docker Commands

```bash
# Start Kafka
docker-compose -f docker-compose-kafka.yml up -d

# View logs
docker-compose -f docker-compose-kafka.yml logs -f

# Stop Kafka
docker-compose -f docker-compose-kafka.yml down

# Remove all data
docker-compose -f docker-compose-kafka.yml down -v
```

## Kafka Commands

```bash
# List topics
docker exec kafka kafka-topics.sh --bootstrap-server localhost:29092 --list

# Create topic
docker exec kafka kafka-topics.sh \
  --bootstrap-server localhost:29092 \
  --create \
  --topic ecommerce-transactions \
  --partitions 3 \
  --replication-factor 1

# Describe topic
docker exec kafka kafka-topics.sh \
  --bootstrap-server localhost:29092 \
  --describe \
  --topic ecommerce-transactions

# Consume messages
docker exec kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:29092 \
  --topic ecommerce-transactions \
  --from-beginning
```

## Background Execution

```bash
# Using nohup
nohup python ecommerce_fraud_data_generator.py > generator.log 2>&1 &

# Check progress
tail -f generator.log

# Stop
pkill -f ecommerce_fraud_data_generator
```

## Data Verification

```python
from ecommerce_fraud_data_generator import verify_data_quality
import json

# Load records
records = []
with open('transactions.jsonl', 'r') as f:
    for line in f:
        records.append(json.loads(line))

# Verify
stats = verify_data_quality(records)
print(f"Total: {stats['total_records']}")
print(f"Missing: {stats['missing_values']}")
print(f"Duplicates: {stats['duplicates']}")
print(f"Outliers: {stats['outliers']}")
print(f"Fraud: {stats['fraud_cases']}")
```

## File Locations

| File | Path |
|------|------|
| Main Script | `/mnt/okcomputer/output/ecommerce_fraud_data_generator.py` |
| Sample Data | `/mnt/okcomputer/output/sample_transactions.jsonl` |
| Data Schema | `/mnt/okcomputer/output/DATA_SCHEMA.md` |
| Kafka Setup | `/mnt/okcomputer/output/KAFKA_SETUP.md` |
| Docker Compose | `/mnt/okcomputer/output/docker-compose-kafka.yml` |
| Full README | `/mnt/okcomputer/output/README.md` |

## Default Configuration

| Parameter | Default | Range |
|-----------|---------|-------|
| Records per hour | 1000 | 1-10000 |
| Duration hours | 6 | 1-168 |
| Missing value rate | 5% | 0-100% |
| Duplicate rate | 2% | 0-100% |
| Outlier rate | 1% | 0-100% |
| Fraud rate | 5% | 0-100% |

## URLs

| Service | URL |
|---------|-----|
| Kafka UI | http://localhost:8080 |
| Kafka Bootstrap | localhost:9092 |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | `pip install faker kafka-python` |
| Kafka connection failed | `docker-compose -f docker-compose-kafka.yml up -d` |
| Permission denied | `chmod +x ecommerce_fraud_data_generator.py` |
| Port 9092 in use | `lsof -ti:9092 \| xargs kill -9` |
