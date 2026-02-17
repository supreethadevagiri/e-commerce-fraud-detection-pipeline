# Kafka Setup Instructions for E-Commerce Fraud Detection

## Prerequisites

- Docker and Docker Compose (recommended)
- OR Apache Kafka 2.8+ installed locally
- Python 3.8+
- kafka-python library

## Option 1: Quick Start with Docker Compose (Recommended)

### Step 1: Create Docker Compose File

Create a file named `docker-compose-kafka.yml`:

```yaml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    hostname: zookeeper
    container_name: zookeeper
    ports:
      - "2181:2181"
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    hostname: kafka
    container_name: kafka
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
      - "29092:29092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: 'zookeeper:2181'
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: 0
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1

  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: kafka-ui
    depends_on:
      - kafka
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:29092
```

### Step 2: Start Kafka Services

```bash
# Start all services
docker-compose -f docker-compose-kafka.yml up -d

# Verify services are running
docker-compose -f docker-compose-kafka.yml ps

# View logs
docker-compose -f docker-compose-kafka.yml logs -f kafka
```

### Step 3: Create the Topic

```bash
# Enter Kafka container
docker exec -it kafka bash

# Create topic with 3 partitions and replication factor 1
kafka-topics --create \
  --bootstrap-server localhost:29092 \
  --topic ecommerce-transactions \
  --partitions 3 \
  --replication-factor 1

# Verify topic creation
kafka-topics --list --bootstrap-server localhost:29092

# View topic details
kafka-topics --describe \
  --bootstrap-server localhost:29092 \
  --topic ecommerce-transactions

# Exit container
exit
```

### Step 4: Access Kafka UI

Open your browser and navigate to: http://localhost:8080

## Option 2: Local Kafka Installation

### Step 1: Download and Extract Kafka

```bash
# Download Kafka (adjust version as needed)
wget https://downloads.apache.org/kafka/3.6.0/kafka_2.13-3.6.0.tgz

# Extract
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0
```

### Step 2: Start Zookeeper

```bash
# Terminal 1: Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties
```

### Step 3: Start Kafka Broker

```bash
# Terminal 2: Start Kafka
bin/kafka-server-start.sh config/server.properties
```

### Step 4: Create Topic

```bash
# Terminal 3: Create topic
bin/kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-transactions \
  --partitions 3 \
  --replication-factor 1

# Verify
bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```

## Python Kafka Consumer (Verification)

Create a simple consumer to verify data is flowing:

```python
# consumer.py
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'ecommerce-transactions',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='fraud-detection-consumer',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Listening for messages...")
for message in consumer:
    print(f"Received: {message.value}")
```

Run the consumer:
```bash
python consumer.py
```

## Running the Data Generator with Kafka

### Install Dependencies

```bash
pip install kafka-python faker
```

### Run Generator with Kafka Output

```bash
# Basic usage with Kafka
python ecommerce_fraud_data_generator.py --kafka

# With custom settings
python ecommerce_fraud_data_generator.py \
  --kafka \
  --kafka-servers localhost:9092 \
  --kafka-topic ecommerce-transactions \
  --rate 1000 \
  --hours 6 \
  --output backup.jsonl
```

## Kafka Topic Configuration

### Recommended Settings for Production

```bash
# Create topic with optimized settings
kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-transactions \
  --partitions 6 \
  --replication-factor 3 \
  --config retention.ms=604800000 \
  --config segment.ms=86400000 \
  --config cleanup.policy=delete
```

### Topic Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| partitions | 3-6 | Number of partitions for parallel processing |
| replication-factor | 1-3 | Data redundancy (3 for production) |
| retention.ms | 604800000 | 7 days retention |
| segment.ms | 86400000 | Daily log segmentation |

## Monitoring Commands

```bash
# Check consumer groups
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list

# Describe consumer group
kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe \
  --group fraud-detection-consumer

# View topic offsets
kafka-run-class.sh kafka.tools.GetOffsetShell \
  --broker-list localhost:9092 \
  --topic ecommerce-transactions

# Produce test message
kafka-console-producer.sh \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-transactions

# Consume messages
kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-transactions \
  --from-beginning
```

## Troubleshooting

### Connection Refused
```bash
# Check if Kafka is running
docker ps

# Check Kafka logs
docker logs kafka

# Verify advertised listeners
docker exec kafka kafka-configs.sh \
  --bootstrap-server localhost:29092 \
  --entity-type brokers \
  --entity-default \
  --describe
```

### Topic Not Found
```bash
# List all topics
docker exec kafka kafka-topics.sh \
  --bootstrap-server localhost:29092 \
  --list

# Create topic if missing
docker exec kafka kafka-topics.sh \
  --bootstrap-server localhost:29092 \
  --create \
  --topic ecommerce-transactions \
  --partitions 3 \
  --replication-factor 1
```

### Reset Consumer Offsets
```bash
# Reset to earliest
docker exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:29092 \
  --group fraud-detection-consumer \
  --reset-offsets \
  --to-earliest \
  --topic ecommerce-transactions \
  --execute
```

## Stopping Kafka

```bash
# Docker Compose
docker-compose -f docker-compose-kafka.yml down

# Remove volumes (WARNING: deletes all data)
docker-compose -f docker-compose-kafka.yml down -v

# Local installation
# Press Ctrl+C in Zookeeper and Kafka terminals
```

## Advanced: Multi-Broker Setup

For production-like environments with multiple brokers:

```yaml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    ports:
      - "2181:2181"
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka-1:
    image: confluentinc/cp-kafka:7.4.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 2

  kafka-2:
    image: confluentinc/cp-kafka:7.4.0
    ports:
      - "9093:9093"
    environment:
      KAFKA_BROKER_ID: 2
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 2
```

## Security Considerations

For production environments:
1. Enable SASL/SSL authentication
2. Configure ACLs for topic access
3. Use encrypted connections
4. Implement proper monitoring with Prometheus/Grafana
