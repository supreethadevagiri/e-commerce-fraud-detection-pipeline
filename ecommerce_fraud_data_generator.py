#!/usr/bin/env python3
"""
E-Commerce Fraud Detection - Synthetic Data Generator
======================================================
Generates realistic e-commerce transaction data with fraud patterns and data quality issues.

Features:
- 1,000+ records/hour generation rate
- 6+ hours continuous operation
- Data quality challenges: missing values (5%), duplicates (2%), outliers (1%)
- Kafka integration for streaming
- CSV output for Airflow DAG compatibility
- Fraud pattern simulation

Author: Data Engineering Team
Version: 1.2.0 (Fixed column names to match DAG)
"""

import json
import random
import time
import uuid
import logging
import argparse
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import threading

from faker import Faker
from faker.providers import internet, misc, date_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Faker
fake = Faker('en_US')
fake.add_provider(internet)
fake.add_provider(misc)
fake.add_provider(date_time)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for the data generator."""
    records_per_hour: int = 1000
    duration_hours: int = 6
    missing_value_rate: float = 0.05  # 5%
    duplicate_rate: float = 0.02      # 2%
    outlier_rate: float = 0.01        # 1%
    fraud_rate: float = 0.05          # 5% fraud cases
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "ecommerce-transactions"
    output_file: Optional[str] = None
    enable_kafka: bool = False

# =============================================================================
# DATA SCHEMA CONSTANTS
# =============================================================================

PRODUCT_CATEGORIES = [
    "Electronics", "Clothing", "Home & Garden", "Sports & Outdoors",
    "Books & Media", "Health & Beauty", "Toys & Games", "Food & Beverages",
    "Automotive", "Jewelry & Watches", "Pet Supplies", "Office Supplies"
]

PAYMENT_METHODS = [
    "Credit Card", "Debit Card", "PayPal", "Apple Pay", "Google Pay",
    "Cryptocurrency", "Bank Transfer", "Buy Now Pay Later"
]

DEVICE_TYPES = [
    "Desktop", "Mobile - iOS", "Mobile - Android", "Tablet - iOS",
    "Tablet - Android", "Smart TV", "Gaming Console", "Unknown Device"
]

COUNTRIES = [
    "USA", "UK", "Canada", "Germany", "France", "Australia", "Japan",
    "Brazil", "India", "Mexico", "Spain", "Italy", "Netherlands", "Singapore"
]

# Fraud indicators for realistic fraud pattern generation
FRAUD_INDICATORS = {
    "high_amount_threshold": 5000,
    "suspicious_countries": ["Unknown", "High Risk Country"],
    "suspicious_payment_methods": ["Cryptocurrency", "Buy Now Pay Later"],
    "rapid_transaction_countries": ["Nigeria", "Romania", "Russia"]
}

# =============================================================================
# DATA GENERATOR CLASS
# =============================================================================

class EcommerceDataGenerator:
    """
    Main data generator class for e-commerce fraud detection data.
    """
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.generated_records: List[Dict[str, Any]] = []
        self.stats = {
            "total_generated": 0,
            "fraud_count": 0,
            "missing_values_count": 0,
            "duplicates_count": 0,
            "outliers_count": 0,
            "start_time": None,
            "end_time": None
        }
        self._lock = threading.Lock()
        
        # Initialize Kafka producer if enabled
        self.kafka_producer = None
        if config.enable_kafka:
            self._init_kafka()
    
    def _init_kafka(self):
        """Initialize Kafka producer."""
        try:
            from kafka import KafkaProducer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                batch_size=16384,
                linger_ms=10
            )
            logger.info(f"Kafka producer initialized: {self.config.kafka_bootstrap_servers}")
        except ImportError:
            logger.error("kafka-python not installed. Run: pip install kafka-python")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def generate_transaction_id(self) -> str:
        """Generate unique transaction ID."""
        return f"TXN-{uuid.uuid4().hex[:16].upper()}"
    
    def generate_customer_id(self) -> str:
        """Generate customer ID."""
        return f"CUST-{fake.random_number(digits=8, fix_len=True)}"
    
    def generate_timestamp(self, base_time: Optional[datetime] = None) -> str:
        """Generate transaction timestamp."""
        if base_time is None:
            base_time = datetime.now()
        offset_seconds = random.randint(0, 3600)
        timestamp = base_time + timedelta(seconds=offset_seconds)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_amount(self, is_fraud: bool = False, make_outlier: bool = False) -> float:
        """Generate transaction amount with fraud and outlier patterns."""
        if make_outlier:
            if random.random() < 0.5:
                return round(random.uniform(15000, 100000), 2)
            else:
                return round(random.uniform(0.01, 0.99), 2)
        
        if is_fraud:
            if random.random() < 0.3:
                return round(random.uniform(0.50, 5.00), 2)
            else:
                return round(random.uniform(500, 15000), 2)
        
        return round(random.uniform(5.00, 500.00), 2)
    
    def generate_category(self, is_fraud: bool = False) -> str:
        """Generate product category with fraud patterns."""
        if is_fraud:
            weighted_categories = [
                "Electronics", "Electronics", "Jewelry & Watches", "Jewelry & Watches",
                "Electronics", "Health & Beauty", "Automotive"
            ]
            return random.choice(weighted_categories)
        return random.choice(PRODUCT_CATEGORIES)
    
    def generate_payment_method(self, is_fraud: bool = False) -> str:
        """Generate payment method with fraud patterns."""
        if is_fraud:
            weighted_methods = [
                "Cryptocurrency", "Cryptocurrency", "Buy Now Pay Later",
                "Credit Card", "Credit Card", "Credit Card"
            ]
            return random.choice(weighted_methods)
        return random.choice(PAYMENT_METHODS)
    
    def generate_device_type(self, is_fraud: bool = False) -> str:
        """Generate device type with fraud patterns."""
        if is_fraud:
            weighted_devices = [
                "Unknown Device", "Unknown Device", "Desktop", "Desktop",
                "Desktop", "Mobile - Android"
            ]
            return random.choice(weighted_devices)
        return random.choice(DEVICE_TYPES)
    
    def generate_country(self, is_fraud: bool = False) -> str:
        """Generate country with fraud patterns."""
        if is_fraud:
            weighted_countries = [
                "Nigeria", "Romania", "Russia", "Unknown", "Unknown",
                "USA", "USA", "UK"
            ]
            return random.choice(weighted_countries)
        return random.choice(COUNTRIES)
    
    def determine_fraud(self, amount: float, payment_method: str, 
                        device_type: str, country: str) -> bool:
        """Determine if transaction is fraudulent based on multiple factors."""
        fraud_score = 0
        
        if amount > 5000:
            fraud_score += 0.4
        elif amount < 1.0:
            fraud_score += 0.2
        
        if payment_method in FRAUD_INDICATORS["suspicious_payment_methods"]:
            fraud_score += 0.3
        
        if device_type == "Unknown Device":
            fraud_score += 0.2
        
        if country in FRAUD_INDICATORS["rapid_transaction_countries"] or country == "Unknown":
            fraud_score += 0.3
        
        fraud_score += random.uniform(0, 0.2)
        
        return random.random() < (fraud_score * self.config.fraud_rate * 5)
    
    def apply_missing_values(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply missing values to random fields (5% of records)."""
        if random.random() < self.config.missing_value_rate:
            fields_to_null = random.sample(
                ['category', 'payment_method', 'device_type', 'country'],
                k=random.randint(1, 2)
            )
            for field in fields_to_null:
                record[field] = None
            with self._lock:
                self.stats["missing_values_count"] += 1
        return record
    
    def create_duplicate(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create a duplicate record with slight timestamp variation."""
        duplicate = record.copy()
        original_ts = datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S")
        new_ts = original_ts + timedelta(seconds=random.randint(1, 60))
        duplicate['timestamp'] = new_ts.strftime("%Y-%m-%d %H:%M:%S")
        duplicate['_is_duplicate'] = True
        with self._lock:
            self.stats["duplicates_count"] += 1
        return duplicate
    
    def generate_single_record(self, base_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a single transaction record with DAG-compatible column names."""
        make_outlier = random.random() < self.config.outlier_rate
        
        is_fraud = random.random() < self.config.fraud_rate
        amount = self.generate_amount(is_fraud=is_fraud, make_outlier=make_outlier)
        payment_method = self.generate_payment_method(is_fraud)
        device_type = self.generate_device_type(is_fraud)
        country = self.generate_country(is_fraud)
        
        is_fraudulent = self.determine_fraud(amount, payment_method, device_type, country)
        
        # FIXED: Column names now match DAG expectations
        record = {
            "transaction_id": self.generate_transaction_id(),
            "customer_id": self.generate_customer_id(),
            "timestamp": self.generate_timestamp(base_time),
            "amount": amount,
            "category": self.generate_category(is_fraudulent),
            "payment_method": payment_method,
            "device_type": device_type,
            "country": country,
            "merchant_id": f"MERCH_{random.randint(1, 100):03d}",
            "is_fraudulent": is_fraudulent
        }
        
        record = self.apply_missing_values(record)
        
        if make_outlier:
            with self._lock:
                self.stats["outliers_count"] += 1
        
        if is_fraudulent:
            with self._lock:
                self.stats["fraud_count"] += 1
        
        return record
    
    def generate_batch(self, batch_size: int, base_time: datetime) -> List[Dict[str, Any]]:
        """Generate a batch of records."""
        batch = []
        
        for _ in range(batch_size):
            record = self.generate_single_record(base_time)
            batch.append(record)
            
            if random.random() < self.config.duplicate_rate:
                duplicate = self.create_duplicate(record)
                batch.append(duplicate)
        
        return batch
    
    def send_to_kafka(self, records: List[Dict[str, Any]]):
        """Send records to Kafka topic."""
        if not self.kafka_producer:
            return
        
        for record in records:
            try:
                key = record['customer_id']
                self.kafka_producer.send(
                    self.config.kafka_topic,
                    key=key,
                    value=record
                )
            except Exception as e:
                logger.error(f"Failed to send to Kafka: {e}")
        
        self.kafka_producer.flush()
    
    def save_to_csv(self, records: List[Dict[str, Any]], csv_path: str):
        """Save records to CSV format for Airflow DAG."""
        import pandas as pd
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        df = pd.DataFrame(records)
        
        if '_is_duplicate' in df.columns:
            df = df.drop(columns=['_is_duplicate'])
        
        # FIX: Append to existing file instead of overwriting
        file_exists = os.path.exists(csv_path)
        
        if file_exists:
            # Append without header
            df.to_csv(csv_path, mode='a', header=False, index=False)
            logger.info(f"Appended {len(records)} records to CSV: {csv_path}")
        else:
            # Create new file with header
            df.to_csv(csv_path, mode='w', header=True, index=False)
            logger.info(f"Created CSV with {len(records)} records: {csv_path}")
    
    def save_to_file(self, records: List[Dict[str, Any]]):
        """Save records to output file (JSONL or CSV based on extension)."""
        if not self.config.output_file:
            return
        
        if self.config.output_file.endswith('.csv'):
            self.save_to_csv(records, self.config.output_file)
        else:
            with self._lock:
                with open(self.config.output_file, 'a') as f:
                    for record in records:
                        record_copy = {k: v for k, v in record.items() if k != '_is_duplicate'}
                        f.write(json.dumps(record_copy) + '\n')
    
    def generate_for_duration(self):
        """Generate data continuously for specified duration."""
        self.stats["start_time"] = datetime.now()
        total_records_needed = self.config.records_per_hour * self.config.duration_hours
        
        logger.info("=" * 60)
        logger.info("E-COMMERCE FRAUD DATA GENERATOR STARTED")
        logger.info("=" * 60)
        logger.info(f"Configuration:")
        logger.info(f"  - Records per hour: {self.config.records_per_hour}")
        logger.info(f"  - Duration: {self.config.duration_hours} hours")
        logger.info(f"  - Total target records: {total_records_needed}")
        logger.info(f"  - Missing value rate: {self.config.missing_value_rate * 100}%")
        logger.info(f"  - Duplicate rate: {self.config.duplicate_rate * 100}%")
        logger.info(f"  - Outlier rate: {self.config.outlier_rate * 100}%")
        logger.info(f"  - Fraud rate: {self.config.fraud_rate * 100}%")
        logger.info(f"  - Kafka enabled: {self.config.enable_kafka}")
        logger.info(f"  - Output format: {'CSV' if self.config.output_file and self.config.output_file.endswith('.csv') else 'JSONL'}")
        logger.info("=" * 60)
        
        records_per_second = self.config.records_per_hour / 3600
        interval = 1.0 / records_per_second if records_per_second > 0 else 1.0
        
        hour_start = datetime.now()
        hour_records = 0
        current_hour = 0
        
        try:
            while current_hour < self.config.duration_hours:
                batch_start = datetime.now()
                
                batch_size = max(1, int(records_per_second * 10))
                
                batch = self.generate_batch(batch_size, hour_start)
                
                with self._lock:
                    self.stats["total_generated"] += len(batch)
                    self.generated_records.extend(batch)
                
                hour_records += len(batch)
                
                if self.config.enable_kafka:
                    self.send_to_kafka(batch)
                
                self.save_to_file(batch)
                
                elapsed = (datetime.now() - hour_start).total_seconds()
                if elapsed >= 3600:
                    current_hour += 1
                    logger.info(f"Hour {current_hour} complete: {hour_records} records generated")
                    hour_start = datetime.now()
                    hour_records = 0
                
                if self.stats["total_generated"] % 1000 == 0:
                    logger.info(f"Progress: {self.stats['total_generated']} records generated")
                
                batch_time = (datetime.now() - batch_start).total_seconds()
                sleep_time = max(0, (batch_size / records_per_second) - batch_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("\nGeneration interrupted by user")
        
        finally:
            self.stats["end_time"] = datetime.now()
            self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final generation statistics."""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        logger.info("\n" + "=" * 60)
        logger.info("GENERATION COMPLETE - FINAL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {duration / 3600:.2f} hours")
        logger.info(f"Total Records Generated: {self.stats['total_generated']}")
        logger.info(f"Records per Hour: {self.stats['total_generated'] / (duration / 3600):.0f}")
        logger.info("")
        logger.info("Data Quality Issues Introduced:")
        logger.info(f"  - Missing Values: {self.stats['missing_values_count']} ({self.stats['missing_values_count']/max(1,self.stats['total_generated'])*100:.2f}%)")
        logger.info(f"  - Duplicates: {self.stats['duplicates_count']} ({self.stats['duplicates_count']/max(1,self.stats['total_generated'])*100:.2f}%)")
        logger.info(f"  - Outliers: {self.stats['outliers_count']} ({self.stats['outliers_count']/max(1,self.stats['total_generated'])*100:.2f}%)")
        logger.info("")
        logger.info("Fraud Statistics:")
        logger.info(f"  - Fraudulent Transactions: {self.stats['fraud_count']} ({self.stats['fraud_count']/max(1,self.stats['total_generated'])*100:.2f}%)")
        logger.info(f"  - Legitimate Transactions: {self.stats['total_generated'] - self.stats['fraud_count']}")
        logger.info("=" * 60)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the data generator."""
    parser = argparse.ArgumentParser(
        description='E-Commerce Fraud Detection Data Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 records/hour for 6 hours to CSV (for Airflow DAG)
  python ecommerce_fraud_data_generator.py --output transactions.csv
  
  # Generate to CSV with custom path
  python ecommerce_fraud_data_generator.py --output ~/airflow/data/raw/transactions_2026-02-12.csv
  
  # Generate and send to Kafka
  python ecommerce_fraud_data_generator.py --kafka --kafka-topic fraud-transactions
  
  # Generate to BOTH CSV and Kafka
  python ecommerce_fraud_data_generator.py --output data.csv --kafka
  
  # Custom rate: 2000 records/hour for 12 hours to CSV
  python ecommerce_fraud_data_generator.py --rate 2000 --hours 12 --output data.csv
        """
    )
    
    parser.add_argument('--rate', type=int, default=1000,
                        help='Records per hour (default: 1000)')
    parser.add_argument('--hours', type=int, default=6,
                        help='Duration in hours (default: 6)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (use .csv extension for CSV format)')
    parser.add_argument('--kafka', action='store_true',
                        help='Enable Kafka output')
    parser.add_argument('--kafka-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers (default: localhost:9092)')
    parser.add_argument('--kafka-topic', type=str, default='ecommerce-transactions',
                        help='Kafka topic name (default: ecommerce-transactions)')
    parser.add_argument('--fraud-rate', type=float, default=0.05,
                        help='Fraud rate 0-1 (default: 0.05)')
    parser.add_argument('--missing-rate', type=float, default=0.05,
                        help='Missing value rate 0-1 (default: 0.05)')
    parser.add_argument('--duplicate-rate', type=float, default=0.02,
                        help='Duplicate rate 0-1 (default: 0.02)')
    parser.add_argument('--outlier-rate', type=float, default=0.01,
                        help='Outlier rate 0-1 (default: 0.01)')
    
    args = parser.parse_args()
    
    config = GeneratorConfig(
        records_per_hour=args.rate,
        duration_hours=args.hours,
        missing_value_rate=args.missing_rate,
        duplicate_rate=args.duplicate_rate,
        outlier_rate=args.outlier_rate,
        fraud_rate=args.fraud_rate,
        kafka_bootstrap_servers=args.kafka_servers,
        kafka_topic=args.kafka_topic,
        output_file=args.output,
        enable_kafka=args.kafka
    )
    
    generator = EcommerceDataGenerator(config)
    generator.generate_for_duration()


if __name__ == "__main__":
    main()