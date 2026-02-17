#!/usr/bin/env python3
"""
Sample Transaction Producer for Fraud Detection Testing
=======================================================
Generates sample e-commerce transactions and sends to Kafka topic.
Can also be used standalone to generate test data files.

Usage:
    python transaction_producer.py [--kafka] [--count 100] [--delay 1.0]
"""

import json
import random
import time
import argparse
from datetime import datetime
from typing import Dict, List

try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: kafka-python not installed. File output mode only.")


class TransactionGenerator:
    """Generates realistic e-commerce transaction data."""
    
    def __init__(self):
        self.merchants = [
            'AMAZON', 'EBAY', 'WALMART', 'TARGET', 'BESTBUY',
            'APPLE', 'GOOGLE', 'NETFLIX', 'SPOTIFY', 'UBER',
            'DOORDASH', 'GRUBHUB', 'AIRBNB', 'EXPEDIA', 'PAYPAL'
        ]
        self.countries = ['US', 'UK', 'CA', 'DE', 'FR', 'JP', 'AU', 'BR']
        self.card_types = ['visa', 'mastercard', 'amex', 'discover']
    
    def generate_transaction(self, user_id: str = None, force_high_value: bool = False) -> Dict:
        """Generate a single transaction."""
        # Generate transaction amount with some high-value transactions
        if force_high_value:
            amount = random.uniform(1001, 8000)
        else:
            # 10% chance of high-value transaction
            if random.random() < 0.1:
                amount = random.uniform(1001, 8000)
            else:
                amount = random.uniform(5, 999)
        
        transaction = {
            'transaction_id': f"TXN-{random.randint(100000000, 999999999)}",
            'user_id': user_id or f"USER-{random.randint(10000, 99999)}",
            'amount': round(amount, 2),
            'timestamp': datetime.now().isoformat(),
            'merchant_id': random.choice(self.merchants),
            'card_last4': f"{random.randint(1000, 9999)}",
            'country': random.choice(self.countries),
            'currency': 'USD',
            'card_type': random.choice(self.card_types)
        }
        
        return transaction
    
    def generate_batch(self, count: int, user_id: str = None) -> List[Dict]:
        """Generate a batch of transactions."""
        return [self.generate_transaction(user_id) for _ in range(count)]


class KafkaTransactionProducer:
    """Producer that sends transactions to Kafka."""
    
    def __init__(
        self,
        bootstrap_servers: List[str] = ['localhost:9092'],
        topic: str = 'ecommerce-transactions'
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.generator = TransactionGenerator()
        self.producer = None
        
        if KAFKA_AVAILABLE:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks='all',
                    retries=3,
                    retry_backoff_ms=1000
                )
                print(f"Connected to Kafka: {bootstrap_servers}")
            except Exception as e:
                print(f"Could not connect to Kafka: {e}")
                self.producer = None
    
    def send_transaction(self, transaction: Dict, key: str = None) -> bool:
        """Send a single transaction to Kafka."""
        if not self.producer:
            print("Kafka producer not available")
            return False
        
        try:
            future = self.producer.send(self.topic, key=key, value=transaction)
            future.get(timeout=10)  # Wait for confirmation
            return True
        except Exception as e:
            print(f"Failed to send transaction: {e}")
            return False
    
    def send_batch(self, transactions: List[Dict]) -> int:
        """Send multiple transactions to Kafka."""
        sent = 0
        for tx in transactions:
            if self.send_transaction(tx, key=tx.get('user_id')):
                sent += 1
        return sent
    
    def produce_continuously(
        self,
        count: int = None,
        delay: float = 1.0,
        batch_size: int = 1
    ) -> None:
        """Continuously produce transactions."""
        print(f"Producing transactions to topic: {self.topic}")
        print(f"Delay between batches: {delay}s")
        print(f"Batch size: {batch_size}")
        print("Press Ctrl+C to stop\n")
        
        produced = 0
        try:
            while count is None or produced < count:
                # Generate and send batch
                transactions = self.generator.generate_batch(batch_size)
                sent = self.send_batch(transactions)
                produced += sent
                
                for tx in transactions:
                    print(f"[{produced}] Sent: {tx['transaction_id']} - ${tx['amount']:.2f}")
                
                time.sleep(delay)
        
        except KeyboardInterrupt:
            print("\nStopping producer...")
        
        finally:
            if self.producer:
                self.producer.flush()
                self.producer.close()
            print(f"Total transactions produced: {produced}")


def generate_test_data_file(count: int = 100, output_file: str = 'test_transactions.json'):
    """Generate test data file."""
    generator = TransactionGenerator()
    transactions = generator.generate_batch(count)
    
    with open(output_file, 'w') as f:
        json.dump(transactions, f, indent=2)
    
    print(f"Generated {count} transactions to {output_file}")
    return transactions


def main():
    parser = argparse.ArgumentParser(description='Transaction Producer for Fraud Detection')
    parser.add_argument('--kafka', action='store_true', help='Send to Kafka')
    parser.add_argument('--servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='ecommerce-transactions', help='Kafka topic')
    parser.add_argument('--count', type=int, default=None, help='Number of transactions (None=infinite)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between batches (seconds)')
    parser.add_argument('--batch-size', type=int, default=1, help='Transactions per batch')
    parser.add_argument('--output-file', default='test_transactions.json', help='Output file for test data')
    
    args = parser.parse_args()
    
    if args.kafka:
        # Kafka mode
        servers = args.servers.split(',')
        producer = KafkaTransactionProducer(
            bootstrap_servers=servers,
            topic=args.topic
        )
        producer.produce_continuously(
            count=args.count,
            delay=args.delay,
            batch_size=args.batch_size
        )
    else:
        # File mode
        count = args.count or 100
        generate_test_data_file(count, args.output_file)


if __name__ == "__main__":
    main()
