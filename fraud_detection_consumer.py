#!/usr/bin/env python3
"""
E-commerce Real-Time Fraud Detection Consumer
==============================================
A Kafka consumer for real-time fraud detection with:
- Transaction counting per minute
- Average amount calculation per 5-minute window
- Threshold violation detection (>$1000)
- Micro-batch processing (30-second windows)
- Error handling and retry mechanism

Author: Stream Processing Expert
"""

import json
import time
import logging
import signal
import sys
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from threading import Lock
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "fraud_detection.log"

# Kafka imports (using kafka-python library)
try:
    from kafka import KafkaConsumer, KafkaError
    from kafka.errors import KafkaConnectionError, NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: kafka-python not installed. Running in simulation mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Transaction:
    """Represents an e-commerce transaction."""
    transaction_id: str
    user_id: str
    amount: float
    timestamp: datetime
    merchant_id: str
    card_last4: str
    country: str
    
    @classmethod
    def from_json(cls, data: dict) -> 'Transaction':
        """Create Transaction from JSON data."""
        return cls(
            transaction_id=data.get('transaction_id', ''),
            user_id=data.get('user_id', ''),
            amount=float(data.get('amount', 0)),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            merchant_id=data.get('merchant_id', ''),
            card_last4=data.get('card_last4', ''),
            country=data.get('country', 'US')
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'transaction_id': self.transaction_id,
            'user_id': self.user_id,
            'amount': self.amount,
            'timestamp': self.timestamp.isoformat(),
            'merchant_id': self.merchant_id,
            'card_last4': self.card_last4,
            'country': self.country
        }


@dataclass
class FraudAlert:
    """Represents a fraud alert."""
    alert_type: str
    transaction: Transaction
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    detected_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'detected_at': self.detected_at.isoformat(),
            'transaction': self.transaction.to_dict()
        }


# ============================================================================
# WINDOWING AND ANALYTICS
# ============================================================================

class TimeWindow:
    """Time-based window for aggregations."""
    
    def __init__(self, window_size_seconds: int):
        self.window_size = timedelta(seconds=window_size_seconds)
        self.data: deque = deque()
        self.lock = Lock()
    
    def add(self, transaction: Transaction) -> None:
        """Add transaction to window."""
        with self.lock:
            self.data.append({
                'transaction': transaction,
                'added_at': datetime.now()
            })
            self._cleanup_old_data()
    
    def _cleanup_old_data(self) -> None:
        """Remove data older than window size."""
        cutoff = datetime.now() - self.window_size
        while self.data and self.data[0]['added_at'] < cutoff:
            self.data.popleft()
    
    def get_transactions(self) -> List[Transaction]:
        """Get all transactions in current window."""
        with self.lock:
            self._cleanup_old_data()
            return [item['transaction'] for item in self.data]
    
    def count(self) -> int:
        """Get count of transactions in window."""
        return len(self.get_transactions())
    
    def average_amount(self) -> float:
        """Calculate average transaction amount."""
        transactions = self.get_transactions()
        if not transactions:
            return 0.0
        return sum(t.amount for t in transactions) / len(transactions)
    
    def clear(self) -> None:
        """Clear all data from window."""
        with self.lock:
            self.data.clear()


class AnalyticsEngine:
    """Real-time analytics engine for fraud detection."""
    
    def __init__(self):
        # 1-minute window for transaction counting
        self.minute_window = TimeWindow(60)
        # 5-minute window for average calculation
        self.five_minute_window = TimeWindow(300)
        # Threshold for high-value transactions
        self.amount_threshold = 1000.0
        # Store alerts
        self.alerts: List[FraudAlert] = []
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_alerts': 0,
            'threshold_violations': 0,
            'start_time': datetime.now()
        }
        self.lock = Lock()
    
    def process_transaction(self, transaction: Transaction) -> List[FraudAlert]:
        """Process a single transaction and return any alerts."""
        alerts = []
        
        # Add to windows
        self.minute_window.add(transaction)
        self.five_minute_window.add(transaction)
        
        # Update statistics
        with self.lock:
            self.stats['total_processed'] += 1
        
        # Check threshold violation
        if transaction.amount > self.amount_threshold:
            alert = FraudAlert(
                alert_type='THRESHOLD_VIOLATION',
                transaction=transaction,
                severity=self._calculate_severity(transaction.amount),
                message=f"Transaction amount ${transaction.amount:.2f} exceeds threshold of ${self.amount_threshold:.2f}"
            )
            alerts.append(alert)
            with self.lock:
                self.stats['threshold_violations'] += 1
        
        # Check for rapid transactions (more than 5 in 1 minute)
        minute_count = self.minute_window.count()
        if minute_count > 5:
            alert = FraudAlert(
                alert_type='RAPID_TRANSACTIONS',
                transaction=transaction,
                severity='medium',
                message=f"User made {minute_count} transactions in the last minute"
            )
            alerts.append(alert)
        
        # Store alerts
        self.alerts.extend(alerts)
        with self.lock:
            self.stats['total_alerts'] += len(alerts)
        
        return alerts
    
    def _calculate_severity(self, amount: float) -> str:
        """Calculate alert severity based on amount."""
        if amount > 5000:
            return 'critical'
        elif amount > 2500:
            return 'high'
        elif amount > 1500:
            return 'medium'
        else:
            return 'low'
    
    def get_minute_stats(self) -> dict:
        """Get statistics for the last minute."""
        return {
            'transactions_last_minute': self.minute_window.count(),
            'average_amount_last_5min': self.five_minute_window.average_amount(),
            'transactions_last_5min': self.five_minute_window.count()
        }
    
    def get_overall_stats(self) -> dict:
        """Get overall processing statistics."""
        with self.lock:
            runtime = datetime.now() - self.stats['start_time']
            return {
                'total_processed': self.stats['total_processed'],
                'total_alerts': self.stats['total_alerts'],
                'threshold_violations': self.stats['threshold_violations'],
                'runtime_seconds': runtime.total_seconds(),
                'processing_rate': self.stats['total_processed'] / max(runtime.total_seconds(), 1)
            }


# ============================================================================
# MICRO-BATCH PROCESSOR
# ============================================================================

class MicroBatchProcessor:
    """Processes transactions in micro-batches."""
    
    def __init__(self, batch_interval_seconds: int = 30):
        self.batch_interval = batch_interval_seconds
        self.buffer: List[Transaction] = []
        self.buffer_lock = Lock()
        self.last_flush = datetime.now()
        self.analytics = AnalyticsEngine()
        self.running = False
    
    def add_transaction(self, transaction: Transaction) -> List[FraudAlert]:
        """Add transaction to buffer and process if needed."""
        alerts = []
        
        with self.buffer_lock:
            self.buffer.append(transaction)
            
            # Process immediately for fraud detection
            alerts = self.analytics.process_transaction(transaction)
            
            # Check if we should flush
            if (datetime.now() - self.last_flush).total_seconds() >= self.batch_interval:
                self._flush_batch()
        
        return alerts
    
    def _flush_batch(self) -> dict:
        """Flush the current batch and return summary."""
        with self.buffer_lock:
            batch_size = len(self.buffer)
            if batch_size == 0:
                return {'batch_size': 0}
            
            # Calculate batch statistics
            amounts = [t.amount for t in self.buffer]
            summary = {
                'batch_size': batch_size,
                'total_amount': sum(amounts),
                'average_amount': sum(amounts) / len(amounts),
                'min_amount': min(amounts),
                'max_amount': max(amounts),
                'timestamp': datetime.now().isoformat()
            }
            
            # Clear buffer
            self.buffer.clear()
            self.last_flush = datetime.now()
            
            return summary
    
    def force_flush(self) -> dict:
        """Force flush the current batch."""
        return self._flush_batch()
    
    def get_analytics(self) -> AnalyticsEngine:
        """Get the analytics engine."""
        return self.analytics


# ============================================================================
# KAFKA CONSUMER WITH RETRY LOGIC
# ============================================================================

class FraudDetectionConsumer:
    """Kafka consumer for fraud detection with retry mechanism."""
    
    def __init__(
        self,
        bootstrap_servers: List[str] = ['localhost:9095'],
        topic: str = 'transactions',
        group_id: str = 'fraud-detection-group',
        batch_interval: int = 30,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.batch_interval = batch_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.consumer = None
        self.processor = MicroBatchProcessor(batch_interval)
        self.running = False
        self.metrics = {
            'messages_consumed': 0,
            'parse_errors': 0,
            'processing_errors': 0,
            'retries': 0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received. Stopping consumer...")
        self.running = False
    
    def _create_consumer(self):
        """Create Kafka consumer with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                if not KAFKA_AVAILABLE:
                    logger.warning("Kafka not available - running in simulation mode")
                    return None
                
                logger.info(f"Connecting to Kafka (attempt {attempt}/{self.max_retries})...")
                consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    auto_commit_interval_ms=5000,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    key_deserializer=lambda m: m.decode('utf-8') if m else None,
                    session_timeout_ms=30000,
                    heartbeat_interval_ms=10000
                )
                logger.info(f"Successfully connected to Kafka: {self.bootstrap_servers}")
                return consumer
            
            except (KafkaConnectionError, NoBrokersAvailable) as e:
                logger.error(f"Kafka connection failed: {e}")
                self.metrics['retries'] += 1
                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Max retries exceeded. Could not connect to Kafka.")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error creating consumer: {e}")
                raise
        
        return None
    
    def _parse_message(self, message) -> Optional[Transaction]:
        """Parse Kafka message into Transaction object."""
        try:
            if isinstance(message, dict):
                data = message
            else:
                data = message.value if hasattr(message, 'value') else message
            
            transaction = Transaction.from_json(data)
            return transaction
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse message: {e}")
            self.metrics['parse_errors'] += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected parsing error: {e}")
            self.metrics['parse_errors'] += 1
            return None
    
    def _process_message(self, transaction: Transaction) -> None:
        """Process a single transaction."""
        try:
            alerts = self.processor.add_transaction(transaction)
            self.metrics['messages_consumed'] += 1
            
            # Log transaction
            logger.info(
                f"Processed transaction: ID={transaction.transaction_id}, "
                f"Amount=${transaction.amount:.2f}, User={transaction.user_id}"
            )
            
            # Log any alerts
            for alert in alerts:
                self._log_alert(alert)
        
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            self.metrics['processing_errors'] += 1
    
    def _log_alert(self, alert: FraudAlert) -> None:
        """Log a fraud alert with formatting."""
        severity_colors = {
            'critical': '🔴 CRITICAL',
            'high': '🟠 HIGH',
            'medium': '🟡 MEDIUM',
            'low': '🟢 LOW'
        }
        severity_label = severity_colors.get(alert.severity, alert.severity.upper())
        
        logger.warning(
            f"\n{'='*60}\n"
            f"FRAUD ALERT - {severity_label}\n"
            f"{'='*60}\n"
            f"Type: {alert.alert_type}\n"
            f"Message: {alert.message}\n"
            f"Transaction ID: {alert.transaction.transaction_id}\n"
            f"User ID: {alert.transaction.user_id}\n"
            f"Amount: ${alert.transaction.amount:.2f}\n"
            f"Merchant: {alert.transaction.merchant_id}\n"
            f"Country: {alert.transaction.country}\n"
            f"Detected At: {alert.detected_at.isoformat()}\n"
            f"{'='*60}"
        )
    
    def _print_batch_summary(self) -> None:
        """Print micro-batch summary."""
        summary = self.processor.force_flush()
        stats = self.processor.get_analytics().get_minute_stats()
        overall = self.processor.get_analytics().get_overall_stats()
        
        logger.info(
            f"\n{'='*60}\n"
            f"MICRO-BATCH SUMMARY (30-second window)\n"
            f"{'='*60}\n"
            f"Batch Statistics:\n"
            f"  - Transactions in batch: {summary.get('batch_size', 0)}\n"
            f"  - Total amount: ${summary.get('total_amount', 0):.2f}\n"
            f"  - Average amount: ${summary.get('average_amount', 0):.2f}\n"
            f"  - Min amount: ${summary.get('min_amount', 0):.2f}\n"
            f"  - Max amount: ${summary.get('max_amount', 0):.2f}\n"
            f"\n"
            f"Window Analytics:\n"
            f"  - Transactions last minute: {stats['transactions_last_minute']}\n"
            f"  - Transactions last 5 min: {stats['transactions_last_5min']}\n"
            f"  - Average amount (5min): ${stats['average_amount_last_5min']:.2f}\n"
            f"\n"
            f"Overall Statistics:\n"
            f"  - Total processed: {overall['total_processed']}\n"
            f"  - Total alerts: {overall['total_alerts']}\n"
            f"  - Threshold violations: {overall['threshold_violations']}\n"
            f"  - Processing rate: {overall['processing_rate']:.2f} tx/sec\n"
            f"{'='*60}"
        )
    
    def _generate_sample_transaction(self) -> Transaction:
        """Generate a sample transaction for simulation mode."""
        return Transaction(
            transaction_id=f"TXN-{random.randint(100000, 999999)}",
            user_id=f"USER-{random.randint(1000, 9999)}",
            amount=random.uniform(10, 5000),
            timestamp=datetime.now(),
            merchant_id=f"MERCH-{random.randint(100, 999)}",
            card_last4=f"{random.randint(1000, 9999)}",
            country=random.choice(['US', 'UK', 'CA', 'DE', 'FR', 'JP'])
        )
    
    def run(self) -> None:
        """Main consumer loop."""
        logger.info("="*60)
        logger.info("FRAUD DETECTION CONSUMER STARTING")
        logger.info("="*60)
        logger.info(f"Topic: {self.topic}")
        logger.info(f"Bootstrap Servers: {self.bootstrap_servers}")
        logger.info(f"Batch Interval: {self.batch_interval} seconds")
        logger.info(f"Amount Threshold: ${self.processor.analytics.amount_threshold}")
        logger.info("="*60)
        
        self.running = True
        last_summary_time = time.time()
        
        try:
            self.consumer = self._create_consumer()
            
            if self.consumer:
                # Real Kafka consumer mode
                logger.info("Running in Kafka consumer mode")
                for message in self.consumer:
                    if not self.running:
                        break
                    
                    transaction = self._parse_message(message)
                    if transaction:
                        self._process_message(transaction)
                    
                    # Print batch summary every 30 seconds
                    if time.time() - last_summary_time >= self.batch_interval:
                        self._print_batch_summary()
                        last_summary_time = time.time()
            
            else:
                # Simulation mode
                logger.info("Running in SIMULATION mode (generating sample transactions)")
                while self.running:
                    # Generate sample transaction
                    transaction = self._generate_sample_transaction()
                    self._process_message(transaction)
                    
                    # Print batch summary every 30 seconds
                    if time.time() - last_summary_time >= self.batch_interval:
                        self._print_batch_summary()
                        last_summary_time = time.time()
                    
                    # Small delay between transactions
                    time.sleep(random.uniform(0.5, 2.0))
        
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        
        finally:
            self._shutdown()
    
    def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down consumer...")
        
        # Print final summary
        self._print_batch_summary()
        
        # Print final metrics
        logger.info(
            f"\n{'='*60}\n"
            f"FINAL METRICS\n"
            f"{'='*60}\n"
            f"Messages consumed: {self.metrics['messages_consumed']}\n"
            f"Parse errors: {self.metrics['parse_errors']}\n"
            f"Processing errors: {self.metrics['processing_errors']}\n"
            f"Retries: {self.metrics['retries']}\n"
            f"{'='*60}"
        )
        
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        
        logger.info("Consumer shutdown complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    # Configuration
    config = {
        'bootstrap_servers': ['localhost:9095'],
        'topic': 'ecommerce-transactions',
        'group_id': 'fraud-detection-consumer-v1',
        'batch_interval': 30,  # 30 seconds
        'max_retries': 3,
        'retry_delay': 5
    }
    
    # Create and run consumer
    consumer = FraudDetectionConsumer(**config)
    consumer.run()


if __name__ == "__main__":
    main()
