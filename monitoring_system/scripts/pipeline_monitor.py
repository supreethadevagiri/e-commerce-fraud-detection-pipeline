#!/usr/bin/env python3
"""
Pipeline Monitoring Script
==========================
Collects and reports key metrics for data pipeline monitoring:
- Data volume (records processed per hour)
- Processing time (batch duration, stream latency)
- Error rates (failed records, exceptions)

Usage:
    python pipeline_monitor.py --component airflow --metric all
    python pipeline_monitor.py --component kafka --metric volume --hours 24
    python pipeline_monitor.py --component ml --metric latency --output json
"""

import os
import sys
import json
import re
import glob
import logging
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import threading
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pipeline.monitor')

# =============================================================================
# CONFIGURATION
# =============================================================================

# User's actual Airflow logs location
AIRFLOW_LOGS_DIR = os.path.expanduser('~/airflow/logs/dag_id=ecommerce_batch_pipeline')

# Expected log locations (for JSON format)
LOG_PATHS = {
    'airflow': os.path.expanduser('~/airflow/logs/airflow_metrics.log'),
    'kafka': '/var/log/kafka/consumer_stats.log',
    'ml': '/var/log/ml/metrics.log',
    'pipeline': '/var/log/pipeline/metrics.log'
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MetricRecord:
    """Data class for metric records"""
    timestamp: datetime
    component: str
    metric_type: str
    value: float
    unit: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'metric_type': self.metric_type,
            'value': self.value,
            'unit': self.unit,
            'metadata': self.metadata or {}
        }


# =============================================================================
# AIRFLOW TEXT LOG PARSER (for existing logs)
# =============================================================================

class AirflowTextLogParser:
    """Parser for Airflow's text-based log format"""
    
    def __init__(self, logs_dir: str = AIRFLOW_LOGS_DIR):
        self.logs_dir = logs_dir
    
    def find_log_files(self) -> List[str]:
        """Find all Airflow log files"""
        if not os.path.exists(self.logs_dir):
            logger.warning(f"Airflow logs directory not found: {self.logs_dir}")
            return []
        
        pattern = os.path.join(self.logs_dir, 'run_id=*', 'task_id=*', 'attempt=*.log')
        files = glob.glob(pattern)
        logger.info(f"Found {len(files)} Airflow log files")
        return files
    
    def parse_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from log line"""
        # Format: [2026-02-12T02:07:36.641+0100]
        pattern = r'\[(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})\.\d+([+-]\d{4})\]'
        match = re.search(pattern, line)
        if match:
            date_str, time_str, tz = match.groups()
            # Parse without timezone for simplicity
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return None
    
    def extract_metric(self, line: str) -> Optional[Dict]:
        """Extract metric from log message"""
        
        # Records ingested: 6000
        match = re.search(r'Records ingested:\s*([\d,]+)', line)
        if match:
            return {'type': 'records_processed', 'value': int(match.group(1).replace(',', ''))}
        
        # Final records: X
        match = re.search(r'Final records:\s*([\d,]+)', line)
        if match:
            return {'type': 'records_cleaned', 'value': int(match.group(1).replace(',', ''))}
        
        # Duplicates marked: X
        match = re.search(r'Duplicates marked:\s*([\d,]+)', line)
        if match:
            return {'type': 'duplicates_removed', 'value': int(match.group(1).replace(',', ''))}
        
        # Outliers flagged: X
        match = re.search(r'Outliers flagged:\s*([\d,]+)', line)
        if match:
            return {'type': 'outliers_flagged', 'value': int(match.group(1).replace(',', ''))}
        
        # Missing values filled: X
        match = re.search(r'Missing values filled:\s*([\d,]+)', line)
        if match:
            return {'type': 'missing_values', 'value': int(match.group(1).replace(',', ''))}
        
        # Hourly category summary: X rows
        match = re.search(r'Hourly category summary:\s*([\d,]+)', line)
        if match:
            return {'type': 'hourly_category_rows', 'value': int(match.group(1).replace(',', ''))}
        
        # Hourly overall summary: X rows
        match = re.search(r'Hourly overall summary:\s*([\d,]+)', line)
        if match:
            return {'type': 'hourly_overall_rows', 'value': int(match.group(1).replace(',', ''))}
        
        # Daily category summary: X rows
        match = re.search(r'Daily category summary:\s*([\d,]+)', line)
        if match:
            return {'type': 'daily_category_rows', 'value': int(match.group(1).replace(',', ''))}
        
        # Exported X records
        match = re.search(r'Exported\s+([\d,]+)', line)
        if match:
            return {'type': 'records_exported', 'value': int(match.group(1).replace(',', ''))}
        
        # Total records generated: X
        match = re.search(r'Total records generated:\s*([\d,]+)', line)
        if match:
            return {'type': 'total_generated', 'value': int(match.group(1).replace(',', ''))}
        
        # Fraudulent transactions: X
        match = re.search(r'Fraudulent [Tt]ransactions:\s*([\d,]+)', line, re.IGNORECASE)
        if match:
            return {'type': 'fraud_count', 'value': int(match.group(1).replace(',', ''))}
        
        # Progress: X records
        match = re.search(r'Progress:\s*([\d,]+)', line)
        if match:
            return {'type': 'progress', 'value': int(match.group(1).replace(',', ''))}
        
        # Hour X complete: Y records
        match = re.search(r'Hour\s+\d+\s+complete:\s*([\d,]+)', line)
        if match:
            return {'type': 'hourly_generated', 'value': int(match.group(1).replace(',', ''))}
        
        return None
    
    def parse_log_file(self, log_path: str, hours: int = 24) -> List[MetricRecord]:
        """Parse a single log file"""
        metrics = []
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Extract task name from path
        task_match = re.search(r'task_id=([^/]+)', log_path)
        task_name = task_match.group(1) if task_match else 'unknown'
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    timestamp = self.parse_timestamp(line)
                    if timestamp and timestamp >= cutoff:
                        metric = self.extract_metric(line)
                        if metric:
                            metrics.append(MetricRecord(
                                timestamp=timestamp,
                                component='airflow',
                                metric_type=metric['type'],
                                value=float(metric['value']),
                                unit='records',
                                metadata={'task': task_name, 'log_file': os.path.basename(log_path)}
                            ))
        except Exception as e:
            logger.error(f"Error reading {log_path}: {e}")
        
        return metrics
    
    def collect_metrics(self, hours: int = 24) -> List[MetricRecord]:
        """Collect all metrics from Airflow logs"""
        all_metrics = []
        log_files = self.find_log_files()
        
        for log_file in log_files:
            metrics = self.parse_log_file(log_file, hours)
            all_metrics.extend(metrics)
        
        logger.info(f"Collected {len(all_metrics)} metrics from Airflow logs")
        return all_metrics


# =============================================================================
# JSON LOG PARSERS (for future use with structured logging)
# =============================================================================

class JSONLogParser:
    """Parser for JSON-formatted log files"""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
    
    def parse_file(self, hours: int = 24) -> List[MetricRecord]:
        """Parse JSON log file"""
        metrics = []
        cutoff = datetime.now() - timedelta(hours=hours)
        
        if not os.path.exists(self.log_path):
            logger.warning(f"Log file not found: {self.log_path}")
            return metrics
        
        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        timestamp_str = data.get('timestamp', '')
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        
                        if timestamp >= cutoff:
                            # Extract metrics based on log type
                            record = self._extract_metric_record(data, timestamp)
                            if record:
                                metrics.append(record)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.debug(f"Failed to parse line: {e}")
        except Exception as e:
            logger.error(f"Error reading {self.log_path}: {e}")
        
        return metrics
    
    def _extract_metric_record(self, data: Dict, timestamp: datetime) -> Optional[MetricRecord]:
        """Extract metric record from JSON data - override in subclasses"""
        return None


class AirflowJSONParser(JSONLogParser):
    """Parser for Airflow JSON logs"""
    
    def _extract_metric_record(self, data: Dict, timestamp: datetime) -> Optional[MetricRecord]:
        message = data.get('message', '')
        
        # Extract records processed
        match = re.search(r'records_processed=(\d+)', message)
        if match:
            return MetricRecord(
                timestamp=timestamp,
                component='airflow',
                metric_type='records_processed',
                value=float(match.group(1)),
                unit='records',
                metadata={'dag_id': data.get('dag_id'), 'task_id': data.get('task_id')}
            )
        
        # Extract duration
        match = re.search(r'duration=(\d+\.?\d*)', message)
        if match:
            return MetricRecord(
                timestamp=timestamp,
                component='airflow',
                metric_type='dag_duration',
                value=float(match.group(1)),
                unit='seconds',
                metadata={'dag_id': data.get('dag_id'), 'task_id': data.get('task_id')}
            )
        
        return None


class KafkaJSONParser(JSONLogParser):
    """Parser for Kafka JSON logs"""
    
    def _extract_metric_record(self, data: Dict, timestamp: datetime) -> Optional[MetricRecord]:
        # Consumer lag
        if 'consumer_lag' in data:
            return MetricRecord(
                timestamp=timestamp,
                component='kafka',
                metric_type='consumer_lag',
                value=float(data['consumer_lag']),
                unit='messages',
                metadata={'topic': data.get('topic'), 'partition': data.get('partition')}
            )
        
        # Latency
        if 'latency_ms' in data:
            return MetricRecord(
                timestamp=timestamp,
                component='kafka',
                metric_type='processing_latency',
                value=float(data['latency_ms']),
                unit='milliseconds',
                metadata={'topic': data.get('topic')}
            )
        
        # Throughput
        if 'messages_per_second' in data:
            return MetricRecord(
                timestamp=timestamp,
                component='kafka',
                metric_type='throughput',
                value=float(data['messages_per_second']),
                unit='messages/sec',
                metadata={'topic': data.get('topic')}
            )
        
        return None


class MLJSONParser(JSONLogParser):
    """Parser for ML JSON logs"""
    
    def _extract_metric_record(self, data: Dict, timestamp: datetime) -> Optional[MetricRecord]:
        # Prediction latency
        if 'latency_ms' in data:
            return MetricRecord(
                timestamp=timestamp,
                component='ml',
                metric_type='prediction_latency',
                value=float(data['latency_ms']),
                unit='milliseconds',
                metadata={'model_version': data.get('model_version')}
            )
        
        # Confidence
        if 'confidence' in data:
            return MetricRecord(
                timestamp=timestamp,
                component='ml',
                metric_type='prediction_confidence',
                value=float(data['confidence']),
                unit='score',
                metadata={'model_version': data.get('model_version')}
            )
        
        return None


# =============================================================================
# METRICS AGGREGATOR
# =============================================================================

class MetricsAggregator:
    """Aggregate and summarize collected metrics"""
    
    @staticmethod
    def aggregate_by_hour(metrics: List[MetricRecord]) -> Dict:
        """Aggregate metrics by hour"""
        hourly = defaultdict(lambda: defaultdict(list))
        
        for metric in metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            key = f"{metric.component}:{metric.metric_type}"
            hourly[hour_key][key].append(metric.value)
        
        result = {}
        for hour, components in hourly.items():
            result[hour.isoformat()] = {
                key: {
                    'count': len(values),
                    'sum': sum(values),
                    'avg': sum(values) / len(values) if values else 0,
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0
                }
                for key, values in components.items()
            }
        
        return result
    
    @staticmethod
    def summarize_volume(metrics: List[MetricRecord]) -> Dict:
        """Summarize data volume metrics"""
        volume_metrics = [m for m in metrics if 'record' in m.metric_type or 'message' in m.metric_type]
        
        if not volume_metrics:
            return {'total_records': 0, 'avg_per_hour': 0}
        
        total = sum(m.value for m in volume_metrics)
        hours = set(m.timestamp.replace(minute=0, second=0, microsecond=0) for m in volume_metrics)
        avg_per_hour = total / len(hours) if hours else 0
        
        return {
            'total_records': int(total),
            'avg_per_hour': round(avg_per_hour, 2),
            'unique_hours': len(hours)
        }
    
    @staticmethod
    def summarize_latency(metrics: List[MetricRecord]) -> Dict:
        """Summarize latency metrics"""
        latency_metrics = [m for m in metrics if 'latency' in m.metric_type or 'duration' in m.metric_type]
        
        if not latency_metrics:
            return {'avg_latency_ms': 0, 'p95_latency_ms': 0, 'p99_latency_ms': 0}
        
        values = sorted([m.value for m in latency_metrics])
        
        def percentile(data, p):
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f]) if f != c else data[f]
        
        return {
            'avg_latency_ms': round(sum(values) / len(values), 2),
            'min_latency_ms': round(values[0], 2),
            'max_latency_ms': round(values[-1], 2),
            'p50_latency_ms': round(percentile(values, 50), 2),
            'p95_latency_ms': round(percentile(values, 95), 2),
            'p99_latency_ms': round(percentile(values, 99), 2),
            'sample_count': len(values)
        }
    
    @staticmethod
    def calculate_error_rate(metrics: List[MetricRecord], hours: int = 24) -> Dict:
        """Calculate error rates"""
        error_metrics = [m for m in metrics if m.metric_type == 'error']
        total_operations = len([m for m in metrics if m.metric_type != 'error'])
        
        error_count = len(error_metrics)
        error_rate = (error_count / total_operations * 100) if total_operations > 0 else 0
        
        return {
            'error_count': error_count,
            'total_operations': total_operations,
            'error_rate_percent': round(error_rate, 4),
            'time_window_hours': hours
        }


# =============================================================================
# METRICS REPORTER
# =============================================================================

class MetricsReporter:
    """Generate reports from collected metrics"""
    
    def __init__(self):
        self.aggregator = MetricsAggregator()
    
    def collect_all_metrics(self, hours: int = 24) -> List[MetricRecord]:
        """Collect metrics from all sources"""
        all_metrics = []
        
        # Airflow (text logs)
        airflow_parser = AirflowTextLogParser()
        all_metrics.extend(airflow_parser.collect_metrics(hours))
        
        # JSON logs (if they exist)
        for component, log_path in LOG_PATHS.items():
            if os.path.exists(log_path):
                if component == 'airflow':
                    parser = AirflowJSONParser(log_path)
                elif component == 'kafka':
                    parser = KafkaJSONParser(log_path)
                elif component == 'ml':
                    parser = MLJSONParser(log_path)
                else:
                    continue
                
                metrics = parser.parse_file(hours)
                all_metrics.extend(metrics)
        
        return all_metrics
    
    def generate_report(self, hours: int = 24, output_format: str = 'text') -> str:
        """Generate comprehensive metrics report"""
        metrics = self.collect_all_metrics(hours)
        
        if output_format == 'json':
            return self._generate_json_report(metrics, hours)
        else:
            return self._generate_text_report(metrics, hours)
    
    def _generate_text_report(self, metrics: List[MetricRecord], hours: int) -> str:
        """Generate human-readable text report"""
        lines = []
        lines.append("=" * 80)
        lines.append("PIPELINE METRICS REPORT")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Time Window: Last {hours} hours")
        lines.append("=" * 80)
        lines.append("")
        
        # Data Volume Summary
        lines.append("-" * 40)
        lines.append("DATA VOLUME METRICS")
        lines.append("-" * 40)
        volume_summary = self.aggregator.summarize_volume(metrics)
        for key, value in volume_summary.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Latency Summary
        lines.append("-" * 40)
        lines.append("LATENCY METRICS")
        lines.append("-" * 40)
        latency_summary = self.aggregator.summarize_latency(metrics)
        for key, value in latency_summary.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Error Rate
        lines.append("-" * 40)
        lines.append("ERROR METRICS")
        lines.append("-" * 40)
        error_summary = self.aggregator.calculate_error_rate(metrics, hours)
        for key, value in error_summary.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Component Breakdown
        lines.append("-" * 40)
        lines.append("COMPONENT BREAKDOWN")
        lines.append("-" * 40)
        component_counts = defaultdict(int)
        for m in metrics:
            component_counts[m.component] += 1
        for component, count in sorted(component_counts.items()):
            lines.append(f"  {component}: {count} metrics")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_json_report(self, metrics: List[MetricRecord], hours: int) -> str:
        """Generate JSON format report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'time_window_hours': hours,
            'summary': {
                'volume': self.aggregator.summarize_volume(metrics),
                'latency': self.aggregator.summarize_latency(metrics),
                'errors': self.aggregator.calculate_error_rate(metrics, hours)
            },
            'metrics': [m.to_dict() for m in metrics[:100]]  # Limit to first 100
        }
        
        return json.dumps(report, indent=2)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline Monitoring Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --hours 24
  %(prog)s --output json --hours 6
  %(prog)s --component airflow --hours 12
        """
    )
    
    parser.add_argument('--component', '-c',
                        choices=['airflow', 'kafka', 'ml', 'all'],
                        default='all',
                        help='Component to monitor (default: all)')
    
    parser.add_argument('--hours', '-t',
                        type=int,
                        default=24,
                        help='Time window in hours (default: 24)')
    
    parser.add_argument('--output', '-o',
                        choices=['text', 'json'],
                        default='text',
                        help='Output format (default: text)')
    
    parser.add_argument('--log-path',
                        help='Custom log file path')
    
    args = parser.parse_args()
    
    # Generate report
    reporter = MetricsReporter()
    report = reporter.generate_report(hours=args.hours, output_format=args.output)
    print(report)


if __name__ == '__main__':
    main()
