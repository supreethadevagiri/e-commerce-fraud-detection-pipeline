#!/usr/bin/env python3
"""
Pipeline Dashboard - CLI Version
=================================
Simple command-line dashboard for pipeline monitoring.
(No Jupyter required)
"""

import os
import sys
import time
import subprocess
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pipeline_monitor import MetricsReporter, AirflowTextLogParser


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    """Print dashboard header"""
    print("=" * 75)
    print("  E-COMMERCE FRAUD DETECTION PIPELINE - MONITORING DASHBOARD")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 75)


def print_summary_cards(metrics_summary):
    """Print summary cards"""
    print("\n" + "─" * 75)
    print("  SUMMARY")
    print("─" * 75)
    
    for key, value in metrics_summary.items():
        if isinstance(value, dict):
            print(f"\n  {key.upper()}:")
            for k, v in value.items():
                print(f"    {k:25s}: {v}")
        else:
            print(f"  {key:25s}: {value}")


def check_component_status():
    """Check status of pipeline components"""
    status = {}
    
    # Check API
    try:
        import requests
        response = requests.get('http://localhost:4500/health', timeout=2)
        status['API'] = '✅ Online' if response.status_code == 200 else '❌ Error'
    except:
        status['API'] = '❌ Down'
    
    # Check Kafka
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        status['Kafka'] = '✅ Running' if 'kafka' in result.stdout.lower() else '❌ Down'
    except:
        status['Kafka'] = '❌ Unknown'
    
    # Check Airflow
    try:
        result = subprocess.run(['pgrep', '-f', 'airflow scheduler'], capture_output=True)
        status['Airflow'] = '✅ Running' if result.returncode == 0 else '❌ Down'
    except:
        status['Airflow'] = '❌ Unknown'
    
    return status


def print_component_status():
    """Print component status"""
    print("\n" + "─" * 75)
    print("  COMPONENT STATUS")
    print("─" * 75)
    
    status = check_component_status()
    for component, state in status.items():
        print(f"  {component:20s}: {state}")


def print_data_flow(metrics_by_task):
    """Print data flow diagram"""
    print("\n" + "─" * 75)
    print("  DATA FLOW")
    print("─" * 75)
    
    # Extract values
    ingest = metrics_by_task.get('ingest_data', {}).get('records_processed', 0)
    clean = metrics_by_task.get('clean_data', {}).get('records_cleaned', 0)
    dup = metrics_by_task.get('clean_data', {}).get('duplicates_removed', 0)
    out = metrics_by_task.get('clean_data', {}).get('outliers_flagged', 0)
    miss = metrics_by_task.get('clean_data', {}).get('missing_values', 0)
    hourly_cat = metrics_by_task.get('aggregate_data', {}).get('hourly_category_rows', 0)
    hourly_over = metrics_by_task.get('aggregate_data', {}).get('hourly_overall_rows', 0)
    daily_cat = metrics_by_task.get('aggregate_data', {}).get('daily_category_rows', 0)
    export = metrics_by_task.get('export_to_snowflake', {}).get('records_exported', 0)
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  INGEST:     {ingest:>8,} records read from CSV                      │
  │      ↓                                                              │
  │  CLEAN:      {clean:>8,} records after cleaning                      │
  │              - {dup:>7,} duplicates removed                           │
  │              - {out:>7,} outliers flagged                             │
  │              - {miss:>7,} missing values filled                        │
  │      ↓                                                              │
  │  AGGREGATE:  {hourly_cat:>8,} hourly category rows                    │
  │              {hourly_over:>8,} hourly overall rows                     │
  │              {daily_cat:>8,} daily category rows                       │
  │      ↓                                                              │
  │  EXPORT:     {export:>8,} total records to Snowflake                 │
  │              (6 tables: RAW, CLEANED, HOURLY_CAT,                    │
  │               HOURLY_OVERALL, DAILY_CAT, ML_FEATURES)               │
  └─────────────────────────────────────────────────────────────────────┘
""")


def collect_metrics_summary(hours=24):
    """Collect and summarize metrics"""
    reporter = MetricsReporter()
    metrics = reporter.collect_all_metrics(hours)
    
    # Group by task
    metrics_by_task = {}
    for m in metrics:
        task = m.metadata.get('task', 'unknown')
        if task not in metrics_by_task:
            metrics_by_task[task] = {}
        
        metric_type = m.metric_type
        if metric_type in metrics_by_task[task]:
            # For exports, sum them; for others, keep max
            if metric_type == 'records_exported':
                metrics_by_task[task][metric_type] += int(m.value)
            else:
                metrics_by_task[task][metric_type] = max(metrics_by_task[task][metric_type], int(m.value))
        else:
            metrics_by_task[task][metric_type] = int(m.value)
    
    # Calculate totals
    total_raw = sum(t.get('records_processed', 0) for t in metrics_by_task.values())
    total_clean = sum(t.get('records_cleaned', 0) for t in metrics_by_task.values())
    total_export = sum(t.get('records_exported', 0) for t in metrics_by_task.values())
    
    summary = {
        'total_metrics': len(metrics),
        'unique_tasks': len(metrics_by_task),
        'total_raw': total_raw,
        'total_cleaned': total_clean,
        'total_exported': total_export
    }
    
    return summary, metrics_by_task


def print_dashboard(refresh=False, interval=60):
    """Print the full dashboard"""
    try:
        while True:
            if refresh:
                clear_screen()
            
            print_header()
            
            # Collect metrics
            summary, metrics_by_task = collect_metrics_summary()
            
            # Print sections
            print_summary_cards(summary)
            print_component_status()
            print_data_flow(metrics_by_task)
            
            print("\n" + "=" * 75)
            
            if not refresh:
                break
            
            print(f"\nRefreshing in {interval} seconds... (Press Ctrl+C to exit)")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pipeline Dashboard')
    parser.add_argument('--refresh', '-r', action='store_true', help='Auto-refresh mode')
    parser.add_argument('--interval', '-i', type=int, default=60, help='Refresh interval in seconds')
    args = parser.parse_args()
    
    print_dashboard(refresh=args.refresh, interval=args.interval)


if __name__ == '__main__':
    main()
