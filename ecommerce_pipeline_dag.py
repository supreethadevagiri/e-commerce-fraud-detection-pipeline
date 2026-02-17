"""
================================================================================
ECOMMERCE BATCH PROCESSING PIPELINE - AIRFLOW DAG (FIXED WITH RSA KEY AUTH)
================================================================================

This DAG implements a complete ETL pipeline for e-commerce transaction data:
1. Data Ingestion: Load CSV files into staging area
2. Data Cleaning: Handle missing values, duplicates, outliers
3. Data Validation: Quality checks and anomaly detection
4. Aggregation: Create hourly/daily summaries by category
5. Feature Engineering: Build features for fraud detection ML model
6. Export to Snowflake: Load processed data into data warehouse

Author: Data Engineering Team
Version: 1.2.0 (Fixed with RSA key authentication)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
import pandas as pd
import numpy as np
import logging
import os
import json
import glob
from typing import Dict, List

# For RSA key authentication
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# DYNAMIC PATH CONFIGURATION - Works anywhere!
# =============================================================================

# Get the directory where this DAG file is located
DAG_FILE_PATH = os.path.abspath(__file__)
DAGS_FOLDER = os.path.dirname(DAG_FILE_PATH)
PROJECT_ROOT = os.path.dirname(DAGS_FOLDER)

# Data directories (relative to project root)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw') + os.sep
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed') + os.sep
ARCHIVE_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'archive') + os.sep

# Pipeline configuration
CONFIG = {
    'raw_data_path': RAW_DATA_PATH,
    'processed_data_path': PROCESSED_DATA_PATH,
    'archive_data_path': ARCHIVE_DATA_PATH,
    'csv_filename': 'transactions_{{ ds }}.csv',
    'snowflake_database': 'ECOMMERCE_DW',
    'snowflake_schema_raw': 'RAW_DATA',
    'snowflake_schema_cleaned': 'CLEANED_DATA',
    'snowflake_schema_analytics': 'ANALYTICS',
    'snowflake_schema_ml': 'ML_FEATURES',
    'batch_size': 10000,
    'outlier_threshold': 3.0,
    'fraud_high_risk_countries': ['RU', 'CN', 'NG', 'PK', 'BD'],
    'fraud_high_risk_payment_methods': ['Cryptocurrency'],
}

# =============================================================================
# DEFAULT ARGS
# =============================================================================

default_args = {
    'owner': 'data_engineering',
    'depends_on_past': False,
    'email': ['data-team@company.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# =============================================================================
# HELPER FUNCTION: Generate Sample Data
# =============================================================================

def generate_sample_data(n_records=5000):
    """Generate sample transaction data for testing."""
    np.random.seed(42)
    
    categories = {
        'Electronics': (50, 2000),
        'Clothing': (10, 500),
        'Home & Garden': (20, 800),
        'Sports': (15, 600),
        'Books': (5, 100),
        'Health & Beauty': (10, 300),
        'Food & Beverage': (5, 200),
        'Toys': (10, 400)
    }
    
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Apple Pay', 'Google Pay', 'Cryptocurrency']
    devices = ['Desktop', 'Mobile', 'Tablet']
    countries = ['US', 'UK', 'CA', 'DE', 'FR', 'AU', 'JP', 'BR', 'IN', 'MX']
    
    start = datetime(2024, 1, 1)
    
    data = []
    for i in range(n_records):
        hours_offset = np.random.randint(0, 7 * 24)
        minutes_offset = np.random.randint(0, 59)
        timestamp = start + timedelta(hours=hours_offset, minutes=minutes_offset)
        
        category = np.random.choice(list(categories.keys()))
        min_price, max_price = categories[category]
        amount = round(np.random.uniform(min_price, max_price), 2)
        
        is_fraudulent = np.random.random() < 0.05
        fraud_type = None
        country = np.random.choice(countries)
        
        if is_fraudulent:
            fraud_type = np.random.choice(['high_amount', 'velocity', 'unusual_location', 'odd_hours'])
            if fraud_type == 'high_amount':
                amount = round(amount * np.random.uniform(5, 15), 2)
            elif fraud_type == 'unusual_location':
                country = np.random.choice(['RU', 'CN', 'NG'])
            elif fraud_type == 'odd_hours':
                timestamp = timestamp.replace(hour=np.random.choice([2, 3, 4, 5]))
        
        customer_id = f"CUST_{np.random.randint(1, 500):05d}"
        transaction_id = f"TXN_{timestamp.strftime('%Y%m%d')}_{i:08d}"
        
        record = {
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'amount': amount,
            'category': category,
            'payment_method': np.random.choice(payment_methods),
            'device_type': np.random.choice(devices),
            'country': country,
            'merchant_id': f"MERCH_{np.random.randint(1, 100):03d}",
            'is_fraudulent': is_fraudulent,
            'fraud_type': fraud_type
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Add data quality issues
    missing_indices = np.random.choice(df.index, 50, replace=False)
    df.loc[missing_indices[:25], 'payment_method'] = None
    df.loc[missing_indices[25:], 'device_type'] = ''
    
    # Add duplicates
    duplicates = df.sample(100)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Add outliers
    outlier_indices = np.random.choice(df.index, 20, replace=False)
    for idx in outlier_indices:
        df.loc[idx, 'amount'] = np.random.choice([0.01, 50000, 99999.99])
    
    return df

# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def ingest_data_task(**context):
    """Task 1: Ingest transaction data from CSV files."""
    logger.info("=" * 60)
    logger.info("TASK 1: DATA INGESTION STARTED")
    logger.info("=" * 60)
    
    ds = context['ds']
    csv_path = f"{CONFIG['raw_data_path']}transactions_{ds}.csv"
    
    # Create directory if it doesn't exist
    os.makedirs(CONFIG['raw_data_path'], exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(csv_path):
        logger.warning(f"File not found: {csv_path}")
        
        # Look for any CSV file in the raw directory
        csv_files = glob.glob(f"{CONFIG['raw_data_path']}*.csv")
        if csv_files:
            csv_path = csv_files[0]
            logger.info(f"Using existing file: {csv_path}")
        else:
            # Generate sample data
            logger.info("Generating sample data...")
            csv_path = f"{CONFIG['raw_data_path']}sample_transactions.csv"
            sample_data = generate_sample_data()
            sample_data.to_csv(csv_path, index=False)
            logger.info(f"Generated sample data: {csv_path}")
    
    # Read CSV file
    logger.info(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert timestamp column
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add metadata
    df['ingestion_timestamp'] = datetime.now()
    df['source_file'] = csv_path
    
    # Log statistics
    logger.info(f"Records ingested: {len(df)}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Unique customers: {df['customer_id'].nunique()}")
    logger.info(f"Unique merchants: {df['merchant_id'].nunique()}")
    logger.info(f"Total amount: ${df['amount'].sum():,.2f}")
    
    # Push to XCom
    context['ti'].xcom_push(key='raw_data', value=df.to_json(date_format='iso'))
    context['ti'].xcom_push(key='record_count', value=len(df))
    
    logger.info("TASK 1: DATA INGESTION COMPLETED")
    return f"Ingested {len(df)} records"


def validate_raw_data_task(**context):
    """Task 2: Validate raw data quality."""
    logger.info("=" * 60)
    logger.info("TASK 2: RAW DATA VALIDATION STARTED")
    logger.info("=" * 60)
    
    ti = context['ti']
    raw_data_json = ti.xcom_pull(task_ids='ingest_data', key='raw_data')
    df = pd.read_json(raw_data_json)
    
    validation_report = {
        'validation_passed': True,
        'checks': {},
        'errors': []
    }
    
    # Check 1: Required columns
    required_columns = [
        'transaction_id', 'customer_id', 'timestamp', 'amount', 'category',
        'payment_method', 'device_type', 'country', 'merchant_id'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_report['checks']['required_columns'] = {
            'status': 'FAIL',
            'details': f"Missing columns: {missing_columns}"
        }
        validation_report['validation_passed'] = False
    else:
        validation_report['checks']['required_columns'] = {
            'status': 'PASS',
            'details': f"All {len(required_columns)} required columns present"
        }
    
    # Check 2: Empty file
    if len(df) == 0:
        validation_report['checks']['empty_file'] = {'status': 'FAIL', 'details': "No records found"}
        validation_report['validation_passed'] = False
    else:
        validation_report['checks']['empty_file'] = {'status': 'PASS', 'details': f"{len(df)} records found"}
    
    # Check 3: Data types
    try:
        pd.to_datetime(df['timestamp'])
        pd.to_numeric(df['amount'])
        validation_report['checks']['data_types'] = {'status': 'PASS', 'details': "All data types valid"}
    except Exception as e:
        validation_report['checks']['data_types'] = {'status': 'FAIL', 'details': str(e)}
        validation_report['validation_passed'] = False
    
    # Log results
    logger.info(f"Validation passed: {validation_report['validation_passed']}")
    for check, result in validation_report['checks'].items():
        logger.info(f"  {check}: {result['status']} - {result['details']}")
    
    ti.xcom_push(key='validation_report', value=json.dumps(validation_report))
    
    logger.info("TASK 2: RAW DATA VALIDATION COMPLETED")
    
    if not validation_report['validation_passed']:
        raise ValueError(f"Data validation failed: {validation_report['errors']}")
    
    return validation_report


def clean_data_task(**context):
    """Task 3: Clean and preprocess the transaction data."""
    logger.info("=" * 60)
    logger.info("TASK 3: DATA CLEANING STARTED")
    logger.info("=" * 60)
    
    ti = context['ti']
    raw_data_json = ti.xcom_pull(task_ids='ingest_data', key='raw_data')
    df = pd.read_json(raw_data_json)
    
    initial_count = len(df)
    cleaning_stats = {
        'initial_records': initial_count,
        'duplicates_removed': 0,
        'missing_values_filled': 0,
        'outliers_flagged': 0,
        'final_records': 0
    }
    
    # Initialize columns
    df['cleaning_action'] = None
    df['is_duplicate'] = False
    df['is_outlier'] = False
    df['had_missing_values'] = False
    
    logger.info(f"Starting cleaning with {initial_count} records")
    
    # Step 1: Remove duplicates
    logger.info("Step 1: Marking duplicates...")
    initial_count = len(df)
    duplicates = df.duplicated(subset=["transaction_id"], keep="first")
    duplicate_count = duplicates.sum()
    df = df[~duplicates].reset_index(drop=True)
    # Duplicates removed
    cleaning_stats['duplicates_removed'] = duplicate_count
    logger.info(f"Found {duplicate_count} duplicate records")
    
    # Step 2: Handle missing values
    logger.info("Step 2: Handling missing values...")
    missing_before = df.isnull().sum().sum()
    
    # Fill missing payment_method
    payment_missing = df['payment_method'].isnull() | (df['payment_method'] == '')
    df.loc[payment_missing, 'payment_method'] = 'Unknown'
    df.loc[payment_missing, 'had_missing_values'] = True
    
    # Fill missing device_type
    device_missing = df['device_type'].isnull() | (df['device_type'] == '')
    df.loc[device_missing, 'device_type'] = 'Unknown'
    df.loc[device_missing, 'had_missing_values'] = True
    
    # Fill missing category
    category_missing = df['category'].isnull() | (df['category'] == '')
    df.loc[category_missing, 'category'] = 'Unknown'
    df.loc[category_missing, 'had_missing_values'] = True
    
    # Fill missing country
    country_missing = df['country'].isnull() | (df['country'] == '')
    df.loc[country_missing, 'country'] = 'Unknown'
    df.loc[country_missing, 'had_missing_values'] = True
    
    cleaning_stats['missing_values_filled'] = missing_before
    logger.info(f"Handled missing values: {missing_before} fields")
    
    # Step 3: Detect outliers
    logger.info("Step 3: Detecting outliers...")
    df['amount_zscore'] = df.groupby('category')['amount'].transform(
        lambda x: np.abs((x - x.mean()) / x.std()) if x.std() > 0 else 0
    )
    
    outlier_threshold = CONFIG['outlier_threshold']
    outliers = df['amount_zscore'] > outlier_threshold
    outlier_count = outliers.sum()
    
    df.loc[outliers, 'is_outlier'] = True
    df.loc[outliers, 'cleaning_action'] = 'flagged_outlier'
    cleaning_stats['outliers_flagged'] = outlier_count
    logger.info(f"Flagged {outlier_count} outliers (Z-score > {outlier_threshold})")
    
    # Step 4: Standardize categorical values
    logger.info("Step 4: Standardizing categorical values...")
    
    payment_mapping = {
        'cc': 'Credit Card', 'credit': 'Credit Card',
        'dc': 'Debit Card', 'debit': 'Debit Card',
        'paypal': 'PayPal', 'crypto': 'Cryptocurrency', 'bitcoin': 'Cryptocurrency'
    }
    df['payment_method'] = df['payment_method'].str.lower().map(payment_mapping).fillna(df['payment_method'])
    
    device_mapping = {
        'phone': 'Mobile', 'cell': 'Mobile',
        'ipad': 'Tablet', 'android tablet': 'Tablet',
        'pc': 'Desktop', 'computer': 'Desktop'
    }
    df['device_type'] = df['device_type'].str.lower().map(device_mapping).fillna(df['device_type'])
    
    # Step 5: Validate amounts
    logger.info("Step 5: Validating amounts...")
    invalid_amounts = df['amount'] <= 0
    if invalid_amounts.any():
        logger.warning(f"Found {invalid_amounts.sum()} invalid amounts")
        df = df[~invalid_amounts]
    
    # Final statistics
    cleaning_stats['final_records'] = len(df)
    df = df.drop(columns=['amount_zscore'], errors='ignore')
    df['cleaning_timestamp'] = datetime.now()
    
    logger.info("=" * 40)
    logger.info("CLEANING STATISTICS:")
    logger.info(f"  Initial records: {cleaning_stats['initial_records']}")
    logger.info(f"  Duplicates marked: {cleaning_stats['duplicates_removed']}")
    logger.info(f"  Missing values filled: {cleaning_stats['missing_values_filled']}")
    logger.info(f"  Outliers flagged: {cleaning_stats['outliers_flagged']}")
    logger.info(f"  Final records: {cleaning_stats['final_records']}")
    logger.info("=" * 40)
    
    ti.xcom_push(key='cleaned_data', value=df.to_json(date_format='iso'))
    
    # Convert numpy types to Python native types for JSON serialization
    cleaning_stats_serializable = {
        'initial_records': int(cleaning_stats['initial_records']),
        'duplicates_removed': int(cleaning_stats['duplicates_removed']),
        'missing_values_filled': int(cleaning_stats['missing_values_filled']),
        'outliers_flagged': int(cleaning_stats['outliers_flagged']),
        'final_records': int(cleaning_stats['final_records'])
    }
    ti.xcom_push(key='cleaning_stats', value=json.dumps(cleaning_stats_serializable))
    
    logger.info("TASK 3: DATA CLEANING COMPLETED")
    return cleaning_stats


def validate_cleaned_data_task(**context):
    """Task 4: Validate cleaned data quality."""
    logger.info("=" * 60)
    logger.info("TASK 4: CLEANED DATA VALIDATION STARTED")
    logger.info("=" * 60)
    
    ti = context['ti']
    cleaned_data_json = ti.xcom_pull(task_ids='clean_data', key='cleaned_data')
    df = pd.read_json(cleaned_data_json)
    
    validation_report = {
        'validation_passed': True,
        'checks': {},
        'warnings': []
    }
    
    # Check 1: No duplicate transaction_ids
    duplicate_ids = df['transaction_id'].duplicated().sum()
    if duplicate_ids > 0:
        validation_report['checks']['unique_transaction_ids'] = {
            'status': 'FAIL', 'details': f"{duplicate_ids} duplicate transaction IDs found"
        }
        validation_report['validation_passed'] = False
    else:
        validation_report['checks']['unique_transaction_ids'] = {
            'status': 'PASS', 'details': 'All transaction IDs are unique'
        }
    
    # Check 2: All required fields populated
    required_fields = ['transaction_id', 'customer_id', 'timestamp', 'amount', 'category']
    empty_fields = []
    for field in required_fields:
        if df[field].isnull().sum() > 0:
            empty_fields.append(f"{field}: {df[field].isnull().sum()}")
    
    if empty_fields:
        validation_report['checks']['required_fields'] = {
            'status': 'FAIL', 'details': f"Empty values found: {empty_fields}"
        }
        validation_report['validation_passed'] = False
    else:
        validation_report['checks']['required_fields'] = {
            'status': 'PASS', 'details': 'All required fields are populated'
        }
    
    logger.info(f"Validation passed: {validation_report['validation_passed']}")
    for check, result in validation_report['checks'].items():
        logger.info(f"  {check}: {result['status']} - {result['details']}")
    
    logger.info("TASK 4: CLEANED DATA VALIDATION COMPLETED")
    
    if not validation_report['validation_passed']:
        raise ValueError(f"Cleaned data validation failed")
    
    return validation_report


def aggregate_data_task(**context):
    """Task 5: Aggregate data for analytics."""
    logger.info("=" * 60)
    logger.info("TASK 5: DATA AGGREGATION STARTED")
    logger.info("=" * 60)
    
    ti = context['ti']
    cleaned_data_json = ti.xcom_pull(task_ids='clean_data', key='cleaned_data')
    df = pd.read_json(cleaned_data_json)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['summary_hour'] = df['timestamp'].dt.floor('H')
    df['summary_date'] = df['timestamp'].dt.date
    
    logger.info(f"Aggregating {len(df)} records...")
    
    # Hourly category summary
    logger.info("Creating hourly category summary...")
    hourly_category = df.groupby(['summary_hour', 'category']).agg({
        'transaction_id': 'count',
        'amount': ['sum', 'mean', 'min', 'max'],
        'customer_id': 'nunique',
        'merchant_id': 'nunique',
        'is_fraudulent': 'sum'
    }).reset_index()
    
    hourly_category.columns = [
        'summary_hour', 'category', 'transaction_count', 'total_amount',
        'avg_amount', 'min_amount', 'max_amount', 'unique_customers',
        'unique_merchants', 'fraud_count'
    ]
    
    fraud_amounts = df[df['is_fraudulent'] == True].groupby(['summary_hour', 'category'])['amount'].sum().reset_index()
    fraud_amounts.columns = ['summary_hour', 'category', 'fraud_amount']
    hourly_category = hourly_category.merge(fraud_amounts, on=['summary_hour', 'category'], how='left')
    hourly_category['fraud_amount'] = hourly_category['fraud_amount'].fillna(0)
    
    logger.info(f"Hourly category summary: {len(hourly_category)} rows")
    
    # Hourly overall summary
    logger.info("Creating hourly overall summary...")
    hourly_overall = df.groupby('summary_hour').agg({
        'transaction_id': 'count',
        'amount': ['sum', 'mean'],
        'customer_id': 'nunique',
        'merchant_id': 'nunique',
        'is_fraudulent': 'sum'
    }).reset_index()
    
    hourly_overall.columns = [
        'summary_hour', 'transaction_count', 'total_amount',
        'avg_amount', 'unique_customers', 'unique_merchants', 'fraud_count'
    ]
    
    hourly_overall['fraud_rate_percent'] = round(
        100.0 * hourly_overall['fraud_count'] / hourly_overall['transaction_count'], 2
    )
    
    logger.info(f"Hourly overall summary: {len(hourly_overall)} rows")
    
    # Daily category summary
    logger.info("Creating daily category summary...")
    daily_category = df.groupby(['summary_date', 'category']).agg({
        'transaction_id': 'count',
        'amount': ['sum', 'mean'],
        'is_fraudulent': 'sum'
    }).reset_index()
    
    daily_category.columns = [
        'summary_date', 'category', 'transaction_count',
        'total_amount', 'avg_amount', 'fraud_count'
    ]
    
    daily_category['fraud_rate_percent'] = round(
        100.0 * daily_category['fraud_count'] / daily_category['transaction_count'], 2
    )
    
    logger.info(f"Daily category summary: {len(daily_category)} rows")
    
    ti.xcom_push(key='hourly_category_summary', value=hourly_category.to_json(date_format='iso'))
    ti.xcom_push(key='hourly_overall_summary', value=hourly_overall.to_json(date_format='iso'))
    ti.xcom_push(key='daily_category_summary', value=daily_category.to_json(date_format='iso'))
    
    logger.info("TASK 5: DATA AGGREGATION COMPLETED")
    
    return {
        'hourly_category_rows': len(hourly_category),
        'hourly_overall_rows': len(hourly_overall),
        'daily_category_rows': len(daily_category)
    }


def engineer_features_task(**context):
    """Task 6: Engineer features for fraud detection ML model."""
    logger.info("=" * 60)
    logger.info("TASK 6: FEATURE ENGINEERING STARTED")
    logger.info("=" * 60)
    
    ti = context['ti']
    cleaned_data_json = ti.xcom_pull(task_ids='clean_data', key='cleaned_data')
    df = pd.read_json(cleaned_data_json)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    logger.info(f"Engineering features for {len(df)} transactions...")
    
    # Time-based features
    logger.info("Creating time-based features...")
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_night_time'] = (df['hour_of_day'] >= 23) | (df['hour_of_day'] <= 5)
    
    # Amount-based features
    logger.info("Creating amount-based features...")
    df['amount_zscore'] = df.groupby('category')['amount'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['amount_percentile'] = df.groupby('category')['amount'].rank(pct=True) * 100
    
    category_avg = df.groupby('category')['amount'].transform('mean')
    df['amount_vs_category_avg'] = df['amount'] / category_avg
    
    customer_avg = df.groupby('customer_id')['amount'].transform('mean')
    df['amount_vs_customer_avg'] = df['amount'] / customer_avg.replace(0, np.nan)
    df['amount_vs_customer_avg'] = df['amount_vs_customer_avg'].fillna(1)
    
    # Customer behavior features (simplified)
    logger.info("Creating customer behavior features...")
    df['customer_txn_count_24h'] = 0
    df['customer_avg_amount_24h'] = 0
    
    window_24h = timedelta(hours=24)
    
    for customer_id in df['customer_id'].unique():
        customer_mask = df['customer_id'] == customer_id
        for idx in df[customer_mask].index:
            current_time = df.loc[idx, 'timestamp']
            mask_24h = (df['timestamp'] < current_time) & (df['timestamp'] >= current_time - window_24h) & customer_mask
            txn_24h = df[mask_24h]
            df.loc[idx, 'customer_txn_count_24h'] = len(txn_24h)
            df.loc[idx, 'customer_avg_amount_24h'] = txn_24h['amount'].mean() if len(txn_24h) > 0 else 0
    
    # Risk indicators
    logger.info("Creating risk indicator features...")
    df['is_high_risk_country'] = df['country'].isin(CONFIG['fraud_high_risk_countries'])
    df['is_high_risk_payment_method'] = df['payment_method'].isin(CONFIG['fraud_high_risk_payment_methods'])
    
    # Composite risk score
    logger.info("Calculating composite risk score...")
    
    def normalize_score(series, inverse=False):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([50] * len(series))
        normalized = (series - min_val) / (max_val - min_val) * 100
        if inverse:
            normalized = 100 - normalized
        return normalized
    
    risk_components = pd.DataFrame(index=df.index)
    risk_components['amount_risk'] = normalize_score(df['amount_zscore'].clip(lower=0))
    risk_components['velocity_risk'] = normalize_score(df['customer_txn_count_24h'])
    risk_components['time_risk'] = df['is_night_time'].astype(int) * 50
    risk_components['country_risk'] = df['is_high_risk_country'].astype(int) * 70
    risk_components['payment_risk'] = df['is_high_risk_payment_method'].astype(int) * 60
    
    weights = {
        'amount_risk': 0.25,
        'velocity_risk': 0.25,
        'time_risk': 0.15,
        'country_risk': 0.15,
        'payment_risk': 0.20
    }
    
    df['risk_score'] = sum(risk_components[component] * weight for component, weight in weights.items())
    df['risk_score'] = df['risk_score'].clip(0, 100).round(2)
    
    df['feature_timestamp'] = datetime.now()
    
    logger.info("Risk score calculated")
    
    # Log statistics
    logger.info("\n" + "=" * 40)
    logger.info("FEATURE STATISTICS:")
    logger.info(f"Risk score: mean={df['risk_score'].mean():.2f}, max={df['risk_score'].max():.2f}")
    logger.info(f"High risk transactions (>70): {(df['risk_score'] > 70).sum()}")
    logger.info("=" * 40)
    
    ti.xcom_push(key='feature_data', value=df.to_json(date_format='iso'))
    
    logger.info("TASK 6: FEATURE ENGINEERING COMPLETED")
    
    return {
        'total_features': 15,
        'records_processed': len(df),
        'avg_risk_score': df['risk_score'].mean(),
        'high_risk_count': (df['risk_score'] > 70).sum()
    }


def export_to_snowflake_task(**context):
    """Task 7: Export processed data to Snowflake tables using RSA key authentication."""
    logger.info("=" * 60)
    logger.info("TASK 7: EXPORT TO SNOWFLAKE STARTED")
    logger.info("=" * 60)
    
    ti = context['ti']
    
    # Get data from XCom
    raw_data_json = ti.xcom_pull(task_ids='ingest_data', key='raw_data')
    cleaned_data_json = ti.xcom_pull(task_ids='clean_data', key='cleaned_data')
    hourly_category_json = ti.xcom_pull(task_ids='aggregate_data', key='hourly_category_summary')
    hourly_overall_json = ti.xcom_pull(task_ids='aggregate_data', key='hourly_overall_summary')
    daily_category_json = ti.xcom_pull(task_ids='aggregate_data', key='daily_category_summary')
    feature_data_json = ti.xcom_pull(task_ids='engineer_features', key='feature_data')
    
    # Convert back to DataFrames
    df_raw = pd.read_json(raw_data_json)
    df_cleaned = pd.read_json(cleaned_data_json)
    df_hourly_category = pd.read_json(hourly_category_json)
    df_hourly_overall = pd.read_json(hourly_overall_json)
    df_daily_category = pd.read_json(daily_category_json)
    df_features = pd.read_json(feature_data_json)
    
    export_stats = {}
    
    try:
        # Load RSA private key
        key_path = os.path.join(PROJECT_ROOT, 'snowflake_key.p8')
        logger.info(f"Loading RSA key from: {key_path}")
        
        with open(key_path, 'rb') as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,  # Add password here if your key is encrypted
                backend=default_backend()
            )
        
        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Connect to Snowflake with RSA key
        import snowflake.connector
        conn = snowflake.connector.connect(
            account='SFEDU02-FEB92475',
            user='BADGER',
            private_key=private_key_bytes,
            database='ECOMMERCE_DW',
            warehouse='ECOMMERCE_LOAD_WH',
            role='TRAINING_ROLE'
        )
        
        cursor = conn.cursor()
        logger.info("Connected to Snowflake successfully with RSA key")
        
        # Helper function to escape SQL strings
        def escape_sql(val):
            if pd.isna(val):
                return 'NULL'
            elif isinstance(val, (bool, np.bool_)):
                return str(val).upper()
            elif isinstance(val, (int, np.integer)):
                return str(val)
            elif isinstance(val, (float, np.floating)):
                return str(val)
            elif isinstance(val, datetime):
                return f"'{val.strftime('%Y-%m-%d %H:%M:%S')}'"
            else:
                return "'" + str(val).replace("'", "''") + "'"
        
        # Export 1: Raw Transactions - USE FULLY QUALIFIED TABLE NAME
        logger.info("Exporting raw transactions to Snowflake...")
        TABLE_RAW = 'ECOMMERCE_DW.RAW_DATA.RAW_TRANSACTIONS'
        cursor.execute(f"TRUNCATE TABLE {TABLE_RAW}")
        
        batch_size = CONFIG['batch_size']
        total_inserted = 0
        
        for i in range(0, len(df_raw), batch_size):
            batch = df_raw.iloc[i:i+batch_size]
            
            values_list = []
            for _, row in batch.iterrows():
                fraud_type_val = escape_sql(row.get('fraud_type'))
                values_parts = [
                    escape_sql(row['transaction_id']),
                    escape_sql(row['customer_id']),
                    escape_sql(row['timestamp']),
                    escape_sql(row['amount']),
                    escape_sql(row['category']),
                    escape_sql(row.get('payment_method', 'Unknown')),
                    escape_sql(row.get('device_type', 'Unknown')),
                    escape_sql(row.get('country', 'Unknown')),
                    escape_sql(row['merchant_id']),
                    escape_sql(row.get('is_fraudulent', False)),
                    fraud_type_val,
                    'CURRENT_TIMESTAMP()',
                    escape_sql(row.get('source_file', 'unknown'))
                ]
                values_list.append('(' + ','.join(values_parts) + ')')
            
            insert_sql = f"""
                INSERT INTO {TABLE_RAW} 
                (transaction_id, customer_id, timestamp, amount, category, payment_method, 
                 device_type, country, merchant_id, is_fraudulent, fraud_type, 
                 ingestion_timestamp, source_file)
                VALUES {','.join(values_list)};
            """
            cursor.execute(insert_sql)
            total_inserted += len(batch)
        
        export_stats['raw_transactions'] = total_inserted
        logger.info(f"Exported {total_inserted} raw transactions to {TABLE_RAW}")
        
        # Export 2: Cleaned Transactions - USE FULLY QUALIFIED TABLE NAME
        logger.info("Exporting cleaned transactions to Snowflake...")
        TABLE_CLEANED = 'ECOMMERCE_DW.CLEANED_DATA.CLEANED_TRANSACTIONS'
        cursor.execute(f"TRUNCATE TABLE {TABLE_CLEANED}")
        
        total_inserted = 0
        for i in range(0, len(df_cleaned), batch_size):
            batch = df_cleaned.iloc[i:i+batch_size]
            
            values_list = []
            for _, row in batch.iterrows():
                cleaning_action_val = escape_sql(row.get('cleaning_action'))
                fraud_type_val = escape_sql(row.get('fraud_type'))
                values_parts = [
                    escape_sql(row['transaction_id']),
                    escape_sql(row['customer_id']),
                    escape_sql(row['timestamp']),
                    escape_sql(row['amount']),
                    escape_sql(row['category']),
                    escape_sql(row['payment_method']),
                    escape_sql(row['device_type']),
                    escape_sql(row['country']),
                    escape_sql(row['merchant_id']),
                    escape_sql(row.get('is_fraudulent', False)),
                    fraud_type_val,
                    escape_sql(row.get('is_duplicate', False)),
                    escape_sql(row.get('is_outlier', False)),
                    escape_sql(row.get('had_missing_values', False)),
                    cleaning_action_val,
                    'CURRENT_TIMESTAMP()',
                    escape_sql(row.get('source_file', 'unknown'))
                ]
                values_list.append('(' + ','.join(values_parts) + ')')
            
            insert_sql = f"""
                INSERT INTO {TABLE_CLEANED} 
                (transaction_id, customer_id, timestamp, amount, category, payment_method, 
                 device_type, country, merchant_id, is_fraudulent, fraud_type, 
                 is_duplicate, is_outlier, had_missing_values, cleaning_action,
                 cleaning_timestamp, source_file)
                VALUES {','.join(values_list)};
            """
            cursor.execute(insert_sql)
            total_inserted += len(batch)
        
        export_stats['cleaned_transactions'] = total_inserted
        logger.info(f"Exported {total_inserted} cleaned transactions to {TABLE_CLEANED}")
        
        # Export 3: Hourly Category Summary - USE FULLY QUALIFIED TABLE NAME
        logger.info("Exporting hourly category summary to Snowflake...")
        TABLE_HOURLY_CAT = 'ECOMMERCE_DW.ANALYTICS.HOURLY_CATEGORY_SUMMARY'
        cursor.execute(f"TRUNCATE TABLE {TABLE_HOURLY_CAT}")
        
        if len(df_hourly_category) > 0:
            values_list = []
            for _, row in df_hourly_category.iterrows():
                values_parts = [
                    escape_sql(row['summary_hour']),
                    escape_sql(row['category']),
                    escape_sql(row['transaction_count']),
                    escape_sql(row['total_amount']),
                    escape_sql(row['avg_amount']),
                    escape_sql(row['min_amount']),
                    escape_sql(row['max_amount']),
                    escape_sql(row['unique_customers']),
                    escape_sql(row['unique_merchants']),
                    escape_sql(row.get('fraud_count', 0)),
                    escape_sql(row.get('fraud_amount', 0)),
                    'CURRENT_TIMESTAMP()'
                ]
                values_list.append('(' + ','.join(values_parts) + ')')
            
            insert_sql = f"""
                INSERT INTO {TABLE_HOURLY_CAT} 
                (summary_hour, category, transaction_count, total_amount, avg_amount, 
                 min_amount, max_amount, unique_customers, unique_merchants, 
                 fraud_count, fraud_amount, created_at)
                VALUES {','.join(values_list)};
            """
            cursor.execute(insert_sql)
        
        export_stats['hourly_category_summary'] = len(df_hourly_category)
        logger.info(f"Exported {len(df_hourly_category)} hourly category records to {TABLE_HOURLY_CAT}")
        
        # Export 4: Hourly Overall Summary - USE FULLY QUALIFIED TABLE NAME
        logger.info("Exporting hourly overall summary to Snowflake...")
        TABLE_HOURLY_OVERALL = 'ECOMMERCE_DW.ANALYTICS.HOURLY_OVERALL_SUMMARY'
        cursor.execute(f"TRUNCATE TABLE {TABLE_HOURLY_OVERALL}")
        
        if len(df_hourly_overall) > 0:
            values_list = []
            for _, row in df_hourly_overall.iterrows():
                values_parts = [
                    escape_sql(row['summary_hour']),
                    escape_sql(row['transaction_count']),
                    escape_sql(row['total_amount']),
                    escape_sql(row['avg_amount']),
                    escape_sql(row['unique_customers']),
                    escape_sql(row['unique_merchants']),
                    escape_sql(row.get('fraud_count', 0)),
                    escape_sql(row.get('fraud_rate_percent', 0)),
                    'CURRENT_TIMESTAMP()'
                ]
                values_list.append('(' + ','.join(values_parts) + ')')
            
            insert_sql = f"""
                INSERT INTO {TABLE_HOURLY_OVERALL} 
                (summary_hour, transaction_count, total_amount, avg_amount, 
                 unique_customers, unique_merchants, fraud_count, fraud_rate_percent, created_at)
                VALUES {','.join(values_list)};
            """
            cursor.execute(insert_sql)
        
        export_stats['hourly_overall_summary'] = len(df_hourly_overall)
        logger.info(f"Exported {len(df_hourly_overall)} hourly overall records to {TABLE_HOURLY_OVERALL}")
        
        # Export 5: Daily Category Summary - USE FULLY QUALIFIED TABLE NAME
        logger.info("Exporting daily category summary to Snowflake...")
        TABLE_DAILY_CAT = 'ECOMMERCE_DW.ANALYTICS.DAILY_CATEGORY_SUMMARY'
        cursor.execute(f"TRUNCATE TABLE {TABLE_DAILY_CAT}")
        
        if len(df_daily_category) > 0:
            values_list = []
            for _, row in df_daily_category.iterrows():
                values_parts = [
                    escape_sql(row['summary_date']),
                    escape_sql(row['category']),
                    escape_sql(row['transaction_count']),
                    escape_sql(row['total_amount']),
                    escape_sql(row['avg_amount']),
                    escape_sql(row.get('fraud_count', 0)),
                    escape_sql(row.get('fraud_rate_percent', 0)),
                    'CURRENT_TIMESTAMP()'
                ]
                values_list.append('(' + ','.join(values_parts) + ')')
            
            insert_sql = f"""
                INSERT INTO {TABLE_DAILY_CAT} 
                (summary_date, category, transaction_count, total_amount, avg_amount, 
                 fraud_count, fraud_rate_percent, created_at)
                VALUES {','.join(values_list)};
            """
            cursor.execute(insert_sql)
        
        export_stats['daily_category_summary'] = len(df_daily_category)
        logger.info(f"Exported {len(df_daily_category)} daily category records to {TABLE_DAILY_CAT}")
        
        # Export 6: ML Features - USE FULLY QUALIFIED TABLE NAME
        logger.info("Exporting ML features to Snowflake...")
        TABLE_FEATURES = 'ECOMMERCE_DW.ML_FEATURES.FRAUD_DETECTION_FEATURES'
        cursor.execute(f"TRUNCATE TABLE {TABLE_FEATURES}")
        
        feature_columns = [
            'transaction_id', 'customer_id', 'timestamp', 'amount', 'category',
            'payment_method', 'device_type', 'country', 'merchant_id',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_night_time',
            'amount_zscore', 'amount_percentile', 'amount_vs_category_avg', 'amount_vs_customer_avg',
            'customer_txn_count_24h', 'customer_avg_amount_24h',
            'is_high_risk_country', 'is_high_risk_payment_method',
            'risk_score', 'is_fraudulent'
        ]
        
        available_columns = [col for col in feature_columns if col in df_features.columns]
        df_features_export = df_features[available_columns].copy()
        
        total_inserted = 0
        for i in range(0, len(df_features_export), batch_size):
            batch = df_features_export.iloc[i:i+batch_size]
            
            values_list = []
            for _, row in batch.iterrows():
                values_parts = []
                for col in available_columns:
                    values_parts.append(escape_sql(row.get(col)))
                
                values_parts.append('CURRENT_TIMESTAMP()')
                values_list.append('(' + ','.join(values_parts) + ')')
            
            columns_sql = ', '.join(available_columns) + ', feature_timestamp'
            insert_sql = f"""
                INSERT INTO {TABLE_FEATURES} 
                ({columns_sql})
                VALUES {','.join(values_list)};
            """
            cursor.execute(insert_sql)
            total_inserted += len(batch)
        
        export_stats['ml_features'] = total_inserted
        logger.info(f"Exported {total_inserted} ML feature records to {TABLE_FEATURES}")
        
        # Commit and close
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("=" * 40)
        logger.info("EXPORT STATISTICS:")
        for table, count in export_stats.items():
            logger.info(f"  {table}: {count} records")
        logger.info("=" * 40)
        
        logger.info("TASK 7: EXPORT TO SNOWFLAKE COMPLETED")
        
        return export_stats
        
    except Exception as e:
        logger.error(f"Error exporting to Snowflake: {str(e)}")
        raise


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id='ecommerce_batch_pipeline',
    default_args=default_args,
    description='E-commerce batch processing pipeline with data cleaning, aggregation, and ML feature engineering',
    schedule_interval='0 */6 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ecommerce', 'batch', 'snowflake', 'fraud_detection'],
    max_active_runs=1,
) as dag:

    # Task definitions
    start_pipeline = BashOperator(
        task_id='start_pipeline',
        bash_command='echo "Starting E-commerce Batch Pipeline at $(date)"',
    )
    
    ingest_data = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data_task,
        provide_context=True,
    )
    
    validate_raw_data = PythonOperator(
        task_id='validate_raw_data',
        python_callable=validate_raw_data_task,
        provide_context=True,
    )
    
    clean_data = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data_task,
        provide_context=True,
    )
    
    validate_cleaned_data = PythonOperator(
        task_id='validate_cleaned_data',
        python_callable=validate_cleaned_data_task,
        provide_context=True,
    )
    
    aggregate_data = PythonOperator(
        task_id='aggregate_data',
        python_callable=aggregate_data_task,
        provide_context=True,
    )
    
    engineer_features = PythonOperator(
        task_id='engineer_features',
        python_callable=engineer_features_task,
        provide_context=True,
    )
    
    export_to_snowflake = PythonOperator(
        task_id='export_to_snowflake',
        python_callable=export_to_snowflake_task,
        provide_context=True,
    )
    
    end_pipeline = BashOperator(
        task_id='end_pipeline',
        bash_command='echo "E-commerce Batch Pipeline completed at $(date)"',
    )
    
    # Task dependencies
    start_pipeline >> ingest_data >> validate_raw_data >> clean_data >> validate_cleaned_data
    validate_cleaned_data >> [aggregate_data, engineer_features]
    [aggregate_data, engineer_features] >> export_to_snowflake >> end_pipeline