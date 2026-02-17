#!/usr/bin/env python3
"""
Fraud Detection - Sample Data Generator
Generates synthetic transaction data for training and testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json
from pathlib import Path

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def generate_transaction_features(n_samples=10000, fraud_rate=0.05):
    """
    Generate synthetic transaction features with fraud labels.
    
    Args:
        n_samples: Number of transactions to generate
        fraud_rate: Percentage of fraudulent transactions
    
    Returns:
        DataFrame with transaction features and fraud labels
    """
    print(f"Generating {n_samples} synthetic transactions (fraud rate: {fraud_rate*100}%)")
    
    n_fraud = int(n_samples * fraud_rate)
    n_legitimate = n_samples - n_fraud
    
    # Generate legitimate transactions
    legitimate = _generate_legitimate_transactions(n_legitimate)
    legitimate['is_fraud'] = 0
    
    # Generate fraudulent transactions
    fraud = _generate_fraudulent_transactions(n_fraud)
    fraud['is_fraud'] = 1
    
    # Combine and shuffle
    df = pd.concat([legitimate, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Add transaction ID and timestamp
    df['transaction_id'] = [f'TXN_{i:08d}' for i in range(len(df))]
    df['timestamp'] = pd.date_range(
        start='2023-01-01', 
        periods=len(df), 
        freq='5min'
    )
    
    # Reorder columns
    columns = ['transaction_id', 'timestamp'] + [c for c in df.columns 
                                                  if c not in ['transaction_id', 'timestamp', 'is_fraud']] + ['is_fraud']
    df = df[columns]
    
    return df


def _generate_legitimate_transactions(n):
    """Generate features for legitimate transactions."""
    data = {
        'transaction_amount': np.random.lognormal(4, 1, n),  # ~$50 average
        'transaction_hour': np.random.choice(range(6, 23), n),  # Business hours
        'day_of_week': np.random.choice(range(5), n, p=[0.20, 0.20, 0.20, 0.20, 0.20]),
        'merchant_category': np.random.choice(
            ['grocery', 'gas', 'restaurant', 'retail', 'online', 'travel'],
            n,
            p=[0.25, 0.15, 0.20, 0.20, 0.15, 0.05]
        ),
        'card_present': np.random.choice([1, 0], n, p=[0.85, 0.15]),
        'international': np.random.choice([0, 1], n, p=[0.95, 0.05]),
        'distance_from_home': np.random.exponential(5, n),  # Usually close to home
        'prev_transaction_count': np.random.poisson(20, n),
        'avg_transaction_amount': np.random.lognormal(4, 0.8, n),
    }
    return pd.DataFrame(data)


def _generate_fraudulent_transactions(n):
    """Generate features for fraudulent transactions."""
    data = {
        # Higher amounts for fraud
        'transaction_amount': np.random.lognormal(5.5, 1.2, n),  # ~$250 average
        # Odd hours for fraud
        'transaction_hour': np.random.choice(range(24), n),
        # Weekend fraud more common
        'day_of_week': np.random.choice(range(7), n, p=[0.12, 0.12, 0.12, 0.12, 0.12, 0.20, 0.20]),
        # High-risk categories
        'merchant_category': np.random.choice(
            ['grocery', 'gas', 'restaurant', 'retail', 'online', 'travel'],
            n,
            p=[0.05, 0.05, 0.10, 0.15, 0.45, 0.20]
        ),
        # Card-not-present more common in fraud
        'card_present': np.random.choice([1, 0], n, p=[0.20, 0.80]),
        # International more common
        'international': np.random.choice([0, 1], n, p=[0.70, 0.30]),
        # Far from home
        'distance_from_home': np.random.exponential(200, n),
        # Unusual patterns
        'prev_transaction_count': np.random.poisson(2, n),
        # Amount differs from average
        'avg_transaction_amount': np.random.lognormal(4, 0.8, n),
    }
    return pd.DataFrame(data)


def save_data_splits(df, output_dir='data', train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train/validation/test sets and save to CSV.
    
    Args:
        df: DataFrame with all data
        output_dir: Directory to save files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate split sizes
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # Split data
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    # Save to CSV
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'validation.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Save full dataset for batch predictions
    full_path = os.path.join(output_dir, 'full_dataset.csv')
    df.to_csv(full_path, index=False)
    
    # Save metadata
    metadata = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'validation_samples': len(val_df),
        'test_samples': len(test_df),
        'fraud_rate': float(df['is_fraud'].mean()),
        'features': [c for c in df.columns if c not in ['transaction_id', 'timestamp', 'is_fraud']],
        'target': 'is_fraud',
        'generated_at': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nData saved to {output_dir}/")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/n*100:.1f}%)")
    print(f"  Validation: {len(val_df):,} samples ({len(val_df)/n*100:.1f}%)")
    print(f"  Test: {len(test_df):,} samples ({len(test_df)/n*100:.1f}%)")
    print(f"\nFraud distribution:")
    print(f"  Overall fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"  Train fraud rate: {train_df['is_fraud'].mean()*100:.2f}%")
    print(f"  Validation fraud rate: {val_df['is_fraud'].mean()*100:.2f}%")
    print(f"  Test fraud rate: {test_df['is_fraud'].mean()*100:.2f}%")
    
    return train_df, val_df, test_df


def generate_batch_input_data(n_samples=1000, output_path='data/batch_input.csv'):
    """
    Generate new data for batch predictions (no labels).
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save batch input
    """
    print(f"\nGenerating {n_samples} samples for batch prediction...")
    
    # Mix of legitimate and fraud (unknown to model)
    n_fraud = int(n_samples * 0.08)  # Slightly higher fraud rate
    n_legit = n_samples - n_fraud
    
    legit = _generate_legitimate_transactions(n_legit)
    fraud = _generate_fraudulent_transactions(n_fraud)
    
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Add transaction ID and timestamp
    df['transaction_id'] = [f'BATCH_{i:08d}' for i in range(len(df))]
    df['timestamp'] = pd.date_range(
        start='2024-01-01', 
        periods=len(df), 
        freq='10min'
    )
    
    # Save with labels (for evaluation) and without (for prediction)
    df_with_labels = df.copy()
    df_with_labels['is_fraud'] = [0] * n_legit + [1] * n_fraud
    
    # Shuffle labels
    df_with_labels = df_with_labels.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Save without labels for prediction input
    df_input = df_with_labels.drop(columns=['is_fraud'])
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_input.to_csv(output_path, index=False)
    
    # Save with labels for ground truth evaluation
    truth_path = output_path.replace('.csv', '_ground_truth.csv')
    df_with_labels.to_csv(truth_path, index=False)
    
    print(f"  Batch input saved: {output_path}")
    print(f"  Ground truth saved: {truth_path}")
    
    return df_input, df_with_labels


if __name__ == '__main__':
    print("=" * 60)
    print("FRAUD DETECTION - SAMPLE DATA GENERATOR")
    print("=" * 60)
    
    # Generate main dataset
    df = generate_transaction_features(n_samples=50000, fraud_rate=0.05)
    
    # Save data splits
    train_df, val_df, test_df = save_data_splits(df, output_dir='data')
    
    # Generate batch prediction input
    batch_input, batch_truth = generate_batch_input_data(
        n_samples=1000, 
        output_path='data/batch_input.csv'
    )
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print("\nSample of training data:")
    print(train_df.head(10).to_string())
