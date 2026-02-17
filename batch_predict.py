#!/usr/bin/env python3
"""
Batch Fraud Prediction Script
==============================
Process multiple transactions via API and save predictions.

Usage:
    python batch_predict.py --input test_transactions.csv --output predictions.csv
"""

import pandas as pd
import numpy as np
import requests
import argparse
from datetime import datetime
import json
import sys

# Default API endpoint
DEFAULT_API_URL = "http://localhost:3001/api/ml/predict"
HEADERS = {
    'Content-Type': 'application/json'
}


def predict_single(transaction, api_url):
    """
    Make prediction via API for a single transaction.
    """
    try:
        response = requests.post(
            api_url, 
            headers=HEADERS,
            json=transaction, 
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API error for transaction: {e}")
        return {
            'is_fraud': False,
            'fraud_probability': 0.0,
            'risk_level': 'LOW',
            'error': str(e)
        }


def predict_batch(df, api_url):
    """
    Make predictions on a batch of transactions via API.
    """
    feature_columns = [
        'transaction_amount', 'transaction_hour', 'days_since_last_transaction',
        'merchant_risk_score', 'customer_age_months', 'num_transactions_24h', 'is_international'
    ]
    
    results = df.copy()
    predictions_list = []
    
    print(f"\nProcessing {len(df)} transactions via API...")
    print(f"API: {api_url}")
    print(f"Headers: {HEADERS}")
    
    for idx, row in df.iterrows():
        transaction = {col: row[col] for col in feature_columns if col in row}
        pred = predict_single(transaction, api_url)
        
        predictions_list.append({
            'is_fraud_predicted': pred.get('is_fraud', False),
            'fraud_probability': pred.get('fraud_probability', 0.0),
            'risk_level': pred.get('risk_level', 'LOW')
        })
        
        if (idx + 1) % 10 == 0 or idx == len(df) - 1:
            print(f"  Processed {idx + 1}/{len(df)}...")
    
    pred_df = pd.DataFrame(predictions_list)
    results = pd.concat([results, pred_df], axis=1)
    results['prediction_timestamp'] = datetime.now().isoformat()
    
    return results


def generate_summary(results):
    """Generate summary statistics for the batch."""
    total = len(results)
    fraud_count = results['is_fraud_predicted'].sum()
    fraud_rate = fraud_count / total * 100 if total > 0 else 0
    
    high_risk = (results['risk_level'] == 'HIGH').sum()
    medium_risk = (results['risk_level'] == 'MEDIUM').sum()
    low_risk = (results['risk_level'] == 'LOW').sum()
    
    summary = {
        'total_transactions': int(total),
        'fraud_detected': int(fraud_count),
        'fraud_rate_percent': round(fraud_rate, 2),
        'risk_distribution': {
            'high_risk': int(high_risk),
            'medium_risk': int(medium_risk),
            'low_risk': int(low_risk)
        },
        'avg_fraud_probability': round(results['fraud_probability'].mean(), 4),
        'max_fraud_probability': round(results['fraud_probability'].max(), 4),
        'timestamp': datetime.now().isoformat()
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Batch Fraud Prediction via API')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', default='predictions.csv', help='Output CSV file')
    parser.add_argument('--summary', '-s', help='Output summary JSON file')
    parser.add_argument('--api-url', '-a', default=DEFAULT_API_URL, help='API endpoint URL')
    
    args = parser.parse_args()
    
    api_url = args.api_url
    
    print("="*60)
    print("BATCH FRAUD PREDICTION - API MODE")
    print("="*60)
    print(f"API Endpoint: {api_url}")
    print(f"Content-Type: application/json")
    
    # Check API health
    try:
        health = requests.get(
            "http://localhost:3001/api/health", 
            headers=HEADERS,
            timeout=5
        )
        print(f"Backend Status: {health.json().get('status', 'unknown')}")
    except Exception as e:
        print(f"WARNING: Backend API not reachable: {e}")
        sys.exit(1)
    
    # Load transactions
    print(f"\nLoading transactions from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} transactions")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    # Validate columns
    required_cols = ['transaction_amount', 'transaction_hour', 'days_since_last_transaction',
                     'merchant_risk_score', 'customer_age_months', 'num_transactions_24h', 'is_international']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)
    
    # Make predictions
    results = predict_batch(df, api_url)
    summary = generate_summary(results)
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total Transactions: {summary['total_transactions']}")
    print(f"Fraud Detected: {summary['fraud_detected']}")
    print(f"Fraud Rate: {summary['fraud_rate_percent']}%")
    print(f"\nRisk Distribution:")
    print(f"  HIGH:   {summary['risk_distribution']['high_risk']}")
    print(f"  MEDIUM: {summary['risk_distribution']['medium_risk']}")
    print(f"  LOW:    {summary['risk_distribution']['low_risk']}")
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"\nPredictions saved to: {args.output}")
    
    if args.summary:
        with open(args.summary, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {args.summary}")
    
    print("\n" + "="*60)
    print("BATCH PREDICTION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()