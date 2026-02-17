#!/usr/bin/env python3
"""
Fraud Detection Model Training Script
======================================
Simple RandomForest classifier for fraud detection.


Usage:
    python train_fraud_model.py

Output:
    - fraud_model.pkl (trained model)
    - Training metrics printed to console
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import json


def load_data(filepath):
    """Load the fraud dataset from CSV."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records")
    return df


def prepare_data(df):
    """
    Prepare features and target for training.
    
    Features used:
    - transaction_amount: Amount of the transaction
    - transaction_hour: Hour of day (0-23)
    - days_since_last_transaction: Days since customer's last transaction
    - merchant_risk_score: Risk score of the merchant (0-100)
    - customer_age_months: How long customer has been with us
    - num_transactions_24h: Number of transactions in last 24 hours
    - is_international: Whether transaction is international (0/1)
    """
    feature_columns = [
        'transaction_amount',
        'transaction_hour',
        'days_since_last_transaction',
        'merchant_risk_score',
        'customer_age_months',
        'num_transactions_24h',
        'is_international'
    ]
    
    X = df[feature_columns]
    y = df['is_fraud']
    
    return X, y, feature_columns


def train_model(X_train, y_train):
    """
    Train a RandomForest classifier.
    
    Why RandomForest?
    - Handles imbalanced data well
    - No need for feature scaling
    - Provides feature importance
    - Robust to outliers
    """
    print("\nTraining RandomForest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Prevent overfitting
        min_samples_split=10,  # Minimum samples to split a node
        min_samples_leaf=5,    # Minimum samples in leaf node
        class_weight='balanced',  # Handle imbalanced fraud data
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate model performance with multiple metrics.
    
    Key Metrics for Fraud Detection:
    - Accuracy: Overall correctness (can be misleading with imbalanced data)
    - Precision: Of predicted frauds, how many are actually fraud
    - Recall: Of actual frauds, how many did we catch
    - F1 Score: Balance between precision and recall
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Print metrics
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"                 Normal  Fraud")
    print(f"Actual Normal    {cm[0,0]:6d}  {cm[0,1]:5d}")
    print(f"       Fraud     {cm[1,0]:6d}  {cm[1,1]:5d}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # Feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    return metrics


def save_model(model, filepath):
    """Save the trained model to disk."""
    joblib.dump(model, filepath)
    print(f"\nModel saved to: {filepath}")


def main():
    """Main training pipeline."""
    print("="*50)
    print("FRAUD DETECTION MODEL TRAINING")
    print("="*50)
    
    # 1. Load data
    df = load_data('fraud_data.csv')
    
    # 2. Prepare features and target
    X, y, feature_names = prepare_data(df)
    
    # 3. Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Fraud rate in training: {y_train.mean()*100:.2f}%")
    
    # 4. Train model
    model = train_model(X_train, y_train)
    
    # 5. Evaluate model
    metrics = evaluate_model(model, X_test, y_test, feature_names)
    
    # 6. Save model
    save_model(model, 'fraud_model.pkl')
    
    # 7. Save metrics
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to: model_metrics.json")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    main()
