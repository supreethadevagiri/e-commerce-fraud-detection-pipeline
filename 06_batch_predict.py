#!/usr/bin/env python3
"""
Fraud Detection - Batch Prediction Script
Run batch predictions on new data and log results
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

import pandas as pd
import numpy as np
import joblib
import requests
from tqdm import tqdm

# Google Cloud
from google.cloud import aiplatform, storage, bigquery
from google.cloud.aiplatform.gapic.schema import predict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchPredictor:
    """Batch prediction handler for fraud detection."""
    
    def __init__(self, model_dir: str = 'models', api_url: str = None):
        """
        Initialize batch predictor.
        
        Args:
            model_dir: Directory containing model artifacts
            api_url: URL of prediction API (if using API mode)
        """
        self.model_dir = model_dir
        self.api_url = api_url
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        
        if api_url is None:
            self._load_local_model()
    
    def _load_local_model(self):
        """Load model for local predictions."""
        logger.info("Loading local model...")
        
        model_path = os.path.join(self.model_dir, 'fraud_detection_model.pkl')
        self.model = joblib.load(model_path)
        
        preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
        self.preprocessor = joblib.load(preprocessor_path)
        
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        self.label_encoders = joblib.load(encoders_path)
        
        logger.info("Local model loaded successfully")
    
    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess data for prediction."""
        categorical_cols = ['merchant_category']
        numerical_cols = [
            'transaction_amount', 'transaction_hour', 'day_of_week',
            'card_present', 'international', 'distance_from_home',
            'prev_transaction_count', 'avg_transaction_amount'
        ]
        
        X = df.copy()
        
        # Drop non-feature columns
        drop_cols = ['transaction_id', 'timestamp']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
        
        # Encode categorical features
        for col in categorical_cols:
            if col in X.columns:
                le = self.label_encoders.get(col)
                if le:
                    X[col] = X[col].apply(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                    )
        
        # Ensure all columns present
        for col in numerical_cols + categorical_cols:
            if col not in X.columns:
                X[col] = 0
        
        X = X[numerical_cols + categorical_cols]
        return self.preprocessor.transform(X)
    
    def predict_local(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run predictions using local model.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Running local predictions on {len(df)} samples...")
        
        X = self.preprocess_data(df)
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = pd.DataFrame({
            'transaction_id': df['transaction_id'].values,
            'predicted_label': predictions,
            'is_fraud': predictions == 1,
            'fraud_probability': probabilities[:, 1],
            'legitimate_probability': probabilities[:, 0],
            'prediction_timestamp': datetime.now().isoformat()
        })
        
        return results
    
    def predict_api(self, df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        """
        Run predictions using API.
        
        Args:
            df: Input DataFrame
            batch_size: Number of records per API call
        
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Running API predictions on {len(df)} samples...")
        
        all_results = []
        
        # Process in batches
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_df = df.iloc[i:i+batch_size]
            
            # Convert to API format
            transactions = []
            for _, row in batch_df.iterrows():
                txn = {
                    'transaction_amount': float(row['transaction_amount']),
                    'transaction_hour': int(row['transaction_hour']),
                    'day_of_week': int(row['day_of_week']),
                    'merchant_category': str(row['merchant_category']),
                    'card_present': int(row['card_present']),
                    'international': int(row['international']),
                    'distance_from_home': float(row['distance_from_home']),
                    'prev_transaction_count': int(row['prev_transaction_count']),
                    'avg_transaction_amount': float(row['avg_transaction_amount'])
                }
                transactions.append(txn)
            
            # Make API call
            try:
                response = requests.post(
                    f"{self.api_url}/predict/batch",
                    json={'transactions': transactions},
                    timeout=30
                )
                response.raise_for_status()
                
                batch_results = response.json()
                
                for j, pred in enumerate(batch_results['predictions']):
                    all_results.append({
                        'transaction_id': batch_df.iloc[j]['transaction_id'],
                        'predicted_label': pred['prediction'],
                        'is_fraud': pred['is_fraud'],
                        'fraud_probability': pred['fraud_probability'],
                        'legitimate_probability': pred['legitimate_probability'],
                        'prediction_timestamp': datetime.now().isoformat()
                    })
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"API call failed for batch {i}: {e}")
                # Add empty predictions for failed batch
                for j in range(len(batch_df)):
                    all_results.append({
                        'transaction_id': batch_df.iloc[j]['transaction_id'],
                        'predicted_label': None,
                        'is_fraud': None,
                        'fraud_probability': None,
                        'legitimate_probability': None,
                        'prediction_timestamp': datetime.now().isoformat(),
                        'error': str(e)
                    })
        
        return pd.DataFrame(all_results)
    
    def predict(self, df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        """
        Run predictions (local or API based on configuration).
        
        Args:
            df: Input DataFrame
            batch_size: Batch size for API mode
        
        Returns:
            DataFrame with predictions
        """
        if self.api_url:
            return self.predict_api(df, batch_size)
        else:
            return self.predict_local(df)
    
    def evaluate_predictions(self, predictions_df: pd.DataFrame, 
                            ground_truth_df: pd.DataFrame = None) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions_df: DataFrame with predictions
            ground_truth_df: DataFrame with true labels (optional)
        
        Returns:
            Evaluation metrics dictionary
        """
        if ground_truth_df is None:
            logger.info("No ground truth provided, skipping evaluation")
            return None
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        # Merge predictions with ground truth
        merged = predictions_df.merge(
            ground_truth_df[['transaction_id', 'is_fraud']],
            on='transaction_id',
            suffixes=('_pred', '_true')
        )
        
        y_true = merged['is_fraud_true'].values
        y_pred = merged['predicted_label'].values
        y_proba = merged['fraud_probability'].values
        
        metrics = {
            'total_samples': len(merged),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_proba)),
        }
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        return metrics
    
    def save_results(self, predictions_df: pd.DataFrame, 
                    output_dir: str = 'predictions',
                    evaluation_metrics: Dict = None):
        """
        Save prediction results.
        
        Args:
            predictions_df: DataFrame with predictions
            output_dir: Directory to save results
            evaluation_metrics: Optional evaluation metrics
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save predictions
        predictions_path = os.path.join(output_dir, f'predictions_{timestamp}.csv')
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")
        
        # Save summary statistics
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions_df),
            'fraud_predictions': int(predictions_df['is_fraud'].sum()),
            'fraud_rate': float(predictions_df['is_fraud'].mean()),
            'avg_fraud_probability': float(predictions_df['fraud_probability'].mean()),
            'high_risk_count': int((predictions_df['fraud_probability'] > 0.7).sum()),
            'medium_risk_count': int(
                ((predictions_df['fraud_probability'] > 0.3) & 
                 (predictions_df['fraud_probability'] <= 0.7)).sum()
            ),
            'low_risk_count': int((predictions_df['fraud_probability'] <= 0.3).sum())
        }
        
        if evaluation_metrics:
            summary['evaluation_metrics'] = evaluation_metrics
        
        summary_path = os.path.join(output_dir, f'summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
        
        # Save high-risk transactions separately
        high_risk = predictions_df[predictions_df['fraud_probability'] > 0.7]
        if len(high_risk) > 0:
            high_risk_path = os.path.join(output_dir, f'high_risk_{timestamp}.csv')
            high_risk.to_csv(high_risk_path, index=False)
            logger.info(f"High-risk transactions saved to {high_risk_path}")
        
        return summary


class VertexAIBatchPredictor:
    """Batch prediction using Vertex AI endpoints."""
    
    def __init__(self, project_id: str, region: str, endpoint_id: str):
        """
        Initialize Vertex AI batch predictor.
        
        Args:
            project_id: GCP project ID
            region: GCP region
            endpoint_id: Vertex AI endpoint ID
        """
        self.project_id = project_id
        self.region = region
        self.endpoint_id = endpoint_id
        
        aiplatform.init(project=project_id, location=region)
        self.endpoint = aiplatform.Endpoint(endpoint_id)
    
    def predict(self, instances: List[Dict]) -> List[Dict]:
        """
        Run predictions using Vertex AI endpoint.
        
        Args:
            instances: List of feature dictionaries
        
        Returns:
            List of prediction results
        """
        # Format instances for Vertex AI
        formatted_instances = []
        for instance in instances:
            formatted_instances.append([
                instance['transaction_amount'],
                instance['transaction_hour'],
                instance['day_of_week'],
                instance['merchant_category'],
                instance['card_present'],
                instance['international'],
                instance['distance_from_home'],
                instance['prev_transaction_count'],
                instance['avg_transaction_amount']
            ])
        
        # Make prediction
        response = self.endpoint.predict(instances=formatted_instances)
        
        # Parse response
        results = []
        for pred in response.predictions:
            results.append({
                'prediction': int(pred),
                'is_fraud': pred == 1
            })
        
        return results


def run_batch_prediction(input_path: str, output_dir: str = 'predictions',
                         model_dir: str = 'models', api_url: str = None,
                         ground_truth_path: str = None):
    """
    Run complete batch prediction workflow.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save predictions
        model_dir: Directory containing model artifacts
        api_url: API URL (optional)
        ground_truth_path: Path to ground truth CSV (optional)
    """
    print("=" * 70)
    print("FRAUD DETECTION - BATCH PREDICTION")
    print("=" * 70)
    
    # Load input data
    logger.info(f"Loading input data from {input_path}...")
    input_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(input_df)} records")
    
    # Initialize predictor
    predictor = BatchPredictor(model_dir=model_dir, api_url=api_url)
    
    # Run predictions
    predictions_df = predictor.predict(input_df)
    
    # Load ground truth if provided
    ground_truth_df = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        logger.info(f"Loading ground truth from {ground_truth_path}...")
        ground_truth_df = pd.read_csv(ground_truth_path)
    
    # Evaluate if ground truth available
    evaluation_metrics = None
    if ground_truth_df is not None:
        evaluation_metrics = predictor.evaluate_predictions(predictions_df, ground_truth_df)
        
        if evaluation_metrics:
            print("\n📊 EVALUATION METRICS:")
            print(f"  Accuracy:  {evaluation_metrics['accuracy']:.4f}")
            print(f"  Precision: {evaluation_metrics['precision']:.4f}")
            print(f"  Recall:    {evaluation_metrics['recall']:.4f}")
            print(f"  F1-Score:  {evaluation_metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {evaluation_metrics['roc_auc']:.4f}")
    
    # Save results
    summary = predictor.save_results(predictions_df, output_dir, evaluation_metrics)
    
    # Print summary
    print("\n📋 PREDICTION SUMMARY:")
    print(f"  Total predictions: {summary['total_predictions']:,}")
    print(f"  Fraud detected: {summary['fraud_predictions']:,} ({summary['fraud_rate']*100:.2f}%)")
    print(f"  High risk (>70%): {summary['high_risk_count']:,}")
    print(f"  Medium risk (30-70%): {summary['medium_risk_count']:,}")
    print(f"  Low risk (<30%): {summary['low_risk_count']:,}")
    print(f"  Avg fraud probability: {summary['avg_fraud_probability']:.4f}")
    
    print("\n" + "=" * 70)
    print("BATCH PREDICTION COMPLETE")
    print("=" * 70)
    
    return predictions_df, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fraud Detection Batch Prediction')
    parser.add_argument('--input', '-i', default='data/batch_input.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', '-o', default='predictions',
                       help='Output directory')
    parser.add_argument('--model-dir', '-m', default='models',
                       help='Model directory')
    parser.add_argument('--api-url', '-a', default=None,
                       help='API URL (if using API mode)')
    parser.add_argument('--ground-truth', '-g', default='data/batch_input_ground_truth.csv',
                       help='Ground truth CSV for evaluation')
    
    args = parser.parse_args()
    
    run_batch_prediction(
        input_path=args.input,
        output_dir=args.output,
        model_dir=args.model_dir,
        api_url=args.api_url,
        ground_truth_path=args.ground_truth
    )
