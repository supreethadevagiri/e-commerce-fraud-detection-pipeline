#!/usr/bin/env python3
"""
Fraud Detection - Model Evaluation
Comprehensive evaluation with accuracy, precision, recall, F1-score, and more
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ML Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_curve,
    matthews_corrcoef, cohen_kappa_score, log_loss
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluator for fraud detection."""
    
    def __init__(self, model_dir: str = 'models'):
        """Initialize evaluator with model artifacts."""
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        self.metrics = {}
        self.predictions = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessors."""
        logger.info("Loading model artifacts...")
        
        # Load model
        model_path = os.path.join(self.model_dir, 'fraud_detection_model.pkl')
        self.model = joblib.load(model_path)
        
        # Load preprocessor
        preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Load label encoders
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        self.label_encoders = joblib.load(encoders_path)
        
        # Load feature names
        feature_path = os.path.join(self.model_dir, 'feature_names.json')
        with open(feature_path, 'r') as f:
            self.feature_names = json.load(f)
        
        logger.info("Model artifacts loaded successfully")
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for evaluation."""
        categorical_cols = ['merchant_category']
        numerical_cols = [
            'transaction_amount', 'transaction_hour', 'day_of_week',
            'card_present', 'international', 'distance_from_home',
            'prev_transaction_count', 'avg_transaction_amount'
        ]
        
        X = df.copy()
        y = X.pop('is_fraud').values if 'is_fraud' in X.columns else None
        
        # Drop non-feature columns
        drop_cols = ['transaction_id', 'timestamp']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
        
        # Encode categorical features
        for col in categorical_cols:
            if col in X.columns:
                le = self.label_encoders.get(col)
                if le:
                    X[col] = X[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        X_processed = self.preprocessor.transform(X)
        return X_processed, y
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Make predictions on data.
        
        Returns:
            Dictionary with predictions, probabilities, and metadata
        """
        X, y = self.preprocess_data(df)
        
        # Get predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        result = {
            'predictions': predictions,
            'probabilities': probabilities,
            'fraud_probability': probabilities[:, 1],
            'transaction_ids': df['transaction_id'].values if 'transaction_id' in df.columns else None,
        }
        
        if y is not None:
            result['true_labels'] = y
        
        return result
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_proba: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
        
        # Advanced metrics
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        metrics['average_precision'] = float(average_precision_score(y_true, y_proba))
        metrics['matthews_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))
        metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
        metrics['log_loss'] = float(log_loss(y_true, y_proba))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Derived metrics
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0
        
        # Business metrics
        metrics['fraud_caught'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0  # Same as recall
        metrics['false_alarm_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0  # Same as FPR
        
        return metrics
    
    def evaluate(self, test_df: pd.DataFrame, output_dir: str = 'evaluation') -> Dict:
        """
        Run complete evaluation on test data.
        
        Args:
            test_df: Test data DataFrame
            output_dir: Directory to save evaluation results
        
        Returns:
            Dictionary of all metrics
        """
        logger.info("Running model evaluation...")
        
        # Make predictions
        result = self.predict(test_df)
        y_true = result['true_labels']
        y_pred = result['predictions']
        y_proba = result['fraud_probability']
        
        # Calculate metrics
        self.metrics = self.calculate_metrics(y_true, y_pred, y_proba)
        self.metrics['total_samples'] = len(y_true)
        self.metrics['fraud_samples'] = int(y_true.sum())
        self.metrics['legitimate_samples'] = int(len(y_true) - y_true.sum())
        self.metrics['evaluation_timestamp'] = datetime.now().isoformat()
        
        # Store predictions
        self.predictions = pd.DataFrame({
            'transaction_id': result['transaction_ids'],
            'true_label': y_true,
            'predicted_label': y_pred,
            'fraud_probability': y_proba
        })
        
        # Save results
        self._save_results(output_dir)
        
        # Generate visualizations
        self._generate_visualizations(y_true, y_pred, y_proba, output_dir)
        
        logger.info("Evaluation complete")
        
        return self.metrics
    
    def _save_results(self, output_dir: str):
        """Save evaluation results to files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save predictions
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        self.predictions.to_csv(predictions_path, index=False)
        
        # Generate classification report
        report = classification_report(
            self.predictions['true_label'],
            self.predictions['predicted_label'],
            target_names=['Legitimate', 'Fraud'],
            output_dict=True
        )
        report_path = os.path.join(output_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}/")
    
    def _generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_proba: np.ndarray, output_dir: str):
        """Generate evaluation visualizations."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        # 2. ROC Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
        plt.close()
        
        # 3. Precision-Recall Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=150)
        plt.close()
        
        # 4. Prediction Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        fraud_proba = y_proba[y_true == 1]
        legit_proba = y_proba[y_true == 0]
        ax.hist(legit_proba, bins=50, alpha=0.5, label='Legitimate', color='green')
        ax.hist(fraud_proba, bins=50, alpha=0.5, label='Fraud', color='red')
        ax.set_xlabel('Fraud Probability')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Probability Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'probability_distribution.png'), dpi=150)
        plt.close()
        
        # 5. Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 6))
            importance = self.model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            ax.bar(range(len(importance)), importance[indices])
            ax.set_xticks(range(len(importance)))
            ax.set_xticklabels([self.feature_names[i] for i in indices], rotation=45, ha='right')
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importance')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}/")
    
    def print_report(self):
        """Print formatted evaluation report."""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION REPORT")
        print("=" * 70)
        
        print("\n📊 DATASET STATISTICS:")
        print(f"  Total samples: {self.metrics['total_samples']:,}")
        print(f"  Fraud samples: {self.metrics['fraud_samples']:,} ({self.metrics['fraud_samples']/self.metrics['total_samples']*100:.2f}%)")
        print(f"  Legitimate samples: {self.metrics['legitimate_samples']:,} ({self.metrics['legitimate_samples']/self.metrics['total_samples']*100:.2f}%)")
        
        print("\n🎯 CLASSIFICATION METRICS:")
        print(f"  Accuracy:  {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {self.metrics['precision']:.4f} ({self.metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {self.metrics['recall']:.4f} ({self.metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:  {self.metrics['f1_score']:.4f}")
        
        print("\n📈 ADVANCED METRICS:")
        print(f"  ROC-AUC:           {self.metrics['roc_auc']:.4f}")
        print(f"  Average Precision: {self.metrics['average_precision']:.4f}")
        print(f"  Matthews Corr Coef: {self.metrics['matthews_corrcoef']:.4f}")
        print(f"  Cohen's Kappa:     {self.metrics['cohen_kappa']:.4f}")
        
        print("\n🔍 CONFUSION MATRIX:")
        cm = self.metrics['confusion_matrix']
        print(f"                 Predicted")
        print(f"                 Legit    Fraud")
        print(f"  Actual Legit   {cm[0][0]:6d}   {cm[0][1]:6d}  (TN={cm[0][0]}, FP={cm[0][1]})")
        print(f"         Fraud   {cm[1][0]:6d}   {cm[1][1]:6d}  (FN={cm[1][0]}, TP={cm[1][1]})")
        
        print("\n💼 BUSINESS METRICS:")
        print(f"  Fraud Detection Rate: {self.metrics['fraud_caught']*100:.2f}% (Recall)")
        print(f"  False Alarm Rate:     {self.metrics['false_alarm_rate']*100:.2f}% (FPR)")
        print(f"  Specificity:          {self.metrics['specificity']*100:.2f}%")
        
        print("\n" + "=" * 70)


def evaluate_thresholds(evaluator: ModelEvaluator, test_df: pd.DataFrame, 
                        thresholds: List[float] = None) -> pd.DataFrame:
    """
    Evaluate model at different probability thresholds.
    
    Args:
        evaluator: ModelEvaluator instance
        test_df: Test data
        thresholds: List of thresholds to evaluate
    
    Returns:
        DataFrame with metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    result = evaluator.predict(test_df)
    y_true = result['true_labels']
    y_proba = result['fraud_probability']
    
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        results.append(metrics)
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("=" * 70)
    print("FRAUD DETECTION - MODEL EVALUATION")
    print("=" * 70)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_dir='models')
    
    # Load test data
    test_df = pd.read_csv('data/test.csv')
    
    # Run evaluation
    metrics = evaluator.evaluate(test_df, output_dir='evaluation')
    
    # Print report
    evaluator.print_report()
    
    # Evaluate different thresholds
    print("\n📊 THRESHOLD ANALYSIS:")
    threshold_df = evaluate_thresholds(evaluator, test_df)
    print(threshold_df.to_string(index=False))
    
    # Save threshold analysis
    threshold_df.to_csv('evaluation/threshold_analysis.csv', index=False)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\nResults saved to:")
    print("  - evaluation/evaluation_metrics.json")
    print("  - evaluation/predictions.csv")
    print("  - evaluation/classification_report.json")
    print("  - evaluation/threshold_analysis.csv")
    print("  - evaluation/*.png (visualizations)")
