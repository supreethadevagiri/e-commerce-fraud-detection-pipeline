#!/usr/bin/env python3
"""
Fraud Detection - Model Training
Trains a classification model using scikit-learn with Vertex AI integration
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Google Cloud
from google.cloud import aiplatform, storage
from google.cloud.aiplatform.gapic.schema import predict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionTrainer:
    """Trainer class for fraud detection model."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        self.feature_names = None
        self.metrics = {}
        
        # Initialize Vertex AI
        self._init_vertex_ai()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML or use defaults."""
        defaults = {
            'project_id': os.getenv('PROJECT_ID', 'your-project-id'),
            'region': os.getenv('REGION', 'us-central1'),
            'bucket_name': os.getenv('BUCKET_NAME', 'fraud-detection-bucket'),
            'model_display_name': 'fraud-detection-classifier',
            'model_description': 'Binary classifier for fraud detection',
        }
        
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            defaults.update(config)
        
        return defaults
    
    def _init_vertex_ai(self):
        """Initialize Vertex AI SDK."""
        try:
            aiplatform.init(
                project=self.config['project_id'],
                location=self.config['region'],
                staging_bucket=f"gs://{self.config['bucket_name']}"
            )
            logger.info(f"Vertex AI initialized: {self.config['project_id']}")
        except Exception as e:
            logger.warning(f"Could not initialize Vertex AI: {e}")
            logger.info("Continuing with local training only...")
    
    def load_data(self, data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, validation, and test data.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Loading data...")
        
        train_path = os.path.join(data_dir, 'train.csv')
        val_path = os.path.join(data_dir, 'validation.csv')
        test_path = os.path.join(data_dir, 'test.csv')
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Loaded {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test samples")
        
        return train_df, val_df, test_df
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for model training/prediction.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit preprocessors (True for training)
        
        Returns:
            Tuple of (X, y) arrays
        """
        # Define feature columns
        categorical_cols = ['merchant_category']
        numerical_cols = [
            'transaction_amount', 'transaction_hour', 'day_of_week',
            'card_present', 'international', 'distance_from_home',
            'prev_transaction_count', 'avg_transaction_amount'
        ]
        
        self.feature_names = numerical_cols + categorical_cols
        
        # Extract features and target
        X = df.copy()
        y = X.pop('is_fraud').values if 'is_fraud' in X.columns else None
        
        # Drop non-feature columns
        drop_cols = ['transaction_id', 'timestamp']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
        
        # Encode categorical features
        for col in categorical_cols:
            if col in X.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    # Handle unseen categories
                    le = self.label_encoders.get(col)
                    if le:
                        X[col] = X[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        # Build preprocessor if fitting
        if fit and self.preprocessor is None:
            numerical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=-1))
            ])
            
            self.preprocessor = ColumnTransformer([
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
            
            X_processed = self.preprocessor.fit_transform(X)
        else:
            X_processed = self.preprocessor.transform(X)
        
        return X_processed, y
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, 
              model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train the fraud detection model.
        
        Args:
            train_df: Training data
            val_df: Validation data (optional)
            model_type: 'random_forest' or 'gradient_boosting'
        
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {model_type} model...")
        
        # Preprocess data
        X_train, y_train = self.preprocess_data(train_df, fit=True)
        
        # Handle class imbalance with class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
        
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Training metrics
        train_preds = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        
        self.metrics = {
            'model_type': model_type,
            'training_samples': len(X_train),
            'cv_roc_auc_mean': float(cv_scores.mean()),
            'cv_roc_auc_std': float(cv_scores.std()),
            'training_accuracy': float((train_preds == y_train).mean()),
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            self.metrics['feature_importance'] = importance
        
        logger.info(f"Training complete. CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return self.metrics
    
    def save_model(self, output_dir: str = 'models') -> str:
        """
        Save model and preprocessors to disk.
        
        Args:
            output_dir: Directory to save model
        
        Returns:
            Path to saved model
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'fraud_detection_model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save preprocessor
        preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_path)
        
        # Save label encoders
        encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
        joblib.dump(self.label_encoders, encoders_path)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save feature names
        feature_path = os.path.join(output_dir, 'feature_names.json')
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        logger.info(f"Model saved to {output_dir}/")
        
        return output_dir
    
    def upload_to_gcs(self, local_dir: str = 'models') -> str:
        """
        Upload model artifacts to Google Cloud Storage.
        
        Args:
            local_dir: Local directory with model files
        
        Returns:
            GCS path where model was uploaded
        """
        try:
            client = storage.Client()
            bucket = client.bucket(self.config['bucket_name'])
            
            gcs_path = f"models/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            for filename in os.listdir(local_dir):
                local_file = os.path.join(local_dir, filename)
                blob_path = f"{gcs_path}/{filename}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_file)
                logger.info(f"Uploaded {filename} to gs://{self.config['bucket_name']}/{blob_path}")
            
            return f"gs://{self.config['bucket_name']}/{gcs_path}"
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            return None
    
    def upload_to_vertex_ai(self, model_path: str = 'models') -> str:
        """
        Upload model to Vertex AI Model Registry.
        
        Args:
            model_path: Path to model directory
        
        Returns:
            Vertex AI model resource name
        """
        try:
            # Upload model to Vertex AI
            model = aiplatform.Model.upload(
                display_name=self.config['model_display_name'],
                description=self.config['model_description'],
                artifact_uri=f"gs://{self.config['bucket_name']}/models/latest",
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
                serving_container_predict_route="/predict",
                serving_container_health_route="/health",
                labels={"task": "fraud-detection", "framework": "sklearn"}
            )
            
            logger.info(f"Model uploaded to Vertex AI: {model.resource_name}")
            return model.resource_name
        except Exception as e:
            logger.error(f"Failed to upload to Vertex AI: {e}")
            return None


def train_with_vertex_ai_automl(data_path: str = 'data/train.csv'):
    """
    Train model using Vertex AI AutoML (alternative approach).
    
    Args:
        data_path: Path to training data CSV
    """
    logger.info("Starting AutoML training on Vertex AI...")
    
    # Initialize Vertex AI
    project_id = os.getenv('PROJECT_ID', 'your-project-id')
    region = os.getenv('REGION', 'us-central1')
    
    aiplatform.init(project=project_id, location=region)
    
    # Create dataset
    dataset = aiplatform.TabularDataset.create(
        display_name="fraud-detection-dataset",
        gcs_source=[f"gs://{project_id}-fraud-detection/data/train.csv"]
    )
    
    # Run AutoML training job
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="fraud-detection-automl",
        optimization_prediction_type="classification",
        optimization_objective="maximize-au-prc",  # Good for imbalanced data
        column_transformations=[
            {"numeric": {"column_name": "transaction_amount"}},
            {"numeric": {"column_name": "transaction_hour"}},
            {"categorical": {"column_name": "merchant_category"}},
            {"numeric": {"column_name": "distance_from_home"}},
        ]
    )
    
    model = job.run(
        dataset=dataset,
        target_column="is_fraud",
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        budget_milli_node_hours=1000,
        model_display_name="fraud-detection-automl-model",
        disable_early_stopping=False
    )
    
    logger.info(f"AutoML training complete. Model: {model.resource_name}")
    return model


if __name__ == '__main__':
    print("=" * 70)
    print("FRAUD DETECTION - MODEL TRAINING")
    print("=" * 70)
    
    # Initialize trainer
    trainer = FraudDetectionTrainer(config_path='config.yaml')
    
    # Load data
    train_df, val_df, test_df = trainer.load_data(data_dir='data')
    
    # Train model
    metrics = trainer.train(train_df, val_df, model_type='random_forest')
    
    # Save model locally
    model_dir = trainer.save_model(output_dir='models')
    
    # Upload to GCS (optional)
    gcs_path = trainer.upload_to_gcs(local_dir='models')
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTraining Metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved to: {model_dir}/")
    if gcs_path:
        print(f"Model uploaded to: {gcs_path}")
