#!/usr/bin/env python3
"""
Fraud Detection - Flask API for Real-time Predictions
Provides REST endpoints for single and batch predictions
"""

import os
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global model variables
MODEL = None
PREPROCESSOR = None
LABEL_ENCODERS = {}
FEATURE_NAMES = []
MODEL_VERSION = "1.0.0"
PREDICTION_LOGS = []


class FraudDetectionAPI:
    """Fraud Detection Prediction API Handler."""
    
    def __init__(self, model_dir: str = 'models'):
        """Initialize with model artifacts."""
        self.model_dir = model_dir
        self.load_model()
    
    def load_model(self):
        """Load model and preprocessors."""
        global MODEL, PREPROCESSOR, LABEL_ENCODERS, FEATURE_NAMES
        
        logger.info("Loading model artifacts...")
        
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'fraud_detection_model.pkl')
            MODEL = joblib.load(model_path)
            
            # Load preprocessor
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
            PREPROCESSOR = joblib.load(preprocessor_path)
            
            # Load label encoders
            encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
            LABEL_ENCODERS = joblib.load(encoders_path)
            
            # Load feature names
            feature_path = os.path.join(self.model_dir, 'feature_names.json')
            with open(feature_path, 'r') as f:
                FEATURE_NAMES = json.load(f)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_features(self, data: Dict) -> np.ndarray:
        """
        Preprocess input features for prediction.
        
        Args:
            data: Dictionary with feature values
        
        Returns:
            Preprocessed feature array
        """
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        categorical_cols = ['merchant_category']
        numerical_cols = [
            'transaction_amount', 'transaction_hour', 'day_of_week',
            'card_present', 'international', 'distance_from_home',
            'prev_transaction_count', 'avg_transaction_amount'
        ]
        
        # Encode categorical features
        for col in categorical_cols:
            if col in df.columns:
                le = LABEL_ENCODERS.get(col)
                if le:
                    df[col] = df[col].apply(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                    )
        
        # Ensure all required columns are present
        for col in numerical_cols + categorical_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training
        df = df[numerical_cols + categorical_cols]
        
        # Transform
        X = PREPROCESSOR.transform(df)
        return X
    
    def predict(self, data: Dict) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            data: Input features
        
        Returns:
            Prediction result dictionary
        """
        # Preprocess
        X = self.preprocess_features(data)
        
        # Predict
        prediction = MODEL.predict(X)[0]
        probabilities = MODEL.predict_proba(X)[0]
        
        return {
            'prediction': int(prediction),
            'is_fraud': bool(prediction == 1),
            'fraud_probability': float(probabilities[1]),
            'legitimate_probability': float(probabilities[0]),
            'confidence': float(max(probabilities)),
            'model_version': MODEL_VERSION
        }
    
    def predict_batch(self, data_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            data_list: List of input feature dictionaries
        
        Returns:
            List of prediction results
        """
        results = []
        for data in data_list:
            result = self.predict(data)
            results.append(result)
        return results


# Initialize API handler
api_handler = FraudDetectionAPI()


def log_prediction(request_data: Dict, prediction: Dict, endpoint: str):
    """Log prediction for monitoring."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'request_id': str(uuid.uuid4()),
        'endpoint': endpoint,
        'input_features': {k: v for k, v in request_data.items() if k not in ['transaction_id']},
        'prediction': prediction['prediction'],
        'fraud_probability': prediction['fraud_probability'],
        'confidence': prediction['confidence'],
        'model_version': MODEL_VERSION
    }
    
    PREDICTION_LOGS.append(log_entry)
    
    # Keep only last 10000 logs in memory
    if len(PREDICTION_LOGS) > 10000:
        PREDICTION_LOGS.pop(0)
    
    # Also log to file
    log_file = Path('logs/predictions.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    logger.info(f"Prediction logged: {log_entry['request_id']}")


# ==================== API ENDPOINTS ====================

@app.route('/')
def index():
    """Root endpoint - API info."""
    return jsonify({
        'service': 'Fraud Detection API',
        'version': MODEL_VERSION,
        'status': 'healthy',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'predict_batch': '/predict/batch (POST)',
            'logs': '/logs (GET)',
            'model_info': '/model/info (GET)'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'model_version': MODEL_VERSION,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/model/info')
def model_info():
    """Get model information."""
    info = {
        'model_version': MODEL_VERSION,
        'model_type': type(MODEL).__name__,
        'features': FEATURE_NAMES,
        'feature_count': len(FEATURE_NAMES),
        'loaded_at': datetime.now().isoformat()
    }
    
    if hasattr(MODEL, 'n_features_in_'):
        info['n_features'] = MODEL.n_features_in_
    
    if hasattr(MODEL, 'n_classes_'):
        info['n_classes'] = MODEL.n_classes_
    
    return jsonify(info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single transaction prediction endpoint.
    
    Request Body (JSON):
        {
            "transaction_amount": 150.00,
            "transaction_hour": 14,
            "day_of_week": 2,
            "merchant_category": "online",
            "card_present": 0,
            "international": 1,
            "distance_from_home": 250.5,
            "prev_transaction_count": 5,
            "avg_transaction_amount": 75.00
        }
    
    Response:
        {
            "prediction": 1,
            "is_fraud": true,
            "fraud_probability": 0.89,
            "legitimate_probability": 0.11,
            "confidence": 0.89,
            "model_version": "1.0.0"
        }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            raise BadRequest("No JSON data provided")
        
        # Validate required fields
        required_fields = [
            'transaction_amount', 'transaction_hour', 'day_of_week',
            'merchant_category', 'card_present', 'international',
            'distance_from_home', 'prev_transaction_count', 'avg_transaction_amount'
        ]
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise BadRequest(f"Missing required fields: {missing_fields}")
        
        # Make prediction
        result = api_handler.predict(data)
        
        # Log prediction
        log_prediction(data, result, '/predict')
        
        return jsonify(result)
        
    except BadRequest as e:
        logger.error(f"Bad request: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint.
    
    Request Body (JSON):
        {
            "transactions": [
                {
                    "transaction_amount": 150.00,
                    "transaction_hour": 14,
                    ...
                },
                ...
            ]
        }
    
    Response:
        {
            "predictions": [
                {
                    "prediction": 1,
                    "is_fraud": true,
                    "fraud_probability": 0.89,
                    ...
                },
                ...
            ],
            "total": 2,
            "fraud_count": 1
        }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            raise BadRequest("Request must contain 'transactions' array")
        
        transactions = data['transactions']
        
        if not isinstance(transactions, list):
            raise BadRequest("'transactions' must be an array")
        
        if len(transactions) > 1000:
            raise BadRequest("Maximum 1000 transactions per batch request")
        
        # Make predictions
        results = api_handler.predict_batch(transactions)
        
        # Log predictions
        for txn, result in zip(transactions, results):
            log_prediction(txn, result, '/predict/batch')
        
        # Count fraud predictions
        fraud_count = sum(1 for r in results if r['is_fraud'])
        
        return jsonify({
            'predictions': results,
            'total': len(results),
            'fraud_count': fraud_count,
            'fraud_rate': fraud_count / len(results) if results else 0,
            'model_version': MODEL_VERSION
        })
        
    except BadRequest as e:
        logger.error(f"Bad request: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/logs', methods=['GET'])
def get_logs():
    """
    Get recent prediction logs.
    
    Query Parameters:
        - limit: Maximum number of logs to return (default: 100)
        - fraud_only: Only return fraud predictions (default: false)
    
    Response:
        {
            "logs": [...],
            "total": 100
        }
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        fraud_only = request.args.get('fraud_only', 'false').lower() == 'true'
        
        logs = PREDICTION_LOGS
        
        if fraud_only:
            logs = [l for l in logs if l['prediction'] == 1]
        
        logs = logs[-limit:]
        
        return jsonify({
            'logs': logs,
            'total': len(logs)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/logs/summary', methods=['GET'])
def get_logs_summary():
    """Get summary statistics from prediction logs."""
    try:
        if not PREDICTION_LOGS:
            return jsonify({
                'total_predictions': 0,
                'fraud_predictions': 0,
                'avg_fraud_probability': 0,
                'avg_confidence': 0
            })
        
        total = len(PREDICTION_LOGS)
        fraud_count = sum(1 for l in PREDICTION_LOGS if l['prediction'] == 1)
        avg_fraud_prob = sum(l['fraud_probability'] for l in PREDICTION_LOGS) / total
        avg_confidence = sum(l['confidence'] for l in PREDICTION_LOGS) / total
        
        return jsonify({
            'total_predictions': total,
            'fraud_predictions': fraud_count,
            'fraud_rate': fraud_count / total,
            'avg_fraud_probability': avg_fraud_prob,
            'avg_confidence': avg_confidence,
            'time_range': {
                'first': PREDICTION_LOGS[0]['timestamp'],
                'last': PREDICTION_LOGS[-1]['timestamp']
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Get configuration from environment
    port = int(os.getenv('PORT', 8080))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print("=" * 70)
    print("FRAUD DETECTION API SERVER")
    print("=" * 70)
    print(f"\nStarting server on {host}:{port}")
    print(f"Debug mode: {debug}")
    print(f"\nAvailable endpoints:")
    print(f"  - http://{host}:{port}/")
    print(f"  - http://{host}:{port}/health")
    print(f"  - http://{host}:{port}/predict (POST)")
    print(f"  - http://{host}:{port}/predict/batch (POST)")
    print(f"  - http://{host}:{port}/logs")
    print(f"  - http://{host}:{port}/model/info")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    
    # Run Flask app
    app.run(host=host, port=port, debug=debug)
