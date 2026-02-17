#!/usr/bin/env python3
"""
Fraud Detection API - Flask Application
========================================
Simple REST API for fraud prediction.

Usage:
    python fraud_api.py
    
API Endpoints:
    POST /predict      - Single transaction prediction
    POST /predict_batch - Batch predictions
    GET  /health       - Health check

Example:
    curl -X POST http://localhost:4500/predict \
         -H "Content-Type: application/json" \
         -d '{"transaction_amount": 500, "transaction_hour": 3, ...}'
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model at startup
MODEL_PATH = 'fraud_model.pkl'
model = None
feature_names = [
    'transaction_amount', 'transaction_hour', 'days_since_last_transaction',
    'merchant_risk_score', 'customer_age_months', 'num_transactions_24h', 'is_international'
]


def load_model():
    """Load the trained model."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def validate_input(data):
    """Validate input data has all required fields."""
    missing = [f for f in feature_names if f not in data]
    if missing:
        return False, f"Missing fields: {missing}"
    return True, None


def make_prediction(features):
    """Make prediction using the model."""
    X = np.array([features])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    return int(prediction), float(probability[1])  # Return fraud probability


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Single transaction prediction endpoint."""
    try:
        data = request.get_json()
        
        # Validate input
        valid, error = validate_input(data)
        if not valid:
            return jsonify({'error': error}), 400
        
        # Extract features in correct order
        features = [data[f] for f in feature_names]
        
        # Make prediction
        prediction, fraud_probability = make_prediction(features)
        
        # Log prediction
        logger.info(f"Prediction: fraud={prediction}, prob={fraud_probability:.4f}")
        
        return jsonify({
            'is_fraud': bool(prediction),
            'fraud_probability': round(fraud_probability, 4),
            'risk_level': 'HIGH' if fraud_probability > 0.7 else 'MEDIUM' if fraud_probability > 0.3 else 'LOW',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint."""
    try:
        data = request.get_json()
        transactions = data.get('transactions', [])
        
        if not transactions:
            return jsonify({'error': 'No transactions provided'}), 400
        
        results = []
        for txn in transactions:
            valid, error = validate_input(txn)
            if not valid:
                results.append({'error': error})
                continue
            
            features = [txn[f] for f in feature_names]
            prediction, fraud_probability = make_prediction(features)
            
            results.append({
                'is_fraud': bool(prediction),
                'fraud_probability': round(fraud_probability, 4),
                'risk_level': 'HIGH' if fraud_probability > 0.7 else 'MEDIUM' if fraud_probability > 0.3 else 'LOW'
            })
        
        fraud_count = sum(1 for r in results if r.get('is_fraud', False))
        
        return jsonify({
            'total_transactions': len(transactions),
            'fraud_detected': fraud_count,
            'predictions': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    print("="*50)
    print("FRAUD DETECTION API")
    print("="*50)
    print("Endpoints:")
    print("  GET  /health       - Health check")
    print("  POST /predict      - Single prediction")
    print("  POST /predict_batch - Batch predictions")
    print("="*50)
    print("Server running on http://localhost:4500")
    app.run(host='0.0.0.0', port=4500, debug=False)
