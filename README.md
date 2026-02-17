# Fraud Detection ML System

A simple, student-friendly fraud detection system using scikit-learn RandomForest.

## Project Structure

```
.
├── fraud_data.csv              # Training dataset (5000 transactions)
├── fraud_model.pkl             # Trained RandomForest model
├── model_metrics.json          # Model evaluation metrics
├── train_fraud_model.py        # Training script
├── fraud_api.py                # Flask API for predictions
├── batch_predict.py            # Batch prediction script
├── test_transactions.csv       # Sample test data
├── predictions.csv             # Sample prediction output
└── prediction_summary.json     # Sample summary output
```

## Quick Start

### 1. Train the Model

```bash
python train_fraud_model.py
```

**Output:**
- `fraud_model.pkl` - Trained model
- `model_metrics.json` - Evaluation metrics

**Sample Output:**
```
==================================================
MODEL EVALUATION
==================================================

Accuracy:  0.9960
Precision: 0.9423
Recall:    0.9800
F1 Score:  0.9608

Confusion Matrix:
                 Predicted
                 Normal  Fraud
Actual Normal       947      3
       Fraud          1     49

Feature Importance:
  num_transactions_24h          : 0.3398
  merchant_risk_score           : 0.2473
  transaction_hour              : 0.1888
```

### 2. Start the Flask API

```bash
python fraud_api.py
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single transaction prediction |
| `/predict_batch` | POST | Batch predictions |

**Example - Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 500.00,
    "transaction_hour": 3,
    "days_since_last_transaction": 45.5,
    "merchant_risk_score": 85.0,
    "customer_age_months": 2.0,
    "num_transactions_24h": 15,
    "is_international": 1
  }'
```

**Response:**
```json
{
  "is_fraud": true,
  "fraud_probability": 0.95,
  "risk_level": "HIGH",
  "timestamp": "2024-01-15T10:30:00"
}
```

**Example - Batch Prediction:**
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"transaction_amount": 100, "transaction_hour": 14, ...},
      {"transaction_amount": 800, "transaction_hour": 2, ...}
    ]
  }'
```

### 3. Run Batch Predictions

```bash
python batch_predict.py --input test_transactions.csv --output predictions.csv --summary summary.json
```

**Sample Output:**
```
============================================================
PREDICTION SUMMARY
============================================================
Total Transactions: 20
Fraud Detected: 5
Fraud Rate: 25.0%

Risk Distribution:
  HIGH:   5
  MEDIUM: 0
  LOW:    15

Avg Fraud Probability: 0.2462
Max Fraud Probability: 1.0
```

## Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `transaction_amount` | float | Transaction amount in dollars |
| `transaction_hour` | int | Hour of day (0-23) |
| `days_since_last_transaction` | float | Days since customer's last transaction |
| `merchant_risk_score` | float | Merchant risk score (0-100) |
| `customer_age_months` | float | Customer account age in months |
| `num_transactions_24h` | int | Number of transactions in last 24 hours |
| `is_international` | int | 1 if international, 0 otherwise |

## Model Details

**Algorithm:** RandomForestClassifier

**Parameters:**
- `n_estimators`: 100 trees
- `max_depth`: 10 (prevents overfitting)
- `class_weight`: 'balanced' (handles imbalanced fraud data)

**Why RandomForest?**
- Handles imbalanced data well
- No need for feature scaling
- Provides feature importance
- Robust to outliers

## Metrics Explained

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Accuracy** | Overall correctness | Can be misleading with imbalanced data |
| **Precision** | Of predicted frauds, how many are actually fraud | Minimizes false alarms |
| **Recall** | Of actual frauds, how many did we catch | Minimizes missed fraud |
| **F1 Score** | Balance between precision and recall | Overall performance |

## Requirements

```
pip install scikit-learn pandas numpy flask joblib
```

## File Descriptions

### train_fraud_model.py
- Loads data from CSV
- Splits into train/test sets
- Trains RandomForest model
- Evaluates with multiple metrics
- Saves model to disk

### fraud_api.py (85 lines)
- Flask REST API
- Single and batch prediction endpoints
- Health check endpoint
- Input validation
- Risk level classification

### batch_predict.py
- Command-line batch processing
- Reads transactions from CSV
- Outputs predictions with probabilities
- Generates summary statistics

## Risk Levels

| Level | Probability Range | Action |
|-------|-------------------|--------|
| LOW | 0.0 - 0.3 | Approve transaction |
| MEDIUM | 0.3 - 0.7 | Review required |
| HIGH | 0.7 - 1.0 | Block transaction |

## Example Usage in Python

```python
import joblib
import numpy as np

# Load model
model = joblib.load('fraud_model.pkl')

# Prepare features
features = [
    500.0,   # transaction_amount
    3,       # transaction_hour
    45.5,    # days_since_last_transaction
    85.0,    # merchant_risk_score
    2.0,     # customer_age_months
    15,      # num_transactions_24h
    1        # is_international
]

# Predict
prediction = model.predict([features])
probability = model.predict_proba([features])[0][1]

print(f"Fraud: {bool(prediction[0])}")
print(f"Probability: {probability:.4f}")
```

## License

Educational use only.
