# Google Vertex AI Fraud Detection - Setup Instructions

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [GCP Project Setup](#gcp-project-setup)
3. [Enable APIs](#enable-apis)
4. [Service Account Setup](#service-account-setup)
5. [Local Environment Setup](#local-environment-setup)
6. [Vertex AI Workbench (Optional)](#vertex-ai-workbench-optional)

---

## Prerequisites

- Google Cloud Platform (GCP) account
- Billing enabled on your GCP project
- Python 3.8+ installed locally
- gcloud CLI installed

---

## GCP Project Setup

### Step 1: Create a New Project (or use existing)

```bash
# Set your project ID
export PROJECT_ID="your-fraud-detection-project"
export REGION="us-central1"

# Create new project (optional)
gcloud projects create $PROJECT_ID --name="Fraud Detection ML"

# Set the project
gcloud config set project $PROJECT_ID
```

### Step 2: Enable Billing

```bash
# Link billing account (replace with your billing account ID)
gcloud billing projects link $PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID
```

---

## Enable APIs

### Step 3: Enable Required Google Cloud APIs

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Cloud Storage API
gcloud services enable storage.googleapis.com

# Enable BigQuery API (for data storage)
gcloud services enable bigquery.googleapis.com

# Enable Cloud Logging API
gcloud services enable logging.googleapis.com

# Enable Cloud Monitoring API
gcloud services enable monitoring.googleapis.com
```

Verify APIs are enabled:
```bash
gcloud services list --enabled
```

---

## Service Account Setup

### Step 4: Create Service Account

```bash
# Create service account
gcloud iam service-accounts create fraud-detection-sa \
    --display-name="Fraud Detection Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:fraud-detection-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:fraud-detection-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:fraud-detection-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:fraud-detection-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"
```

### Step 5: Download Service Account Key

```bash
# Create keys directory
mkdir -p ~/.gcp

# Download service account key
gcloud iam service-accounts keys create ~/.gcp/fraud-detection-sa-key.json \
    --iam-account=fraud-detection-sa@$PROJECT_ID.iam.gserviceaccount.com

# Set environment variable for authentication
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.gcp/fraud-detection-sa-key.json"
```

---

## Local Environment Setup

### Step 6: Create Virtual Environment

```bash
# Create project directory
mkdir -p ~/fraud-detection-vertex-ai
cd ~/fraud-detection-vertex-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 7: Install Dependencies

Create `requirements.txt`:
```
# Core ML Libraries
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0

# Google Cloud / Vertex AI
google-cloud-aiplatform>=1.35.0
google-cloud-storage>=2.10.0
google-cloud-bigquery>=3.11.0
google-cloud-logging>=3.8.0

# Flask API
flask>=2.3.0
gunicorn>=21.2.0

# Model Serialization
joblib>=1.3.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0.1

# Monitoring
prometheus-client>=0.17.0
```

Install packages:
```bash
pip install -r requirements.txt
```

### Step 8: Create Configuration File

Create `config.yaml`:
```yaml
# Vertex AI Configuration
project_id: "your-fraud-detection-project"
region: "us-central1"

# Storage Configuration
bucket_name: "fraud-detection-data-bucket"

# Model Configuration
model_display_name: "fraud-detection-classifier"
model_description: "Binary classifier for fraud detection"

# Training Configuration
training_fraction: 0.8
validation_fraction: 0.1
test_fraction: 0.1

# Feature Configuration
features:
  - transaction_amount
  - transaction_hour
  - day_of_week
  - merchant_category
  - card_present
  - international
  - distance_from_home
  - prev_transaction_count
  - avg_transaction_amount
  
target_column: "is_fraud"

# API Configuration
api_port: 8080
api_host: "0.0.0.0"

# Logging
log_level: "INFO"
log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## Vertex AI Workbench (Optional)

### Step 9: Create Managed Notebook (Alternative to Local)

```bash
# Create Vertex AI Workbench instance
gcloud notebooks instances create fraud-detection-notebook \
    --vm-image-project=deeplearning-platform-release \
    --vm-image-family=tf-latest-cpu \
    --machine-type=n1-standard-4 \
    --location=$REGION \
    --boot-disk-size=100GB
```

Or use Google Cloud Console:
1. Navigate to **Vertex AI** > **Workbench**
2. Click **Create New**
3. Select **Managed Notebook**
4. Choose configuration (recommend: n1-standard-4)
5. Click **Create**

---

## Create Cloud Storage Bucket

### Step 10: Setup Storage for Model Artifacts

```bash
# Create bucket for model artifacts
gcloud storage buckets create gs://$PROJECT_ID-fraud-detection \
    --location=$REGION \
    --uniform-bucket-level-access

# Create folders for organization
gsutil mkdir gs://$PROJECT_ID-fraud-detection/data
gsutil mkdir gs://$PROJECT_ID-fraud-detection/models
gsutil mkdir gs://$PROJECT_ID-fraud-detection/predictions
gsutil mkdir gs://$PROJECT_ID-fraud-detection/logs
```

---

## Verification

### Step 11: Test Setup

Create `test_setup.py`:
```python
import os
from google.cloud import aiplatform

# Test authentication
print("Testing Vertex AI setup...")

# Initialize Vertex AI
project_id = os.getenv("PROJECT_ID", "your-project-id")
region = os.getenv("REGION", "us-central1")

aiplatform.init(project=project_id, location=region)

# List existing models (should work if setup is correct)
try:
    models = aiplatform.Model.list()
    print(f"✓ Successfully connected to Vertex AI")
    print(f"  Found {len(models)} existing models")
except Exception as e:
    print(f"✗ Connection failed: {e}")

# Test Cloud Storage
try:
    from google.cloud import storage
    client = storage.Client()
    buckets = list(client.list_buckets())
    print(f"✓ Successfully connected to Cloud Storage")
    print(f"  Found {len(buckets)} buckets")
except Exception as e:
    print(f"✗ Cloud Storage connection failed: {e}")

print("\nSetup verification complete!")
```

Run the test:
```bash
python test_setup.py
```

---

## Expected Output

```
Testing Vertex AI setup...
✓ Successfully connected to Vertex AI
  Found 0 existing models
✓ Successfully connected to Cloud Storage
  Found 3 buckets

Setup verification complete!
```

---

## Next Steps

1. **Generate Sample Data**: Run `python generate_sample_data.py`
2. **Train Model**: Run `python train_model.py`
3. **Evaluate Model**: Run `python evaluate_model.py`
4. **Deploy API**: Run `python app.py`
5. **Run Batch Predictions**: Run `python batch_predict.py`

---

## Troubleshooting

### Issue: Permission Denied
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

### Issue: API Not Enabled
```bash
# Check enabled APIs
gcloud services list --enabled | grep aiplatform

# Enable if missing
gcloud services enable aiplatform.googleapis.com
```

### Issue: Quota Exceeded
- Visit GCP Console > IAM & Admin > Quotas
- Request quota increase for Vertex AI training

---

## Cost Considerations

| Resource | Estimated Cost |
|----------|---------------|
| Vertex AI Training (n1-standard-4) | ~$0.19/hour |
| Vertex AI Prediction (n1-standard-2) | ~$0.045/hour |
| Cloud Storage (Standard) | ~$0.02/GB/month |
| BigQuery Storage | ~$0.02/GB/month |

**Tip**: Use preemptible VMs for training to save ~70% on costs.
