#!/bin/bash
# Data Pipeline Monitoring System - Setup Script
# ===============================================

set -e  # Exit on error

echo "=========================================="
echo "Data Pipeline Monitoring System Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi
print_info "Python version check passed: $PYTHON_VERSION"

# Check if pip is installed
print_info "Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip first."
    exit 1
fi
print_info "pip3 is installed"

# Create virtual environment
print_info "Creating virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    print_info "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_info "Installing Python dependencies..."
pip install -r requirements.txt

print_info "Dependencies installed successfully"

# Create log directories
print_info "Creating log directories..."
LOG_DIRS=(
    "/var/log/airflow"
    "/var/log/kafka"
    "/var/log/ml"
    "/var/log/pipeline"
)

for dir in "${LOG_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_warning "Directory $dir already exists"
    else
        print_info "Creating directory: $dir"
        sudo mkdir -p "$dir"
        sudo chown -R "$USER:$USER" "$dir"
        sudo chmod 755 "$dir"
    fi
done

print_info "Log directories created"

# Create local log directories for testing
print_info "Creating local log directories for testing..."
mkdir -p logs/airflow
mkdir -p logs/kafka
mkdir -p logs/ml
mkdir -p logs/pipeline

# Copy example logs for testing
print_info "Setting up example log files..."
if [ -f "logs/example_airflow_logs.jsonl" ]; then
    cp logs/example_airflow_logs.jsonl logs/airflow/airflow_metrics.log
fi
if [ -f "logs/example_kafka_logs.jsonl" ]; then
    cp logs/example_kafka_logs.jsonl logs/kafka/consumer_stats.log
fi
if [ -f "logs/example_ml_logs.jsonl" ]; then
    cp logs/example_ml_logs.jsonl logs/ml/metrics.log
fi

print_info "Example logs copied"

# Make scripts executable
print_info "Setting up executable permissions..."
chmod +x scripts/pipeline_monitor.py

# Create configuration symlinks (optional)
print_info "Setting up configuration files..."
if [ ! -f "/etc/pipeline/logging.json" ]; then
    print_warning "Creating system-wide config directory requires sudo"
    sudo mkdir -p /etc/pipeline
    sudo cp configs/unified_logging.json /etc/pipeline/logging.json
    print_info "System configuration created"
fi

# Verify installation
print_info "Verifying installation..."
python3 -c "import pandas; import matplotlib; import yaml; print('All required packages imported successfully')"

# Run a quick test
print_info "Running quick test..."
python3 scripts/pipeline_monitor.py --component all --hours 1 --output text > /dev/null 2>&1 && print_info "Monitor script test passed" || print_warning "Monitor script test had issues (expected if no logs exist)"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Test the monitoring script:"
echo "   python scripts/pipeline_monitor.py --component all --hours 24"
echo ""
echo "3. Start the Jupyter dashboard:"
echo "   jupyter notebook notebooks/pipeline_dashboard.ipynb"
echo ""
echo "4. Run the monitoring server (optional):"
echo "   python scripts/pipeline_monitor.py --server --interval 60"
echo ""
echo "For more information, see README.md"
echo ""
