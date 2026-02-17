#!/bin/bash

# ============================================
# Fraud Dashboard Backend Setup Script
# ============================================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Fraud Detection Pipeline Dashboard - Setup             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18+ required. Found: $(node -v)"
    exit 1
fi

echo "✅ Node.js version: $(node -v)"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    
    # Try to detect username for default paths
    USERNAME=$(whoami)
    HOME_DIR=$HOME
    
    # Update default paths
    sed -i "s|/home/user|$HOME_DIR|g" .env
    
    echo "✅ .env file created at: $(pwd)/.env"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env with your actual configuration:"
    echo "   nano .env"
    echo ""
    echo "   Key settings to update:"
    echo "   - AIRFLOW_DATA_PATH (path to your airflow/data folder)"
    echo "   - AIRFLOW_USER and AIRFLOW_PASS"
    echo "   - KAFKA_BROKERS (if not localhost:9092)"
    echo "   - SNOWFLAKE_KEY_PATH (path to your .p8 key file)"
    echo ""
else
    echo "✅ .env file already exists"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                      Setup Complete!                       ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Next steps:                                               ║"
echo "║  1. Edit .env with your configuration                      ║"
echo "║  2. Start the server: npm start                            ║"
echo "║  3. Open http://localhost:3001 to test API                 ║"
echo "║                                                            ║"
echo "║  To update your React app:                                 ║"
echo "║  - Copy hooks/useRealData.ts to your React app's src/hooks/║"
echo "║  - Replace App.tsx with App.tsx.real                       ║"
echo "║  - Add VITE_API_URL=http://localhost:3001 to your .env     ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
