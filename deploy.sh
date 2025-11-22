#!/bin/bash

# Deployment script for Breast Cancer Detection API

echo "========================================="
echo "Breast Cancer Detection - Deployment"
echo "========================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

echo "✓ Python found: $(python --version)"

# Check if model files exist
if [ ! -f "best_model.pkl" ] || [ ! -f "model_metadata.pkl" ]; then
    echo ""
    echo "⚠ Model files not found. Training model..."
    python train_models.py
    
    if [ ! -f "best_model.pkl" ]; then
        echo "✗ Error: Model training failed!"
        exit 1
    fi
    echo "✓ Model trained successfully"
else
    echo "✓ Model files found"
fi

# Install dependencies if needed
echo ""
echo "Checking dependencies..."
pip install -r requirements.txt --quiet

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "✗ Error: app.py not found!"
    exit 1
fi

echo ""
echo "========================================="
echo "Starting server..."
echo "========================================="
echo "Web Interface: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python app.py

