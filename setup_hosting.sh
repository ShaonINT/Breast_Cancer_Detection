#!/bin/bash

# Setup script for traditional web hosting deployment
# Breast Cancer Detection API

echo "========================================="
echo "Breast Cancer Detection - Hosting Setup"
echo "========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "✓ Pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Check if model files exist
if [ ! -f "best_model.pkl" ] || [ ! -f "model_metadata.pkl" ]; then
    echo "⚠ Model files not found. Training model..."
    echo "This may take a few minutes..."
    python train_models.py
    
    if [ $? -ne 0 ]; then
        echo "Error: Model training failed!"
        exit 1
    fi
    
    if [ ! -f "best_model.pkl" ]; then
        echo "Error: Model file still not found after training!"
        exit 1
    fi
    
    echo "✓ Model trained successfully"
else
    echo "✓ Model files found"
fi

echo ""

# Set permissions
echo "Setting file permissions..."
chmod 755 app.py train_models.py 2>/dev/null
chmod 644 *.pkl requirements.txt 2>/dev/null
chmod +x deploy.sh 2>/dev/null

echo "✓ Permissions set"
echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To start the server:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start server: python app.py"
echo ""
echo "Or use nohup to run in background:"
echo "  nohup python app.py > app.log 2>&1 &"
echo ""
echo "Check if running:"
echo "  ps aux | grep python"
echo ""
echo "View logs:"
echo "  tail -f app.log"
echo ""

