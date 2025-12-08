#!/bin/bash

echo "╔═══════════════════════════════════════╗"
echo "║   OUROBOROS Setup Script              ║"
echo "║   Topological Malware Analysis        ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment."
    exit 1
fi

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "[4/6] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "[5/6] Installing dependencies..."
echo "This may take several minutes (GUDHI compilation requires time)..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies."
    echo "Common issues:"
    echo "  - GUDHI requires C++ compiler (install build-essential on Linux, Xcode on macOS)"
    echo "  - Try: pip install --no-cache-dir gudhi"
    exit 1
fi

# Create test samples
echo ""
echo "[6/6] Creating test samples..."
python create_test_samples.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run OUROBOROS:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run: streamlit run app.py"
echo ""
echo "Test samples created in: test_samples/"
echo ""
