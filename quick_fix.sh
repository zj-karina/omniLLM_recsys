#!/bin/bash
# Quick fix script for missing dependencies

echo "🔧 QUICK FIX FOR MISSING DEPENDENCIES"
echo "====================================="

echo "Installing missing packages..."

# Install requests
pip install requests

# Install pyyaml
pip install pyyaml

# Install rich
pip install rich

# Install wandb
pip install wandb

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "Now you can run:"
echo "  wandb login"
echo "  make test-logging"
echo "  make train"







