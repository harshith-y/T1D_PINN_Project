#!/bin/bash
# Install all dependencies on Ubuntu

set -e

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    wget

echo "Creating virtual environment..."
python3.9 -m venv venv
source venv/bin/activate

echo "Installing Python packages..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

echo "Done! Activate with: source venv/bin/activate"
