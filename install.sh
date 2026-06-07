#!/bin/bash
set -e

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install PyTorch CPU only
pip install torch~=2.4.0 --index-url https://download.pytorch.org/whl/cpu

echo "Installation complete. Run 'source venv/bin/activate' to use."