#!/bin/bash

echo "Starting AI Governance Lab Setup..."

# Create Virtual Environment
python3 -m venv venv

# Activate Virtual Environment
source venv/bin/activate

# Install Dependencies
echo "Installing required Python packages (Deepchecks, Jupyter, etc.)..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p audit_scripts
mkdir -p reports
mkdir -p notebooks

echo "Setup Complete! To start: source venv/bin/activate"
echo "To run the CI/CD Gate: python3 audit_scripts/run_governance_gate.py"