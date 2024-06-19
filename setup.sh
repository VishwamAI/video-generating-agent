#!/bin/bash

# Check if virtual environment directory exists
if [ ! -d "venv" ]; then
  # Create a virtual environment
  python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required dependencies
pip install -r requirements.txt

# Additional setup steps
# (Add any additional setup commands here)

echo "Setup complete. Virtual environment is ready and dependencies are installed."
