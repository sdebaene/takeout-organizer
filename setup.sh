#!/bin/bash

# Exit on error
set -e

echo "Setting up environment for media_matcher.py..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo "Creating requirements.txt..."
    cat > requirements.txt << EOF
Pillow
piexif
ffmpeg-python
pillow-heif
EOF
    echo "requirements.txt created."
fi

# Install required Python packages
echo "Installing Python packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "All dependencies installed successfully!"
echo "Virtual environment is now active. To deactivate, run 'deactivate'"
echo "You can now run media_matcher.py"

# Keep the environment activated for the current shell
exec "$SHELL" 