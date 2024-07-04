#!/bin/bash

# Script to download the pre-trained model file for Real-ESRGAN

MODEL_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx4.pth"
MODEL_DIR="models"
MODEL_PATH="${MODEL_DIR}/RealESRGANv2-animevideo-xsx4.pth"

# Print the current working directory
echo "Current working directory: $(pwd)"

# Create the model directory if it does not exist
mkdir -p ${MODEL_DIR}

# Download the model file with verbose output
curl -L -v ${MODEL_URL} -o ${MODEL_PATH}

# Check if the download was successful
if [ -f "${MODEL_PATH}" ]; then
    echo "Model downloaded to ${MODEL_PATH}"
    # List the contents of the model directory
    ls -l ${MODEL_DIR}
else
    echo "Error: Model download failed"
    exit 1
fi
