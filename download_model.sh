#!/bin/bash

# Script to download the pre-trained model file for Real-ESRGAN

MODEL_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth"
MODEL_DIR="models"
MODEL_PATH="${MODEL_DIR}/RealESRGAN_x4plus.pth"

# Create the model directory if it does not exist
mkdir -p ${MODEL_DIR}

# Download the model file
curl -L ${MODEL_URL} -o ${MODEL_PATH}

echo "Model downloaded to ${MODEL_PATH}"
