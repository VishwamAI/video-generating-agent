import os
import cv2
import numpy as np

# Define the directory containing the training frames
train_dir = "/home/ubuntu/CamVid/train"

# Define the output directory for preprocessed frames
output_dir = "/home/ubuntu/CamVid/preprocessed_train"
os.makedirs(output_dir, exist_ok=True)

# Define the target size for resizing frames
target_size = (256, 256)

def preprocess_frame(frame_path, output_path):
    try:
        # Read the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Unable to read frame {frame_path}")
            return

        # Resize the frame
        resized_frame = cv2.resize(frame, target_size)

        # Normalize pixel values to the range [0, 1]
        normalized_frame = resized_frame / 255.0

        # Save the preprocessed frame
        cv2.imwrite(output_path, (normalized_frame * 255).astype(np.uint8))
    except Exception as e:
        print(f"Error processing frame {frame_path}: {e}")

# Process all frames in the training directory
for frame_name in os.listdir(train_dir):
    frame_path = os.path.join(train_dir, frame_name)
    output_path = os.path.join(output_dir, frame_name)
    preprocess_frame(frame_path, output_path)

print("Preprocessing complete. Preprocessed frames are saved in:", output_dir)
