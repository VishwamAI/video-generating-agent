import os
import cv2
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Preprocess frames for training")
parser.add_argument("--input", type=str, required=True, help="Path to the input directory containing frames")
args = parser.parse_args()

# Define the directories containing the frames
train_dir = os.path.join(args.input, "train")
val_dir = os.path.join(args.input, "val")
test_dir = os.path.join(args.input, "test")

# Define the output directories for preprocessed frames
output_train_dir = os.path.join(args.input, "preprocessed_train")
output_val_dir = os.path.join(args.input, "preprocessed_val")
output_test_dir = os.path.join(args.input, "preprocessed_test")

os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

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
    output_path = os.path.join(output_train_dir, frame_name)
    preprocess_frame(frame_path, output_path)

# Process all frames in the validation directory
for frame_name in os.listdir(val_dir):
    frame_path = os.path.join(val_dir, frame_name)
    output_path = os.path.join(output_val_dir, frame_name)
    preprocess_frame(frame_path, output_path)

# Process all frames in the testing directory
for frame_name in os.listdir(test_dir):
    frame_path = os.path.join(test_dir, frame_name)
    output_path = os.path.join(output_test_dir, frame_name)
    preprocess_frame(frame_path, output_path)

print("Preprocessing complete. Preprocessed frames are saved in the respective directories.")
