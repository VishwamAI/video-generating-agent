import os
import glob
import argparse
import imageio
import numpy as np

def preprocess_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_dir, '*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}")

    for image_file in image_files:
        image = imageio.imread(image_file)
        output_file = os.path.join(output_dir, os.path.basename(image_file))
        imageio.imwrite(output_file, image)

    print(f"Preprocessed {len(image_files)} images from {input_dir} to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for NeRF training")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing raw images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for preprocessed images")
    args = parser.parse_args()

    # Corrected the input directory path
    args.input = args.input.replace('nerf_llff_data/nerf_llff_data', 'nerf_llff_data')

    preprocess_images(args.input, args.output)
