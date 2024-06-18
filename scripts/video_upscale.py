import argparse
import cv2
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def upscale_video(input_video, output_video, model_path, scale=4, tile=0, tile_pad=10, pre_pad=0, fp32=False, gpu_id=None):
    # Load the model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)

    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width * scale, height * scale))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_idx + 1}/{total_frames}")
        frame_idx += 1

        # Enhance the frame
        try:
            output, _ = upsampler.enhance(frame, outscale=scale)
        except RuntimeError as error:
            print(f"Error: {error}")
            print("If you encounter CUDA out of memory, try to set --tile with a smaller number.")
            break

        # Write the frame to the output video
        out.write(output)

    # Release everything if job is finished
    cap.release()
    out.release()
    print(f"Video saved to {output_video}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input video file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output video file')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('-s', '--scale', type=int, default=4, help='Upscaling factor')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('-g', '--gpu_id', type=int, default=None, help='GPU device to use (default=None) can be 0,1,2 for multi-GPU')

    args = parser.parse_args()

    upscale_video(args.input, args.output, args.model_path, args.scale, args.tile, args.tile_pad, args.pre_pad, args.fp32, args.gpu_id)
