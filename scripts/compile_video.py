import os
import json
import moviepy.editor as mp
import logging
from PIL import Image

def compile_video(project_name, resolution="1080p", fps=24):
    # Load project content
    with open(f"projects/{project_name}/content.json", "r") as f:
        content = json.load(f)
        story = content["content"]

    # Load generated audio
    audio_path = f"projects/{project_name}/audio.mp3"
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load generated images
    images_dir = f"projects/{project_name}/images"
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_files = sorted([os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        raise FileNotFoundError(f"No images found in directory: {images_dir}")

    # Create video clips from images
    image_clips = [mp.ImageClip(img).set_duration(2).set_fps(fps) for img in image_files]  # Set each image duration to 2 seconds and fps

    # Concatenate image clips
    video = mp.concatenate_videoclips(image_clips, method="compose")

    # Add audio to video
    audio = mp.AudioFileClip(audio_path)
    video = video.set_audio(audio)

    # Set resolution
    resolution_map = {
        "1080p": (1920, 1080),
        "2K": (2048, 1080),
        "3K": (3072, 1620),
        "4K": (3840, 2160)
    }
    if resolution in resolution_map:
        video = video.resize(newsize=resolution_map[resolution])
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    # Save the final video
    output_path = f"projects/{project_name}/final_video_{resolution}.mp4"
    video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)

    return output_path

if __name__ == "__main__":
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler("compile_video.log"),
        logging.StreamHandler()
    ])

    parser = argparse.ArgumentParser(description="Compile video from generated content.")
    parser.add_argument("project_name", type=str, help="Name of the project to compile.")
    parser.add_argument("--resolution", type=str, default="1080p", help="Resolution of the output video (e.g., 1080p, 2K, 3K, 4K).")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second of the output video.")
    args = parser.parse_args()

    output_video = compile_video(args.project_name, args.resolution, args.fps)
    logging.info(f"Video compiled successfully: {output_video}")
