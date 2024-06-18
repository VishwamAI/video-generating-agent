import os
import json
import moviepy.editor as mp

def compile_video(project_name):
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
    image_clips = [mp.ImageClip(img).set_duration(2) for img in image_files]  # Set each image duration to 2 seconds

    # Concatenate image clips
    video = mp.concatenate_videoclips(image_clips, method="compose")

    # Add audio to video
    audio = mp.AudioFileClip(audio_path)
    video = video.set_audio(audio)

    # Save the final video
    output_path = f"projects/{project_name}/final_video.mp4"
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return output_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile video from generated content.")
    parser.add_argument("project_name", type=str, help="Name of the project to compile.")
    args = parser.parse_args()

    output_video = compile_video(args.project_name)
    print(f"Video compiled successfully: {output_video}")
