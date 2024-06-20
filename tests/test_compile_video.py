import unittest
import os
import json
from scripts.compile_video import compile_video
from PIL import Image
import wave

class TestCompileVideo(unittest.TestCase):
    def setUp(self):
        self.project_name = "test_project"
        self.project_dir = f"projects/{self.project_name}"
        self.content_file = f"{self.project_dir}/content.json"
        self.audio_file = f"{self.project_dir}/audio.mp3"
        self.images_dir = f"{self.project_dir}/images"
        self.output_video = f"{self.project_dir}/final_video.mp4"

        # Create test project directory and files
        os.makedirs(self.images_dir, exist_ok=True)
        with open(self.content_file, "w") as f:
            json.dump({"content": "Test content"}, f)

        # Create a valid audio file
        with wave.open(self.audio_file, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(44100)
            f.writeframes(b'\x00\x00' * 44100 * 5)  # 5 seconds of silence

        for i in range(5):
            img = Image.new('RGB', (100, 100), color = (73, 109, 137))
            img.save(f"{self.images_dir}/image_{i}.png")

    def test_compile_video_1080p(self):
        output_path = compile_video(self.project_name, resolution="1080p")
        self.assertTrue(os.path.exists(output_path))

    def test_compile_video_2K(self):
        output_path = compile_video(self.project_name, resolution="2K")
        self.assertTrue(os.path.exists(output_path))

    def test_compile_video_3K(self):
        output_path = compile_video(self.project_name, resolution="3K")
        self.assertTrue(os.path.exists(output_path))

    def test_compile_video_4K(self):
        output_path = compile_video(self.project_name, resolution="4K")
        self.assertTrue(os.path.exists(output_path))

    def tearDown(self):
        # Clean up test project directory and files
        if os.path.exists(self.project_dir):
            for root, dirs, files in os.walk(self.project_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.project_dir)

if __name__ == "__main__":
    unittest.main()
