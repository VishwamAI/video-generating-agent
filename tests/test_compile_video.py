import unittest
import os
import json
from scripts.compile_video import compile_video

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
        with open(self.audio_file, "w") as f:
            f.write("Test audio content")
        for i in range(5):
            with open(f"{self.images_dir}/image_{i}.png", "w") as f:
                f.write("Test image content")

    def test_compile_video(self):
        output_path = compile_video(self.project_name)
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
