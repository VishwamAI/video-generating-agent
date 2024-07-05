import unittest
import os
import cv2
import numpy as np
from scripts.video_upscale import upscale_video

class TestVideoUpscale(unittest.TestCase):
    def setUp(self):
        self.input_video = "tests/sample_input.mp4"
        self.output_video = "tests/sample_output.mp4"
        self.model_path = "models/RealESRGAN_x4plus.pth"
        self.scale = 4

        # Create a sample input video for testing
        if not os.path.exists(self.input_video):
            cap = cv2.VideoWriter(self.input_video, cv2.VideoWriter_fourcc(*'mp4v'), 1, (64, 64))
            for _ in range(10):
                cap.write(np.zeros((64, 64, 3), dtype=np.uint8))
            cap.release()

    def test_upscale_video(self):
        upscale_video(self.input_video, self.output_video, self.model_path, self.scale)
        self.assertTrue(os.path.exists(self.output_video))

    def test_upscale_video_invalid_input(self):
        with self.assertRaises(FileNotFoundError):
            upscale_video("invalid_input.mp4", self.output_video, self.model_path, self.scale)

    def test_upscale_video_invalid_model(self):
        with self.assertRaises(FileNotFoundError):
            upscale_video(self.input_video, self.output_video, "invalid_model.pth", self.scale)

    def test_upscale_video_check_resolution(self):
        upscale_video(self.input_video, self.output_video, self.model_path, self.scale)
        cap = cv2.VideoCapture(self.output_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self.assertEqual(width, 64 * self.scale)
        self.assertEqual(height, 64 * self.scale)

    def tearDown(self):
        if os.path.exists(self.input_video):
            os.remove(self.input_video)
        if os.path.exists(self.output_video):
            os.remove(self.output_video)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == "__main__":
    unittest.main()
