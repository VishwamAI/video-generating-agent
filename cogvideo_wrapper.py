import os
import torch
from cogvideo_pipeline import InferenceModel_Sequential, InferenceModel_Interpolate, process_stage1, process_stage2

class CogVideoWrapper:
    def __init__(self, stage1_model_path, stage2_model_path, device='cuda'):
        self.device = device
        self.model_stage1 = InferenceModel_Sequential.from_pretrained(stage1_model_path)
        self.model_stage2 = InferenceModel_Interpolate.from_pretrained(stage2_model_path)
        self.model_stage1.to(self.device).eval()
        self.model_stage2.to(self.device).eval()

    def generate_video(self, text_prompt, output_dir, duration_stage1=4.0, duration_stage2=2.0, batch_size=1):
        os.makedirs(output_dir, exist_ok=True)
        parent_given_tokens = process_stage1(self.model_stage1, text_prompt, duration=duration_stage1, video_raw_text=text_prompt, video_guidance_text="视频", image_text_suffix=" 高清摄影", outputdir=output_dir, batch_size=batch_size)
        process_stage2(self.model_stage2, text_prompt, duration=duration_stage2, video_raw_text=text_prompt+" 视频", video_guidance_text="视频", parent_given_tokens=parent_given_tokens, outputdir=output_dir, gpu_rank=0, gpu_parallel_size=1)

# Example usage:
# wrapper = CogVideoWrapper(stage1_model_path='cogvideo-stage1', stage2_model_path='cogvideo-stage2')
# wrapper.generate_video("A beautiful sunset over the mountains", output_dir="./output")
