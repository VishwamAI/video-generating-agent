import os
import torch
import argparse
from cogvideo_pipeline import InferenceModel_Sequential, InferenceModel_Interpolate, process_stage1, process_stage2

def initialize_models(args):
    if args.stage_1 or args.both_stages:
        model_stage1, args = InferenceModel_Sequential.from_pretrained(args, 'cogvideo-stage1')
        model_stage1.eval()
        if args.both_stages:
            model_stage1 = model_stage1.cpu()
    else:
        model_stage1 = None

    if args.stage_2 or args.both_stages:
        model_stage2, args = InferenceModel_Interpolate.from_pretrained(args, 'cogvideo-stage2')
        model_stage2.eval()
        if args.both_stages:
            model_stage2 = model_stage2.cpu()
    else:
        model_stage2 = None

    return model_stage1, model_stage2, args

def run_cogvideo_pipeline(text_description, output_dir, stage_1=True, stage_2=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-frame-num', type=int, default=5)
    parser.add_argument('--coglm-temperature2', type=float, default=0.89)
    parser.add_argument('--use-guidance-stage1', action='store_true')
    parser.add_argument('--use-guidance-stage2', action='store_true')
    parser.add_argument('--guidance-alpha', type=float, default=3.0)
    parser.add_argument('--stage-1', action='store_true')
    parser.add_argument('--stage-2', action='store_true')
    parser.add_argument('--both-stages', action='store_true')
    parser.add_argument('--parallel-size', type=int, default=1)
    parser.add_argument('--stage1-max-inference-batch-size', type=int, default=-1)
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--input_source', type=str, default="interactive")
    parser.add_argument('--output_path', type=str, default=output_dir)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    if stage_1:
        args.stage_1 = True
    if stage_2:
        args.stage_2 = True
    if stage_1 and stage_2:
        args.both_stages = True

    model_stage1, model_stage2, args = initialize_models(args)

    if args.stage_1 or args.both_stages:
        parent_given_tokens = process_stage1(model_stage1, text_description, duration=4.0, video_raw_text=text_description, video_guidance_text="视频",
                                             image_text_suffix=" 高清摄影", outputdir=output_dir, batch_size=args.batch_size)
        if args.both_stages:
            process_stage2(model_stage2, text_description, duration=2.0, video_raw_text=text_description + " 视频",
                           video_guidance_text="视频", parent_given_tokens=parent_given_tokens, outputdir=output_dir,
                           gpu_rank=0, gpu_parallel_size=1)
    elif args.stage_2:
        parent_given_tokens = torch.load(os.path.join(output_dir, "frame_tokens.pt"))
        process_stage2(model_stage2, text_description, duration=2.0, video_raw_text=text_description + " 视频",
                       video_guidance_text="视频", parent_given_tokens=parent_given_tokens, outputdir=output_dir,
                       gpu_rank=0, gpu_parallel_size=1)

if __name__ == "__main__":
    text_description = "A beautiful sunset over the mountains."
    output_dir = "./output"
    run_cogvideo_pipeline(text_description, output_dir, stage_1=True, stage_2=True)
