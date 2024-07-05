import torch
import torchvision
import cv2
import moviepy.editor as mp
import basicsr
import transformers
import realesrgan

def print_versions():
    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)
    print("opencv version:", cv2.__version__)
    print("basicsr version:", basicsr.__version__)
    print("transformers version:", transformers.__version__)

if __name__ == "__main__":
    print_versions()
