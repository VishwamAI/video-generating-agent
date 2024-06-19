import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.text_descriptions = []
        self.genres = []

        # Load data
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".png"):
                    self.image_files.append(os.path.join(root, file))
                elif file.endswith(".json"):
                    with open(os.path.join(root, file), "r") as f:
                        data = json.load(f)
                        self.text_descriptions.append(data["text"])
                        self.genres.append(data["genre"])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = self.text_descriptions[idx]
        genre = self.genres[idx]
        return image, text, genre

def get_data_loader(data_dir, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = VideoDataset(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

if __name__ == "__main__":
    data_dir = "../data"
    data_loader = get_data_loader(data_dir)

    for images, texts, genres in data_loader:
        print(f"Batch of images: {images.shape}")
        print(f"Batch of texts: {texts}")
        print(f"Batch of genres: {genres}")
