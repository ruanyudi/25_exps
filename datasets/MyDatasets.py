import torch
import torchvision
from torch import nn
import lightning as L
from torch.utils.data import Dataset
import os
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, transform, path):
        super().__init__()
        if transform == None:
            transform = torchvision.transforms.ToTensor()
        self.transform = transform
        self.files = os.listdir(path)
        self.length = len(self.files)
        self.path = path

    def __getitem__(self, index):
        filename = self.files[index]
        image = Image.open(os.path.join(self.path, filename))
        label = int(filename[1])
        image = self.transform(image)
        return image, label

    def __len__(self):
        return self.length
