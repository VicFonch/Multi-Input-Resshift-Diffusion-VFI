import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import transforms

class DavisDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        video_dir = [os.path.join(data_dir, foder) for foder in os.listdir(data_dir)]
        
        self.triplets_dir = {"I0":[], "I1":[], "I2":[]}
        for folder_dir in video_dir:
            images = [os.path.join(folder_dir, image) for image in os.listdir(folder_dir)]
            self.triplets_dir["I0"] += [images[0], images[6], images[12], images[18], images[22]] 
            self.triplets_dir["I1"] += [images[1], images[7], images[13], images[19], images[23]]
            self.triplets_dir["I2"] += [images[2], images[8], images[14], images[20], images[24]]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else: self.transform = transform

    def __len__(self):
        return len(self.triplets_dir["I0"])
    
    def __getitem__(self, index):
        I0 = self.transform(Image.open(self.triplets_dir["I0"][index]))
        I1 = self.transform(Image.open(self.triplets_dir["I1"][index]))
        I2 = self.transform(Image.open(self.triplets_dir["I2"][index]))

        return I0, I1, I2
    
class UCF101Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        video_dir = [os.path.join(data_dir, foder) for foder in os.listdir(data_dir)]
        
        self.triplets_dir = {"I0":[], "I1":[], "I2":[]}
        for folder_dir in video_dir:
            images = [os.path.join(folder_dir, image) for image in os.listdir(folder_dir)]
            self.triplets_dir["I0"].append(images[1])  
            self.triplets_dir["I1"].append(images[2])
            self.triplets_dir["I2"].append(images[3])

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else: self.transform = transform

    def __len__(self):
        return len(self.triplets_dir["I0"])
    
    def __getitem__(self, index):
        print(self.triplets_dir["I0"][index])
        
        I0 = self.transform(Image.open(self.triplets_dir["I0"][index]))
        I1 = self.transform(Image.open(self.triplets_dir["I1"][index]))
        I2 = self.transform(Image.open(self.triplets_dir["I2"][index]))

        return I0, I1, I2

class MiddleburyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        video_dir = [os.path.join(data_dir, foder) for foder in os.listdir(data_dir)]
        
        self.triplets_dir = {"I0":[], "I1":[], "I2":[]}
        for folder_dir in video_dir:
            images = [os.path.join(folder_dir, image) for image in os.listdir(folder_dir)]
            self.triplets_dir["I0"].append(images[0])  
            self.triplets_dir["I1"].append(images[1])
            self.triplets_dir["I2"].append(images[2])

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else: self.transform = transform

    def __len__(self):
        return len(self.triplets_dir["I0"])
    
    def __getitem__(self, index):
        I0 = self.transform(Image.open(self.triplets_dir["I0"][index]))
        I1 = self.transform(Image.open(self.triplets_dir["I1"][index]))
        I2 = self.transform(Image.open(self.triplets_dir["I2"][index]))

        return I0, I1, I2


class SnuFilmDataset(Dataset):
    def __init__(self, data_dir, difficulty = 'easy', transform=None):        
        with open(data_dir + f"/test-{difficulty}.txt") as f:
            paths = f.readlines()

        self.images_dir = []
        for path in paths:
            self.images_dir.append([(data_dir + "/" + direction).replace('\n', "") for direction in path.split(" ")])

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else: self.transform = transform

    def __len__(self):
        return len(self.images_dir)
    
    def __getitem__(self, index):
        
        I0 = self.transform(Image.open(self.images_dir[index][0]))
        I1 = self.transform(Image.open(self.images_dir[index][1]))
        I2 = self.transform(Image.open(self.images_dir[index][2]))

        return I0, I1, I2