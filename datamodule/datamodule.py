import os
from PIL import Image
import random

import torch
from torch.utils.data import Dataset

from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    Resize,
    RandomHorizontalFlip,
    RandomRotation,
    ColorJitter,
)

class TargetImageDataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 size: tuple[int, int] = (256, 512),
                 mean: list[float] = [0.5, 0.5, 0.5], 
                 sd: list[float] = [0.5, 0.5, 0.5],
                 amount_augmentations: int = 1,
                 horizontal_flip: float = 0.5,
                 rotation: float = 15,
                 brightness: float = 0.2,
                 contrast: float = 0.2,
                 saturation: float = 0.2,
                 hue: float = 0.1,
        ):
        assert amount_augmentations >= 1, 'amount_augmentations must be greater than or equal to 1'

        self.data_dir = data_dir
        self.amount_augmentations = amount_augmentations

        # Obtener la lista de carpetas en el directorio de datos
        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        # Crear una lista de rutas de acceso a las imágenes
        self.image_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            image_path = os.path.join(image_folder_path, f'frame2.jpg')
            self.image_paths.append(image_path)

        # Transformaciones básicas
        self.transform_img = Compose([
            ToTensor(),
            Resize(size),
            Normalize(mean, sd)
        ])

        # Transformaciones de data augmentation
        self.augmentation_transform = Compose([
            RandomHorizontalFlip(p=horizontal_flip),
            RandomRotation(degrees=rotation),
            ColorJitter(brightness=brightness, 
                                    contrast=contrast, 
                                    saturation=saturation, 
                                    hue=hue),
        ])

    def __len__(self):
        return len(self.image_paths) * self.amount_augmentations

    def __getitem__(self, index):
        original_index = index // self.amount_augmentations
        is_augmented = index % self.amount_augmentations == 1

        image_path = self.image_paths[original_index]
        image = Image.open(image_path)

        if is_augmented:
            image = self.augmentation_transform(image)

        image = self.transform_img(image)
        return image

class TwoImagesDataset(Dataset):
    def __init__(self, data_dir: str, 
                 size: tuple[int, int] = (256, 256),
                 mean: list[float] = [0.5, 0.5, 0.5], 
                 sd: list[float] = [0.5, 0.5, 0.5],
                 amount_augmentations: int = 1,
                 horizontal_flip: float = 0.5,
                 time_flip: bool = True,
                 rotation: float = 0.5,
                 brightness: float = 0.1,
                 contrast: float = 0.1,
                 saturation: float = 0.1,
                 hue: float = 0.1,
        ):
        assert amount_augmentations >= 1, 'amount_augmentations must be greater than or equal to 1'

        self.data_dir = data_dir
        self.time_flip = time_flip
        self.amount_augmentations = amount_augmentations

        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        self.image_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            image_paths = [os.path.join(image_folder_path, f'frame{i}.jpg') for i in range(1, 3)]
            self.image_paths.append(image_paths)

        self.transform_img = Compose([
            ToTensor(),
            Resize(size),
        ])

        # Data augmentation transformations
        self.augmentation_transform = Compose([
            RandomHorizontalFlip(p=horizontal_flip),
            RandomRotation(degrees=rotation),
            ColorJitter(brightness=brightness, 
                        contrast=contrast, 
                        saturation=saturation, 
                        hue=hue),
            Normalize(mean, sd)
        ])

    def __len__(self):
        return len(self.image_paths) * self.amount_augmentations

    def __getitem__(self, index):
        original_index = index // self.amount_augmentations
        is_augmented = index % self.amount_augmentations == 1

        image_paths = self.image_paths[original_index]
        images = [Image.open(image_path) for image_path in image_paths]
        images = [self.transform_img(image) for image in images]

        if is_augmented:
            if self.time_flip and torch.rand(1) > 0.5:
                images = images[::-1] 
            images = [self.augmentation_transform(image) for image in images]

        return images

class TripletImagesDataset(Dataset):
    def __init__(self,
                 data_dir: str, 
                 size: tuple[int, int] = (256, 256),
                 mean: list[float] = [0.5, 0.5, 0.5], 
                 sd: list[float] = [0.5, 0.5, 0.5],
                 amount_augmentations: int = 1,
                 horizontal_flip: float = 0.5,
                 time_flip: bool = True,
                 rotation: float = 0,  
                 brightness: float = 0.2,
                 contrast: float = 0.2,
                 saturation: float = 0.2,
                 hue: float = 0.1):

        assert amount_augmentations >= 1, 'amount_augmentations must be at least 1'

        self.data_dir = data_dir
        self.time_flip = time_flip
        self.amount_augmentations = amount_augmentations

        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        self.image_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            I0 = os.path.join(image_folder_path, 'frame1.jpg')
            It = os.path.join(image_folder_path, 'frame2.jpg')
            I1 = os.path.join(image_folder_path, 'frame3.jpg')
            if os.path.exists(I0) and os.path.exists(It) and os.path.exists(I1):
                self.image_paths.append([I0, It, I1])

        # Transformaciones combinadas para evitar aplicar doble transformación
        if amount_augmentations > 1:
            self.transform = Compose([
                RandomHorizontalFlip(p=horizontal_flip),
                RandomRotation(degrees=rotation),
                ColorJitter(brightness=brightness, 
                            contrast=contrast, 
                            saturation=saturation, 
                            hue=hue),
                Resize(size),
                ToTensor(),
                Normalize(mean, sd)
            ])
        else:
            self.transform = Compose([
                Resize(size),
                ToTensor(),
                Normalize(mean, sd)
            ])

    def __len__(self):
        return len(self.image_paths) * self.amount_augmentations

    def __getitem__(self, index):
        original_index = index // self.amount_augmentations

        try:
            image_paths = self.image_paths[original_index]
            images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        except Exception as e:
            print(f"Error loading images from {self.image_paths[original_index]}: {e}")
            return None 
        
        if self.amount_augmentations > 1:
            if torch.rand(1) < self.time_flip:
                images = images[::-1]
            seed = torch.randint(0, 10000, (1,)).item()
            torch.manual_seed(seed) 
            random.seed(seed)
        images = [self.transform(image) for image in images]

        return images


class TwoImagesWithFlowDataset(Dataset):
    def __init__(self, data_dir: str, 
                 mean: list[float] = [0.5, 0.5, 0.5], 
                 sd: list[float] = [0.5, 0.5, 0.5],
        ):
        self.data_dir = data_dir
        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        self.image_paths = []
        self.flow_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            flow_folder_path = os.path.join(data_dir, folder)
            
            image_paths = [os.path.join(image_folder_path, f'frame{i}.jpg') for i in range(1, 3)]
            flow_path = os.path.join(flow_folder_path, f'flow0tot.pt')
            
            self.image_paths.append(image_paths)
            self.flow_paths.append(flow_path)

        self.transform_img = Compose([
            ToTensor(),
            Resize((256, 256)),
            Normalize(mean, sd)
        ])
        self.transform_flow = Resize((256, 256))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_paths = self.image_paths[index]
        flow_path = self.flow_paths[index]
        
        images = [Image.open(image_path) for image_path in image_paths]
        flow = torch.load(flow_path)
        
        images = [self.transform_img(image) for image in images]
        flow = self.transform_flow(flow.squeeze(0))

        images = torch.stack(images)

        return {'images': images, 'flow': flow}


class TripletImagesWithFlowsDataset(Dataset):
    def __init__(self, data_dir: str, 
                 mean: list[float] = [0.5, 0.5, 0.5], 
                 sd: list[float] = [0.5, 0.5, 0.5]
        ):
        self.data_dir = data_dir
        subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        self.image_paths = []
        self.flow_paths = []
        for folder in subfolders:
            image_folder_path = os.path.join(data_dir, folder)
            flow_folder_path = os.path.join(data_dir, folder)
            
            image_paths = [os.path.join(image_folder_path, f'frame{i}.jpg') for i in range(1, 4)]
            flow_paths = [os.path.join(flow_folder_path, f'flow{i}.pt') for i in range(2)]
            
            self.image_paths.append(image_paths)
            self.flow_paths.append(flow_paths)

        self.transform_img = Compose([
            ToTensor(),
            Resize((256, 256)),
            Normalize(mean, sd)
        ])
        self.transform_flow = Resize((256, 256))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_paths = self.image_paths[index]
        flow_paths = self.flow_paths[index]
        
        images = [Image.open(image_path) for image_path in image_paths]
        flows = [torch.load(flow_path) for flow_path in flow_paths]
        
        images = [self.transform_img(image) for image in images]
        flows = [self.transform_flow(flow.squeeze(0)) for flow in flows]

        images = torch.stack(images)
        flows = torch.stack(flows)

        return {'images': images, 'flows': flows}
