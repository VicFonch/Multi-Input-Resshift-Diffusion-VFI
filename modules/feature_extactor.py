import torch
import torch.nn as nn

import torchvision.models as models

from modules.basic_layers import GroupNorm

class Extractor(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=3, stride=2, padding=1),
                GroupNorm(channels[i + 1]),
                nn.SiLU(),
                nn.Conv2d(in_channels=channels[i + 1], out_channels=channels[i + 1], kernel_size=3, stride=1, padding=1),
                GroupNorm(channels[i + 1]),
                nn.SiLU()
            ) for i in range(len(channels) - 1)
        ])

        self.residual = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=3, stride=2, padding=1),
            ) for i in range(len(channels) - 1)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for residual, layer in zip(self.residual, self.layers):
            x = layer(x) + residual(x)
            features.append(x)
        return features
    

class ResNetExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, layers_to_extract: list[str] = ["layer1", "layer2", "layer3"]):
        super(ResNetExtractor, self).__init__()
        
        resnet = models.resnet18(pretrained=pretrained)
        
        self.initial_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        
        self.layers = nn.ModuleDict({
            "layer1": resnet.layer1,
            "layer2": resnet.layer2,
            "layer3": resnet.layer3,
        })
        
        self.layers_to_extract = layers_to_extract

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        x = self.initial_layers(x)
        
        for name, layer in self.layers.items():
            x = layer(x)
            if name in self.layers_to_extract:
                features.append(x)
        
        return features
    
class VGGExtractor(nn.Module):
    def __init__(self, layers_to_extract: list[int] = [8, 15, 22, 29]):
        super(VGGExtractor, self).__init__()
        
        self.vgg = models.vgg16(pretrained=True).features
        self.layers_to_extract = layers_to_extract
        self.selected_layers = [self.vgg[i] for i in layers_to_extract]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers_to_extract:
                features.append(x)
        return features
