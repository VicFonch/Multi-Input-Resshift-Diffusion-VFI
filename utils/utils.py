import os
import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.utils import make_grid, save_image

from typing import Any

def exist(val: Any) -> bool:
    return val is not None

def morph_open(x: torch.Tensor, k: int) -> torch.Tensor:
    if k==0:
        return x
    else:
        with torch.no_grad():
            return kornia.morphology.opening(x, torch.ones(k,k,device=x.device))

def make_grid_images(images: list[torch.Tensor], **kwargs) -> torch.Tensor:
    concatenated_images = torch.cat(images, dim=3)
    grid_concatenated = make_grid(concatenated_images, **kwargs)
    return grid_concatenated

def save_images(images: tuple[torch.Tensor, torch.Tensor], path: str, **kwargs) -> None:
    gen, real = images
    concatenated_images = torch.cat((gen, real), dim=3)
    grid_concatenated = make_grid(concatenated_images, **kwargs)

    ndarr_concatenated = grid_concatenated.permute(1, 2, 0).to("cpu").numpy()
    ndarr_concatenated = (ndarr_concatenated * 255).astype(np.uint8)

    save_image(torch.from_numpy(ndarr_concatenated).permute(2, 0, 1) / 255, path)

def save_triplet(images: tuple[torch.Tensor, ...], path: str, **kwargs) -> None:
    concatenated_images = torch.cat(images, dim=3)
    grid_concatenated = make_grid(concatenated_images, **kwargs)
    
    ndarr_concatenated = grid_concatenated.permute(1, 2, 0).to("cpu").numpy()
    ndarr_concatenated = (ndarr_concatenated * 255).astype(np.uint8)

    save_image(torch.from_numpy(ndarr_concatenated).permute(2, 0, 1) / 255, path)

def plot_images(images: torch.Tensor) -> None:
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def make_graphic(metric_name: str, metrics: list[torch.Tensor], path: str) -> None:
    plt.figure(figsize=(32, 32))
    metrics = [m.cpu().numpy() for m in metrics]
    plt.plot(metrics)
    plt.title(metric_name)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    path = os.path.join(path, f"{metric_name}.png")
    plt.savefig(path)
    plt.close()

def norm(
    img: torch.Tensor, 
    mean: list[float] = [0.5, 0.5, 0.5], 
    std: list[float] = [0.5, 0.5, 0.5]
) -> torch.Tensor:
    normalize = transforms.Normalize(mean, std)
    return normalize(img)

def denorm(
    img: torch.Tensor, 
    mean: list[float] = [0.5, 0.5, 0.5], 
    std: list[float] = [0.5, 0.5, 0.5]
) -> torch.Tensor:
    mean = torch.tensor(mean, device=img.device)
    std = torch.tensor(std, device=img.device)
    return img*std[None][...,None,None] + mean[None][...,None,None]