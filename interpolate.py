import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

import yaml
from PIL import Image
import matplotlib.pyplot as plt

from model.model import MultiInputResShift

from utils.utils import denorm

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--confg_path", type=str, default=r"config\confg.yaml")
    parser.add_argument("--checkpoint_path", type=str, default=r"_checkpoint\residual_diff_rfr.pth")
    parser.add_argument("--img0_path", type=str, default=r"_data\example_images\frame1.png")
    parser.add_argument("--img2_path", type=str, default=r"_data\example_images\frame3.png")
    parser.add_argument("--tau_val", type=float, default=0.5)
    parser.add_argument("--output_path", type=str, default=r"_data\example_images")
    parser.add_argument("--num_samples", type=int, default=1)
    return parser.parse_args()

def load_image(img_path):
    transforms = Compose([
        Resize((256, 448)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda().requires_grad_(False)

def save_image(tensor, path):
    tensor = denorm(tensor.clamp(-1, 1), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    tensor = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.imsave(path, tensor)

def load_model(confg_path, checkpoint_path):
    with open(confg_path, "r") as f:
        confg = yaml.safe_load(f)
    model = MultiInputResShift(**confg["model_confg"])
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval().requires_grad_(False).cuda()
    return model

def main():
    args = parse_args()
    model = load_model(args.confg_path, args.checkpoint_path)

    img0 = load_image(args.img0_path)
    img2 = load_image(args.img2_path)

    if args.num_samples > 1:
        It_list = []
        tau_list = torch.linspace(0, 1, args.num_samples + 2)[1:-1]
        for i in range(args.num_samples):
            It_list.append(model.reverse_process([img0, img2], tau_list[i]))
        save_image(It_list, f"{args.output_path}/It_{i}.png")
    else:
        It = model.reverse_process([img0, img2], args.tau_val)
        save_image(It, f"{args.output_path}/frame2.png")

if __name__ == "__main__":
    main()